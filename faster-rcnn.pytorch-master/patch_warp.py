import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class PatchWarp:
	def __init__(self, ph, param, scale, x, y):
		self.scale = scale  # path size
		self.x = x
		self.y = y
		self.ph = torch.Tensor([[ph]]).cuda()
		self.param = torch.Tensor([[param]]).cuda()

	def tf_integral(self, x, a):
		res = (0.5 * (x * torch.sqrt(x ** 2 + a) + a * torch.log(torch.abs(x + torch.sqrt(x ** 2 + a))))).cuda()
		return res

	def tf_pre_parabol(self, x, par, size):
		x = x - size / 2.
		par = par.cuda()
		prev = 2. * par * (self.tf_integral(torch.abs(x), 0.25 / (par ** 2)) - self.tf_integral(0, 0.25 / (par ** 2)))
		prev = prev.cuda()
		return prev + size / 2.

	def th_gather_nd(self, x, coords):
		x = x.contiguous()
		inds = coords.mv(torch.Tensor(x.stride()).float().cuda())
		x_gather = torch.index_select(x.contiguous().view(-1), 0, inds)
		return x_gather

	def projector(self, logo, size, height):
		tmp_size = size / 2.
		right_cumsum = F.pad(torch.cumsum(logo[:, :, int(tmp_size):], dim=2), (0, 0, 1, 0, 0, 0, 0, 0), 'constant')	
		right_cumsum = right_cumsum.permute(0, 2, 1, 3).cuda()
	

		logo = np.array(logo.detach().cpu())
		left_cumsum = np.pad(np.cumsum(logo[:, :, :int(tmp_size)][:, :, ::-1], axis=2),
		                     ((0, 0), (0, 0), (1, 0), (0, 0)), 'constant')
		left_cumsum = torch.Tensor(left_cumsum).permute(0, 2, 1, 3).cuda()

		tmp_anchors = torch.arange(int(tmp_size), size + 1).float().cuda()

		anchors = torch.clamp(
			self.tf_pre_parabol(tmp_anchors.unsqueeze(0), self.param, size.cuda()) - tmp_size.cuda(), 0,
			tmp_size.cuda()).round().unsqueeze(2)

		anch_inds = torch.arange(self.param.size()[0]).float().cuda().unsqueeze(1).unsqueeze(2).repeat(
			[1, int(tmp_size) + 1, 1])
		new_anchors = torch.cat([anch_inds, anchors], 2).cuda()
		anchors_div = (anchors[:, 1:] - anchors[:, :-1]).clamp(1, size).unsqueeze(3)
		new_anchors = new_anchors.squeeze().transpose(1, 0)

		# right_anchors_cumsum = right_cumsum[new_anchors[0,:].cpu().numpy(), new_anchors[1, :].cpu().numpy()].cuda().unsqueeze(0)
		right_anchors_cumsum = right_cumsum[new_anchors[0,:].long(), new_anchors[1, :].long()].cuda().unsqueeze(0)

		right_anchors_diffs = right_anchors_cumsum[:, 1:] - right_anchors_cumsum[:, :-1]

		right = right_anchors_diffs / anchors_div

		# left_anchors_cumsum = left_cumsum[new_anchors[0,:].cpu().numpy(), new_anchors[1, :].cpu().numpy()].cuda().unsqueeze(0)
		left_anchors_cumsum = left_cumsum[new_anchors[0,:].long(), new_anchors[1, :].long()].cuda().unsqueeze(0)
		left_anchors_diffs = left_anchors_cumsum[:, 1:] - left_anchors_cumsum[:, :-1]
		left = left_anchors_diffs / anchors_div

		tmp_result = (
			torch.cat([torch.Tensor(np.array(np.array(left.detach().cpu())[:, ::-1])), torch.Tensor(np.array(np.array(right.detach().cpu())))],
			          1)).permute(0, 2, 1, 3)

		cumsum = torch.Tensor(
			np.pad(np.cumsum(np.array(tmp_result), axis=1), ((0, 0), (1, 0), (0, 0), (0, 0)), 'constant'))
		
		angle = (torch.tensor(np.pi).cuda() / torch.tensor(180.) * self.ph).unsqueeze(2)

		z = self.param * ((torch.arange(size).float() - 449.5) ** 2).cuda()
		z_tile = z.unsqueeze(1).repeat([1, size + 1, 1])

		y_coord = torch.arange(self.y, self.y + size + 1).float().cuda()
		y_tile = y_coord.unsqueeze(1).unsqueeze(0).repeat([self.param.size()[0], 1, size])
		y_prev = (y_tile + z_tile * torch.sin(-angle)) / torch.cos(angle)
		y_round = torch.clamp(y_prev, 0, float(height)).round()
		y_div = torch.clamp(y_round[:, 1:] - y_round[:, :-1], 1, size)

		x_coord = torch.arange(size).float().cuda()
		x_tile = x_coord.unsqueeze(0).unsqueeze(0).repeat([self.param.size()[0], size + 1, 1])

		b_tmp = torch.arange(self.param.size()[0]).float().cuda()
		b_coord = b_tmp.unsqueeze(1).unsqueeze(2).repeat([1, size + 1, size])

		
		indices = torch.stack([b_coord, y_round, x_tile], dim=3).cuda()

		shape = indices.size()
		indices = indices.view(-1, shape[-1]).transpose(1, 0)

		chosen_cumsum = cumsum[indices[0,:].detach().cpu().numpy(), indices[1, :].detach().cpu().numpy(), indices[2, :].detach().cpu().numpy()].cuda().unsqueeze(0).view(shape)

		chosen_cumsum_diffs = chosen_cumsum[:, 1:] - chosen_cumsum[:, :-1]
		final_results = torch.clamp(chosen_cumsum_diffs / y_div.unsqueeze(3), 0., 1.)
		return final_results

	def spatial_transformer_network(self, input_fmap, theta, out_dims=None):
		B = input_fmap.size()[0]
		H = input_fmap.size()[1]
		W = input_fmap.size()[2]

		theta = theta.view([B, 2, 3])

		# generate grids of same size or upsample/downsample if specified
		if out_dims:
			out_H = out_dims[0]
			out_W = out_dims[1]
			batch_grids = self.affine_grid_generator(out_H, out_W, theta)
		else:
			batch_grids = self.affine_grid_generator(H, W, theta)

		x_s = batch_grids[:, 0, :, :]
		y_s = batch_grids[:, 1, :, :]
		# sample input with grid to get output
		out_fmap = self.bilinear_sampler(input_fmap, x_s, y_s)
		return out_fmap

	def get_pixel_value(self, img, x, y):
		shape = x.size()
		batch_size = shape[0]
		height = shape[1]
		width = shape[2]

		batch_idx = torch.arange(0, batch_size).float().cuda()
		batch_idx = batch_idx.view(batch_size, 1, 1)
		b = batch_idx.repeat(1, height, width)

		indices = torch.stack([b, y, x], 3).cuda()
		shape = indices.size()
		indices = indices.view(-1, shape[-1]).transpose(1, 0)
		chosen_cumsum = img[indices[0,:].detach().cpu().numpy(), indices[1, :].detach().cpu().numpy(), indices[2, :].detach().cpu().numpy()].cuda().unsqueeze(0).view(shape)

		return chosen_cumsum

	def affine_grid_generator(self, height, width, theta):
		num_batch = theta.size()[0]

		# create normalized 2D grid
		x = torch.linspace(-1.0, 1.0, width).cuda()
		y = torch.linspace(-1.0, 1.0, height).cuda()
		x_t, y_t = torch.meshgrid(x, y)

		# flatten
		x_t_flat = x_t.contiguous().view([-1])
		y_t_flat = y_t.contiguous().view([-1])

		ones = torch.ones_like(x_t_flat).cuda()
		sampling_grid = torch.stack([x_t_flat, y_t_flat, ones]).cuda()
		sampling_grid = sampling_grid.unsqueeze(0)
		sampling_grid = sampling_grid.repeat(np.stack([num_batch, 1, 1]).tolist())
		batch_grids = torch.matmul(theta, sampling_grid)
		batch_grids = batch_grids.view([num_batch, 2, height, width])

		return batch_grids

	def bilinear_sampler(self, img, x, y):
		H = img.size()[1]
		W = img.size()[2]

		max_y = torch.tensor(H - 1).cuda()
		max_x = torch.tensor(W - 1).cuda()
		zero = torch.zeros([]).cuda()

		# rescale x and y to [0, W-1/H-1]
		x = 0.5 * ((x + 1.0) * (max_x - 1))
		y = 0.5 * ((y + 1.0) * (max_y - 1))

		# grab 4 nearest corner points for each (x_i, y_i)
		x0 = torch.floor(x)
		x1 = x0 + 1
		y0 = torch.floor(y)
		y1 = y0 + 1

		# clip to range [0, H-1/W-1] to not violate img boundaries
		x0 = torch.clamp(x0, zero, max_x)
		x1 = torch.clamp(x1, zero, max_x)
		y0 = torch.clamp(y0, zero, max_y)
		y1 = torch.clamp(y1, zero, max_y)
		# get pixel value at corner coords
		Ia = self.get_pixel_value(img, x0, y0)
		Ib = self.get_pixel_value(img, x0, y1)
		Ic = self.get_pixel_value(img, x1, y0)
		Id = self.get_pixel_value(img, x1, y1)

		# calculate deltas
		wa = (x1 - x) * (y1 - y)
		wb = (x1 - x) * (y - y0)
		wc = (x - x0) * (y1 - y)
		wd = (x - x0) * (y - y0)

		# add dimension for addition
		wa = wa.unsqueeze(3)
		wb = wb.unsqueeze(3)
		wc = wc.unsqueeze(3)
		wd = wd.unsqueeze(3)

		# compute output
		out = torch.add(torch.add(wa * Ia, wb * Ib), torch.add(wc * Ic, wd * Id))
		return out


	def main(self, image, width, height):
		width = torch.tensor(width)
		height = torch.tensor(height)
		logo_mask = image.permute(1, 2, 0).unsqueeze(0).cuda()
		result = self.projector(logo_mask, size=width, height=height)
		part = width / 2.
		theta = 1. / self.scale * torch.Tensor([[1., 0., - self.x / part, 0., 1., - self.y / part]]).cuda()
		prepared = self.spatial_transformer_network(result, theta)
		patch = prepared[0].squeeze().permute(2, 1, 0)
		patch = patch[:, :height, :width]
		
		return patch


if __name__ == '__main__':
	ph = 30.  # 5.
	param = 0.0001  # 13
	scale = 1.0  # path size
	x = 0.1
	y = -15.  # -15.
	warp = PatchWarp(ph, param, scale, x, y)

	name = 'patch.png'
	img = Image.open(name)
	# width, height = img.size
	# width = torch.tensor(width)
	# height = torch.tensor(height)

	img = transforms.ToTensor()(img).cuda()
	width = img.size()[2]
	height = img.size()[1]
	print(width, height)

	res_patch = warp.main(img, width, height)
	res_patch = transforms.ToPILImage()(res_patch.cpu())

	res_patch.save('patch_warp.png')