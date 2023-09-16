
import torch.nn as nn
import numpy as np
import torch

def light_filter(img, param, steps, img_max, img_min):
    param=param[:,:,None,None]
    light_curve_sum = torch.sum(param, 4) + 1e-30
    image = img * 0

    # lightness range constraint
    used_tensor = img_max - img_min
    img = img - img_min
    for i in range(steps):
        delta = torch.clamp(img - used_tensor.item() * i / steps, 0, used_tensor.item() / steps) * param[:, :, :, :, i]
        image += delta
    image *= steps / light_curve_sum
    image += img_min
    return image

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]



M = np.array([[0.412453, 0.357580, 0.180423],
              [0.212671, 0.715160, 0.072169],
              [0.019334, 0.119193, 0.950227]])

M_neg = np.linalg.inv(M)
M_neg_t = torch.from_numpy(M_neg)
M_neg_t = M_neg_t.type(torch.FloatTensor)
M_neg_t = M_neg_t.cuda()

M_t = torch.from_numpy(M)
M_t = M_t.type(torch.FloatTensor)
M_t = M_t.cuda()

def f_t(im_tensor):
    return torch.where(im_tensor > 0.008856, im_tensor.pow(1/3), 7.787 * im_tensor + 0.137931)

def anti_f_t(im_tensor):
    return torch.where(im_tensor > 0.206893, im_tensor.pow(3), (im_tensor - 0.137931)/7.787)

def __rgb2xyz__t(rgb_tensor):
    rgb_tensor = rgb_tensor.permute(0, 2, 1)
    XYZ = torch.matmul(M_t, rgb_tensor)

    XYZ = XYZ.permute(1,0,2)
    XYZ = XYZ / 255.0
    a,b,c = torch.split(XYZ, [1,1,1])
    a = a / 0.95047
    b = b / 1.0
    c = c / 1.08883
    ab = torch.cat((a, b), 0)
    abc = torch.cat((ab, c), 0)
    return abc

def __xyz2lab__t(xyz_tensor):
    F_XYZ = f_t(xyz_tensor)
    F_X, F_Y, F_Z = torch.split(F_XYZ, [1, 1, 1])
    x, y, z = torch.split(xyz_tensor, [1, 1, 1])
    L = torch.where(y > 0.008856, 116 * F_Y - 16, 903.3 * y)
    a = 500 * (F_X - F_Y)
    b = 200 * (F_Y - F_Z)
    La_tensor = torch.cat((L,a), 0)
    Lab_tensor = torch.cat((La_tensor, b), 0)
    return Lab_tensor

def RGB2Lab_t(rgb_tensor):
    xyz_tensor = __rgb2xyz__t(rgb_tensor)
    Lab = __xyz2lab__t(xyz_tensor)
    return Lab


def __lab2xyz__t(Lab_tensor):
    L, a, b = torch.split(Lab_tensor, [1, 1, 1])
    fY = (L + 16) / 116
    fX = a/500 + fY
    fZ = fY - b / 200

    x = anti_f_t(fX)
    y = anti_f_t(fY)
    z = anti_f_t(fZ)

    x = x * 0.95047
    y = y * 1.0
    z = z * 1.0883

    xy_tensor = torch.cat((x, y), 0)
    xyz_tensor = torch.cat((xy_tensor, z), 0)
    return  xyz_tensor


def __xyz2rgb_t(xyz_tensor):
    xyz_tensor = xyz_tensor * 255
    xyz_tensor = xyz_tensor.permute(2, 0, 1)
    rgb_tensor = torch.matmul(M_neg_t, xyz_tensor)
    rgb_tensor = rgb_tensor.permute(1, 2, 0)
    rgb_tensor = torch.clamp(rgb_tensor, 0, 255)
    return rgb_tensor


def Lab2RGB_t(lab_tensor):
    xyz_tensor = __lab2xyz__t(lab_tensor)
    rgb_tensor = __xyz2rgb_t(xyz_tensor)
    return rgb_tensor

def update_paras(param_region, lr, batch_size):
    if param_region.grad != None:
        grad_a_region = param_region.grad.clone()
        param_region.data = param_region.data - lr * (grad_a_region.permute(1, 2, 0) / (
                torch.norm(grad_a_region.view(batch_size, -1), dim=1) + 0.00000001)).permute(2, 0, 1)
        param_region.grad.zero_()


