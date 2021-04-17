import numpy as np
import torch
import torch.nn as nn


# 衡量色差/颜色相似性
# https://zh.wikipedia.org/wiki/%E9%A2%9C%E8%89%B2%E5%B7%AE%E5%BC%82
def loss_cs(y_true, y_pred):
    # mse loss
    # mse_loss = nn.MSELoss(y_true, y_pred)
    # perceptual loss
    y_true *= 255 # [-1,1] => [0,255]
    y_pred *= 255 # [-1,1] => [0,255]
    rmean = (y_true[:, :, :, 0] + y_pred[:, :, :, 0]) / 2
    r = y_true[:, :, :, 0] - y_pred[:, :, :, 0]
    g = y_true[:, :, :, 1] - y_pred[:, :, :, 1]
    b = y_true[:, :, :, 2] - y_pred[:, :, :, 2]
    percep_loss = torch.mean(torch.sqrt((((512+rmean)*r*r)/256) + 4*g*g + (((767-rmean)*b*b)/256)))/255.0
    # gen_total_err = 0.8*mse_loss+0.2*percep_loss
    return percep_loss

# 衡量色彩恒常性
# https://www.cnblogs.com/wangyong/p/9141075.html
class loss_cc(nn.Module):

    def __init__(self):
        super(loss_cc, self).__init__()

    def forward(self, x):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


#
class loss_t(nn.Module):
    def __init__(self):
        super(loss_t, self).__init__()

    def forward(self, x, y):
        L2_temp = nn.MSELoss()(x, y)
        L1_temp = nn.L1Loss()(x, y)

        L_total = 0.3 * L1_temp + 0.7 * L2_temp  # + 0.3*Le1_temp + 0.3*Le2_temp
        return L_total