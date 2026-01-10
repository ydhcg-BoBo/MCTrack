from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import cv2

# 生成一个随机的二维矩阵
import torch
from torch import nn
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()


        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]
            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes,
                    kernel_size=1, stride=1, padding=0, bias=True)
              conv = nn.Conv2d(last_channel, head_conv[0],
                               kernel_size=head_kernel,
                               padding=head_kernel // 2, bias=True)
              convs = [conv]

              for k in range(1, len(head_conv)):
                  convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k],
                               kernel_size=1, bias=True))
              if len(convs) == 1:
                fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              elif len(convs) == 2:
                fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True),
                  convs[1], nn.ReLU(inplace=True), out)
              elif len(convs) == 3:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True),
                    convs[1], nn.ReLU(inplace=True),
                    convs[2], nn.ReLU(inplace=True), out)
              elif len(convs) == 4:
                fc = nn.Sequential(
                    convs[0], nn.ReLU(inplace=True),
                    convs[1], nn.ReLU(inplace=True),
                    convs[2], nn.ReLU(inplace=True),
                    convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes,
                  kernel_size=1, stride=1, padding=0, bias=True)

              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def img2feats(self, x_rgb, x_t):
      raise NotImplementedError

    def imgpre2feats(self, x_rgb, x_t, pre_img_rgb=None, pre_img_t=None, pre_hm=None, img_id=None):
      raise NotImplementedError

    def forward(self,  x_rgb, x_t, pre_img_rgb=None, pre_img_t=None, pre_hm=None, img_id=None):

    # image_size [3,544,960]
      if (pre_hm is not None) or (pre_img_rgb is not None) or (pre_img_t is not None):
        feats = self.imgpre2feats(x_rgb, x_t, pre_img_rgb, pre_img_t, pre_hm)
      else:
        feats = self.img2feats(x_rgb, x_t)
      out = []

      # x_rgb_show = x_rgb[0].cpu().numpy()
      # plt.imshow(x_rgb_show.transpose((1, 2, 0)))
      # plt.show()
      # Heat = feats[0][0].cpu().numpy().sum(axis=0)
      # Heat = (Heat - np.min(Heat)) / (np.max(Heat) - np.min(Heat))
      # Heat = cv2.resize(Heat, (960,544))
      # plt.imshow(Heat, interpolation='nearest')  # 添加颜色条
      # plt.show()

      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)

      else:
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
      return out
