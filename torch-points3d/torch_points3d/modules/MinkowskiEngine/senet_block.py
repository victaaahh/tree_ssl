# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import torch
import torch.nn as nn

import MinkowskiEngine as ME
from torch.cuda.amp import custom_fwd

from .resnet_block import BasicBlock, Bottleneck


class SELayer(nn.Module):

    def __init__(self, channel, act_fn, reduction=16, dimension=-1):
        # Global coords does not require coords_key
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            ME.MinkowskiLinear(channel, channel // reduction),
            act_fn,
            ME.MinkowskiLinear(channel // reduction, channel),
            ME.MinkowskiSigmoid())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.broadcast_mul = ME.MinkowskiBroadcastMultiplication()

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        y = self.pooling(x)
        y = self.fc(y)
        return self.broadcast_mul(x, y)


class SEBasicBlock(BasicBlock):

    def __init__(self,
                 inplanes,
                 planes,
                 act_fn,
                 norm_layer,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 reduction=16,
                 drop_path=0.0,
                 bias: bool = True,
                 dimension=-1):
        super(SEBasicBlock, self).__init__(
            inplanes,
            planes,
            act_fn,
            norm_layer,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            drop_path=drop_path,
            bias=bias,
            dimension=dimension)
        self.se = SELayer(planes, act_fn, reduction=reduction, dimension=dimension)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)

        residual = self.downsample(residual)

        out = self.drop_path(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):

    def __init__(self,
                 inplanes,
                 planes,
                 act_fn,
                 norm_layer,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 dimension=-1,
                 drop_path=0.0,
                 bias: bool = True,
                 reduction=16):
        super(SEBottleneck, self).__init__(
            inplanes,
            planes,
            act_fn,
            norm_layer,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            drop_path=drop_path,
            bias=bias,
            dimension=dimension)
        self.se = SELayer(planes * self.expansion, act_fn, reduction=reduction, dimension=dimension)

    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.se(out)

        residual = self.downsample(residual)

        out = self.drop_path(out) + residual
        out = self.relu(out)

        return out