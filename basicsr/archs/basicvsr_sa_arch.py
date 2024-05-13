import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, flow_warp, make_layer
from .edvr_arch import PCDAlignment, TSAFusion
from .spynet_arch import SpyNet


@ARCH_REGISTRY.register()
class BasicVSRSA(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_block=15, spynet_path=None, agnostic=True):
        super().__init__()
        self.num_feat = num_feat

        # alignment
        self.spynet = SpyNet(spynet_path)

        # propagation
        self.backward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)
        self.forward_trunk = ConvResidualBlocks(num_feat + 3, num_feat, num_block)

        # reconstruction
        self.fusion = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.upsample = Upsample(num_feat, agnostic)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Default scale = 4
        self.scale = 4

    def get_flow(self, x):
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(x_1, x_2).view(b, n - 1, 2, h, w)
        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)

        return flows_forward, flows_backward

    def set_scale(self, scale):
        self.scale = scale

    def forward(self, x):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        flows_forward, flows_backward = self.get_flow(x)
        b, n, _, h, w = x.size()

        # backward branch
        out_l = []
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(n - 1, -1, -1):
            x_i = x[:, i, :, :, :]
            if i < n - 1:
                flow = flows_backward[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.backward_trunk(feat_prop)
            out_l.insert(0, feat_prop)

        # forward branch
        feat_prop = torch.zeros_like(feat_prop)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

            feat_prop = torch.cat([x_i, feat_prop], dim=1)
            feat_prop = self.forward_trunk(feat_prop)

            # fuse layers then upsample
            out = torch.cat([out_l[i], feat_prop], dim=1)
            out = self.lrelu(self.fusion(out))
            out_l[i] = self.upsample(x_i, out, scale=self.scale)

        return torch.stack(out_l, dim=1)


class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)

class Upsample(nn.Module):
    def __init__(self, channels, num_experts=4, bias=False, agnostic=True):
        super(Upsample, self).__init__()
        self.agnostic = agnostic
        if self.agnostic:
            self.bias = bias
            self.num_experts = num_experts
            self.channels = channels

            # experts
            weight_compress = []
            for i in range(num_experts):
                weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
                nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
            self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

            weight_expand = []
            for i in range(num_experts):
                weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
                nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
            self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

            # two FC layers
            self.body = nn.Sequential(
                nn.Conv2d(4, 64, 1, 1, 0, bias=True),
                nn.ReLU(True),
                nn.Conv2d(64, 64, 1, 1, 0, bias=True),
                nn.ReLU(True),
            )
            # routing head
            self.routing = nn.Sequential(
                nn.Conv2d(64, num_experts, 1, 1, 0, bias=True),
                nn.Sigmoid()
            )
            # offset head
            self.offset = nn.Conv2d(64, 2, 1, 1, 0, bias=True)
        else:
            ## reconstruction
            #self.upconv1 = nn.Conv2d(channels, channels * 4, 3, 1, 1, bias=True)
            #self.upconv2 = nn.Conv2d(channels, 64 * 4, 3, 1, 1, bias=True)
            #self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
            #self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)
            #self.pixel_shuffle = nn.PixelShuffle(2)
            #self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            raise Exception("Unimplemented")

    def forward(self, x, fused, scale):
        b, c, h, w = fused.size()
        out = fused

        if self.agnostic:
            # (1) coordinates in LR space
            ## coordinates in HR space
            coor_hr = [torch.arange(0, round(h * scale), 1).unsqueeze(0).float().to(out.device),
                       torch.arange(0, round(w * scale), 1).unsqueeze(0).float().to(out.device)]

            ## coordinates in LR space
            coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
            coor_h = coor_h.permute(1, 0)
            coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

            inp = torch.cat((
                torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
                torch.ones_like(coor_h).expand([-1, round(scale * w)]).unsqueeze(0) / scale,
                coor_h.expand([-1, round(scale * w)]).unsqueeze(0),
                coor_w.expand([round(scale * h), -1]).unsqueeze(0)
            ), 0).unsqueeze(0)

            # (2) predict filters and offsets
            embedding = self.body(inp)
            ## offsets
            offset = self.offset(embedding)

            ## filters
            routing_weights = self.routing(embedding)
            routing_weights = routing_weights.view(self.num_experts, round(scale*h) * round(scale*w)).transpose(0, 1)      # (h*w) * n

            weight_compress = self.weight_compress.view(self.num_experts, -1)
            weight_compress = torch.matmul(routing_weights, weight_compress)
            weight_compress = weight_compress.view(1, round(scale*h), round(scale*w), self.channels//8, self.channels)

            weight_expand = self.weight_expand.view(self.num_experts, -1)
            weight_expand = torch.matmul(routing_weights, weight_expand)
            weight_expand = weight_expand.view(1, round(scale*h), round(scale*w), self.channels, self.channels//8)

            # (3) grid sample & spatially varying filtering
            ## grid sample
            fea0 = grid_sample(out, offset, scale, scale)               ## b * h * w * c * 1
            fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1

            ## spatially varying filtering
            out = torch.matmul(weight_compress.expand([b, -1, -1, -1, -1]), fea)
            out = torch.matmul(weight_expand.expand([b, -1, -1, -1, -1]), out).squeeze(-1)

            return out.permute(0, 3, 1, 2) + fea0
        else:
            #if scale != 4:
            #    raise Exception("Incorrect scale passed to static upsampler")
            #
            ## upsample
            #out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            #out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            #out = self.lrelu(self.conv_hr(out))
            #out = self.conv_last(out)
            #base = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
            #out += base
            raise Exception("Unimplemented")
