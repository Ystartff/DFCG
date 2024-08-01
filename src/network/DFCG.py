
import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform
import numpy as np
from torch.nn.modules import activation

from lib.gcn_lib import Grapher as GCB



class FrequencySeparator:
    def __init__(self, fs, high_freq_threshold=30, low_freq_threshold=5):
        self.fs = fs
        self.high_freq_threshold = high_freq_threshold
        self.low_freq_threshold = low_freq_threshold

    def separate(self, signal):
        signal = signal.to(signal.device)

        fft_result = torch.fft.fft(signal)
        freqs = torch.fft.fftfreq(len(fft_result), 1 / self.fs)

        high_freq_mask = torch.abs(freqs) > self.high_freq_threshold
        low_freq_mask = torch.abs(freqs) < self.low_freq_threshold

        high_freq_data = fft_result.clone()
        high_freq_data[low_freq_mask] = 0
        low_freq_data = fft_result.clone()
        low_freq_data[high_freq_mask] = 0

        high_freq_signal = torch.fft.ifft(high_freq_data)
        low_freq_signal = torch.fft.ifft(low_freq_data)

        high_freq_signal = high_freq_signal.to(torch.float32)
        low_freq_signal = low_freq_signal.to(torch.float32)
        return high_freq_signal, low_freq_signal


class FFParser(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.complex_weight = None

    def forward(self, x, spatial_size=None):
        B, C, H, W = x.shape
        assert H == W, "height and width are not equal"

        if spatial_size is None:
            a = b = H
        else:
            a, b = spatial_size

        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(2, 3), norm='ortho')
        b, c, h, w = x.shape
        if self.complex_weight is None:
            self.complex_weight = nn.Parameter(
                torch.randn(self.dim, h, w, 2, dtype=torch.float32, device=x.device) * 0.02)

        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm='ortho')
        x = x.reshape(B, C, H, W)

        return x


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


class MDAG(nn.Module):
    """
    Multi-scale attention gate
    Arxiv: https://arxiv.org/abs/2210.13012
    """

    def __init__(self, channel):
        super(MDAG, self).__init__()
        self.channel = channel
        self.pointwiseConv = Residual(nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),

        ))
        self.ordinaryConv = Residual(nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),

        ))
        self.dilationConv = Residual(nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),

        ))

        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=(1, 1)),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

        self.CBAM = CBAMLayer(self.channel, self.channel)

        self.freq_separator = FrequencySeparator(fs=2000)

    def forward(self, x):
        xcm = self.CBAM(x)
        high_freq, low_freq = self.freq_separator.separate(x)
        x1_high = self.pointwiseConv(high_freq)
        x1_low = self.pointwiseConv(low_freq)
        x2_high = self.ordinaryConv(high_freq)
        x2_low = self.ordinaryConv(low_freq)
        x3_high = self.dilationConv(high_freq)
        x3_low = self.dilationConv(low_freq)
        x1 = x1_high + x1_low
        x1 = torch.add(x1, xcm)
        x2 = x2_high + x2_low
        x2 = torch.add(x2, xcm)
        x3 = x3_high + x3_low
        x3 = torch.add(x3, xcm)
        _x = self.relu(torch.cat((x1, x2, x3), dim=1))
        _x = self.relu(x1)
        _x = self.voteConv(_x)
        x = x + x * _x
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class FCMxierBlock(nn.Module):
    def __init__(self, dim=1024, depth=7, k=7):  # 1，3
        super(FCMxierBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim * 4),
                nn.Conv2d(dim * 4, dim, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]
        )
        self.up = conv_block(dim, dim)
        self.FFP = FFParser(dim)

    def forward(self, x):
#        x = self.FFP(x)
        x = self.block(x)
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x


def Dropout(x, p=0.3):
    x = torch.nn.functional.dropout(x, p)
    return x


def FeatureDropout(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    max_val, _ = torch.max(attention.view(
        x.size(0), -1), dim=1, keepdim=True)
    threshold = max_val * np.random.uniform(0.7, 0.9)
    threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
    drop_mask = (attention < threshold).float()
    x = x.mul(drop_mask)
    return x



class SPA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class GCD(nn.Module):
    def __init__(self, dim_mult=4, with_masg=True, channels=[1024, 512, 256, 128, 64], drop_path_rate=0.0, img_size=256, k=11, padding=5, conv='mr',
                 gcb_act='gelu', activation='relu'):
        super(GCD, self).__init__()

        #  Graph convolution block (GCB) parameters
        self.padding = padding
        self.k = k  # neighbor num (default:9)
        self.conv = conv  # graph conv layer {edge, mr, sage, gin} # default mr
        self.gcb_act = gcb_act  # activation layer for graph convolution block {relu, prelu, leakyrelu, gelu, hswish}
        self.gcb_norm = 'batch'  # batch or instance normalization for graph convolution block {batch, instance}
        self.bias = True  # bias of conv layer True or False
        self.dropout = 0.0  # dropout rate
        self.use_dilation = True  # use dilated knn or not
        self.epsilon = 0.2  # stochastic epsilon for gcn
        self.use_stochastic = False  # stochastic for gcn, True or False
        self.drop_path = drop_path_rate
        self.reduce_ratios = [1, 1, 4, 2, 1]
        self.dpr = [self.drop_path, self.drop_path, self.drop_path, self.drop_path, self.drop_path]  # stochastic depth decay rule
        self.num_knn = [self.k, self.k, self.k, self.k, self.k]  # number of knn's k
        self.max_dilation = 18 // max(self.num_knn)
        self.HW = img_size // 4 * img_size // 4

        self.gcb5 = nn.Sequential(
            GCB(channels[0], self.num_knn[0], min(0 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[0], n=self.HW // (4 * 4 * 4),
                drop_path=self.dpr[0],
                relative_pos=True, padding=self.padding),
            )

        self.gcb4 = nn.Sequential(
            GCB(channels[1], self.num_knn[1], min(3 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[1], n=self.HW // (4 * 4),
                drop_path=self.dpr[1],
                relative_pos=True, padding=self.padding),
            )

        self.gcb3 = nn.Sequential(
            GCB(channels[2], self.num_knn[2], min(8 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[2], n=self.HW // (4),
                drop_path=self.dpr[2],
                relative_pos=True, padding=self.padding),
            )

        self.gcb2 = nn.Sequential(
            GCB(channels[3], self.num_knn[3], min(11 // 4 + 1, self.max_dilation), self.conv, self.gcb_act,
                self.gcb_norm,
                self.bias, self.use_stochastic, self.epsilon, self.reduce_ratios[3], n=self.HW, drop_path=self.dpr[3],
                relative_pos=True, padding=self.padding),
            )



        self.with_masg = with_masg
        self.Up5 = up_conv(ch_in=256 * dim_mult, ch_out=128 * dim_mult)
        self.Up_conv5 = conv_block(ch_in=128 * 2 * dim_mult, ch_out=128 * dim_mult)
        self.Up4 = up_conv(ch_in=128 * dim_mult, ch_out=64 * dim_mult)
        self.Up_conv4 = conv_block(ch_in=64 * 2 * dim_mult, ch_out=64 * dim_mult)
        self.Up3 = up_conv(ch_in=64 * dim_mult, ch_out=32 * dim_mult)
        self.Up_conv3 = conv_block(ch_in=32 * 2 * dim_mult, ch_out=32 * dim_mult)
        self.Up2 = up_conv(ch_in=32 * dim_mult, ch_out=16 * dim_mult)
        self.Up_conv2 = conv_block(ch_in=16 * 2 * dim_mult, ch_out=16 * dim_mult)
        self.Conv_1x1 = nn.Conv2d(16 * dim_mult, 1, kernel_size=1, stride=1, padding=0)

        self.MDAG4 = MDAG(128 * dim_mult)
        self.MDAG3 = MDAG(64 * dim_mult)
        self.MDAG2 = MDAG(32 * dim_mult)
        self.MDAG1 = MDAG(16 * dim_mult)
        self.spa = SPA()

    def forward(self, x):
        x1, x2, x3, x4, x5 = x
        a = x5
        if self.with_masg:
            x4 = self.MDAG4(x4)
            x3 = self.MDAG3(x3)
            x2 = self.MDAG2(x2)
            x1 = self.MDAG1(x1)

        d6 = a
        d5 = self.gcb5(d6)
        d5 = self.spa(d5) * d5
        d5 = self.Up5(d5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.gcb4(d5)
        d4 = self.spa(d4) * d4
        d4 = self.Up4(d4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.gcb3(d4)
        d3 = self.spa(d3) * d3
        d3 = self.Up3(d3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.gcb2(d3)
        d2 = self.spa(d2) * d2
        d2 = self.Up2(d2)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)

        return d1



class Decoder(nn.Module):

    def __init__(self, dim_mult=4, with_masg=True):
        super(Decoder, self).__init__()
        self.with_masg = with_masg
        self.Up5 = up_conv(ch_in=256 * dim_mult, ch_out=128 * dim_mult)
        self.Up_conv5 = conv_block(ch_in=128 * 2 * dim_mult, ch_out=128 * dim_mult)
        self.Up4 = up_conv(ch_in=128 * dim_mult, ch_out=64 * dim_mult)
        self.Up_conv4 = conv_block(ch_in=64 * 2 * dim_mult, ch_out=64 * dim_mult)
        self.Up3 = up_conv(ch_in=64 * dim_mult, ch_out=32 * dim_mult)
        self.Up_conv3 = conv_block(ch_in=32 * 2 * dim_mult, ch_out=32 * dim_mult)
        self.Up2 = up_conv(ch_in=32 * dim_mult, ch_out=16 * dim_mult)
        self.Up_conv2 = conv_block(ch_in=16 * 2 * dim_mult, ch_out=16 * dim_mult)
        self.Conv_1x1 = nn.Conv2d(16 * dim_mult, 1, kernel_size=1, stride=1, padding=0)

        self.MDAG4 = MDAG(128 * dim_mult)
        self.MDAG3 = MDAG(64 * dim_mult)
        self.MDAG2 = MDAG(32 * dim_mult)
        self.MDAG1 = MDAG(16 * dim_mult)

    def forward(self, feature):
        x1, x2, x3, x4, x5 = feature
        a = x5

        if self.with_masg:
            x4 = self.MDAG4(x4)
            x3 = self.MDAG3(x3)
            x2 = self.MDAG2(x2)
            x1 = self.MDAG1(x1)

        d6 = a
        d5 = self.Up5(d6)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1


class DFCG(nn.Module):
    def __init__(self, img_ch=3, length=(3, 3, 3), k=7, dim_mult=4):
        """
        Multi-Level Global Context Cross Consistency Model
        Args:
            img_ch : input channel.
            output_ch: output channel.
            length: number of FCMxierBlock layers
            k: kernal size of FCMxierBlock

        """
        super(DFCG, self).__init__()

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=16 * dim_mult)
        self.Conv2 = conv_block(ch_in=16 * dim_mult, ch_out=32 * dim_mult)
        self.Conv3 = conv_block(ch_in=32 * dim_mult, ch_out=64 * dim_mult)
        self.Conv4 = conv_block(ch_in=64 * dim_mult, ch_out=128 * dim_mult)
        self.Conv5 = conv_block(ch_in=128 * dim_mult, ch_out=256 * dim_mult)


        self.FCMxierBlock1 = FCMxierBlock(dim=256 * dim_mult, depth=length[0], k=k)
        self.FCMxierBlock2 = FCMxierBlock(dim=256 * dim_mult, depth=length[1], k=k)
        self.FCMxierBlock3 = FCMxierBlock(dim=256 * dim_mult, depth=length[2], k=k)

        # main Decoder
        # self.main_decoder = Decoder(dim_mult=dim_mult, with_masg=True)
        self.main_decoder = GCD(dim_mult=dim_mult, with_masg=True, channels=[1024, 512, 256, 128, 64], drop_path_rate=0.0, img_size=256, k=11, padding=5, conv='mr',
                 gcb_act='gelu', activation='relu')

        # aux Decoder
        self.aux_decoder1 = Decoder(dim_mult=dim_mult, with_masg=True)
        self.aux_decoder2 = Decoder(dim_mult=dim_mult, with_masg=True)
        self.aux_decoder3 = Decoder(dim_mult=dim_mult, with_masg=True)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        if not self.training:
            x5 = self.FCMxierBlock1(x5)
            x5 = self.FCMxierBlock2(x5)
            x5 = self.FCMxierBlock3(x5)
            feature = [x1, x2, x3, x4, x5]
            main_seg = self.main_decoder(feature)
            return main_seg

        feature = [x1, x2, x3, x4, x5]
        aux_seg1 = self.aux_decoder1(feature)

        x5 = self.FCMxierBlock1(x5)
        feature = [x1, x2, x3, x4, x5]
        aux_seg2 = self.aux_decoder2(feature)

        x5 = self.FCMxierBlock2(x5)
        feature = [x1, x2, x3, x4, x5]
        aux_seg3 = self.aux_decoder3(feature)

        x5 = self.FCMxierBlock3(x5)
        feature = [x1, x2, x3, x4, x5]
        main_seg = self.main_decoder(feature)
        return main_seg, aux_seg1, aux_seg2, aux_seg3


