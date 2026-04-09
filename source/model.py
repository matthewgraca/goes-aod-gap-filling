import torch
from torch import nn
from torch.nn import init


def init_kaiming(m):
    classname = m.__class__.__name__
    if classname.__contains__('Conv'):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.__contains__('Linear'):
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.__contains__('BatchNorm'):
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(module, init_type):
    if init_type == 'kaiming':
        module.apply(init_kaiming)
    else:
        raise NotImplementedError(f'Initialization {init_type} not implemented.')


class UNETConv(nn.Module):
    def __init__(self, in_size, out_size, batchnorm, n=2, kernel_size=3, stride=1, padding=1):
        super(UNETConv, self).__init__()
        self.n = n
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if batchnorm:
            for i in range(n):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride, padding),
                                     nn.BatchNorm2d(out_size), nn.ReLU(inplace=True))
                setattr(self, f'conv{i}', conv)
                in_size = out_size
        else:
            for i in range(n):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size, stride, padding), nn.ReLU(inplace=True))
                setattr(self, f'conv{i}', conv)
                in_size = out_size

        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, x):
        res = x
        for i in range(self.n):
            res = getattr(self, f'conv{i}')(res)
        return res


class Unet3p(nn.Module):
    def __init__(self, in_channels, batchnorm=True):
        super(Unet3p, self).__init__()
        self.in_channels = in_channels
        self.batchnorm = batchnorm
        channels = [64, 128, 256, 512, 1024]

        # encoder
        self.conv1 = UNETConv(self.in_channels, channels[0], self.batchnorm)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = UNETConv(channels[0], channels[1], self.batchnorm)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = UNETConv(channels[1], channels[2], self.batchnorm)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = UNETConv(channels[2], channels[3], self.batchnorm)
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = UNETConv(channels[3], channels[4], self.batchnorm)

        self.cat_channels = channels[0]
        self.dec_channels = self.cat_channels * 5

        # decoder 4
        self.enc1_dec4_pool = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.enc1_dec4_conv = nn.Conv2d(channels[0], self.cat_channels, kernel_size=3, padding=1)
        self.enc1_dec4_bn = nn.BatchNorm2d(self.cat_channels)
        self.enc1_dec4_relu = nn.ReLU(inplace=True)

        self.enc2_dec4_pool = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.enc2_dec4_conv = nn.Conv2d(channels[1], self.cat_channels, kernel_size=3, padding=1)
        self.enc2_dec4_bn = nn.BatchNorm2d(self.cat_channels)
        self.enc2_dec4_relu = nn.ReLU(inplace=True)

        self.enc3_dec4_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.enc3_dec4_conv = nn.Conv2d(channels[2], self.cat_channels, kernel_size=3, padding=1)
        self.enc3_dec4_bn = nn.BatchNorm2d(self.cat_channels)
        self.enc3_dec4_relu = nn.ReLU(inplace=True)

        self.enc4_dec4_conv = nn.Conv2d(channels[3], self.cat_channels, kernel_size=3, padding=1)
        self.enc4_dec4_bn = nn.BatchNorm2d(self.cat_channels)
        self.enc4_dec4_relu = nn.ReLU(inplace=True)

        self.dec5_dec4_up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec5_dec4_conv = nn.Conv2d(channels[4], self.cat_channels, kernel_size=3, padding=1)
        self.dec5_dec4_bn = nn.BatchNorm2d(self.cat_channels)
        self.dec5_dec4_relu = nn.ReLU(inplace=True)

        self.dec4_conv = nn.Conv2d(self.dec_channels, self.dec_channels, kernel_size=3, padding=1)
        self.dec4_bn = nn.BatchNorm2d(self.dec_channels)
        self.dec4_relu = nn.ReLU(inplace=True)

        # decoder 3
        self.enc1_dec3_pool = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.enc1_dec3_conv = nn.Conv2d(channels[0], self.cat_channels, kernel_size=3, padding=1)
        self.enc1_dec3_bn = nn.BatchNorm2d(self.cat_channels)
        self.enc1_dec3_relu = nn.ReLU(inplace=True)

        self.enc2_dec3_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.enc2_dec3_conv = nn.Conv2d(channels[1], self.cat_channels, kernel_size=3, padding=1)
        self.enc2_dec3_bn = nn.BatchNorm2d(self.cat_channels)
        self.enc2_dec3_relu = nn.ReLU(inplace=True)

        self.enc3_dec3_conv = nn.Conv2d(channels[2], self.cat_channels, kernel_size=3, padding=1)
        self.enc3_dec3_bn = nn.BatchNorm2d(self.cat_channels)
        self.enc3_dec3_relu = nn.ReLU(inplace=True)

        self.dec4_dec3_up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec4_dec3_conv = nn.Conv2d(self.dec_channels, self.cat_channels, kernel_size=3, padding=1)
        self.dec4_dec3_bn = nn.BatchNorm2d(self.cat_channels)
        self.dec4_dec3_relu = nn.ReLU(inplace=True)

        self.dec5_dec3_up = nn.Upsample(scale_factor=4, mode='bilinear')
        self.dec5_dec3_conv = nn.Conv2d(channels[4], self.cat_channels, kernel_size=3, padding=1)
        self.dec5_dec3_bn = nn.BatchNorm2d(self.cat_channels)
        self.dec5_dec3_relu = nn.ReLU(inplace=True)

        self.dec3_conv = nn.Conv2d(self.dec_channels, self.dec_channels, kernel_size=3, padding=1)
        self.dec3_bn = nn.BatchNorm2d(self.dec_channels)
        self.dec3_relu = nn.ReLU(inplace=True)

        # decoder 2
        self.enc1_dec2_pool = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.enc1_dec2_conv = nn.Conv2d(channels[0], self.cat_channels, kernel_size=3, padding=1)
        self.enc1_dec2_bn = nn.BatchNorm2d(self.cat_channels)
        self.enc1_dec2_relu = nn.ReLU(inplace=True)

        self.enc2_dec2_conv = nn.Conv2d(channels[1], self.cat_channels, kernel_size=3, padding=1)
        self.enc2_dec2_bn = nn.BatchNorm2d(self.cat_channels)
        self.enc2_dec2_relu = nn.ReLU(inplace=True)

        self.dec3_dec2_up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec3_dec2_conv = nn.Conv2d(self.dec_channels, self.cat_channels, kernel_size=3, padding=1)
        self.dec3_dec2_bn = nn.BatchNorm2d(self.cat_channels)
        self.dec3_dec2_relu = nn.ReLU(inplace=True)

        self.dec4_dec2_up = nn.Upsample(scale_factor=4, mode='bilinear')
        self.dec4_dec2_conv = nn.Conv2d(self.dec_channels, self.cat_channels, kernel_size=3, padding=1)
        self.dec4_dec2_bn = nn.BatchNorm2d(self.cat_channels)
        self.dec4_dec2_relu = nn.ReLU(inplace=True)

        self.dec5_dec2_up = nn.Upsample(scale_factor=8, mode='bilinear')
        self.dec5_dec2_conv = nn.Conv2d(channels[4], self.cat_channels, kernel_size=3, padding=1)
        self.dec5_dec2_bn = nn.BatchNorm2d(self.cat_channels)
        self.dec5_dec2_relu = nn.ReLU(inplace=True)

        self.dec2_conv = nn.Conv2d(self.dec_channels, self.dec_channels, kernel_size=3, padding=1)
        self.dec2_bn = nn.BatchNorm2d(self.dec_channels)
        self.dec2_relu = nn.ReLU(inplace=True)

        # decoder 1
        self.enc1_dec1_conv = nn.Conv2d(channels[0], self.cat_channels, kernel_size=3, padding=1)
        self.enc1_dec1_bn = nn.BatchNorm2d(self.cat_channels)
        self.enc1_dec1_relu = nn.ReLU(inplace=True)

        self.dec2_dec1_up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dec2_dec1_conv = nn.Conv2d(self.dec_channels, self.cat_channels, kernel_size=3, padding=1)
        self.dec2_dec1_bn = nn.BatchNorm2d(self.cat_channels)
        self.dec2_dec1_relu = nn.ReLU(inplace=True)

        self.dec3_dec1_up = nn.Upsample(scale_factor=4, mode='bilinear')
        self.dec3_dec1_conv = nn.Conv2d(self.dec_channels, self.cat_channels, kernel_size=3, padding=1)
        self.dec3_dec1_bn = nn.BatchNorm2d(self.cat_channels)
        self.dec3_dec1_relu = nn.ReLU(inplace=True)

        self.dec4_dec1_up = nn.Upsample(scale_factor=8, mode='bilinear')
        self.dec4_dec1_conv = nn.Conv2d(self.dec_channels, self.cat_channels, kernel_size=3, padding=1)
        self.dec4_dec1_bn = nn.BatchNorm2d(self.cat_channels)
        self.dec4_dec1_relu = nn.ReLU(inplace=True)

        self.dec5_dec1_up = nn.Upsample(scale_factor=16, mode='bilinear')
        self.dec5_dec1_conv = nn.Conv2d(channels[4], self.cat_channels, kernel_size=3, padding=1)
        self.dec5_dec1_bn = nn.BatchNorm2d(self.cat_channels)
        self.dec5_dec1_relu = nn.ReLU(inplace=True)

        self.dec1_conv = nn.Conv2d(self.dec_channels, self.dec_channels, kernel_size=3, padding=1)
        self.dec1_bn = nn.BatchNorm2d(self.dec_channels)
        self.dec1_relu = nn.ReLU(inplace=True)

        # output
        self.out_conv = nn.Conv2d(self.dec_channels, 1, kernel_size=3, padding=1)

        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        # encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(self.pool1(enc1))
        enc3 = self.conv3(self.pool2(enc2))
        enc4 = self.conv4(self.pool3(enc3))
        enc5 = self.conv5(self.pool4(enc4))

        # decoder 4
        enc1_dec4 = self.enc1_dec4_relu(self.enc1_dec4_bn(self.enc1_dec4_conv(self.enc1_dec4_pool(enc1))))
        enc2_dec4 = self.enc2_dec4_relu(self.enc2_dec4_bn(self.enc2_dec4_conv(self.enc2_dec4_pool(enc2))))
        enc3_dec4 = self.enc3_dec4_relu(self.enc3_dec4_bn(self.enc3_dec4_conv(self.enc3_dec4_pool(enc3))))
        enc4_dec4 = self.enc4_dec4_relu(self.enc4_dec4_bn(self.enc4_dec4_conv(enc4)))
        dec5_dec4 = self.dec5_dec4_relu(self.dec5_dec4_bn(self.dec5_dec4_conv(self.dec5_dec4_up(enc5))))
        dec4 = self.dec4_relu(
            self.dec4_bn(self.dec4_conv(torch.cat([enc1_dec4, enc2_dec4, enc3_dec4, enc4_dec4, dec5_dec4], dim=1))))

        # decoder 3
        enc1_dec3 = self.enc1_dec3_relu(self.enc1_dec3_bn(self.enc1_dec3_conv(self.enc1_dec3_pool(enc1))))
        enc2_dec3 = self.enc2_dec3_relu(self.enc2_dec3_bn(self.enc2_dec3_conv(self.enc2_dec3_pool(enc2))))
        enc3_dec3 = self.enc3_dec3_relu(self.enc3_dec3_bn(self.enc3_dec3_conv(enc3)))
        dec4_dec3 = self.dec4_dec3_relu(self.dec4_dec3_bn(self.dec4_dec3_conv(self.dec4_dec3_up(dec4))))
        dec5_dec3 = self.dec5_dec3_relu(self.dec5_dec3_bn(self.dec5_dec3_conv(self.dec5_dec3_up(enc5))))
        dec3 = self.dec3_relu(
            self.dec3_bn(self.dec3_conv(torch.cat([enc1_dec3, enc2_dec3, enc3_dec3, dec4_dec3, dec5_dec3], dim=1))))

        # decoder 2
        enc1_dec2 = self.enc1_dec2_relu(self.enc1_dec2_bn(self.enc1_dec2_conv(self.enc1_dec2_pool(enc1))))
        enc2_dec2 = self.enc2_dec2_relu(self.enc2_dec2_bn(self.enc2_dec2_conv(enc2)))
        dec3_dec2 = self.dec3_dec2_relu(self.dec3_dec2_bn(self.dec3_dec2_conv(self.dec3_dec2_up(dec3))))
        dec4_dec2 = self.dec4_dec2_relu(self.dec4_dec2_bn(self.dec4_dec2_conv(self.dec4_dec2_up(dec4))))
        dec5_dec2 = self.dec5_dec2_relu(self.dec5_dec2_bn(self.dec5_dec2_conv(self.dec5_dec2_up(enc5))))
        dec2 = self.dec2_relu(
            self.dec2_bn(self.dec2_conv(torch.cat([enc1_dec2, enc2_dec2, dec3_dec2, dec4_dec2, dec5_dec2], dim=1))))

        # decoder 1
        enc1_dec1 = self.enc1_dec1_relu(self.enc1_dec1_bn(self.enc1_dec1_conv(enc1)))
        dec2_dec1 = self.dec2_dec1_relu(self.dec2_dec1_bn(self.dec2_dec1_conv(self.dec2_dec1_up(dec2))))
        dec3_dec1 = self.dec3_dec1_relu(self.dec3_dec1_bn(self.dec3_dec1_conv(self.dec3_dec1_up(dec3))))
        dec4_dec1 = self.dec4_dec1_relu(self.dec4_dec1_bn(self.dec4_dec1_conv(self.dec4_dec1_up(dec4))))
        dec5_dec1 = self.dec5_dec1_relu(self.dec5_dec1_bn(self.dec5_dec1_conv(self.dec5_dec1_up(enc5))))
        dec1 = self.dec1_relu(
            self.dec1_bn(self.dec1_conv(torch.cat([enc1_dec1, dec2_dec1, dec3_dec1, dec4_dec1, dec5_dec1], dim=1))))

        # output
        return self.out_conv(dec1)
