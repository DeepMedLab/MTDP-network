import torch.nn as nn
import torch
import torch.nn.functional as F
class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.GroupNorm(16, out_size),
                nn.ReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1),
                nn.BatchNorm2d(out_size),
                nn.ReLU(),
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU()
            )

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        if offset % 2 != 0:
            offset = offset + 1
            inputs1 = F.pad(inputs1, [-1, 0, -1, 0])
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

class SharedEncoder(nn.Module):
    def __init__(
            self, feature_scale=1, is_deconv=True, in_channels=2, is_batchnorm=True):
        super(SharedEncoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self._init_weight()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)#2,64,512,512
        maxpool1 = self.maxpool1(conv1)#2,64,256,256

        conv2 = self.conv2(maxpool1)#2,128,256,256
        maxpool2 = self.maxpool2(conv2)#2,128,128,128

        conv3 = self.conv3(maxpool2)#2,256,128,128
        maxpool3 = self.maxpool3(conv3)#2,256,64,64

        conv4 = self.conv4(maxpool3)#2,512,64,64
        maxpool4 = self.maxpool4(conv4)#2,512,32,32

        center = self.center(maxpool4)#2,1024,32,32

        return conv1,conv2,conv3,conv4,center

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Dose_prediction(nn.Module):
    def __init__(
            self, feature_scale=1, n_classes=1, is_deconv = True, in_channels=6, is_batchnorm=True):
        super(Dose_prediction, self).__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        self.final = nn.Conv2d(filters[0], n_classes, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weight()


    def forward(self, conv1, conv2, conv3, conv4, center):

        up4 = self.up_concat4(conv4, center)  # 2,512,64,64
        up3 = self.up_concat3(conv3, up4)  # 2,256,128,128
        up2 = self.up_concat2(conv2, up3)  # 2,128,256,256
        up1 = self.up_concat1(conv1, up2)  # 2,64,512,512

        final = self.final(up1)  # 2,1,512,512

        out = self.sigmoid(final)  # 2,1,512,512
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Discretization_Dose_prediction(nn.Module):
    def __init__(
            self, feature_scale=1, n_classes=7, is_deconv = True, in_channels=2, is_batchnorm=True):
        super(Discretization_Dose_prediction, self).__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        self.final = nn.Conv2d(filters[0], n_classes, 1)

        self.softmax = nn.Softmax(dim = 1)
        self._init_weight()


    def forward(self, conv1, conv2, conv3, conv4, center):

        up4 = self.up_concat4(conv4, center)  # 2,512,64,64
        up3 = self.up_concat3(conv3, up4)  # 2,256,128,128
        up2 = self.up_concat2(conv2, up3)  # 2,128,256,256
        up1 = self.up_concat1(conv1, up2)  # 2,64,512,512

        final = self.final(up1)  # 2,1,512,512

        # out = self.softmax(final)  # 2,1,512,512
        return final

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class Gradient_regression(nn.Module):
    def __init__(
            self, feature_scale=1, n_classes=1, is_deconv = True, in_channels=6, is_batchnorm=True):
        super(Gradient_regression, self).__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]

        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        self.final = nn.Conv2d(filters[0], n_classes, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weight()


    def forward(self, conv1, conv2, conv3, conv4, center):

        up4 = self.up_concat4(conv4, center)  # 2,512,64,64
        up3 = self.up_concat3(conv3, up4)  # 2,256,128,128
        up2 = self.up_concat2(conv2, up3)  # 2,128,256,256
        up1 = self.up_concat1(conv1, up2)  # 2,64,512,512

        final = self.final(up1)  # 2,1,512,512

        out = self.sigmoid(final)  # 2,1,512,512

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

