from torchvision import models
from .layers_fusion import *

class AE_Encoder(nn.Module):
    def __init__(self, channel=64):
        super(AE_Encoder, self).__init__()
        # Feature Encoder 中的FEM部分, 提取浅层特征
        self.inception_res_1 = Inception_ResidualBlock(1, channel // 2)
        self.inception_res_2 = Inception_ResidualBlock(channel // 2, channel)
        self.inception_res_3 = Inception_ResidualBlock(channel, channel)

        # Common Feature Extractor 中的FDM部分,将浅层特征分解为模态不变的共同特征和模态特定的独特特征
        # common，集成了空间和通道注意力
        self.CATT_ir = ChannelAttention(channel)
        self.SATT_ir = SpatialAttention()
        self.CATT_vi = ChannelAttention(channel)
        self.SATT_vi = SpatialAttention()

        # unique，由三个卷积层组成
        self.conv0_D1 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv0_D2 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_D1 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)
        self.conv1_D2 = ConvBnLeakyRelu2d(channel, channel, kernel_size=3, padding=1, stride=1)

        # output
        self.conv_fB_ir = nn.Conv2d(channel * 2, channel, 1, bias=False)
        self.conv_fD_ir = nn.Conv2d(channel * 2, channel, 1, bias=False)
        self.conv_fB_vi = nn.Conv2d(channel * 2, channel, 1, bias=False)
        self.conv_fD_vi = nn.Conv2d(channel * 2, channel, 1, bias=False)

    def forward(self, ir, vi):
        # IR的FEM部分, 提取浅层特征
        f0_ir = self.inception_res_1(ir)
        f1_ir = self.inception_res_2(f0_ir)
        f2_ir = self.inception_res_3(f1_ir) # FDM 的输入，用于分解共同和独特特征

        # VI的FEM部分, 提取浅层特征
        f0_vi = self.inception_res_1(vi)
        f1_vi = self.inception_res_2(f0_vi)
        f2_vi = self.inception_res_3(f1_vi)

        fB_ir_CATT = f2_ir * self.CATT_ir(f2_ir)
        fB_ir_SATT = f2_ir * self.SATT_ir(f2_ir)
        fB_ir = self.conv_fB_ir(torch.cat([fB_ir_CATT, fB_ir_SATT], 1))

        fB_vi_CATT = f2_vi * self.CATT_vi(f2_vi)
        fB_vi_SATT = f2_vi * self.SATT_vi(f2_vi)
        fB_vi = self.conv_fB_vi(torch.cat([fB_vi_CATT, fB_vi_SATT], 1))

        fD_ir = self.conv0_D1(self.conv1_D1(f2_ir))
        fD_vi = self.conv0_D2(self.conv1_D2(f2_vi))

        
        # fB_ir共同特征，fB_vi共同特征
        # fD_ir独特特征，fD_vi独特特征
        # f1_ir浅层特征，f1_vi浅层特征
        # f2_ir深层特征，f2_vi深层特征
        # 返回八个特征，用于后续的配准和融合
        # ir浅层特征，ir深层特征，ir共同特征，ir独特特征， vi浅层特征，vi深层特征，vi共同特征，vi独特特征
        return f1_ir, f2_ir, fB_ir, fD_ir, f1_vi, f2_vi, fB_vi, fD_vi
    
