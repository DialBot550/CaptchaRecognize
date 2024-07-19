import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class BidirectionalLSTM(nn.Module):
 
    def __init__(self, nIn, nHidden, nOut):
        '''
        param
            nIn 输入向量维度
            nHidden 隐层数量
            nOut 输出向量维度        
        '''
        super(BidirectionalLSTM, self).__init__()
 
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut) # 这里的线性变换是为了降低维度，好像是不希望因为双向LSTM导致维度升高一倍？
 
    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h) # 为什么把序列长度和批量大小两个维度合在一起？
 
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1) # 又把序列长度和批量大小解开，那么看来是为了线性变换能够顺利进行才合在一起的。但是合在一起后线性变换的结果还正确吗？
        return output

def conv3x3(nIn, nOut, stride=1):
    '''
    param
        nIn 输入图像通道数
        nOut 输出图像通道数
    '''
    # "3x3 convolution with padding"
    return nn.Conv2d( nIn, nOut, kernel_size=3, stride=stride, padding=1, bias=False )

class basic_res_block(nn.Module):
 
    def __init__(self, nIn, nOut, stride=1, downsample=None):
        super( basic_res_block, self ).__init__()
        m = OrderedDict()
        m['conv1'] = conv3x3( nIn, nOut, stride )
        m['bn1'] = nn.BatchNorm2d( nOut )
        m['relu1'] = nn.ReLU( inplace=True )
        m['conv2'] = conv3x3( nOut, nOut )
        m['bn2'] = nn.BatchNorm2d( nOut )
        self.group1 = nn.Sequential( m )
 
        self.relu = nn.Sequential( nn.ReLU( inplace=True ) )
        self.downsample = downsample
 
    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample( x )
        else:
            residual = x
        out = self.group1( x ) + residual # 这里不会因为形状不匹配导致问题吗？
        out = self.relu( out )
        return out
 
 
class CRNN_res(nn.Module):
 
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN_res, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
 
        self.conv1 = nn.Conv2d(nc, 64, 3, 1, 1)
        self.relu1 = nn.ReLU(True)
        self.res1 = basic_res_block(64, 64)
        # 3x64x200
 
        down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(128))
        self.res2_1 = basic_res_block( 64, 128, 2, down1 )
        self.res2_2 = basic_res_block(128,128)
        # 64x32x100
 
        down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(256))
        self.res3_1 = basic_res_block(128, 256, 2, down2)
        self.res3_2 = basic_res_block(256, 256)
        self.res3_3 = basic_res_block(256, 256)
        # 128x16x50
 
        down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, stride=(2, 1), bias=False),nn.BatchNorm2d(512))
        self.res4_1 = basic_res_block(256, 512, (2, 1), down3)
        self.res4_2 = basic_res_block(512, 512)
        self.res4_3 = basic_res_block(512, 512)
        # 256x8x50 在用于下采样的conv2d处，指定了宽度方向上1的步长，所以在这个维度上不会被下采样

        down4 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=1, stride=(2, 1), bias=False),nn.BatchNorm2d(1024))
        self.res5_1 = basic_res_block(512, 1024, (2, 1), down4)
        self.res5_2 = basic_res_block(1024, 1024)
        self.res5_3 = basic_res_block(1024, 1024)
        # 512x4x50
 
        self.pool = nn.AvgPool2d((2, 2), (2, 1), (0, 1))
        # 1024x2x51
 
        self.conv5 = nn.Conv2d(1024, 1024, 2, 1, 0)
        self.bn5 = nn.BatchNorm2d(1024)
        self.relu5 = nn.ReLU(True)
        # 1024x1x50
 
        self.rnn = nn.Sequential(
            BidirectionalLSTM(1024, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
 
    def forward(self, input):
        # conv features
        x = self.res1(self.relu1(self.conv1(input)))
        x = self.res2_2(self.res2_1(x))
        x = self.res3_3(self.res3_2(self.res3_1(x)))
        x = self.res4_3(self.res4_2(self.res4_1(x)))
        x = self.res5_3(self.res5_2(self.res5_1(x)))
        x = self.pool(x)
        conv = self.relu5(self.bn5(self.conv5(x)))
        # print(conv.size())
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
 
        # rnn features
        output = self.rnn(conv)
 
        return output
    
if __name__ == '__main__':
    model = CRNN_res(imgH=64,nc=3,nclass=27,nh=100)
    # print(model)
    from torchinfo import summary
    summary(model,(64,3,64,200))