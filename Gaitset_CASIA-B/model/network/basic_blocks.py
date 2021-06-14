import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)
class FCONV_4(nn.Module):
    def __init__(self, in_channels, out_channels, p=4):
        super(FCONV_4, self).__init__()
        self.p = p
        self.out_channels = out_channels
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )

    def forward(self, x, batch_frame=None):        
        batch, c, h, w = x.size()
        out_list = []
        x_part = x.view(batch, c, self.p, -1, w).permute(0, 2, 1, 3, 4).contiguous()
        batch, p, c, h, w = x_part.size()
        x_part = x_part.view(-1, c, h, w)
        output = self.conv(x_part).view(batch, p, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        batch, c, p, h, w = output.size()
        output = output.view(batch, c, -1, w)
        return output

class FCONV_8(nn.Module):
    def __init__(self, in_channels, out_channels, p=4):
        super(FCONV_8, self).__init__()
        self.p = p
        self.out_channels = out_channels
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU()
            )

    def forward(self, x, batch_frame=None):        
        batch, c, h, w = x.size()
        out_list = []
        x_part = x.view(batch, c, self.p, -1, w).permute(0, 2, 1, 3, 4).contiguous()
        batch, p, c, h, w = x_part.size()
        x_part = x_part.view(-1, c, h, w)
        output = self.conv(x_part).view(batch, p, -1, h, w).permute(0, 2, 1, 3, 4).contiguous()
        batch, c, p, h, w = output.size()
        output = output.view(batch, c, -1, w)
        return output

class MCM(nn.Module):
    def __init__(self, in_channels, out_channels, p=8, div=4):
        super(MCM, self).__init__()
        self.p = p
        self.div = div
        self.conv_list_1 = []
        self.conv_list_2 = []
        for i in range(self.p):
            conv1 = nn.Sequential(
                nn.Conv1d(in_channels, in_channels // self.div, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels // self.div, in_channels, kernel_size=1, padding=0),
            )
            
            conv_name = 'self.conv1_' + str(i)
            exec("{} = conv1".format(conv_name))
            
            conv2 = nn.Sequential(
                nn.Conv1d(in_channels, in_channels // self.div, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
                nn.Conv1d(in_channels // self.div, in_channels, kernel_size=3, padding=1, stride=1),
            )
            
            conv_name = 'self.conv2_' + str(i)
            exec("{} = conv2".format(conv_name))

        self.max_pool1d_3 = nn.MaxPool1d(kernel_size=3, padding=1, stride=1)
        self.avg_pool1d_3 = nn.AvgPool1d(kernel_size=3, padding=1, stride=1)
        self.max_pool1d_5 = nn.MaxPool1d(kernel_size=5, padding=2, stride=1)
        self.avg_pool1d_5 = nn.AvgPool1d(kernel_size=5, padding=2, stride=1)

    def forward(self, x):
        self.conv_list_1 = [self.conv1_0, self.conv1_1, self.conv1_2, self.conv1_3, self.conv1_4, self.conv1_5, self.conv1_6, self.conv1_7, self.conv1_8, self.conv1_9, self.conv1_10, self.conv1_11, self.conv1_12, self.conv1_13, self.conv1_14, self.conv1_15]
        self.conv_list_2 = [self.conv2_0, self.conv2_1, self.conv2_2, self.conv2_3, self.conv2_4, self.conv2_5, self.conv2_6, self.conv2_7, self.conv2_8, self.conv2_9, self.conv2_10, self.conv2_11, self.conv2_12, self.conv2_13, self.conv2_14, self.conv2_15]
        n, s, c, p = x.size()
        out_list = []
        for i in range(self.p):
            # MTB1
            x_part = x[:,:,:,i].permute(0, 2, 1).contiguous()
            mtb1_attention = self.conv_list_1[i](x_part).sigmoid()
            tmp_out_1 = self.max_pool1d_3(x_part) + self.avg_pool1d_3(x_part)
            out_1 = tmp_out_1 * mtb1_attention
            
            # MTB2
            mtb2_attention = self.conv_list_2[i](x_part).sigmoid()
            tmp_out_2 = self.max_pool1d_5(x_part) + self.avg_pool1d_5(x_part)
            out_2 = tmp_out_2 * mtb2_attention

            out = (out_1 + out_2).max(2)[0].unsqueeze(-1)
            out_list.append(out)

        out = torch.cat(out_list, 2).permute(2, 0, 1).contiguous()
        return out

    
    

class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)
