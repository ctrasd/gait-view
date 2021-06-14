import torch
import torch.nn as nn
import numpy as np

from .basic_blocks import SetBlock, BasicConv2d, HPM
from torch.nn import functional as F, ParameterList, ModuleList, Parameter
from torch.nn.parameter import Parameter
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        #self.p = Parameter(torch.ones(1)*p)
        self.p=1
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class SetNet(nn.Module):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None
        
        _in_channels = 1
        _channels = [64,128,256]
        self.set_layer1 = SetBlock(BasicConv2d(_in_channels, _channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_channels[0], _channels[0], 3, padding=1), True)
        self.set_layer3 = SetBlock(BasicConv2d(_channels[0], _channels[1], 3, padding=1))
        self.set_layer4 = SetBlock(BasicConv2d(_channels[1], _channels[1], 3, padding=1), True)
        self.set_layer5 = SetBlock(BasicConv2d(_channels[1], _channels[2], 3, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(_channels[2], _channels[2], 3, padding=1))
        
        self.gl_layer1 = BasicConv2d(_channels[0], _channels[1], 3, padding=1)
        self.gl_layer2 = BasicConv2d(_channels[1], _channels[1], 3, padding=1)
        self.gl_layer3 = BasicConv2d(_channels[1], _channels[2], 3, padding=1)
        self.gl_layer4 = BasicConv2d(_channels[2], _channels[2], 3, padding=1)
        self.gl_pooling = nn.MaxPool2d(2)
        
        bin_level_num=5
        self.bin_num_g = [2**i for i in range(bin_level_num)]
        self.fc_bin_g = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num_g), 256, hidden_dim)))])
        self.bin_num_x = [2**i for i in range(bin_level_num)]
        self.fc_bin_x = nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(sum(self.bin_num_x), 256, hidden_dim)))])
        self.trans_view_g=nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(14,sum(self.bin_num_g), 256, hidden_dim)))])
        self.trans_view_x=nn.ParameterList([
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(14,sum(self.bin_num_x), 256, hidden_dim)))])
        self.gem=GeM()
        self.cls=nn.Linear(in_features=256*2, out_features=14)
        
        
        #self.gl_hpm = HPM(_channels[-1], hidden_dim)
        #self.x_hpm = HPM(_channels[-1], hidden_dim)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)
                    

    def frame_max(self, x):
        if self.batch_frame is None:
            return torch.max(x, 1)
        else:
            _tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i+1], :, :, :], 1)
                for i in range(len(self.batch_frame)-1)
            ]
            max_list = torch.cat([_tmp[i][0] for i in range(len(_tmp))], 0)
            arg_max_list = torch.cat([_tmp[i][1] for i in range(len(_tmp))], 0)
            return max_list, arg_max_list

                
    def forward(self, silho, batch_frame=None):
        #silho = silho/255
        # n: batch_size, s: frame_num, k: keypoints_num, c: channel
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            _ = len(batch_frame)
            for i in range(len(batch_frame)):
                if batch_frame[-(i+1)] != 0:
                    break
                else:
                    _ -= 1
            batch_frame = batch_frame[:_]
            frame_sum = np.sum(batch_frame)
            if frame_sum < silho.size(1):
                silho = silho[:, :frame_sum,:,:]
            self.batch_frame = [0]+np.cumsum(batch_frame).tolist()
        n = silho.size(0)
        x = silho.unsqueeze(2)
        del silho
        
        x = self.set_layer1(x)
        x = self.set_layer2(x)
        gl = self.gl_layer1(self.frame_max(x)[0])
        gl = self.gl_layer2(gl)
        gl = self.gl_pooling(gl)
        
        x = self.set_layer3(x)
        x = self.set_layer4(x)
        gl = self.gl_layer3(gl+self.frame_max(x)[0])
        gl = self.gl_layer4(gl)
        
        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x)[0]
        gl = gl+x
        
        
        x_feat=self.gem(x)
        gl_feat=self.gem(gl)
        x_feat=torch.cat((x_feat,gl_feat),1)
        x_feat = x_feat.view(x_feat.size(0), -1)
        angle_probe=self.cls(x_feat) # n 11
        _,angle= torch.max(angle_probe, 1)
        
        feature_rt=[]
        
        feature = list()
        n, c, h, w = gl.size()
        for num_bin in self.bin_num_g:
            z = gl.view(n, c, num_bin, -1)
            z = z.mean(3)+z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous() # 62,n,128
        feature = feature.matmul(self.fc_bin_g[0]) # 62 n 128 -> 62 n 256
        feature=feature.permute(1, 0, 2).contiguous()  #  n 62 256
        #feature = torch.cat(feature, 2).permute(0, 2, 1).contiguous()  # n 62 128
        for j in range(feature.shape[0]):
            #print(feature[j].shape)
            feature_now=((feature[j].unsqueeze(1)).bmm(self.trans_view_g[0][angle[j]])).squeeze(1) # 62*256
            feature_rt.append(feature_now)
        gl_f = torch.cat([x.unsqueeze(0) for x in feature_rt]) # n 31 256
        
        
        
        feature_rt=[]
        feature = list()
        n, c, h, w = x.size()
        for num_bin in self.bin_num_x:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3)+z.max(3)[0]
            feature.append(z)
        
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        feature = feature.matmul(self.fc_bin_x[0])
        feature=feature.permute(1, 0, 2).contiguous()
        
        #feature = torch.cat(feature, 2).permute(0, 2, 1).contiguous()  # n 62 128
        for j in range(feature.shape[0]):
            #print(feature[j].shape)
            feature_now=((feature[j].unsqueeze(1)).bmm(self.trans_view_x[0][angle[j]])).squeeze(1) # 62*256
            feature_rt.append(feature_now)
        x_f = torch.cat([x.unsqueeze(0) for x in feature_rt]) # n 31 256
        
        
        
        return torch.cat([gl_f, x_f], 1), None,angle_probe