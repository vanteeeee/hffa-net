import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from .FFA import FFA

class DehazeFFA(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, padding_type='reflect'):
        super(DehazeFFA, self).__init__()

        ###### downsample
        self.ffanet=FFA(gps=3,blocks=19)
        #ckp=torch.load('trained/its_train_ffa_3_19.pk',map_location='cuda')
        #self.ffanet=nn.DataParallel(self.ffanet)
        #self.ffanet.load_state_dict(ckp['model'])
        

        ###### FFA blocks
        self.conv768=nn.Conv2d(64, 768, 1, 1,0)
        

    def forward(self, input):

        #print(input.tpye)
        dehazed, prior = self.ffanet(input)
        g = self.conv768(prior)
        g=F.adaptive_avg_pool2d(g, 1)

        return dehazed, g