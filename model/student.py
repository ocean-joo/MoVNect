import torch
import torch.nn as nn
import torchvision
from utils import *

class MoVNect(nn.Module) :
    """
        MoVNect class.
    """
    def __init__(self, num_joints=15) :
        """
            Initialize VNect class
            
            Args:
                num_joints(int) : number of joints to predict
            
            TODO: implement loading pretrained weight part
        """
        super(MoVNect, self).__init__()
        self.num_joints = num_joints
        self.extractor = MobileNetv2_extractor()

        self.Block13_a1 = nn.Sequential(
            ConvBlock(160, 368, 1, "relu"),
            DepthwiseSeparableConv(368, 368, 3),
            ConvBlock(368, 256, 1)
        )
        
        self.Block13_a2 = ConvBlock(160, 256, 1)
        
        self.Block13_b = nn.Sequential(
            ConvBlock(256, 192, 1, "relu"),
            ConvBlock(192, 192, 3, "relu"),
            ConvBlock(192, 192, 1, "relu")
        )
        
        self.Block14_a = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(192, 128, 3, "relu")
        )
        
        self.Block14_b = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            ConvBlock(192, 3*self.num_joints, 3)
        )
        
        self.Block15 = nn.Sequential(
            ConvBlock(128 + 4*self.num_joints, 128, 1, "relu"),
            DepthwiseSeparableConv(128, 128, 3),
            ConvBlock(128, 4*self.num_joints, 1)
        )
        
        
    
    def forward(self, x) :
        """
            Return :
                #keypoint * [heatmap, location map x, location map y, location map z] 
        """
        feature = self.extractor(x) # (#batch, 160, h/32, w/32)
        result_13a1 = self.Block13_a1(feature)
        result_13a2 = self.Block13_a2(feature)
        result_13b = self.Block13_b(torch.add(result_13a1, result_13a2)) 
        # (#batch, 192, h/32, w/32)
        
        result_14a = self.Block14_a(result_13b)
        result_14b = self.Block14_b(result_13b)
        
        delta_x, delta_y, delta_z = result_14b.split(dim=1, split_size=self.num_joints)
        
        bone_length = torch.abs(delta_x) + torch.abs(delta_y) + torch.abs(delta_z)
        
        result_14 = torch.cat((result_14a, result_14b, bone_length), dim=1)
        # (#batch, 128 + 4 * #joints, h/16, h/16)
        
        result = self.Block15(result_14)
        H, X, Y, Z = result.split(dim=1, split_size=self.num_joints)
        return (H, X, Y, Z)
    