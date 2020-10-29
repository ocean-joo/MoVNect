import torch
import torch.nn as nn
import torchvision


class VNect(nn.Module) :
    """
        VNect class.
    """
    def __init__(self, num_joints=17) :
        """
            Initialize VNect class
            
            Args:
                num_joints(int) : number of joints to predict
            
            TODO: implement loading pretrained weight part
        """
        super(VNect, self).__init__()
        self.num_joints = num_joints
        
        self.stage1 = resnet50_extractor()
        self.stage2a = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 1024, kernel_size=(1, 1)),
            nn.BatchNorm2d(1024)
        )
        self.stage2b = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=(1, 1)),
            nn.BatchNorm2d(1024)
        )
        
        self.stage3 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self.stage4a = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        
        self.stage4b = nn.Sequential(
            nn.ConvTranspose2d(256, 3*self.num_joints, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(3*self.num_joints),
        )
        
        self.stage5 = nn.Sequential(
            nn.Conv2d(128 + 4*self.num_joints, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 4*self.num_joints, kernel_size=(1, 1)),
            nn.BatchNorm2d(4*self.num_joints)
        )

    
    def forward(self, x) :
        """
            Return :
                #keypoint * [heatmap, location map x, location map y, location map z] 
        """
        feature = self.stage1(x) # (#batch, 1024, h/16, w/16)
        result_stage2a = self.stage2a(feature)
        result_stage2b = self.stage2b(feature)
        result_stage3 = self.stage3(torch.add(result_stage2a, result_stage2b)) 
        # (#batch, 256, h/16, w/16)
        
        result_stage4a = self.stage4a(result_stage3)
        result_stage4b = self.stage4b(result_stage3)
        
        delta_x, delta_y, delta_z = result_stage4b.split(dim=1, split_size=self.num_joints)
        
        bone_length_sqaure = self.hadamard(delta_x, delta_x) + self.hadamard(delta_y, delta_y) + self.hadamard(delta_z, delta_z)
        bone_length = torch.sqrt(bone_length_sqaure)
        
        result_stage4 = torch.cat((result_stage4b, result_stage4a, bone_length), dim=1)
        # (#batch, 128 + 4 * #joints, h/8, h/8)
        
        result_stage5 = self.stage5(result_stage4)
        H, X, Y, Z = result_stage5.split(dim=1, split_size=self.num_joints)
        return (H, X, Y, Z)
    
    def hadamard(self, x, y) :
        """
            returns element-wise multiplication of given matrix.
        """
        assert x.shape == y.shape, "{0}.shape({2}) != {1}.shape({3})".format(x, y, x.shape, y.shape)
        return x * y 
    
    
class resnet50_extractor(nn.Module) :
    """
        Class that loads pretrained resnet50 from torchvision 
              and returns feature map from res4f layer.
    """
    
    def __init__(self, finetune=False) :
        super(resnet50_extractor, self).__init__()

        original_model = torchvision.models.resnet50(pretrained=not finetune)
        self.extractor = torch.nn.Sequential(*list(original_model.children())[:-3])
        
        for param in self.extractor.parameters() :
            param.requires_grad = finetune
    
    def forward(self, x) :
        return self.extractor(x)

