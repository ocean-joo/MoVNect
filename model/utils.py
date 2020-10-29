import torch
import torch.nn as nn
import torchvision

def ConvBlock(in_features, out_features, kernel, activation=True) :
    pad = 0 if kernel == 1 else 1
    
    if activation == "relu" :
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
    else :
        return nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=kernel, padding=pad),
            nn.BatchNorm2d(out_features),
        )
    
    
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels * kernel_size, kernel_size=kernel_size, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels * kernel_size, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
    
class MobileNetv2_extractor(nn.Module) :
    """
        Class that loads pretrained MobileNetv2 from torchvision 
              and returns feature map from Block12.
    """
    
    def __init__(self, finetune=False) :
        super(MobileNetv2_extractor, self).__init__()

        original_model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=not finetune)
        self.extractor = torch.nn.Sequential(*list(original_model.children())[0][:15])
        
        for param in self.extractor.parameters() :
            param.requires_grad = finetune
    
    def forward(self, x) :
        return self.extractor(x)

    
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
