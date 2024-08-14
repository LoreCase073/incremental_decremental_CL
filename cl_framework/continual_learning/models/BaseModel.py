 
import torch
from .movinets.models import MoViNetIncDec
from .movinets.config import _C
from torch import nn
import sys

         
class BaseModel(nn.Module):
    def __init__(self, backbone,dataset):
        super(BaseModel, self).__init__()
        self.backbone_type = backbone
        self.dataset = dataset
 
        if self.backbone_type == 'movinetA0':
            self.backbone = MoViNetIncDec(_C.MODEL.MoViNetA0, causal = False, pretrained = False)
        elif self.backbone_type == 'movinetA1':
            self.backbone = MoViNetIncDec(_C.MODEL.MoViNetA1, causal = False, pretrained = False)
        elif self.backbone_type == 'movinetA2':
            self.backbone = MoViNetIncDec(_C.MODEL.MoViNetA2, causal = False, pretrained = False)
        else:
            sys.exit("Model Not Implemented")

        self.heads = nn.ModuleList()
    
    def get_feat_size(self):
        return self.backbone.feature_space_size

    def add_classification_head(self, n_out):
        if self.backbone_type == "movinetA0" or self.backbone_type == "movinetA1" or self.backbone_type == "movinetA2":
            self.heads.append(
                torch.nn.Sequential(self.backbone.add_head(num_classes=n_out)))
        

    def reset_backbone(self, backbone = None):

        if backbone == "movinetA0":
            self.backbone = MoViNetIncDec(_C.MODEL.MoViNetA0, causal = False, pretrained = False)
        elif backbone == "movinetA1":
            self.backbone = MoViNetIncDec(_C.MODEL.MoViNetA1, causal = False, pretrained = False)
        elif backbone == "movinetA2":
            self.backbone = MoViNetIncDec(_C.MODEL.MoViNetA2, causal = False, pretrained = False)
        

    def forward(self, x):
        results = {}
        features = self.backbone(x)

        if self.backbone_type == 'movinetA0' or self.backbone_type == "movinetA1" or self.backbone_type == "movinetA2":
            for id, head in enumerate(self.heads):
                x = head(features)
                results[id] = x.flatten(1)
        
        return results, features
    
    
    def freeze_all(self):
        """Freeze all parameters from the model, including the heads"""
        for param in self.parameters():
            param.requires_grad = False
    

    def freeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    
    def unfreeze_backbone(self):
        """Freeze all parameters from the main model, but not the heads"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
 
    def freeze_bn(self):
        """Freeze all Batch Normalization layers from the model and use them in eval() mode"""
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.eval()
                for param in m.parameters(): 
                    param.requires_grad=False
               
