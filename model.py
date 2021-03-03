import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class CAMModel(nn.Module):
    def __init__(self, args):
        super(CAMModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        
        ###################################################
        # Initialize a pretrained ResNet-18 model from torchvision.models
        self.model = torchvision.models.resnet18(pretrained=True)
        #  Initialize the Global Average Pooling (GAP) 
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # Initialize a linear projection layer WCLS 
        self.Wcls = nn.Linear(in_features=128, out_features=10)
        # YOUR CODE HERE
        # INITIALIZE THE MODEL

        ###################################################

    def forward(self, img_L, img_S=None):

        if self.args.mode == 'CAM':
            pass

            ###################################################
            # Feed the input images into the CAM model
            # Backbone
            print(img_L.shape)
            out = self.model.conv1(img_L)
            out = self.model.bn1(out)
            out = self.model.relu(out)
            out = self.model.maxpool(out)
            out = self.model.layer1(out)
            out = self.model.layer2(out)
            self.featuremap = out            
            # GAP
            out = self.gap(out)
            # Wcls
            out = out.view(out.size(0),out.size(1))
            out = self.Wcls(out)
            self.weight = self.Wcls.weight
            feature_maps = self.featuremap
            weight = self.weight
            # print("classification scores")
            # print(out)
            # print("feature maps shape")
            # print(feature_maps.shape)
            # print("feature maps")
            # print(feature_maps)
            # print("weighting coefficients")
            # print(weight.shape)
            return out,{feature_maps,weight}

            # YOUR CODE HERE
            # FORWARD PATH FOR CAM

            ###################################################

        elif self.args.mode == 'SEG':
            pass
            ###################################################

            # Feed the original-size input images IL and their down-scaled copies IS into the same CAM model
            out_L = self.model.conv1(img_L)
            out_L = self.model.bn1(out_L)
            out_L = self.model.relu(out_L)
            out_L = self.model.maxpool(out_L)
            out_L = self.model.layer1(out_L)
            out_L = self.model.layer2(out_L)
            self.featuremap_L = out_L
            # GAP
            out_L = self.gap(out_L)
            # Wcls
            out_L = out_L.view(out_L.size(0),out_L.size(1))
            out_L = self.Wcls(out_L)
            # self.weight = self.Wcls.weight
            feature_maps_L = self.featuremap_L
            weight_L = self.Wcls.weight
            # print("classification scores")
            # print(out_L)
            # print("feature_L maps shape")
            # print(feature_maps_L.shape)
            # print("feature_L maps")
            # print(feature_maps_L)
            # print("weighting coefficients")
            # print(weight_L)



            out_S = self.model.conv1(img_S)
            out_S = self.model.bn1(out_S)
            out_S = self.model.relu(out_S)
            out_S = self.model.maxpool(out_S)
            out_S = self.model.layer1(out_S)
            out_S = self.model.layer2(out_S)
            self.featuremap_S = out_S
            # GAP
            out_S = self.gap(out_S)
            # Wcls
            out_S = out_S.view(out_S.size(0),out_S.size(1))
            out_S = self.Wcls(out_S)
            # self.weight = self.Wcls.weight
            feature_maps_S = self.featuremap_S
            weight_S = self.Wcls.weight
            # print("classification scores")
            # print(out_S)
            # print("feature_S maps shape")
            # print(feature_maps_S.shape)
            # print("feature_S maps")
            # print(feature_maps_S)
            # print("weighting coefficients")
            # print(weight_S)


            return {out_L,out_S}, {feature_maps_L,feature_maps_S,weight_L,weight_S}

            # YOUR CODE HERE
            # FORWARD PATH FOR SEG

            ###################################################

        else:
            NotImplementedError
