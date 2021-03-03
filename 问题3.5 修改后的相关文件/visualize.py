# author: Yicong Hong (yicong.hong@anu.edu.au) for Lab4, 2020, ENGN8536 @ANU

import argparse
from train import resume
from model import CAMModel
import os
import glob
import cv2
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms

''' Refer to CAM visualization code by BoleiZhou (bzhou@csail.mit.edu),
    the author of paper Learning Deep Features for Discriminative Localization,
    see https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py '''


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='exp_SEG')
    parser.add_argument('--mode', type=str, default='SEG', help='CAM or SEG')
    args = parser.parse_args()
    return args

def returnCAM(feature_conv, weights):
    # generate the class activation maps upsample to 224x224
    size_upsample = (224, 224)
    nc, h, w = feature_conv.shape

    weight_softmax = F.softmax(weights, dim=0)
    cam = weight_softmax.unsqueeze(-1)*(feature_conv.reshape((nc, h*w)))
    cam = cam.sum(0).detach().cpu().numpy()

    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam = cv2.resize(cam_img, size_upsample)
    return output_cam

preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# # Transform_S = transforms.Compose([transforms.Resize((112,112)),
# #                                 transforms.ToTensor(),
# #                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
Transform_S = transforms.Compose([transforms.Resize((224,224)),
                                transforms.RandomHorizontalFlip(p=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
if __name__ == '__main__':
    args = parse_args()

    # network
    model = CAMModel(args).to(device)
    model.eval()
    # optimizer (useless)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00, betas=(0.9,0.999))
    # resume the trained model (assume trained CAM/SEG models exist)
    model, optimizer = resume(args, model, optimizer)

    # source images
    img_list = glob.glob('./data/test_seg_source/*')
    for img_path in img_list:
        img = Image.open(img_path)
        img_tensor = preprocess(img).unsqueeze(0).cuda()

        with torch.no_grad():
            if args.mode == 'CAM':
                out_cls_score_L, cam_source = model(img_tensor)
                feature_conv_L, weights_L = cam_source

            elif args.mode == 'SEG':
                out_cls_scores, _ = model(img_tensor,Transform_S(img).unsqueeze(0).cuda())
                out_cls_score_L,_ = out_cls_scores
                feature_conv_L = model.featuremap_L
                # feature_conv_S = F.interpolate(model.featuremap_S,size=(28,28))
                # # 将特征图水平翻转回来
                feature_conv_S = torch.flip(model.featuremap_S,dims=[3])
                feature_conv_L = (0.2*feature_conv_L+0.8*feature_conv_S)
                weights_L = model.Wcls.weight

        # render the CAM and output
        # the location of the predicted class
        out_cls_score_L_soft = F.softmax(out_cls_score_L,dim=1).data.squeeze()
        idx = out_cls_score_L_soft.argmax()
        output_cam = returnCAM(feature_conv_L.squeeze(), weights_L[idx])

        out_img = cv2.imread(img_path)
        out_img = cv2.resize(out_img,(224, 224))
        heatmap = cv2.applyColorMap(output_cam, cv2.COLORMAP_JET)
        result = heatmap * 0.15 + out_img * 0.3
        img_name = img_path.split('/')[-1]

        # threshold the heatmap, use it as the segmentation map
        ret, thresh_img = cv2.threshold(output_cam, 175, 255, cv2.THRESH_BINARY)

        if args.mode == 'CAM':
            cv2.imwrite('./visualize/CAM/'+img_name, result)
            thresh_img_name = img_name.split('.')[0] + '_seg.png'
            cv2.imwrite('./visualize/CAM/'+thresh_img_name, thresh_img)

        elif args.mode == 'SEG':
            cv2.imwrite('./visualize/SEG/'+img_name, result)
            thresh_img_name = img_name.split('.')[0] + '_seg.png'
            cv2.imwrite('./visualize/SEG/'+thresh_img_name, thresh_img)

    else:
        NotImplementedError
