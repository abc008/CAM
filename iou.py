import cv2
import numpy as np
import torch

def compute_iou(seg_path_gt, seg_path_cam, seg_path_seg):

    ###################################################
    img_seg_gt_raw = cv2.imread(seg_path_gt)
    img_seg_cam_raw = cv2.imread(seg_path_cam)
    img_seg_seg_raw = cv2.imread(seg_path_seg)
    h,w,deep = img_seg_cam_raw.shape
    img_seg_gt_raw = cv2.resize(img_seg_gt_raw,(h,w),interpolation=cv2.INTER_NEAREST)
    # print(img_seg_gt_raw[112])
    # print(img_seg_gt_raw.shape)
    # print(img_seg_cam_raw.shape)
    # print(img_seg_seg_raw.shape)
    img_seg_gt = np.zeros((h,w))
    img_seg_cam = np.zeros((h,w))
    img_seg_seg = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            # print(img_seg_cam_raw[i][j])
            if (img_seg_gt_raw[i][j]==np.array([255,255,255])).all():
                img_seg_gt[i][j]=1
            if (img_seg_cam_raw[i][j]==np.array([255,255,255])).all():
                img_seg_cam[i][j]=1
            if (img_seg_seg_raw[i][j]==np.array([255,255,255])).all():
                img_seg_seg[i][j]=1
    img_seg_gt = img_seg_gt.flatten()
    img_seg_cam = img_seg_cam.flatten()
    img_seg_seg = img_seg_seg.flatten()

    hist_gt_cam = np.zeros((2,2))
    hist_gt_seg = np.zeros((2,2))
    # 混淆矩阵
    hist_gt_cam = np.bincount(2*img_seg_gt.astype(int)+img_seg_cam.astype(int),minlength=4).reshape(2,2)
    hist_gt_seg = np.bincount(2*img_seg_gt.astype(int)+img_seg_seg.astype(int),minlength=4).reshape(2,2)
    # 计算[背景，object]各自的IOU
    iou_gt_cam = np.diag(hist_gt_cam)/(hist_gt_cam.sum(1)+hist_gt_cam.sum(0)-np.diag(hist_gt_cam))
    iou_CAM = iou_gt_cam[1]
    iou_gt_seg = np.diag(hist_gt_seg)/(hist_gt_seg.sum(1)+hist_gt_seg.sum(0)-np.diag(hist_gt_seg))
    iou_SEG = iou_gt_seg[1]

    # YOUR CODE HERE
    # COMPUTE THE INTERSECTION OVER UNION

    ###################################################

    return iou_CAM, iou_SEG


if __name__ == '__main__':
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    for cls in classes:
        seg_path_gt = './data/test_seg/{}.png'.format(cls)       # ground-truth seg map
        seg_path_cam = './visualize/CAM/{}_seg.png'.format(cls)  # output seg map from CAM
        seg_path_seg = './visualize/SEG/{}_seg.png'.format(cls)  # output seg map from SEG

        iou_CAM, iou_SEG = compute_iou(seg_path_gt, seg_path_cam, seg_path_seg)

        print('Class: {} | CAM IoU: {:.3f} | SEG IoU: {:.3f}'.format(cls, iou_CAM, iou_SEG))
