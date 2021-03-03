import argparse
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import tensorboard

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################################
class CEloss(nn.Module):
    def __init__(self):
        super(CEloss,self).__init__()
        self.cerition = nn.CrossEntropyLoss()
    def forward(self, outputs, targets):
        loss = self.cerition(outputs,targets)
        return loss
CEloss = CEloss()

def returnCAM(feature,weight):
    nc,h,w = feature.shape
    weight = weight.cpu().detach().numpy()
    feature = feature.cpu().detach().numpy()
    cam = weight.dot(feature.reshape(nc,h*w))
    cam = torch.FloatTensor(cam.reshape(h,w)).to(device)
    return cam

        
# YOUR CODE HERE
# DEFINE THE LOSS FUNCTIONS

###################################################

writer = tensorboard.SummaryWriter(log_dir='log')

def train(args, model, optimizer, dataloaders):
    trainloader, testloader = dataloaders

    best_testing_accuracy = 0.0

    # training
    print('Network training starts ...')
    global_step = 0
    test_step = 0
    for epoch in range(args.epochs):
        model.train()

        batch_time = time.time(); iter_time = time.time()

        for i, data in enumerate(trainloader):

            img_L = data['img_L']; img_S = data['img_S']; labels = data['label']
            img_L, img_S, labels = img_L.to(device), img_S.to(device), labels.to(device)

            if args.mode == 'CAM':
                pass
                ###################################################
                # outputs = model(img_L)
                outputs, _ = model(img_L, img_S=None)
                loss = CEloss(outputs,labels)

                # YOUR CODE HERE
                # INPUT TO CAM MODEL AND COMPUTE THE LOSS

                ###################################################

            elif args.mode == 'SEG':
                pass
                ###################################################

                outputs, cam_source = model(img_L, img_S)
                # print(cam_source(3))
                outputs_L, outputs_S = outputs
                loss_CLS_L = CEloss(outputs_L,labels)
                loss_CLS_S = CEloss(outputs_S,labels)

                feature_L = model.featuremap_L
                feature_S = model.featuremap_S
                weight_L = model.Wcls.weight
                weight_S = model.Wcls.weight
                # downscale the large feature map
                feature_L_down = F.interpolate(feature_L,scale_factor=0.5,mode='bilinear',align_corners=True)
                feature_L = feature_L_down
                # softmax over the weighting coefficients
                weight_L_softmax = F.softmax(weight_L,dim=0)
                weight_S_softmax = F.softmax(weight_S,dim=0)
                # generate CAM 
                batch_size,class_num = outputs_L.shape
                _,_,h_L,w_L = feature_L.shape
                _,_,h_S,w_S = feature_S.shape
                CAMs_L = torch.zeros((batch_size,class_num,h_L,w_L))
                for batch in range(batch_size):
                    for cla in range(class_num):
                        CAMs_L[batch][cla]= returnCAM(feature_L[batch],weight_L_softmax[cla])
                CAMs_S = torch.zeros((batch_size,class_num,h_S,w_S))
                for batch in range(batch_size):
                    for cla in range(class_num):
                        CAMs_S[batch][cla]= returnCAM(feature_S[batch],weight_S_softmax[cla])

                # Compute the Mean Squared Error between the two set of CAMs
                loss_seg = F.mse_loss(CAMs_L,CAMs_S)
                # Compute the total loss
                loss = (loss_CLS_L+loss_CLS_S)/2+loss_seg

                # YOUR CODE HERE
                # INPUT TO SEG MODEL, DEFINE THE SCALE EQUIVARIANT LOSS
                # AND COMPUTE THE TOTAL LOSS

                ###################################################
                
            else:
                NotImplementedError

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('seg_loss', loss_seg.item(), global_step)
            writer.add_scalar('total_loss', loss.item(), global_step)
            writer.add_scalar('cls_loss', loss_CLS_L.item(), global_step)
            global_step += 1

            # logging
            if i % 100 == 0 and i != 0:
                print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f},loss_seg:{:.5f},loss_L:{:.5f},loss_S:{:.5f}'.format(epoch, i,
                    time.time()-iter_time, loss.item(),loss_seg.item(),loss_CLS_L.item(),loss_CLS_S.item()))
                # print('epoch:{}, iter:{}, time:{:.2f}, loss:{:.5f}'.format(epoch, i,
                #     time.time()-iter_time, loss.item()))
                iter_time = time.time()
        batch_time = time.time() - batch_time
        print('[epoch {} | time:{:.2f} | loss:{:.5f}]'.format(epoch, batch_time, loss.item()))

        # evaluation
        if epoch % 1 == 0:
            testing_accuracy = evaluate(args, model, testloader)
            print('testing accuracy: {:.3f}'.format(testing_accuracy))
            print('-------------------------------------------------')

            if testing_accuracy > best_testing_accuracy:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, './checkpoints/{}_checkpoint.pth'.format(args.exp_id))
                best_testing_accuracy = testing_accuracy
                print('new best model saved at epoch: {}'.format(epoch))
                print('-------------------------------------------------')
            writer.add_scalar('test_acc', testing_accuracy, test_step)
            test_step += 1
    print('-------------------------------------------------')
    print('best testing accuracy achieved: {:.3f}'.format(best_testing_accuracy))


def evaluate(args, model, testloader):
    total_count = torch.tensor([0.0]).to(device)
    correct_count = torch.tensor([0.0]).to(device)

    for i, data in enumerate(testloader):
        # img_L = data['img_L']; labels = data['label']
        # img_L, labels = img_L.to(device), labels.to(device)
        img_L = data['img_L']; img_S = data['img_S']; labels = data['label']
        img_L, img_S, labels = img_L.to(device), img_S.to(device), labels.to(device)
        total_count += labels.size(0)

        with torch.no_grad():
            if args.mode == 'CAM':
                cls_L_scores, _ = model(img_L, img_S=None)
                predict_L = torch.argmax(cls_L_scores, dim=1)
                correct_count += (predict_L == labels).sum()
            elif args.mode == 'SEG':
                cls_scores, _ = model(img_L, img_S)
                cls_L_scores,cls_S_scores = cls_scores
                predict_L = torch.argmax(cls_L_scores, dim=1)
                correct_count += (predict_L == labels).sum()
    testing_accuracy = correct_count / total_count

    return testing_accuracy.item()


def resume(args, model, optimizer):
    checkpoint_path = './checkpoints/{}_checkpoint.pth'.format(args.exp_id)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    checkpoint_saved = torch.load(checkpoint_path)


    model.load_state_dict(checkpoint_saved['model_state_dict'])
    optimizer.load_state_dict(checkpoint_saved['optimizer_state_dict'])

    print('Resume completed for the model\n')

    return model, optimizer
