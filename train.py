import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from data_loader_trn import ImagerLoader
import numpy as np
import random
import cv2
import math
import sklearn.metrics
import torchvision.utils as vutils
from LSTM_model import GazeLSTM

source_path = "/data/vision/torralba/scratch2/recasens/toyota/data/datasets/Ladybug/crops"
test_file = "/data/vision/torralba/scratch2/recasens/opengaze_iccv/val_iccv_cr.txt"
train_file = "/data/vision/torralba/scratch2/recasens/opengaze_iccv/train_iccv_cr.txt"




workers = 30;
epochs = 80
batch_size = 80
base_lr = 1e-4
momentum = 0.9
weight_decay = 1e-4
print_freq = 10
prec1 = 0
best_prec1 = 40
lr = base_lr

scale = 7

network_name = 'ICCV_video_LSTM_CR2'

from tensorboardX import SummaryWriter
foo = SummaryWriter(comment=network_name)


count_test = 0
count = 0
side_x = 40
side_y = 40

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = GazeLSTM()
        #self.features.fc2 = nn.Linear(1000, 3)
        #self.features.conv1 = nn.Conv2d(21, 64, kernel_size=7, stride=2, padding=3,bias=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1,3)
        x[:,0:1] = 3.1415*nn.Tanh()(x[:,0:1])
        x[:,1:2] = (3.1415/2)*nn.Tanh()(x[:,1:2])
        var = 3.1415*nn.Sigmoid()(x[:,2:3])
        var = var.view(-1,1).expand(var.size(0),2)

        return x[:,0:2],var

class WL2(nn.Module):
    def __init__(self):
        super(WL2, self).__init__()
        self.q3 = 0.1
        self.q7 = 1-self.q3

    def compute_l2(self,output,target):
        loss = output-target
        loss = torch.norm(loss).view(-1)
        return loss

    def compute_angular_error(self,output,target):

        input = torch.cos(output)
        target = torch.cos(target)
        target = nn.functional.normalize(target)
        input = nn.functional.normalize(input)

        input = input.view(-1,3,1)
        target = target.view(-1,1,3)
        output_dot = torch.bmm(target,input)
        output_dot = output_dot.view(-1)
        output_dot = torch.clamp(output_dot,-0.999999,0.999999)
        output_dot = torch.acos(output_dot)
        return output_dot

    def forward(self, output_o,target_o,var_o):
        print('Target',target_o.size())
        print('Output',output_o.size())
        print('Variance',var_o.size())
        #import pdb; pdb.set_trace();
        q_30 = target_o-(output_o-var_o)
        q_70 = target_o-(output_o+var_o)

        loss_30 = torch.max(self.q3*q_30, (self.q3-1)*q_30)
        loss_70 = torch.max(self.q7*q_70, (self.q7-1)*q_70)


        loss_30 = torch.mean(loss_30)
        loss_70 = torch.mean(loss_70)
        print(loss_30)

        return loss_30+loss_70



def main():
    global args, best_prec1,weight_decay,momentum,input_scales

    model_v = AlexNet()
    model = torch.nn.DataParallel(model_v).cuda()
    model.cuda()
    #checkpoint = torch.load('/data/vision/torralba/scratch2/recasens/opengaze_iccv/checkpoint_ICCV_video_LSTM.pth.tar')
    #model.load_state_dict(checkpoint['state_dict'])

    resume_epoch = 0




    cudnn.benchmark = True

    image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(source_path,train_file,transforms.Compose([
            transforms.RandomResizedCrop(size=224,scale=(0.8,1)),transforms.ToTensor(),image_normalize,
        ]),square=(224,224),spherical=True,noise_backhead = True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(source_path,test_file,transforms.Compose([
            transforms.Resize((224,224)),transforms.ToTensor(),image_normalize,
        ]),square=(224,224),spherical=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)



    criterion = WL2().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(resume_epoch, epochs):

        #adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


def train(train_loader, model, criterion,optimizer, epoch):
    global count,w
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    prediction_error = AverageMeter()
    angular = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i,  (source_frame,target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        source_frame = source_frame.cuda(async=True)
        target = target.cuda(async=True)


        source_frame_var = torch.autograd.Variable(source_frame)
        target_var = torch.autograd.Variable(target)

        # compute output
        print(source_frame_var.size())
        output,ang_error = model(source_frame_var)


        loss = criterion(output, target_var,ang_error)

        angular_error = compute_angular_error(output,target_var)
        pred_error = ang_error[:,0]*180/math.pi
        pred_error = torch.mean(pred_error,0)

        angular.update(angular_error, source_frame.size(0))

        losses.update(loss.item(), source_frame.size(0))

        prediction_error.update(pred_error, source_frame.size(0))

        image = source_frame[0,:,:,:].view(7,3,224,224)

        image = vutils.make_grid(image, normalize=True, scale_each=True)
        foo.add_image('Input Image',image, count)

        foo.add_scalar("loss", losses.val, count)
        foo.add_scalar("angular", angular.val, count)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        count = count +1

        print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Angular {angular.val:.3f} ({angular.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prediction Error {prediction_error.val:.4f} ({prediction_error.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,angular=angular,prediction_error=prediction_error))

def validate(val_loader, model, criterion):
    global count_test,w
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    topb = AverageMeter()
    l2a= AverageMeter()
    l2e = AverageMeter()
    prediction_error = AverageMeter()
    # switch to evaluate mode
    model.eval()
    end = time.time()
    angular = AverageMeter()

    for i, (source_frame,target) in enumerate(val_loader):

        source_frame = source_frame.cuda(async=True)
        target = target.cuda(async=True)

        source_frame_var = torch.autograd.Variable(source_frame,volatile=True)
        target_var = torch.autograd.Variable(target,volatile=True)
        with torch.no_grad():
            # compute output
            output,ang_error = model(source_frame_var)

            loss = criterion(output, target_var,ang_error)
            angular_error = compute_angular_error(output,target_var)
            pred_error = ang_error[:,0]*180/math.pi
            pred_error = torch.mean(pred_error,0)

            angular.update(angular_error, source_frame.size(0))
            prediction_error.update(pred_error, source_frame.size(0))

            losses.update(loss.item(), source_frame.size(0))

            # measure elapsed time

            # compute gradient and do SGD step
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


        print('Epoch: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Angular {angular.val:.4f} ({angular.avg:.4f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(val_loader), batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5,l2a=l2a,l2e=l2e,angular=angular))

    foo.add_scalar("predicted error", prediction_error.avg, count)
    foo.add_scalar("angular-test", angular.avg, count)
    foo.add_scalar("loss-test", losses.avg, count)
    return angular.avg

def save_checkpoint(state, is_best, filename='checkpoint_'+network_name+'.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_'+network_name+'.pth.tar')


def compute_pred_error(mean,sigma):
    input = torch.cos(mean+sigma)
    target = torch.cos(mean-sigma)
    target = nn.functional.normalize(target)
    input = nn.functional.normalize(input)

    input = input.view(-1,3,1)
    target = target.view(-1,1,3)
    output_dot = torch.bmm(target,input)
    output_dot = output_dot.view(-1)
    print(output_dot[0:3])
    output_dot = torch.clamp(output_dot,-0.999999,0.999999)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180*torch.mean(output_dot)/3.14515
    print(output_dot)
    return output_dot


def spherical2cartesial(x):
    
    output = torch.zeros(x.size(0),3)
    output[:,2] = -torch.cos(x[:,1])*torch.cos(x[:,0])
    output[:,0] = torch.cos(x[:,1])*torch.sin(x[:,0])
    output[:,1] = torch.sin(x[:,1])

    return output


def compute_angular_error(input,target):

    input = spherical2cartesial(input)
    target = spherical2cartesial(target)
    #target = nn.functional.normalize(target)
    #input = nn.functional.normalize(input)

    print(input[0,:])
    print(target[0,:])

    input = input.view(-1,3,1)
    target = target.view(-1,1,3)
    output_dot = torch.bmm(target,input)
    output_dot = output_dot.view(-1)
    print(output_dot[0:3])
    output_dot = torch.clamp(output_dot,-0.999999,0.999999)
    output_dot = torch.acos(output_dot)
    output_dot = output_dot.data
    output_dot = 180*torch.mean(output_dot)/3.14515
    print(output_dot)
    return output_dot




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr








if __name__ == '__main__':
    main()
