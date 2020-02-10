import math
import os

import torch
import time 

import torch.optim as optim
import torchvision
from torch.autograd import Variable

from models import GetModel
from datahandler import GetDataloaders

from plotting import testAndMakeCombinedPlots

from options import opt


def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary

	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict



def train(dataloader, validloader, net, nepoch=10):
    
    start_epoch = 0
    if opt.task == 'segment':
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    loss_function.cuda()

    loss_function_custom = nn.MSELoss()
    loss_function_custom.cuda()


    if len(opt.weights) > 0: # load previous weights?
        checkpoint = torch.load(opt.weights)
        print('loading checkpoint',opt.weights)
        if opt.undomulti:
            checkpoint['state_dict'] = remove_dataparallel_wrapper(checkpoint['state_dict'])
        if opt.modifyPretrainedModel:
            pretrained_dict = checkpoint['state_dict']
            model_dict = net.state_dict()
            # 1. filter out unnecessary keys
            for k,v in list(pretrained_dict.items()):
                print(k)
            pretrained_dict = {k: v for k, v in list(pretrained_dict.items())[:-2]}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            net.load_state_dict(model_dict)

            # optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        else:
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']


    if len(opt.scheduler) > 0:
        stepsize, gamma = int(opt.scheduler.split(',')[0]), float(opt.scheduler.split(',')[1])
        scheduler = optim.lr_scheduler.StepLR(optimizer, stepsize, gamma=gamma, last_epoch=start_epoch-1)

    count = 0
    opt.t0 = time.perf_counter()

    for epoch in range(start_epoch, nepoch):
        mean_loss = 0

        for i, bat in enumerate(dataloader):
            lr, hr = bat[0], bat[1]
            
            optimizer.zero_grad()
            if opt.model == 'ffdnet':
                stdvec = torch.zeros(lr.shape[0])
                for j in range(lr.shape[0]):
                    noise = lr[j] - hr[j]
                    stdvec[j] = torch.std(noise)
                noise = net(lr.cuda(), stdvec.cuda())
                sr = torch.clamp( lr.cuda() - noise,0,1 )
                gt_noise = lr.cuda() - hr.cuda()
                loss = loss_function(noise, gt_noise)
            elif opt.task == 'residualdenoising':
                noise = net(lr.cuda())
                gt_noise = lr.cuda() - hr.cuda()
                loss = loss_function(noise, gt_noise)
            else:
                sr = net(lr.cuda())
                if opt.task == 'segment':
                    hr_classes = torch.round(2*hr).long()
                    loss = loss_function(sr.squeeze(), hr_classes.squeeze().cuda())
                else:
                    loss = loss_function(sr, hr.cuda())

            loss.backward()
            optimizer.step()
            
            
            ######### Status and display #########
            mean_loss += loss.data.item()
            print('\r[%d/%d][%d/%d] Loss: %0.6f' % (epoch+1,nepoch,i+1,len(dataloader),loss.data.item()),end='')
            
            count += 1
            if opt.log and count*opt.batchSize // 1000 > 0:
                t1 = time.perf_counter() - opt.t0
                mem = torch.cuda.memory_allocated()
                print(epoch, count*opt.batchSize, t1, mem, mean_loss / count, file=opt.train_stats)
                opt.train_stats.flush()
                count = 0



        # ---------------- Scheduler -----------------
        if len(opt.scheduler) > 0:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('\nLearning rate',param_group['lr'])
                break        


        # ---------------- Printing -----------------
        print('\nEpoch %d done, %0.6f' % (epoch,(mean_loss / len(dataloader))))
        print('\nEpoch %d done, %0.6f' % (epoch,(mean_loss / len(dataloader))),file=opt.fid)
        opt.fid.flush()


        # ---------------- TEST -----------------
        if (epoch + 1) % opt.testinterval == 0:
            testAndMakeCombinedPlots(net,validloader,opt,epoch)
            # if opt.scheduler:
                # scheduler.step(mean_loss / len(dataloader))

        if (epoch + 1) % opt.saveinterval == 0:
            # torch.save(net.state_dict(), opt.out + '/prelim.pth')
            checkpoint = {'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict() }
            torch.save(checkpoint, opt.out + '/prelim.pth')
    
    checkpoint = {'epoch': nepoch,
    'state_dict': net.state_dict(),
    'optimizer' : optimizer.state_dict() }
    torch.save(checkpoint, opt.out + '/final.pth')


if __name__ == '__main__':

    try:
        os.makedirs(opt.out)
    except IOError:
        pass

    opt.fid = open(opt.out + '/log.txt','w')
    print(opt)
    print(opt,'\n',file=opt.fid)
        
    dataloader, validloader = GetDataloaders(opt)        
    net = GetModel(opt)
    
    if opt.log:
        opt.train_stats = open(opt.out.replace('\\','/') + '/train_stats.csv','w')
        opt.test_stats = open(opt.out.replace('\\','/') + '/test_stats.csv','w')
        print('iter,nsample,time,memory,meanloss',file=opt.train_stats)
        print('iter,time,memory,psnr,ssim',file=opt.test_stats)

    import time
    t0 = time.perf_counter()
    if not opt.test:
        train(dataloader, validloader, net, nepoch=opt.nepoch)
    else:
        if len(opt.weights) > 0: # load previous weights?
            checkpoint = torch.load(opt.weights)
            print('loading checkpoint',opt.weights)
            if opt.undomulti:
                checkpoint['state_dict'] = remove_dataparallel_wrapper(checkpoint['state_dict'])
            net.load_state_dict(checkpoint['state_dict'])
            print('time: ',time.perf_counter()-t0)
        testAndMakeCombinedPlots(net,validloader,opt)
    print('time: ',time.perf_counter()-t0)

