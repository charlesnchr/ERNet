import torch
import matplotlib.pyplot as plt
import torchvision
import skimage
from skimage.measure import compare_ssim
import torchvision.transforms as transforms
import numpy as np
import time
from PIL import Image
import scipy.ndimage as ndimage
import torch.nn as nn
import os

plt.switch_backend('agg')

toTensor = transforms.ToTensor()  
toPIL = transforms.ToPILImage()      

def testAndMakeCombinedPlots(net,loader,opt,idx=None):

    def PSNR_numpy(p0,p1):
        I0,I1 = np.array(p0)/255.0, np.array(p1)/255.0
        MSE = np.mean( (I0-I1)**2 )
        PSNR = 20*np.log10(1/np.sqrt(MSE))
        return PSNR

    def SSIM_numpy(p0,p1):
        I0,I1 = np.array(p0)/255.0, np.array(p1)/255.0
        # return structural_similarity(I0, I1, multichannel=True)
        return compare_ssim(I0, I1, multichannel=True)

    def makesubplot(idx, img, hr=None, title=''):
        plt.subplot(1,4,idx)
        plt.gca().axis('off')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.imshow(img,cmap='gray')
        if not hr == None:
            psnr,ssim = PSNR_numpy(img,hr),SSIM_numpy(img,hr)
            plt.title('%s (%0.2fdB/%0.3f)' % (title,psnr,ssim))
            return psnr,ssim
        plt.title(r'hr ($\infty$/1.000)')


    count, mean_bc_psnr, mean_sr_psnr, mean_bc_ssim, mean_sr_ssim = 0,0,0,0,0

    for i, bat in enumerate(loader):
        lr_bat, hr_bat = bat[0], bat[1]
        with torch.no_grad():
            if opt.model == 'ffdnet':
                stdvec = torch.zeros(lr_bat.shape[0])
                for j in range(lr_bat.shape[0]):
                    noise = lr_bat[j] - hr_bat[j]
                    stdvec[j] = torch.std(noise)
                noise_bat = net(lr_bat.cuda(), stdvec.cuda())
                sr_bat = torch.clamp( lr_bat.cuda() - noise_bat,0,1 )
            elif opt.task == 'residualdenoising':
                for j in range(lr_bat.shape[0]):
                    noise = lr_bat[j] - hr_bat[j]
                noise_bat = net(lr_bat.cuda())
                sr_bat = torch.clamp( lr_bat.cuda() - noise_bat,0,1 )
            else:
                if not opt.cpu:
                    sr_bat = net(lr_bat.cuda())
                else:
                    sr_bat = net(lr_bat)
        sr_bat = sr_bat.cpu()

        for j in range(len(lr_bat)):
            lr, sr, hr = lr_bat.data[j], sr_bat.data[j], hr_bat.data[j]
            
            if opt.task == 'segment':
                if torch.max(hr.long()) == 0: 
                    continue # all black, ignore
                m = nn.LogSoftmax(dim=0)
                sr = m(sr)
                # print(sr)
                sr = sr.argmax(dim=0, keepdim=True)
                # print(sr.shape)

                lr, sr, hr = toPIL(lr), toPIL(sr.float() / (opt.nch_out - 1)), toPIL(hr.float())

                plt.figure(figsize=(10,5))
                makesubplot(1, lr, hr, 'ns')
                bc_psnr, bc_ssim = makesubplot(2, lr, hr,'bc')
                sr_psnr, sr_ssim = makesubplot(3, sr, hr, 're')
                makesubplot(4, hr)
            else:                
                sr = torch.clamp(sr,min=0,max=1)
                
                if lr.shape[0] > 3:
                    lr = lr[lr.shape[0] // 2] # channels are not for colours but separate grayscale frames, take middle
                    hr = hr[hr.shape[0] // 2]

                img = toPIL(lr)
                lr = toTensor(img.resize((hr.shape[2],hr.shape[1]),Image.NEAREST))
                bc = toTensor(img.resize((hr.shape[2],hr.shape[1]),Image.BICUBIC))

                # ---- Plotting -----

                lr, bc, sr, hr = toPIL(lr), toPIL(bc), toPIL(sr), toPIL(hr)
                
                if count % opt.plotinterval == 0:
                    plt.figure(figsize=(10,5))
                    makesubplot(1, lr, hr,'lr')
                    bc_psnr, bc_ssim = makesubplot(2, bc, hr,'bc')
                    sr_psnr, sr_ssim = makesubplot(3, sr, hr,'sr')
                    makesubplot(4, hr)
                else:
                    sr = torch.clamp(sr,min=0,max=1)

                    if lr.shape[0] > 3:
                        lr = lr[lr.shape[0] // 2].unsqueeze(0) # channels are not for colours but separate grayscale frames, take middle
                        sr = sr[sr.shape[0] // 2].unsqueeze(0)
                        hr = hr[hr.shape[0] // 2].unsqueeze(0)
                    ns, re, gt = lr, sr, hr

                    img = toPIL(ns)
                    
                    if ns.shape[0] == 1:
                        sm = ndimage.gaussian_filter(img, sigma=(0.6, 0.6), order=0)
                        sm = np.expand_dims(sm, 2)
                    else:
                        sm = ndimage.gaussian_filter(img, sigma=(0.5, 0.5, 0.2), order=0)

                    # ---- Plotting -----
                    ns, sm, re, gt = toPIL(ns), toPIL(sm), toPIL(re), toPIL(gt)

                    if count % opt.plotinterval == 0:
                        plt.figure(figsize=(10,5))
                        makesubplot(1, ns, gt, 'ns')
                        bc_psnr, bc_ssim = makesubplot(2, sm, gt, 'sm')
                        sr_psnr, sr_ssim = makesubplot(3, re, gt, 're')
                        makesubplot(4, gt)
            
            mean_bc_psnr += bc_psnr
            mean_sr_psnr += sr_psnr
            mean_bc_ssim += bc_ssim
            mean_sr_ssim += sr_ssim

            if count % opt.plotinterval == 0:
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
                if idx is None:  # for tests
                    plt.savefig('%s/combined_%d.png' % (opt.out,count), dpi=300, bbox_inches = 'tight', pad_inches = 0)
                    lr.save('%s/lr_%d.png' % (opt.out,count))
                    bc.save('%s/bc_%d.png' % (opt.out,count))
                    sr.save('%s/sr_%d.png' % (opt.out,count))
                    hr.save('%s/hr_%d.png' % (opt.out,count))
                else:
                    plt.savefig('%s/combined_epoch%d_%d.png' % (opt.out,idx,count), dpi=300, bbox_inches = 'tight', pad_inches = 0)
                plt.close()

            count += 1
            if count == opt.ntest: break
        if count == opt.ntest: break
    
    summarystr = ""
    if count == 0: 
        summarystr += 'Warning: all test samples skipped - count forced to 1 -- '
        count = 1
    summarystr += 'Testing of %d samples complete. bc: %0.2f dB / %0.4f, sr: %0.2f dB / %0.4f' % (count, mean_bc_psnr / count, mean_bc_ssim / count, mean_sr_psnr / count, mean_sr_ssim / count)
    print(summarystr)
    print(summarystr,file=opt.fid)
    opt.fid.flush()
    if opt.log and not opt.test:
        t1 = time.perf_counter() - opt.t0
        mem = torch.cuda.memory_allocated()
        print(idx,t1,mem,mean_sr_psnr / count, mean_sr_ssim / count, file=opt.test_stats)
        opt.test_stats.flush()

