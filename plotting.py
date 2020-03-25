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
        plt.subplot(1,3,idx)
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
            if not opt.cpu:
                sr_bat = net(lr_bat.cuda())
            else:
                sr_bat = net(lr_bat)
        sr_bat = sr_bat.cpu()

        for j in range(len(lr_bat)):
            lr, sr, hr = lr_bat.data[j], sr_bat.data[j], hr_bat.data[j]
            
            if torch.max(hr.long()) == 0: 
                continue # all black, ignore
            m = nn.LogSoftmax(dim=0)
            sr = m(sr)
            sr = sr.argmax(dim=0, keepdim=True)

            lr, sr, hr = toPIL(lr), toPIL(sr.float() / (opt.nch_out - 1)), toPIL(hr.float())

            plt.figure(figsize=(10,5))
            bc_psnr, bc_ssim = makesubplot(1, lr, hr,'input')
            sr_psnr, sr_ssim = makesubplot(2, sr, hr, 'output')
            makesubplot(3, hr)
            
            mean_bc_psnr += bc_psnr
            mean_sr_psnr += sr_psnr
            mean_bc_ssim += bc_ssim
            mean_sr_ssim += sr_ssim

            if count % opt.plotinterval == 0:
                plt.tight_layout()
                plt.subplots_adjust(wspace=0.01, hspace=0.01)
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
        opt.writer.add_scalar('data/psnr', mean_sr_psnr / count,idx)
        opt.writer.add_scalar('data/ssim', mean_sr_ssim / count,idx)
        t1 = time.perf_counter() - opt.t0
        mem = torch.cuda.memory_allocated()
        print(idx,t1,mem,mean_sr_psnr / count, mean_sr_ssim / count, file=opt.test_stats)
        opt.test_stats.flush()