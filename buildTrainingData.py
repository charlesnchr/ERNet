import glob
from PIL import Image
import random
import parser
import os
import scipy.ndimage as ndimage

import numpy as np
import pickle

from skimage.measure import compare_ssim
from skimage import io


def noisy(noise_typ,image,opts=[0,0.005]):
    if noise_typ == "gauss":
        mean = opts[0]
        var = opts[1]
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,image.shape)
        gauss = gauss.reshape(image.shape)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
        for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
        for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(opts[0]))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy/np.max(noisy)*np.max(image)
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)     
        noisy = image + image * gauss
        return noisy


def degrade(img,dim):
    # gaussian darkness
    X,Y = np.meshgrid(np.linspace(0,1,dim),np.linspace(0,1,dim))
    mu_x, mu_y = np.random.rand(), np.random.rand()
    var_x = np.max( [0, 0.05*np.random.randn() + 0.5] )
    var_y = np.max( [0, 0.05*np.random.randn() + 0.5] )
    Z = np.exp( -(X - mu_x)**2 / (2*var_x) ) * np.exp( -(Y - mu_y)**2 / (2*var_y) )
    Z = np.expand_dims(Z, 2)

    darkimg = Z*img
    # darkimg = (0.2*np.random.rand()+0.8)*darkimg  # overall level between 0.5 and 0.1
    
    # poisson_param = np.max([0,3*np.random.randn() + 10])
    # noisyimg = noisy('poisson',darkimg,[poisson_param])

    # gauss_param = np.max([0,0.0001*np.random.randn() + 0.0005])
    # noisyimg = noisy('gauss',darkimg,[0,gauss_param])

    # noisyimg = np.clip(noisyimg,0,1)
    return darkimg


def partitionDataset(imgs,outdir,nreps,dim,degradeBool=True):
    try:
        os.makedirs(outdir)
    except OSError:
        pass

    for i in range(0,len(imgs),2):
        
        src_img = Image.open(imgs[i])
        src_img = np.array(src_img)
        src_gt_img = Image.open(imgs[i+1])
        src_gt_img = np.array(src_gt_img)[:,:,3]
                
        # get rid of gba channels and invert
        # src_gt_img = src_gt_img[:,:,0]

        foreground = src_gt_img > 10
        background = src_gt_img < 10

        src_gt_img[foreground] = 255
        src_gt_img[background] = 0
        
        # normalize and add dimension
        # src_img = (src_img - np.min(src_img)) / (np.max(src_img) - np.min(src_img)) 
        # src_gt_img = src_gt_img/255

   
        h,w = src_img.shape

        j = 0
        while j < nreps:
            r_rand = np.random.randint(0,h-dim)
            c_rand = np.random.randint(0,w-dim)
            img = src_img[r_rand:r_rand+dim,c_rand:c_rand+dim]
            gt_img = src_gt_img[r_rand:r_rand+dim,c_rand:c_rand+dim]

            if np.mean(gt_img) < 0.05*255:
                # print('redoing')
                continue

            # adding random brightness
            brightness = 1 + 0.1*np.random.randn()
            img = np.clip(img* brightness,0,255)

            if np.random.rand() > 0.5:
                poisson_param = np.max([0,3*np.random.randn() + 10])
                img = noisy('poisson',img,[poisson_param])

                gauss_param = np.max([0,0.0001*np.random.randn() + 0.0005])
                img = noisy('gauss',img,[0,gauss_param])
            else:
                sigma_param = np.random.rand()
                img = ndimage.gaussian_filter(img, sigma=(sigma_param,sigma_param), order=0)


            # img = (img - np.min(img)) / (np.max(img) - np.min(img)) 
            # gt_img = (gt_img - np.min(gt_img)) / (np.max(gt_img) - np.min(gt_img)) 

            # img = np.expand_dims(img, 2)
            # if degradeBool:
            #     img = degrade(img,dim)
            # img = img.squeeze()

            filename = '%s/%d-%d.npy' % (outdir,i,j)

            print(i,j,r_rand,c_rand,img.shape,gt_img.shape)

            img = Image.fromarray(img.astype('uint8'))
            gt_img = Image.fromarray(gt_img.astype('uint8'))
            pickle.dump((img,gt_img), open(filename,'wb'))

            combined = np.concatenate((np.array(img),np.array(gt_img)),axis=1)
            io.imsave(filename.replace(".npy",".png"),combined)

            j += 1

        print('[%d/%d]' % (i+2,len(imgs)))


# --------------------------------------------

nreps = 100
dim = 256

allimgs = [
    "trainingdata/labelled_data/stack1/raw_0000.jpg",
    "trainingdata/labelled_data/stack1/0_segmented.png",
    "trainingdata/labelled_data/stack1/126_input_0.png",
    "trainingdata/labelled_data/stack1/126_segmented.png",

    "trainingdata/labelled_data/stack2/raw_0000.jpg",
    "trainingdata/labelled_data/stack2/0_segmented.png",
    "trainingdata/labelled_data/stack2/raw_0126.jpg",
    "trainingdata/labelled_data/stack2/126_segmented.png",
    
    "trainingdata/labelled_data/stack3/raw_0000.jpg",
    "trainingdata/labelled_data/stack3/segmented_0.png"
]

outdir = 'trainingdata/testpartitioned_' + str(dim)
print('Training data')
partitionDataset(allimgs,outdir,nreps,dim,False)