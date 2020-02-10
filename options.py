import argparse
import os


# ---------- OPTIONS ----------------

parser = argparse.ArgumentParser()

# general options
parser.add_argument('--model', type=str, default='edsr', help='model to use')
parser.add_argument('--log', action='store_true')

# training options
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--saveinterval', type=int, default=10, help='number of epochs between saves')
parser.add_argument('--ntrain', type=int, default=0, help='number of samples to train on')
parser.add_argument('--scheduler', type=str, default='', help='options for a scheduler, format: stepsize,gamma')

# data
parser.add_argument('--dataset', type=str, default='imagedataset', help='dataset to train')
parser.add_argument('--imageSize', type=int, default=24, help='the low resolution image size')
parser.add_argument('--weights', type=str, default='', help='model to retrain from')
parser.add_argument('--root', type=str, default='/auto/homes/cnc39/phd/datasets', help='dataset to train')
parser.add_argument('--out', type=str, default='results', help='folder to output model training results')

# computation 
parser.add_argument('--workers', type=int, default=6, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--multigpu', action='store_true')
parser.add_argument('--undomulti', action='store_true')

# restoration options
parser.add_argument('--nch_in', type=int, default=3, help='colour channels in input') 
parser.add_argument('--nch_out', type=int, default=3, help='colour channels in output') 

# architecture options 
parser.add_argument('--narch', type=int, default=0, help='architecture-dependent parameter') 
parser.add_argument('--n_resblocks', type=int, default=10, help='number of residual blocks')
parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')

# test options
parser.add_argument('--ntest', type=int, default=10, help='number of images to test per epoch or test run')
parser.add_argument('--testinterval', type=int, default=1, help='number of epochs between tests during training')
parser.add_argument('--test', action='store_true')
parser.add_argument('--cpu', action='store_true') # not supported for training
parser.add_argument('--batchSize_test', type=int, default=1, help='input batch size for test loader')
parser.add_argument('--plotinterval', type=int, default=1, help='number of test samples between plotting')

opt = parser.parse_args()


# ---------- Model loading convenience function ----------------
# function to infer model options when a output directory from training is provided in opt.weights
if len(opt.weights) > 0 and not os.path.isfile(opt.weights):
    # folder provided, trying to infer model options
    
    logfile = opt.weights + '/log.txt'
    opt.weights += '/final.pth'
    if not os.path.isfile(opt.weights):
        opt.weights = opt.weights.replace('final.pth','prelim.pth')

    if os.path.isfile(logfile):
        fid = open(logfile,'r')
        optstr = fid.read()
        optlist = optstr.split(', ')

        def getopt(optname,typestr):
            opt_e = [e.split('=')[-1].strip("\'") for e in optlist if (optname.split('.')[-1] + '=') in e]
            return eval(optname) if len(opt_e) == 0 else typestr(opt_e[0])
            
        opt.model = getopt('opt.model',str)
        opt.nch_in = getopt('opt.nch_in',int)
        opt.nch_out = getopt('opt.nch_out',int)
        opt.n_resgroups = getopt('opt.n_resgroups',int)
        opt.n_resblocks = getopt('opt.n_resblocks',int)
        opt.n_feats = getopt('opt.n_feats',int)


