import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import os
import numpy as np

from time import time
from networks.CBDNet import CBDNet

from framework import MyFrame
#from loss.dice_loss import SSLoss
from dice_bce_loss import dice_bce_loss
from boundary_bce_loss2 import jointloss
#from boundary_loss import BoundaryLoss
from data import ImageFolder
from hyjoint_loss import hyjoint_loss
SHAPE = (256,256)
ROOT = '.datasets/train/palsar/'
imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(ROOT))
trainlist = list(map(lambda x: x[:-8], imagelist))
NAME = 'palsar_CBDNet'
BATCHSIZE_PER_CARD = 4
EPOCHS_PER_RUN = 25

WEIGHTS_PATH = 'weights/' + NAME + '.th'
EPOCH_FILE = 'weights/' + NAME + '_epoch.txt'

solver = MyFrame(CBDNet, dice_bce_loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    #shuffle=True,
    shuffle=False,
    num_workers=4)

total_epoch = 80

start_epoch = 0
if os.path.exists(EPOCH_FILE):
    with open(EPOCH_FILE, 'r') as f:
        start_epoch = int(f.read().strip())

if os.path.exists(WEIGHTS_PATH):
    solver.load(WEIGHTS_PATH)
    print(f'Resuming from existing weights at {WEIGHTS_PATH} (epoch {start_epoch})')
else:
    print('No existing weights found, starting from random initialization')

end_epoch = min(start_epoch + EPOCHS_PER_RUN, total_epoch)

mylog = open('logs/' + NAME + '.log', 'a' if start_epoch > 0 else 'w')
tic = time()
no_optim = 0
train_epoch_best_loss = 100.
last_epoch = start_epoch
for epoch in range(start_epoch + 1, end_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask in data_loader_iter:
        solver.set_input(img, mask)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)

    print('--------')
    print(f'epoch:{epoch}, time：{int(time()-tic)}， train_loss{train_epoch_loss}')
    # print >> mylog, 'epoch:',epoch, '    time:',int(time()-tic), '    train_loss:',train_epoch_loss
    # print
    # 'epoch:', epoch, '    time:', int(time() - tic)
    # print
    # 'train_loss:', train_epoch_loss
    # print
    # 'SHAPE:', SHAPE

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save(WEIGHTS_PATH)

    last_epoch = epoch
    with open(EPOCH_FILE, 'w') as f:
        f.write(str(epoch))

    if no_optim > 6:
        print (mylog, 'early stop at %d epoch' % epoch)
        # print
        # 'early stop at %d epoch' % epoch
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load(WEIGHTS_PATH)
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()

solver.save(WEIGHTS_PATH)
print( mylog, 'Finish!')
# print
# 'Finish!'
mylog.close()

if last_epoch >= total_epoch:
    print(f'Training complete: reached final epoch {last_epoch}/{total_epoch}.')
else:
    print(f'Stopped after epoch {last_epoch} (target for this run was epoch {end_epoch}).')
    print(f'Run this script again to resume training from epoch {last_epoch}.')