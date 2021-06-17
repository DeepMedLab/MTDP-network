from MydataSets import myData, myData_test
from collections import OrderedDict
import torch.optim as optim
import os
import time
import torch
from torch import nn
from torch.autograd import Variable
from network import SharedEncoder, Dose_prediction, Discretization_Dose_prediction, Gradient_regression
from PIL import Image
import numpy as np
if __name__ == '__main__':
    param = OrderedDict()
    param['gpu_ids'] = [0]
    gpuIns = 0
    testData = myData_test()
    Encoder = SharedEncoder()
    Dose_decoder = Dose_prediction()
    savefile = r'/home/scuse/ts/dose_pre_datasets/result_tot/show/unet+gra+loss3+loss5+dis+loss2/2d/prediction_slice/'

    Encoder.load_state_dict(torch.load('/home/scuse/ts/withoutoars/model/unet+gra+loss3+loss5+dis+loss2/Encoder/117_encoder.pth',map_location={'cuda:1':'cuda:0'}))
    Dose_decoder.load_state_dict(torch.load('/home/scuse/ts/withoutoars/model/unet+gra+loss3+loss5+dis+loss2/Dose_decoder/117_decoder.pth',map_location={'cuda:1':'cuda:0'}))

    if len(param['gpu_ids']) > 0:
        assert (torch.cuda.is_available())
        torch.cuda.set_device(device=param['gpu_ids'][0])
        Encoder.cuda()
        Dose_decoder.cuda()
    for ii, batch_sample in enumerate(testData):
        input, target,name = batch_sample['inputs'], batch_sample['rd'], batch_sample['name']
        # input : <class 'torch.Tensor'> torch.Size([3, 2, 512, 512])
        # target : <class 'torch.Tensor'> torch.Size([3, 1, 512, 512])
        # disDose : <class 'torch.Tensor'> torch.Size([3, 1, 512, 512])
        # disDose_to_show : <class 'torch.Tensor'> torch.Size([3, 1, 512, 512])
        # gra : <class 'torch.Tensor'> torch.Size([3, 1, 512, 512])
        loss_t = 0
        inputs = Variable(input).cuda(gpuIns)
        conv1, conv2, conv3, conv4, center = Encoder.forward(inputs)
        doseFake = Dose_decoder(conv1, conv2, conv3, conv4, center)
        result = doseFake[0,0,:,:]
        result1_to_show = result.detach().cpu().numpy()
        result1_to_show = result1_to_show * 255
        im = Image.fromarray(np.uint8(result1_to_show))
        im.save(savefile + '/' + name[0] + '_prediction' + '.png')
        print(name)
