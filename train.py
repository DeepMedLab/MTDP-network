from MydataSets import myData, myData_test
from collections import OrderedDict
import torch.optim as optim
import os
import time
import torch
from torch import nn
from torch.autograd import Variable
from network import SharedEncoder, Dose_prediction, Discretization_Dose_prediction, Gradient_regression
from dataOperation import Myloss2, dis_to_show, sobel

if __name__ == '__main__':
    param = OrderedDict()
    param['gpu_ids'] = [1]
    fp_lossG = open('/data1/ptang/dose/withoutoars/unet+gra+loss3+loss5+dis+loss2/results/loss.txt','w')
    gpuIns = 1
    a1 = 5
    a2 = 20
    a3 = 5
    a4 = 5
    trainData = myData()
    Encoder = SharedEncoder()
    Dose_decoder = Dose_prediction()
    Dis_dose_decoder = Discretization_Dose_prediction()
    Gra_regress_decoder = Gradient_regression()

    if len(param['gpu_ids']) > 0:
        assert (torch.cuda.is_available())
        torch.cuda.set_device(device=param['gpu_ids'][0])
        Encoder.cuda()
        Dose_decoder.cuda()
        Dis_dose_decoder.cuda()
        Gra_regress_decoder.cuda()

    # Encoder.load_state_dict(torch.load('/home/scuse/ts/dose_prediction/results/Encoder/155_encoder_revise.pth'))
    # Dose_decoder.load_state_dict(torch.load('/home/scuse/ts/dose_prediction/results/Dose_decoder/155_dose_decoder_revise.pth'))
    # Dis_dose_decoder.load_state_dict(torch.load('/home/scuse/ts/dose_prediction/results/Dis_dose_decoder/155_dis_dose_decoder_revise.pth'))
    # Gra_regress_decoder.load_state_dict(torch.load('/home/scuse/ts/dose_prediction/results/Gra_regress_decoder/155_gra_revise.pth'))

    seg = [1, 0.9, 0.8, 0.6, 0.45, 0.35, 0.2, 0]
    myLoss = Myloss2(seg)
    criterionL1= nn.L1Loss()
    criterionCross = nn.CrossEntropyLoss()
    criterionMSE = nn.MSELoss()
    Encoder.train()
    Dose_decoder.train()
    Dis_dose_decoder.train()
    Gra_regress_decoder.train()
    optimizerEncoder = optim.Adam(Encoder.parameters(), lr=1e-5, betas=(0.9, 0.999))
    optimizerDose = optim.Adam(Dose_decoder.parameters(), lr=1e-5, betas=(0.9, 0.999))
    optimizerDisDose = optim.Adam(Dis_dose_decoder.parameters(), lr=4e-4, betas=(0.9, 0.999))
    optimizerGraReg = optim.Adam(Gra_regress_decoder.parameters(), lr=5e-6, betas=(0.9, 0.999))
    count=0
    for epoch in range(200):
        if(epoch == 50):
            a2 = 5
        if(epoch >= 50 and epoch%2 == 0):
            optimizerEncoder.param_groups[0]['lr'] = optimizerEncoder.param_groups[0]['lr'] - 1e-7
            optimizerDose.param_groups[0]['lr'] = optimizerDose.param_groups[0]['lr'] - 1e-7
            optimizerDisDose.param_groups[0]['lr'] = optimizerDisDose.param_groups[0]['lr'] - 4e-6
            optimizerGraReg.param_groups[0]['lr'] = optimizerGraReg.param_groups[0]['lr'] - 5e-8
        count = 0
        loss_tot = 0
        epoch_start_time = time.time()
        iter_data_time = time.time()
        for ii, batch_sample in enumerate(trainData):
            input, target, disDose, disDose_to_show,gra  = batch_sample['inputs'], batch_sample['rd'], batch_sample['disDose'], batch_sample['disDose_to_show'],batch_sample['gra']
            # input : <class 'torch.Tensor'> torch.Size([3, 2, 512, 512])
            # target : <class 'torch.Tensor'> torch.Size([3, 1, 512, 512])
            # disDose : <class 'torch.Tensor'> torch.Size([3, 1, 512, 512])
            # disDose_to_show : <class 'torch.Tensor'> torch.Size([3, 1, 512, 512])
            # gra : <class 'torch.Tensor'> torch.Size([3, 1, 512, 512])
            loss_t = 0
            inputs, dosetargets = Variable(input).cuda(gpuIns), Variable(target).cuda(gpuIns) 
            gra = Variable(gra).cuda(gpuIns)
            disdosetarget =  Variable(disDose).cuda(gpuIns)  
            graFake = torch.zeros([2, 1, 512, 512])
            graFake =Variable(graFake).cuda(gpuIns)
            conv1,conv2,conv3,conv4,center = Encoder.forward(inputs)
            doseFake = Dose_decoder(conv1, conv2, conv3, conv4, center)
            disdoseFake = Dis_dose_decoder(conv1, conv2, conv3, conv4, center) #2,7,512,512
            graFake = sobel(doseFake,graFake)
            # print(disdoseFake.shape)
            graregFake = Gra_regress_decoder(conv1, conv2, conv3, conv4, center)

            loss1 = criterionL1(doseFake, dosetargets)
            x = torch.squeeze(disdosetarget).long()
            # x = torch.unsqueeze(x,dim=0)
            # print(disdoseFake.shape)
            # print(disdosetarget.shape)
            # print(x.shape)
            loss2 = criterionCross(disdoseFake,x)
            # graregFake1 = graregFake.pow(1/2)
            # gra1 = gra.pow(1/2)
            # graFake1 = graFake.pow(1/2)
            graregFake1 = (graregFake+1e-9).sqrt().sqrt()
            gra1 = (gra+1e-9).sqrt().sqrt()
            graFake1 = (graFake+1e-9).sqrt().sqrt()
            loss3 = criterionMSE(graregFake1,gra1)
            loss5 = criterionMSE(graFake1,gra1)

            loss = 10*loss1 + a1 * loss2 + a2 * loss3  + a4*loss5
            loss.backward()
            print(loss)
            optimizerEncoder.step()
            optimizerDose.step()
            optimizerDisDose.step()
            optimizerGraReg.step()
            optimizerEncoder.zero_grad()
            optimizerDose.zero_grad()
            optimizerDisDose.zero_grad()
            optimizerGraReg.zero_grad()
            count += 1
            loss_tot += loss.item()
        print('save_model....')
        print('epoch: %d' %epoch)
        print('epoch: %d' %epoch,'loss:',loss_tot/count)
        print('epoch: %d' %epoch,'loss:',loss_tot/count,file=fp_lossG)
        torch.save(Encoder.state_dict(), os.path.join('/data1/ptang/dose/withoutoars/unet+gra+loss3+loss5+dis+loss2/results/Encoder/', '%s_encoder.pth' % epoch))
        torch.save(Dose_decoder.state_dict(), os.path.join('/data1/ptang/dose/withoutoars/unet+gra+loss3+loss5+dis+loss2/results/Dose_decoder/', '%s_decoder.pth' % epoch))
        torch.save(Dis_dose_decoder.state_dict(), os.path.join('/data1/ptang/dose/withoutoars/unet+gra+loss3+loss5+dis+loss2/results/Dis_decoder/', '%s_dis_decoder.pth' % epoch))
        torch.save(Gra_regress_decoder.state_dict(), os.path.join('/data1/ptang/dose/withoutoars/unet+gra+loss3+loss5+dis+loss2/results/Gra_decoder/', '%s_gra_decoder.pth' % epoch))
