from datasets import make_datasetS
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable
import SimpleITK as sit
from network import SharedEncoder,Dose_prediction
import torch
if __name__ == '__main__':
    param = OrderedDict()
    param['gpu_ids'] = [0]
    save_path = r'/home/scuse/ts/dose_pre_datasets/result_tot/withoutOARs/unet+gra+loss3/3d/'
    # fp_lossG = open('.\\result\\netG_losses.txt','w')
    # fp_lossD = open('.\\result\\netD_losses.txt', 'w')


    Encoder = SharedEncoder()
    Dose_decoder = Dose_prediction()
    if len(param['gpu_ids']) > 0:
        assert (torch.cuda.is_available())
        torch.cuda.set_device(device=param['gpu_ids'][0])
        Encoder.cuda()
        Dose_decoder.cuda()
    Encoder.load_state_dict(torch.load('/home/scuse/ts/withoutoars/model/unet+gra+loss3/Encoder/55_encoder.pth',map_location={'cuda:1':'cuda:0'}))
    Dose_decoder.load_state_dict(torch.load('/home/scuse/ts/withoutoars/model/unet+gra+loss3/Dose_decoder/55_decoder.pth',map_location={'cuda:1':'cuda:0'}))
    batch = 1
    testData = make_datasetS()
    for ii, batch_sample in enumerate(testData):
        input,target,c,name = batch_sample['inputs'],batch_sample['rd'],batch_sample['channel'],batch_sample['name']
        print(name)
        input1 = input.squeeze(0)
        input2 = input1.squeeze(0)  # 185 6 512 512
        target1 = target.squeeze(0)
        target2 = target1.squeeze(0) # 185 1 512 512
        #print(c)

        outputs = np.zeros(shape=(c, 512, 512))
        real = np.zeros(shape=(c, 512, 512))
        for i in range(c):
            main_inputs = input2[i,:,:,:] #6 512 512
            main_target = target2[i,:,:,:] #1 512 512


            inputs, targets = Variable(main_inputs).cuda(), Variable(main_target).cuda()
            #print(torch.max(targets))
            conv1, conv2, conv3, conv4, center = Encoder.forward(inputs.unsqueeze(0))
            doseFake = Dose_decoder(conv1, conv2, conv3, conv4, center)
            fake = doseFake.squeeze(0) #c 1 512 512
            fake = fake.detach().cpu().numpy()
            t = targets.detach().cpu().numpy()

            outputs[i,:,:] = fake
            real[i,:,:] = t

        # fake_mha = sit.GetImageFromArray(outputs)
        # real_mha = sit.GetImageFromArray(real)
        fake_name = str(name[0]) + '_PreRD.mha'
        real_name = str(name[0]) + '_ReaRD.mha'

        # print(real.min())
        # print(real.max())
        # print(real.median())
        fakes: sit.Image = sit.GetImageFromArray(outputs)
        fakes.SetSpacing(spacing=(0.9766, 0.9766, 3))
        s = sit.ImageFileWriter()
        s.SetFileName(save_path + '/' +fake_name)
        s.Execute(fakes)
        sit.WriteImage(fakes, save_path + '/' +fake_name)

        reals: sit.Image = sit.GetImageFromArray(real)
        reals.SetSpacing(spacing=(0.9766, 0.9766, 3))
        s = sit.ImageFileWriter()
        s.SetFileName(save_path + '/' +real_name)
        s.Execute(reals)
        sit.WriteImage(reals, save_path + '/' +real_name)


