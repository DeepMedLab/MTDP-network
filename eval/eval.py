import os
import os.path
import numpy as np
from PIL import Image
import seaborn as sns
import torch
import cv2
import matplotlib.pyplot as plt

from dataOperation import dis_dose


def make_files(rd):
    allRs = []
    names = []
    allRs_rd = []
    names_rd = []
    for root,_,fnames in sorted(os.walk(rd)):
        for fname in fnames:
            pathrs =os.path.join(root,fname)
            name = fname.split('_')
            name = name[0]
            names.append(name)
            allRs.append(pathrs)
    print(len(allRs))
    for i in range(len(allRs)):
        # print(names[i])
        x = 'D:\\DataSet\\result_tot\\Test\\rd_slice\\'+names[i]+'_rd.png'
        # x = 'D:\\DataSet\\dose_pre_datasets\\test\\rd_slice\\'+names[i]+'_rd.png'
        allRs_rd.append(x)
    return allRs,names,allRs_rd

def diff(x,names,x_pre):
    for i in range(len(x)):
        image = cv2.imread(x[i], 2)
        # image =cv2.bilateralFilter(image, 9, 75, 75)
        image_pre = cv2.imread(x_pre[i], 2)
        # print(image.shape)
        try:
            dif = image_pre-image
            # dif = dif
            # dif = np.array(dif, dtype='uint8')
            # print(dif.max())
            # hotmap1 = np.abs(dif)
            # heatmap = sns.heatmap(hotmap1, cmap='jet', xticklabels=False, yticklabels=False, vmin=0, vmax=100)
            # plt.savefig(savefile_dif_JET + '/' + names[i] + '_gt' + '.png')
            # plt.show()
            # plt.close()
            # image_ = sns.heatmap(image, cmap='jet', xticklabels=False, yticklabels=False, vmin=0, vmax=100)
            # plt.savefig(savefile_gt + '/' + names[i] + '_gt' + '.png')
            # plt.show()
            # plt.close()
            # image_pre_ = sns.heatmap(image_pre, cmap='jet', xticklabels=False, yticklabels=False, vmin=0, vmax=100)
            # plt.savefig(savefile_pre + '/' + names[i] + '_predictuon' + '.png')
            # plt.show()
            # plt.close()


            image_ = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            image_pre_ = cv2.applyColorMap(image_pre, cv2.COLORMAP_JET)
            dif_PARULA = cv2.applyColorMap(dif, cv2.COLORMAP_JET)

            cv2.imwrite(savefile_gt + '/' + names[i] + '_gt' + '.png', image_)
            cv2.imwrite(savefile_pre + '/' + names[i] + '_predictuon' + '.png', image_pre_)
            cv2.imwrite(savefile_dif_JET + '/' + names[i] + '_dif' + '.png', dif_PARULA)
            print(names[i])
        except:
            print('------')

def dis(x,names,x_pre):
    for i in range(len(x)):
        image = cv2.imread(x[i], 2)
        image = cv2.bilateralFilter(image, 9, 75, 75)
        image_pre = cv2.imread(x_pre[i], 2)#512*512
        image = image/255
        image_pre = image_pre/255
        image = torch.from_numpy(image)
        image_pre = torch.from_numpy(image_pre)
        image = image.unsqueeze(dim=0)
        image_pre = image_pre.unsqueeze(dim=0)
        _,image = dis_dose(image)
        _,image_pre = dis_dose(image_pre)
        # print(image_pre.max())
        image = image.squeeze()
        image_pre = image_pre.squeeze()
        image = image.numpy()*255
        image_pre = image_pre.numpy()*255
        image = np.array(image, dtype='uint8')
        image_pre = np.array(image_pre, dtype='uint8')
        image = cv2.applyColorMap(image, cv2.COLORMAP_JET)
        image_pre = cv2.applyColorMap(image_pre, cv2.COLORMAP_JET)
        cv2.imwrite(savefile_dis_rd + '/' + names[i] + '_dis_rd' + '.png', image)
        cv2.imwrite(savefile_dis_pre + '/' + names[i] + '_dis_pre' + '.png', image_pre)
        print(names[i])


def gra_(x,names,x_pre):
    for i in range(len(x)):
        image = cv2.imread(x[i])
        image = cv2.bilateralFilter(image, 9, 75, 75)
        image_pre = cv2.imread(x_pre[i])#512*512
        xx = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(image, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(xx)
        absY = cv2.convertScaleAbs(y)
        grad_image = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        xx = cv2.Sobel(image_pre, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(image_pre, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(xx)
        absY = cv2.convertScaleAbs(y)
        grad_image_pre = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        img_rd = cv2.applyColorMap(grad_image, cv2.COLORMAP_JET)
        img_pre = cv2.applyColorMap(grad_image_pre, cv2.COLORMAP_JET)
        # cv2.imwrite(savefile_gra_rd + '/' + names[i] + '_gra_rd' + '.png', img_rd)
        cv2.imwrite(savefile_gra_pre + '/' + names[i] + '_gra_pre' + '.png', img_pre)
        print(names[i])
def smooth(x,names):
    for i in range(len(x)):
        image = cv2.imread(x[i], 0)
        bilateralFilter = cv2.bilateralFilter(image, 9, 75, 75)
        cv2.imwrite(filebilaterlFilter + '/' + names[i] + '_filter' + '.png', bilateralFilter)

# file = 'D:\\DataSet\\dose_pre_datasets\\test\\rd_slice'
file = 'D:\\DataSet\\result_tot\\Test\\rd_slice\\'
filebilaterlFilter = 'D:\\DataSet\\result_tot\\Test\\rd_slice'
# file = 'D:\\DataSet\\dose_pre_datasets\\trainn\\rd_slice'
method = 'unet+gra+loss3+loss5+dis+loss2+loss4'
file_pre = 'D:\\DataSet\\result_tot\\show\\' + method + '\\2d\\prediction_slice'
savefile_gt = 'D:\\DataSet\\result_tot\\show\\gt\\2d\\relitu_prediction_slice'
savefile_pre = 'D:\\DataSet\\result_tot\\show\\' + method + '\\2d\\relitu_prediction_slice'
savefile_dif_JET = 'D:\\DataSet\\result_tot\\show\\' + method + '\\2d\\different_slice'
# savefile_dif_WINTER = 'D:\\DataSet\\result_tot\\new11.29\\' + method + '\\2d\\different_slice_WINTER'
# savefile_dif_PARULA = 'D:\\DataSet\\result_tot\\withoutOARs\\' + method + '\\2d\\different_map'

savefile_dis_rd = 'D:\\DataSet\\result_tot\\show\\gt\\2d\\dis_pre'
savefile_dis_pre = 'D:\\DataSet\\result_tot\\show\\' + method + '\\2d\\dis'
savefile_gra_rd = 'D:\\DataSet\\result_tot\\show\\gt\\2d\\gra_pre'
savefile_gra_pre = 'D:\\DataSet\\result_tot\\show\\' + method + '\\2d\\gra_pre'
# x , names = make_files(file)
x_pre,names,x = make_files(file_pre)
# for i in range(len(x)):
#     print(x_pre[i])
#     print(x[i])
#     print('-----')
# diff(x,names,x_pre)
# dis(x,names,x_pre)
gra_(x,names,x_pre)