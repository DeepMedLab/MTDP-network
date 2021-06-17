import xlwt
import os
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import torch.nn.functional as F

def down_sample(img):
    tensors = torch.from_numpy(img.astype(np.float32))
    tensors = tensors.unsqueeze(0)
    inputs = F.interpolate(tensors, size=(220, 372), mode='nearest')
    return inputs.squeeze(0).numpy()

xls_savepath = '/home/scuse/ts/dose_pre_datasets/result_tot/withoutOARs/unet+gra+loss3/3d/DVH/val.xls'
datapath = r'/home/scuse/ts/dose_pre_datasets/result_tot/withoutOARs/unet+gra+loss3/3d/'
rsdatapath = r'/home/scuse/ts/dose_pre_datasets/test3d/rs/'
writebook = xlwt.Workbook()               
test= writebook.add_sheet('test')         
label = ['patient_name','pre_d98','pre_d95','pre_d50','pre_d2','gt_d98','gt_d95','gt_d50','gt_d2','diff_percent_d98','diff_percent_d95','diff_percent_d50','diff_percent_d2'
         ,'HI_Pre','HI_Rea','CI_Pre_PTV','CI_Rea_PTV'
        ,'Bladder_V40-Rea','Bladder_V40-Pre','Bladder_V50-Rea','Bladder_V50-Pre','Bladder_Dmean-Rea','Bladder_Dmean-Pre','Bladder_Dmax-Rea','Bladder_Dmax-Pre'
         ,'FemoralHeadL_V40-Rea','FemoralHeadL_V40-Pre','FemoralHeadL_V50-Rea','FemoralHeadL_V50-Pre','FemoralHeadL_Dmean-Rea','FemoralHeadL_Dmean-Pre','FemoralHeadL_Dmax-Rea','FemoralHeadL_Dmax-Pre'
,'FemoralHeadR_V40-Rea','FemoralHeadR_V40-Pre','FemoralHeadR_V50-Rea','FemoralHeadR_V50-Pre','FemoralHeadR_Dmean-Rea','FemoralHeadR_Dmean-Pre','FemoralHeadR_Dmax-Rea','FemoralHeadR_Dmax-Pre'
,'PTV_V40-Rea','PTV_V40-Pre','PTV_V50-Rea','PTV_V50-Pre','PTV_Dmean-Rea','PTV_Dmean-Pre','PTV_Dmax-Rea','PTV_Dmax-Pre'
, 'Smallintestine_V40-Rea', 'Smallintestine_V40-Pre', 'Smallintestine_V50-Rea', 'Smallintestine_V50-Pre', 'Smallintestine_Dmean-Rea', 'Smallintestine_Dmean-Pre', 'Smallintestine_Dmax-Rea', 'Smallintestine_Dmax-Pre']


p_name = ["baoyongyuan","caidaiping","chengang",
          "chenjinlian","chenjixiu","chenshude","chensifeng","chenyongni","cuijian"
          ,"daishengrong","dingxiuyun","duanchengsheng","gaojiarui","guoshoumei"]

l = len(label)
for i in range(l):
    test.write(0,i,label[i])
writebook.save(xls_savepath)


pathDVH = os.path.join(datapath, "DVH")
for kkk, patient_name in enumerate(p_name):
    count = 0
    test.write(kkk+1,count,patient_name)
    count+= 1
    GT = os.path.join(datapath, patient_name + '_ReaRD.mha')
    prediction = os.path.join(datapath, patient_name + '_PreRD.mha')

    OARs_1 = os.path.join(rsdatapath, patient_name + '_Bladder_rs.mha')
    OARs_2 = os.path.join(rsdatapath, patient_name + '_FemoralHeadL_rs.mha')
    OARs_3 = os.path.join(rsdatapath, patient_name + '_FemoralHeadR_rs.mha')
    OARs_4 = os.path.join(rsdatapath, patient_name + '_PTV_plan_rs.mha')
    OARs_5 = os.path.join(rsdatapath, patient_name + '_Smallintestine_rs.mha')

    OARs_list = []
    OARs_list.append(OARs_1)
    OARs_list.append(OARs_2)
    OARs_list.append(OARs_3)
    OARs_list.append(OARs_4)
    OARs_list.append(OARs_5)

    max_dose = 5040 * 1.03
    GT_itk = sitk.ReadImage(GT)
    GT_array = sitk.GetArrayFromImage(GT_itk)
    GT_array = GT_array * max_dose / 100

    prediction_itk = sitk.ReadImage(prediction)
    prediction_array = sitk.GetArrayFromImage(prediction_itk)
    prediction_array = prediction_array * max_dose / 100

    plt.title('DVH')

    d98_y = []
    d98_x = []
    d95_y = []
    d95_x = []
    d2_y = []
    d2_x = []
    d50_y = []
    d50_x = []
    HI_ROI_list = []
    d98_y_rea = []
    d98_x_rea = []
    d95_y_rea = []
    d95_x_rea = []
    d2_y_rea = []
    d2_x_rea = []
    d50_y_rea = []
    d50_x_rea = []
    HI_ROI_list_rea = []
    myLog = os.path.join(pathDVH, "lr.txt")
    message = "--"
    try:
        with open(myLog, "a") as logFile:
            for i in range(5):

                OARs = OARs_list[i]
                OARs_itk = sitk.ReadImage(OARs)
                OARs_array = sitk.GetArrayFromImage(OARs_itk)

                OARs_num = np.count_nonzero(OARs_array)
                GT_intersect_OARs = GT_array * OARs_array
                pre_intersect_OARs = prediction_array * OARs_array

                GT_max_dose = np.max(GT_array)
                x = np.linspace(0, GT_max_dose, 500)
                y1 = []
                for j in range(len(x)):
                    y1.append(np.count_nonzero(GT_intersect_OARs >= x[j])/OARs_num)
                y1[0] = 1.0


                pre_max_dose = np.max(prediction_array)
                xx = np.linspace(0, pre_max_dose, 500)
                y2 = []
                for j in range(len(x)):
                    y2.append(np.count_nonzero(pre_intersect_OARs >= xx[j])/OARs_num)
                y2[0] = 1.0

                if i == 3:
                    d98_flag = False
                    d2_flag = False
                    d50_flag = False
                    d95_flag = False
                    for j in range(len(x)-1):
                        if y2[j] > 0.98 and y2[j+1] <= 0.98:
                            d98_index = j + 1
                            d98_flag = True

                            break
                    if d98_flag:
                        d98_y.append(y2[d98_index])
                        d98_x.append(xx[d98_index])
                    else:
                        print("d98_flag_pre",xx)
                        message += "d98_flag_pre:" + str(xx) + "\n"

                    for j in range(len(x)-1):
                        if y2[j] > 0.95 and y2[j+1] <= 0.95:
                            d95_index = j + 1
                            d95_flag = True
                            break
                    if d95_flag:
                        d95_y.append(y2[d95_index])
                        d95_x.append(xx[d95_index])
                    else:
                        print("d95_flag_pre",xx)
                        message += "d95_flag_pre:" + str(xx) + "\n"

                    for j in range(len(x) - 1):
                        if y2[j] > 0.02 and y2[j+1] <= 0.02:
                            d2_index = j + 1
                            d2_flag = True

                            break
                    if d2_flag:
                        d2_y.append(y2[d2_index])
                        d2_x.append(xx[d2_index])
                    else:
                        print("d2_flag_pre",x)
                        message += "d2_flag_pre:" + str(xx) + "\n"

                    for j in range(len(x) - 1):
                        if y2[j] > 0.5 and y2[j + 1] <= 0.5:
                            d50_index = j +1
                            d50_flag = True

                            break
                    if d50_flag:
                        d50_y.append(y2[d50_index])
                        d50_x.append(xx[d50_index])
                    else:
                        print("d50_flag_pre",xx)
                        message += "d50_flag_pre:" + str(xx) + "\n"

                    if d98_flag and d2_flag and d50_flag and d95_flag:

                        HI_ROI_list.append(os.path.basename(OARs_list[i]))


                    d98_flag = False
                    d2_flag = False
                    d50_flag = False
                    d95_flag = False
                    for j in range(len(x)-1):
                        if y1[j] > 0.98 and y1[j+1] <= 0.98:
                            d98_index = j + 1
                            d98_flag = True
                            break
                    if d98_flag:
                        d98_y_rea.append(y1[d98_index])
                        d98_x_rea.append(x[d98_index])
                    else:
                        print("d98_flag",x)
                        message += "d98_flag:" + str(x) + "\n"

                    for j in range(len(x)-1):
                        if y1[j] > 0.95 and y1[j+1] <= 0.95:
                            d95_index = j + 1
                            d95_flag = True
                            break
                    if d95_flag:
                        d95_y_rea.append(y1[d95_index])
                        d95_x_rea.append(x[d95_index])
                    else:
                        print("d95_flag", x)
                        message += "d95_flag:" + str(x) + "\n"

                    for j in range(len(x) - 1):
                        if y1[j] > 0.02 and y1[j+1] <= 0.02:
                            d2_index = j + 1
                            d2_flag = True
                            # d2_y.append(y2[j + 1])
                            # d2_x.append(x[j + 1])
                            break
                    if d2_flag:
                        d2_y_rea.append(y1[d2_index])
                        d2_x_rea.append(x[d2_index])
                    else:
                        print("d2_flag",x)
                        message += "d2_flag:" + str(x) + "\n"

                    for j in range(len(x) - 1):
                        if y1[j] > 0.5 and y1[j + 1] <= 0.5:
                            d50_index = j +1
                            d50_flag = True
                            # d50_y.append(y2[j + 1])
                            # d50_x.append(x[j + 1])
                            break
                    if d50_flag:
                        d50_y_rea.append(y1[d50_index])
                        d50_x_rea.append(x[d50_index])
                    else:
                        print("d50_flag",x)
                        message += "d50_flag:" + str(x) + "\n"
                    if d98_flag and d2_flag and d50_flag and d95_flag:
                    # d98_y_rea.append(y1[d98_index])
                    # d98_x_rea.append(x[d98_index])
                    # d95_y_rea.append(y1[d95_index])
                    # d95_x_rea.append(x[d95_index])
                    # d2_y_rea.append(y1[d2_index])
                    # d2_x_rea.append(x[d2_index])
                        HI_ROI_list_rea.append(os.path.basename(OARs_list[i]))

                plt.xlabel('Gy', fontweight='bold')
                plt.ylabel('Volume%', fontweight='bold')
                if i == 0:
                    plt.plot(x, y1, color='cyan', linewidth=1.0, linestyle='-', label='Bladder')
                    plt.plot(x, y2, color='cyan', linewidth=1.0, linestyle='--')
                    plt.xlim(x.min(), x.max())
                    plt.ylim(0, 1.005)
                    plt.legend(bbox_to_anchor=(1, 0), loc='best', borderaxespad=0)
                    # plt.show()
                elif i == 1:
                    plt.plot(x, y1, color='blue', linewidth=1.0, linestyle='-', label='FHL')
                    plt.plot(x, y2, color='blue', linewidth=1.0, linestyle='--')
                    plt.legend()
                    # plt.show()
                elif i == 2:
                    plt.plot(x, y1, color='green', linewidth=1.0, linestyle='-', label='FHR')
                    plt.plot(x, y2, color='green', linewidth=1.0, linestyle='--')
                    # plt.legend()
                    plt.legend()
                    # plt.show()
                elif i == 3:
                    plt.plot(x, y1, color='red', linewidth=1.0, linestyle='-',label='PTV')
                    plt.plot(x, y2, color='red', linewidth=1.0, linestyle='--')
                    # plt.legend()
                    plt.legend()
                    # plt.show()
                else:
                    plt.plot(x, y1, color='black', linewidth=1.0, linestyle='-', label='ST')
                    plt.plot(x, y2, color='black', linewidth=1.0, linestyle='--')
                    # plt.legend()
                    plt.legend()
                    # plt.show()
            plt.savefig(os.path.join(pathDVH, patient_name))
            plt.show()
            logFile.write("%s\n" % patient_name)

            print("Pre:")
            print("d98", d98_x)  #42.765531062124246
            # print(d98_y) #0.9785376146110867
            print("d95",d95_x)
            # print(d95_y)
            print("d50",d50_x) #51.25250501002004
            # print(d50_y) #0.49560370518450325
            print("d2",d2_x)  #53.677354709418836
            # print(d2_y) #0.017279437704317675
            print("Rea:")
            print("d98", d98_x_rea)
            # print(d98_y)
            print("d95",d95_x_rea)
            # print(d95_y)
            print("d50",d50_x_rea)
            # print(d50_y)
            print("d2",d2_x_rea)
            # print(d2_y)

            message += "Pre:\n"
            message += "d98:" + str(d98_x) + "\n"
            message += "d95:" + str(d95_x) + "\n"
            message += "d50:" + str(d50_x) + "\n"
            message += "d2:" + str(d2_x) + "\n"
            test.write(kkk + 1, count, str(d98_x))
            count += 1
            test.write(kkk + 1, count, str(d95_x))
            count += 1
            test.write(kkk + 1, count, str(d50_x))
            count += 1
            test.write(kkk + 1, count, str(d2_x))
            count += 1
            message += "Rea:\n"
            message += "d98:" + str(d98_x_rea) + "\n"
            message += "d95:" + str(d95_x_rea) + "\n"
            message += "d50:" + str(d50_x_rea) + "\n"
            message += "d2:" + str(d2_x_rea) + "\n"
            test.write(kkk + 1, count, str(d98_x_rea))
            count += 1
            test.write(kkk + 1, count, str(d95_x_rea))
            count += 1
            test.write(kkk + 1, count, str(d50_x_rea))
            count += 1
            test.write(kkk + 1, count, str(d2_x_rea))
            count += 1

            print("diff_percent:")
            message += "diff_percent\n"

            for i in range(len(HI_ROI_list)):
                print((d98_x_rea[i]-d98_x[i])/(max_dose/100))
                print((d95_x_rea[i]-d95_x[i])/(max_dose/100))
                print((d50_x_rea[i]-d50_x[i])/(max_dose/100))
                print((d2_x_rea[i]-d2_x[i])/(max_dose/100))
                print("----------------")
                message += "d98:" + str((d98_x_rea[i]-d98_x[i])/(max_dose/100)) + "\t" #55.44
                test.write(kkk + 1, count, str((d98_x_rea[i]-d98_x[i])/(max_dose/100)))
                count += 1
                message += "d95:" + str((d95_x_rea[i]-d95_x[i])/(max_dose/100)) + "\t"
                message += "d50:" + str((d50_x_rea[i]-d50_x[i])/(max_dose/100)) + "\t"
                message += "d2:" + str((d2_x_rea[i]-d2_x[i])/(max_dose/100)) + "\n"
                test.write(kkk + 1, count, str((d95_x_rea[i]-d95_x[i])/(max_dose/100)))
                count += 1
                test.write(kkk + 1, count, str((d50_x_rea[i]-d50_x[i])/(max_dose/100)))
                count += 1
                test.write(kkk + 1, count, str((d2_x_rea[i]-d2_x[i])/(max_dose/100)))
                count += 1
            # [13.11623246492986, 8.376753507014028, 13.006012024048097, 42.765531062124246, 5.290581162324649]
            # [0.9799835793138924, 0.9760298392847347, 0.9796194056149419, 0.9785376146110867, 0.9794749901630919]
            # [28.877755511022045, 15.761523046092185, 21.60320641282565, 51.25250501002004, 17.965931863727455]
            # [0.49948561706168637, 0.4980527672645494, 0.4951838364853753, 0.49560370518450325, 0.4982935975781131]
            # [53.236472945891784, 37.254509018036075, 43.31663326653307, 53.677354709418836, 45.521042084168336]
            # [0.01943279389071341, 0.019691733859908946, 0.018912251850111594, 0.017279437704317675, 0.0192642795769728]


            print('HI_Pre:')
            message += "HI_Pre\n"
            # print(HI_ROI_list)
            for i in range(len(HI_ROI_list)):
                print((d2_x[i]-d98_x[i])/d50_x[i])
                message += str((d2_x[i]-d98_x[i])/d50_x[i]) + "\t"
                test.write(kkk + 1, count, str((d2_x[i]-d98_x[i])/d50_x[i]))
                count += 1
            print('HI_Rea:')
            message += "\nHI_Rea\n"
            # print(HI_ROI_list_rea)
            for i in range(len(HI_ROI_list)):
                print((d2_x_rea[i]-d98_x_rea[i])/d50_x_rea[i])
                message += str((d2_x_rea[i]-d98_x_rea[i])/d50_x_rea[i]) + "\t"
                test.write(kkk + 1, count, str((d2_x_rea[i]-d98_x_rea[i])/d50_x_rea[i]) )
                count += 1
            # 1.3893129770992367
            # 1.832167832167832
            # 1.403061224489796
            # 0.21290322580645163
            # 2.2392638036809815

            # CI
            PTV_itk = sitk.ReadImage(OARs_4)
            PTV_array = sitk.GetArrayFromImage(PTV_itk)
            # PTV_array = down_sample(PTV_array)
            # if PTV_array.shape[1] == 512:
            #     PTV_array = down_sample(PTV_array)
            OARs_num = np.count_nonzero(PTV_array)
            A = OARs_num
            B = np.count_nonzero(prediction_array > (5040 / 110))
            A_intersect_B = np.count_nonzero(PTV_array * (prediction_array > (5040 / 110)))
            B_rea = np.count_nonzero(GT_array > (5040 / 110))
            A_intersect_B_rea = np.count_nonzero(PTV_array * (GT_array > (5040 / 110)))
            # B = np.count_nonzero(prediction_array > (5040 / 100))
            # A_intersect_B = np.count_nonzero(PTV_array * (prediction_array > (5040 / 100)))
            # B = np.count_nonzero(prediction_array > ((5040 * 65535) / 6622))
            # A_intersect_B = np.count_nonzero(PTV_array * (prediction_array > ((5040 * 65535) / 6622)))
            #print(A_intersect_B)
            CI = A_intersect_B * A_intersect_B / (A * B)
            CI_rea = A_intersect_B_rea * A_intersect_B_rea / (A * B_rea)
            print('CI_Pre_PTV')
            print(CI)
            print('CI_Rea_PTV')
            print(CI_rea)
            message += "\nCI_Pre_PTV\n" + str(CI)
            test.write(kkk + 1, count, str(CI))
            count += 1
            message += "\nCI_Rea_PTV\n" + str(CI_rea) +"\n"
            test.write(kkk + 1, count, str(CI_rea))
            count += 1

            #V40 V50 Dmean
            for oar in OARs_list:
                # print(oar)
                OARs_itk = sitk.ReadImage(oar)
                OARs_array = sitk.GetArrayFromImage(OARs_itk)
                OARs_num = np.count_nonzero(OARs_array)

                GT_intersect_OARs = GT_array * OARs_array
                pre_intersect_OARs = prediction_array * OARs_array
                # print(GT_intersect_OARs.shape) #(142, 512, 512)

                GT_max_dose = np.max(GT_array)
                # print(GT_max_dose) #55.44
                x = np.linspace(0, int(GT_max_dose), 500)
                # print(x)
                # print(len(x)) #500
                y1 = []
                for j in range(len(x)):
                    y1.append(np.count_nonzero(GT_intersect_OARs >= x[j])/OARs_num)
                y1[0] = 1.0

                pre_max_dose = np.max(prediction_array)
                xx = np.linspace(0, int(pre_max_dose), 500)
                y2 = []
                for j in range(len(x)):
                    y2.append(np.count_nonzero(pre_intersect_OARs >= xx[j])/OARs_num)
                y2[0] = 1.0

                # print(pre_max_dose) #54.53903311729431
                print("-----", os.path.basename(oar), "-----")
                message += "-----" + str(os.path.basename(oar)) + "-----"
                for i in range(len(x)):
                    if x[i]>=40 and x[i]<41:
                        # print(i)
                        print("V40-Rea", y1[i])
                        print("V40-Pre", y2[i])
                        # print("V40-diff-percent", (y1[i] - y2[i])/y1[i])
                        message += "V40-Rea" +str(y1[i]) + "\n"
                        message += "V40-Pre" +str(y2[i]) + "\n"
                        test.write(kkk + 1, count, str(y1[i]))
                        count += 1
                        test.write(kkk + 1, count, str(y2[i]))
                        count += 1

                        break
                for i in range(len(x)):
                    if x[i]>=50 and x[i]<51:
                        # print(i)
                        print("V50-Rea", y1[i])
                        print("V50-Pre", y2[i])
                        message += "V50-Rea" + str(y1[i]) + "\n"
                        test.write(kkk + 1, count, str(y1[i]))
                        count += 1
                        message += "V50-Pre" + str(y2[i]) + "\n"
                        test.write(kkk + 1, count, str(y2[i]))
                        count += 1
                        # print("V50-diff-percent", (y1[i] - y2[i]) / y1[i])
                        break


                Dmean1 = GT_intersect_OARs.sum() / OARs_num
                Dmean2 = pre_intersect_OARs.sum() / OARs_num
                Dmax1 = np.max(GT_intersect_OARs)
                Dmax2 = np.max(pre_intersect_OARs)
                print("Dmean-Rea", Dmean1)
                print("Dmean-Pre", Dmean2)
                print("Dmean-diff-percent", (Dmean1 - Dmean2) / Dmean2)
                print("Dmax-Rea", Dmax1)
                print("Dmax-pre", Dmax2)
                print("Dmax-diff-percent", (Dmax1 - Dmax2) / Dmax2)
                message += "Dmean-Rea" + str(Dmean1) + "\n"
                test.write(kkk + 1, count,str(Dmean1))
                count += 1
                message += "Dmean-Pre" + str(Dmean2) + "\n"
                test.write(kkk + 1, count, str(Dmean2))
                count += 1
                message += "Dmean-diff-percent" + str((Dmean1 - Dmean2) / Dmean2) + "\n"
                message += "Dmax-Rea" + str(Dmax1) + "\n"
                test.write(kkk + 1, count, str(Dmax1))
                count += 1
                message += "Dmax-Pre" + str(Dmax2) + "\n"
                test.write(kkk + 1, count, str(Dmax2))
                count += 1
                message += "Dmax-diff-percent" + str((Dmax1 - Dmax2) / Dmax2) + "\n"
            message += "============================================"
            logFile.write("%s\n\n" % message)
            writebook.save(xls_savepath)
    except:
        print('----------------------------------')
        print(patient_name)
        print('----------------------------------')