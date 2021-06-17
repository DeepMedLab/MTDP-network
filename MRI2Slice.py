import SimpleITK as sitk
import os
import numpy as np
import cv2

# 例子：BraTS_2018数据集（MRI）中的标签：0是背景、1是坏疽、2是浮肿、4是增强肿瘤区
# 选择标签值，输出相应的标签切片图
LABEL_NUM = 1

# 图片输出路径及格式
IMAGE_OUT_PATH = '/dataset/dose_pre_datasets/origin_slice'
IMAGE_OUT_FORMAT = '.jpg'
LABEL_OUT_PATH = '/dataset/dose_pre_datasets/rs_slice'
LABEL_OUT_FORMAT = '.jpg'
DOSE_OUT_PATH = '/dataset/dose_pre_datasets/rd_slice'
DOSE_OUT_FORMAT = '.jpg'

np.set_printoptions(threshold=np.inf)


def make_files(origin, rs, rd):
    names = []
    images = {}
    allRs = []
    for root, _, fnames in sorted(os.walk(rs)):
        for fname in fnames:
            pathrs = os.path.join(root, fname)
            allRs.append(pathrs)
    for root, _, fnames in sorted(os.walk(origin)):
        for fname in fnames:
            name = fname.split('_')[0]
            opath = os.path.join(root, fname)
            # names = os.path.join(root,name)
            names.append(name)
            # if name not in images.keys():
            images[name] = []
            images[name].append(opath)
            for item in allRs:
                # print(item)
                t = item.find(name+'_PCTV_plan_rs.mha')
                if t > 0:
                    images[name].append(item)
            rdname = str(name) + '_rd.mha'
            pathrd = os.path.join(rd, rdname)
            images[name].append(pathrd)
    names_ = []
    images_ = {}
    for i in range(len(names)):
        if len(images[names[i]]) == 3:
            names_.append(names[i])
            images_[names[i]] = images[names[i]]
    return names_, images_


def main():
    # 判断输出路径是否存在，若不存在创建一个目录
    if not os.path.exists(IMAGE_OUT_PATH):
        os.mkdir(IMAGE_OUT_PATH)
    if not os.path.exists(LABEL_OUT_PATH):
        os.mkdir(LABEL_OUT_PATH)

    origin = '/home/user/LiZhiang/NPC_Segmentation/res/origin'
    rd = '/home/user/LiZhiang/NPC_Segmentation/res/rd'
    rs = '/home/user/LiZhiang/NPC_Segmentation/res/rs'
    names, images = make_files(origin, rs, rd)

    for name in names:
        path_origin = images[name][0]
        path_label = images[name][1]
        path_dose = images[name][2]
        origin_img = sitk.ReadImage(path_origin)
        label_img = sitk.ReadImage(path_label)
        dose_img = sitk.ReadImage(path_dose)
        origin_array = sitk.GetArrayFromImage(origin_img).astype(np.float32)
        label_array = sitk.GetArrayFromImage(label_img)
        dose_array = sitk.GetArrayFromImage(dose_img).astype(np.float32)
        origin_array = ((origin_array - origin_array.min()) / (origin_array.max() - origin_array.min())) * 255.0
        dose_array = ((dose_array - dose_array.min()) / (dose_array.max() - dose_array.min())) * 255.0

        index = -1
        # 生成与label并且靶区维度一样的array，且元素都为1.
        like_label_one_array = np.ones_like(label_array)

        for m in range(dose_array.shape[0]):
            # 找到有dose剂量的切片。
            flag = np.sum(dose_array[m] > 0) == 0
            true_sum = np.sum(label_array[m] == LABEL_NUM * like_label_one_array[m])
            if true_sum > 0:  # 没有剂量的不切片，避免类别不平衡问题。
                index += 1
                print(m)

                # 处理图像
                origin = origin_array[m]

                # 二值化label
                label = np.zeros(label_array[1].shape, dtype=np.uint8)
                for x in range(label.shape[0]):
                    for y in range(label.shape[1]):
                        if label_array[m][x, y] == LABEL_NUM:
                            label[x, y] = 255

                # 处理剂量图
                # dose = dose_array[m]
                dose = dose_array[m]

                cv2.imwrite("%s/%s%d.png" % (LABEL_OUT_PATH, name+'_PCTV_plan_rs', index), label)
                cv2.imwrite("%s/%s%d.png" % (IMAGE_OUT_PATH, name+'o', index), origin)
                cv2.imwrite("%s/%s%d.png" % (DOSE_OUT_PATH, name+'rd', index), dose)


if __name__ == "__main__":
    main()