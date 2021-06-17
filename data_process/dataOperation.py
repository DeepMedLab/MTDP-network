import numpy as np
import torch
import torch.nn as nn
import cv2


def gra(d):
    sp = d.shape
    height = sp[0]
    weight = sp[1]
    sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    s1 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    s2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])

    dSobel = np.zeros((height, weight))
    dSobelx = np.zeros((height, weight))
    dSobely = np.zeros((height, weight))
    dSobel1 = np.zeros((height, weight))
    dSobel2 = np.zeros((height, weight))
    Gx = np.zeros([height,weight])
    Gy = np.zeros([height,weight])
    G1 = np.zeros([height,weight])
    G2 = np.zeros([height,weight])
    for i in range(height - 2):
        for j in range(weight - 2):
            Gx[i + 1, j + 1] = abs(np.sum(d[i:i + 3, j:j + 3] * sx))
            Gy[i + 1, j + 1] = abs(np.sum(d[i:i + 3, j:j + 3] * sy))
            G1[i + 1, j + 1] = abs(np.sum(d[i:i + 3, j:j + 3] * s1))
            G2[i + 1, j + 1] = abs(np.sum(d[i:i + 3, j:j + 3] * s2))
            dSobel[i + 1, j + 1] = (Gx[i + 1, j + 1] * Gx[i + 1, j + 1] + Gy[i + 1, j + 1] * Gy[i + 1, j + 1] + G1[
                i + 1, j + 1] * G1[i + 1, j + 1] + G2[i + 1, j + 1] * G2[i + 1, j + 1]) ** 0.5
            dSobelx[i + 1, j + 1] = np.sqrt(Gx[i + 1, j + 1])
            dSobely[i + 1, j + 1] = np.sqrt(Gy[i + 1, j + 1])
            dSobel1[i + 1, j + 1] = np.sqrt(G1[i + 1, j + 1])
            dSobel2[i + 1, j + 1] = np.sqrt(G2[i + 1, j + 1])
    d= np.uint8(dSobel)
    return d


def dis_dose(d):
    seg = [1,0.9,0.8,0.6,0.45,0.35,0.2,0]
    s = [1,0.8,0.64,0.48,0.32,0.16,0]
    l = len(seg)   #8
    sp = d.shape
    # print('d[0][10][10]')
    # print(d[0][10][10])
    h = sp[1]
    w = sp[2]
    a = torch.ones([l-1,h,w])   #7,512,512
    b = torch.ones([l,h,w])     #8,512,512
    c = torch.zeros([1,h,w])    #1,512,512
    for i in range(l):
        b[i] = b[i] * seg[i]
    for i in range(l-1):
        if i <l-1-1 :
            a[i] = (d<=b[i])*(d>b[i+1])
            c[0]=c[0]+a[i]*s[i]
        else:
            a[i] = (d <= b[i]) * (d >= b[i + 1])
            c[0] = c[0] + a[i] * s[i]
    e = torch.tensor([[[0]],[[1]],[[2]],[[3]],[[ 4]],[[5]],[[6]]])
    out = a * e.float()
    out = out.sum(dim=0)
    # print('out[10][10]')
    # print(out[10][10])
    return out,c

a = torch.randn([1,512,512])

# tensor([[[1.0000, 0.8000, 0.6400],
#          [0.4800, 0.4800, 0.3200],
#          [0.1600, 0.0000, 0.0000],
#          [0.0000, 0.0000, 1.0000]]])
out,c = dis_dose(a)
print(out)
print(c)
def dis_to_show(a):
    a = a.cpu()
    softmax = nn.Softmax(dim=0)
    a = softmax(a)
    y = (a>0.5).float()
    s = torch.tensor([[[1]],[[0.8]],[[0.64]],[[0.48]],[[0.32]],[[0.16]],[[0]]])
    c = y * s
    c = c.sum( dim =0 )
    return c

# tensor([[0., 1., 2.],
#         [3., 3., 4.],
#         [5., 6., 6.],
#         [6., 6., 0.]])
# a = torch.tensor([[[ 1,  0,  0],
#          [0, 0, 0],
#          [0, 0,  0]],
#
#         [[0, 1, 0],
#          [0, 0,  0],
#          [0,  0, 0]],
#
#           [[0, 0, 1],
#            [0, 0, 0],
#            [0, 0, 0]],
#                   [[0, 0, 0],
#                    [1, 1, 0],
#                    [0, 0, 0]],
#                   [[0, 0, 0],
#                    [0, 0, 1],
#                    [0, 0, 0]],
#                   [[0, 0, 0],
#                    [0, 0, 0],
#                    [1, 1, 0]],
#                   [[0, 0, 0],
#                    [0, 0, 0],
#                    [0, 0, 1]]])
# c = dis_to_show(a.long())
# print(c)
#input:[x,y,z,z],output:[x,y,z,z]
def sobel(img,ii):
    img_ = img.detach().cpu().numpy()*255
    shape = img.shape
    batch = shape[0]
    for i in range(batch):
        x = cv2.Sobel(img_[i][0],cv2.CV_16S,1,0)
        y =  cv2.Sobel(img_[i][0],cv2.CV_16S,0,1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)

        dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
        ii[i][0] = torch.from_numpy(dst)
    ii = ii
    ii = (ii - ii.min()) / (ii.max() - (ii.min()))
    return ii


#x:剂量预测结果 [3,1,512,512]
#y:离散结果 [3,1,512,512]
#     seg = [1, 0.9, 0.8, 0.6, 0.45, 0.35, 0.2, 0]
class Myloss2(nn.Module):
    def __init__(self,seg):
        super(Myloss2,self).__init__()
        self.seg = seg
        return
    def forward(self, x,y,Loss):
        spy = y.shape #3,1,512,512
        batch = spy[0]#
        # c = spy[1]
        c = len(self.seg)-1
        for j in range(batch):
            for i in range(c):
                z = (y[j]==i).float()
                xx = x[j] * z
                min = z * self.seg[i+1]
                max = z * self.seg[i]

                a_min = min - xx
                a_min_p = a_min >= 0
                a_min = a_min * a_min_p.float()
                a_min = a_min ** 2

                a_max = xx - max
                a_max_p = a_max > 0
                a_max = a_max * a_max_p.float()
                a_max = a_max ** 2

                a_loss = a_max + a_min #1,512,512
                Loss[j] = Loss[j] + a_loss
        loss = torch.mean(Loss)
        return loss

# x = torch.randn([2,1,512,512])
# y = torch.randn([2,1,512,512])
# loss = torch.randn([2,1,512,512])
# seg = [1, 0.9, 0.8, 0.6, 0.45, 0.35, 0.2, 0]
# xx = Myloss2(seg)
# yy = xx(x,y,loss)
