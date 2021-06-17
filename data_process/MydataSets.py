import os.path
from torch.utils.data import DataLoader
import torch
import numpy as np
from PIL import Image
from dataOperation import dis_dose


def make_dataset(dir,phase):
    paths = os.path.join(dir,phase)
    rd = ""
    origin = ""
    rs = ""
    gra = ""
    for root,files, _ in sorted(os.walk(paths)):
        for file in files:
            if "origin" in file:
                origin = os.path.join(root, file)
            elif "rd" in file:
                rd = os.path.join(root, file)
            elif "gra" in file:
                gra = os.path.join(root, file)
            elif "rs" in file:
                rs = os.path.join(root, file)
    return origin,rs,rd,gra
def make_files(origin,rs,rd,gra):
    names = []
    images = {}
    allRs = []
    gras = []
    rds = []
    for root,_,fnames in sorted(os.walk(rs)):
        for fname in fnames:
            pathrs =os.path.join(root,fname)
            allRs.append(pathrs)
    for root,_,fnames in sorted(os.walk(gra)):
        for fname in fnames:
            pathgra =os.path.join(root,fname)
            gras.append(pathgra)
    for root,_,fnames in sorted(os.walk(rd)):
        for fname in fnames:
            pathrd =os.path.join(root,fname)
            rds.append(pathrd)
    for root,_,fnames in sorted(os.walk(origin)):
        for fname in fnames:
            name = fname.split('_')[0]
            opath = os.path.join(root,fname)
            names.append(name)
            images[name] = []
            images[name].append(opath)
            for item in allRs:
                t = item.find(name+'_')
                if t > 0:
                    # print(item)
                    images[name].append(item)
            # rdname = str(name) + '_rd.png'
            # pathrd = os.path.join(rd,rdname)
            # # print(pathrd)
            # images[name].append(pathrd)
            for item in rds:
                t = item.find(name+'_')
                if t > 0:
                    # print(item)
                    images[name].append(item)
            # graname = str(name)+'_gra.png'
            for item in gras:
                t = item.find(name+'_')
                if t > 0:
                    # print(item)
                    images[name].append(item)
            # pathgra = os.path.join(gra,graname)
            # images[name].append(pathgra)
    names_ = [] 
    images_ = {}
    for i in range(len(names)):
        if len(images[names[i]]) == 4: 
            names_.append(names[i])
            images_[names[i]] = images[names[i]]
    return names_,images_

class TrainDataset():
    def __init__(self, dir,phase):
        super(TrainDataset, self).__init__()
        self.dir = dir
        self.phase = phase
        # print(sorted(make_dataset(self.dir,self.phase)))
        self.gra,self.origin,self.rd,self.rs = sorted(make_dataset(self.dir,self.phase)) 
        self.names,self.images = make_files(self.origin,self.rs,self.rd,self.gra) #name=["laisong", ..] images{"baisong":["/saisong_o.mha", "/saisong_PTV_rs.mha", ... "/saisong_rd.mha",]}

        self.transform = torch.from_numpy

    def __getitem__(self, index):
        image_path = self.names[index]
        images = self.images[image_path]

        # print(images)
        origin = images[0]
        PCTV = images[1]
        gra = images[-1]
        rd = images[2]

        origin_ = Image.open(origin)
        PCTV_ = Image.open(PCTV)
        gra_ = Image.open(gra)
        rd_ = Image.open(rd)

        origin_numpy = np.array(origin_) #512,512
        PCTV_numpy = np.array(PCTV_)/255    #512,512
        gra_numpy = np.array(gra_)
        gra_numpy = gra_numpy 
        rd_numpy = np.array(rd_)    #512,512

        if rd_numpy.max() != 0:
            origin_numpy = (origin_numpy - origin_numpy.min()) / (origin_numpy.max() - (origin_numpy.min()))
            rd_numpy = (rd_numpy - rd_numpy.min()) / (rd_numpy.max() - (rd_numpy.min()))
        else:
            origin_numpy = (origin_numpy - origin_numpy.min()) / (origin_numpy.max() + 1 - (origin_numpy.min()))
            rd_numpy = (rd_numpy - rd_numpy.min()) / (rd_numpy.max() + 1 - (rd_numpy.min()))
        if gra_numpy.max() != 0:
            gra_numpy = (gra_numpy - gra_numpy.min()) / (gra_numpy.max() - (gra_numpy.min()))
        else:
            gra_numpy = (gra_numpy - gra_numpy.min()) / (gra_numpy.max() + 1 - (gra_numpy.min()))

        inputs = np.zeros(shape=(2, 512, 512))
        inputs[0] = origin_numpy
        inputs[1] = PCTV_numpy
        inputs = self.transform(inputs).type(torch.FloatTensor)

        gt = np.zeros(shape=(1,512,512))
        gt[0] = rd_numpy
        rd = self.transform(gt).type(torch.FloatTensor) #torch.Size([1, 512, 512])

        disDose,disDose_to_show = dis_dose(rd) #disDose_to_show :1,512,512
        disDose = disDose.unsqueeze(0) #1 512 512

        g = np.zeros(shape=(1,512,512))
        g[0] = gra_numpy
        gra = self.transform(g).type(torch.FloatTensor) #torch.Size([1, 512, 512])
        return {'inputs': inputs, 'rd': rd,'disDose':disDose,'disDose_to_show':disDose_to_show,'gra':gra, "name": image_path}

    def __len__(self):
        return len(self.names)

def myData():
    dir = r'/home/scuse/ts/dose_pre_datasets/dose_datasets/'
    batch_size = 2
    train = TrainDataset(dir,'train')
    SynData_train = DataLoader(dataset=train,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)
    return SynData_train


def myData_test():
    dir = r'/home/scuse/ts/dose_pre_datasets/dose_datasets/'
    batch_size = 1
    test = TrainDataset(dir,'Test')
    SynData_test = DataLoader(dataset=test,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)
    return SynData_test
if __name__ == "__main__":
    test = myData()
    print('---')
    for ii, batch_sample in enumerate(test):
        input, target, disDose, disDose_to_show, gra,name = batch_sample['inputs'], batch_sample['rd'],batch_sample['disDose'], batch_sample['disDose_to_show'],batch_sample['gra'],batch_sample['name']
        print(name)
