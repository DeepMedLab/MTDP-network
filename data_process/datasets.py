import os.path
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
import SimpleITK as sit
from torch.autograd import Variable



def make_dataset(dir,phase):
    paths = os.path.join(dir,phase)
    #assert os.path.isdir(paths), '%s is not a valid directory' % dir
    rd = ""
    origin = ""
    rs = ""
    for root,files, _ in sorted(os.walk(paths)):
        for file in files:
            if "origin" in file:
                origin = os.path.join(root, file)
            elif "rd" in file:
                rd = os.path.join(root, file)
            elif "rs" in file:
                rs = os.path.join(root, file)
    return origin,rs,rd

def make_files(origin,rs,rd):
    names = []
    images = {}
    allRs = []
    for root,_,fnames in sorted(os.walk(rs)):
        for fname in fnames:
            pathrs =os.path.join(root,fname)
            allRs.append(pathrs)
    for root,_,fnames in sorted(os.walk(origin)):
        for fname in fnames:
            name = fname.split('_')[0]
            opath = os.path.join(root,fname)
            #names = os.path.join(root,name)
            names.append(name)
            #if name not in images.keys():
            images[name] = []
            images[name].append(opath)
            for item in allRs:
                # print(item)
                t = item.find(name + '_PTV_plan_rs.mha')
                if t > 0:
                    images[name].append(item)
                t = item.find(name + '_Bladder_rs.mha')
                if t > 0:
                    images[name].append(item)
                t = item.find(name + '_FemoralHeadL_rs.mh')
                if t > 0:
                    images[name].append(item)
                t = item.find(name + '_FemoralHeadR_rs.mha')
                if t > 0:
                    images[name].append(item)
                t = item.find(name + '_Smallintestine_rs.mha')
                if t > 0:
                    images[name].append(item)
            rdname = str(name) + '_rd.mha'
            pathrd = os.path.join(rd,rdname)
            images[name].append(pathrd)
    names_ = []
    images_ = {}
    for i in range(len(names)):
        if len(images[names[i]]) == 7:
            names_.append(names[i])
            images_[names[i]] = images[names[i]]
            sorted(images_[names[i]])
    return names_,images_



class TrainDataset():
    def __init__(self,dir,phase):
        super(TrainDataset, self).__init__()
        self.dir = dir
        self.phase = phase
        self.origin,self.rd,self.rs = sorted(make_dataset(self.dir,self.phase))
        self.names,self.images = make_files(self.origin,self.rs,self.rd)
        print(len(self.names))
        print(len(self.images))

        self.transform = torch.from_numpy


    def __getitem__(self, index):
        rsdatapath = r'/home/scuse/ts/dose_pre_datasets/test3d/rs/'
        image_path = self.names[index]
        images = self.images[image_path]
        origin = images[0]
        rd = images[-1]
        Bladder = os.path.join(rsdatapath, image_path + '_Bladder_rs.mha')
        FemoralHeadL = os.path.join(rsdatapath, image_path + '_FemoralHeadL_rs.mha')
        FemoralHeadR = os.path.join(rsdatapath, image_path + '_FemoralHeadR_rs.mha')
        PCTV = os.path.join(rsdatapath, image_path + '_PTV_plan_rs.mha')
        Smallintestine = os.path.join(rsdatapath, image_path + '_Smallintestine_rs.mha')
        print(origin)
        print(rd)
        print(Bladder)
        print(FemoralHeadL)
        print(FemoralHeadR)
        print(PCTV)
        print(Smallintestine)


        origin_ = sit.ReadImage(origin)
        rd_ = sit.ReadImage(rd)
        Bladder_ = sit.ReadImage(Bladder)
        FemoralHeadL_ = sit.ReadImage(FemoralHeadL)
        FemoralHeadR_ = sit.ReadImage(FemoralHeadR)
        PCTV_ = sit.ReadImage(PCTV)
        Smallintestine_ = sit.ReadImage(Smallintestine)


        origin_numpy = sit.GetArrayFromImage(origin_)
        rd_numpy = sit.GetArrayFromImage(rd_)
        Bladder_numpy = sit.GetArrayFromImage(Bladder_)
        FemoralHeadL_numpy = sit.GetArrayFromImage(FemoralHeadL_)
        FemoralHeadR_numpy = sit.GetArrayFromImage(FemoralHeadR_)
        PCTV_numpy = sit.GetArrayFromImage(PCTV_)
        Smallintestine_numpy = sit.GetArrayFromImage(Smallintestine_)
        # print("ori:",rd_numpy.max()) 65535
        # print("ori:",rd_numpy.min()) 0

        if rd_numpy.max() != 0:
            origin_numpy = (origin_numpy - origin_numpy.min())/ (origin_numpy.max() - (origin_numpy.min()))
            rd_numpy = (rd_numpy - rd_numpy.min()) / (rd_numpy.max() - (rd_numpy.min()))
        else:
            origin_numpy = (origin_numpy - origin_numpy.min()) / (origin_numpy.max() + 1 - (origin_numpy.min()))
            rd_numpy = (rd_numpy - rd_numpy.min()) / (rd_numpy.max() + 1 - (rd_numpy.min()))

        c = min(origin_numpy.shape[0], rd_numpy.shape[0], Bladder_numpy.shape[0], FemoralHeadL_numpy.shape[0], FemoralHeadR_numpy.shape[0],PCTV_numpy.shape[0],Smallintestine_numpy.shape[0])

        inputs = np.zeros(shape=(c,6,512,512)) #<class 'tuple'>: (185, 6, 512, 512)
        for i in range(c):
            channel = origin_numpy[np.newaxis,i,:,:]
            channel = np.append(channel,PCTV_numpy[np.newaxis,i,:,:], axis=0)
            channel = np.append(channel,Bladder_numpy[np.newaxis,i,:,:], axis=0) #<class 'tuple'>: (2, 512, 512)
            channel = np.append(channel,FemoralHeadL_numpy[np.newaxis,i,:,:], axis=0)
            channel = np.append(channel,FemoralHeadR_numpy[np.newaxis,i,:,:], axis=0)
            channel = np.append(channel,Smallintestine_numpy[np.newaxis,i,:,:], axis=0) #<class 'tuple'>: (6, 512, 512)
            inputs[i,:,:,:] = channel[:,:,:]
        #print(inputs.shape)
        rd = rd_numpy[:c,np.newaxis,:,:] #<class 'tuple'>: (148, 1, 512, 512)
        #print(rd.shape)
        inputs = self.transform(inputs).unsqueeze(0).type(torch.FloatTensor) #torch.Size([1, 148, 6, 512, 512])

        rd = self.transform(rd).unsqueeze(0).type(torch.FloatTensor) #torch.Size([1, 185, 1, 512, 512])




        #return {'original': orig,'rd':rd,'B':Bladder_numpy,'FHL':FemoralHeadL_numpy,'FHR':FemoralHeadR_numpy,'P':PCTV_numpy,'S':Smallintestine_numpy,'channel':c}
        return {'inputs': inputs, 'rd': rd, 'channel': c, "name": image_path}
        #input ,rd torch.Size([1, 185, 1, 512, 512]),c 148 , name bailasuguo
    def __len__(self):
        return len(self.names)

   # def name(self):
     #   return str(self.kind)+'Dataset'



class TestDataset():
    def __init__(self, dir, phase):
        super(TestDataset, self).__init__()
        self.dir = dir
        self.phase = phase
        self.origin, self.rd, self.rs = sorted(make_dataset(self.dir, self.phase))
        self.names, self.images = make_files(self.origin, self.rs,self.rd)


        self.transform = torch.from_numpy

    def __getitem__(self, index):
        rsdatapath = r'/home/scuse/ts/dose_pre_datasets/test3d/rs/'
        image_path = self.names[index]
        images = self.images[image_path]
        origin = images[0]
        rd = images[-1]
        #Bladder = images[1]
        #FemoralHeadL = images[2]
        #FemoralHeadR = images[3]
        #PCTV = images[4]
        #Smallintestine = images[5]
        #Bladder = os.path.join(rsdatapath, image_path + '_Bladder_rs.mha')
        #FemoralHeadL = os.path.join(rsdatapath, image_path + '_FemoralHeadL_rs.mha')
        #FemoralHeadR = os.path.join(rsdatapath, image_path + '_FemoralHeadR_rs.mha')
        PCTV = os.path.join(rsdatapath, image_path + '_PTV_plan_rs.mha')
        #Smallintestine = os.path.join(rsdatapath, image_path + '_Smallintestine_rs.mha')
        print(origin)
        print(rd)
        #print(Bladder)
        #print(FemoralHeadL)
        #print(FemoralHeadR)
        print(PCTV)
        #print(Smallintestine)
        
        origin_ = sit.ReadImage(origin)
        rd_ = sit.ReadImage(rd)
        #Bladder_ = sit.ReadImage(Bladder)
        #FemoralHeadL_ = sit.ReadImage(FemoralHeadL)
        #FemoralHeadR_ = sit.ReadImage(FemoralHeadR)
        PCTV_ = sit.ReadImage(PCTV)
        #Smallintestine_ = sit.ReadImage(Smallintestine)

        origin_numpy = sit.GetArrayFromImage(origin_)
        rd_numpy = sit.GetArrayFromImage(rd_)
        #Bladder_numpy = sit.GetArrayFromImage(Bladder_)
        #FemoralHeadL_numpy = sit.GetArrayFromImage(FemoralHeadL_)
        #FemoralHeadR_numpy = sit.GetArrayFromImage(FemoralHeadR_)
        PCTV_numpy = sit.GetArrayFromImage(PCTV_)
        #Smallintestine_numpy = sit.GetArrayFromImage(Smallintestine_)

        # print("ori:",rd_numpy.max()) 65535
        # print("ori:",rd_numpy.min()) 0

        if rd_numpy.max() != 0:
            origin_numpy = (origin_numpy - origin_numpy.min()) / (origin_numpy.max() - (origin_numpy.min()))
            rd_numpy = (rd_numpy - rd_numpy.min()) / (rd_numpy.max() - (rd_numpy.min()))
        else:
            origin_numpy = (origin_numpy - origin_numpy.min()) / (origin_numpy.max() + 1 - (origin_numpy.min()))
            rd_numpy = (rd_numpy - rd_numpy.min()) / (rd_numpy.max() + 1 - (rd_numpy.min()))
        # print(origin_numpy.min())
        # print(rd_numpy.min())
        # print(Bladder_numpy.min())
        # print(FemoralHeadL_numpy.min())
        # print(FemoralHeadR_numpy.min())
        # print(Smallintestine_numpy.min())
        # print(PCTV_numpy.min())
        c = min(origin_numpy.shape[0], rd_numpy.shape[0], PCTV_numpy.shape[0])

        inputs = np.zeros(shape=(c, 2, 512, 512))  # <class 'tuple'>: (185, 6, 512, 512)
        for i in range(c):
            channel = origin_numpy[np.newaxis, i, :,:]

            channel = np.append(channel, PCTV_numpy[np.newaxis, i, :, :], axis=0)  # <class 'tuple'>: (2, 512, 512)
            #channel = np.append(channel, Bladder_numpy[np.newaxis, i, :, :], axis=0)
            #channel = np.append(channel, FemoralHeadL_numpy[np.newaxis, i, :, :], axis=0)
            #channel = np.append(channel, FemoralHeadR_numpy[np.newaxis, i, :, :], axis=0)
            #channel = np.append(channel, Smallintestine_numpy[np.newaxis, i, :, :],
                               # <class 'tuple'>: (6, 512, 512)

            inputs[i, :, :, :] = channel[:, :, :]
        # print(inputs.shape)
        rd = rd_numpy[:c, np.newaxis, :, :]  # <class 'tuple'>: (148, 1, 512, 512)
        # print(rd.shape)
        inputs = self.transform(inputs).unsqueeze(0).type(torch.FloatTensor)  # torch.Size([1, 148, 6, 512, 512])

        rd = self.transform(rd).unsqueeze(0).type(torch.FloatTensor)  # torch.Size([1, 185, 1, 512, 512])

        # return {'original': orig,'rd':rd,'B':Bladder_numpy,'FHL':FemoralHeadL_numpy,'FHR':FemoralHeadR_numpy,'P':PCTV_numpy,'S':Smallintestine_numpy,'channel':c}
        return {'inputs': inputs, 'rd': rd, 'channel': c, "name": image_path}
        # input ,rd torch.Size([1, 185, 1, 512, 512]),c 148 , name bailasuguo


    def __len__(self):
        return len(self.names)

def make_datasetS():
    dir = r'/home/scuse/ts/dose_pre_datasets/'
    batch_size = 1
    # Syn_train = TrainDataset(dir,"res")
    Syn_test = TestDataset(dir,"test3d")
    # print(len(Syn_train))
    # SynData_train = DataLoader(dataset=Syn_train,batch_size=batch_size,shuffle=True,drop_last=True,num_workers=0)
    SynData_test = DataLoader(dataset=Syn_test,batch_size=batch_size,shuffle=False,drop_last=False,num_workers=0)
    return SynData_test


if __name__ == "__main__":
    tra= make_datasetS()
    batch = 1
    for ii, batch_sample in enumerate(tra):
        # inputs, target, c ,name = batch_sample['inputs'], batch_sample['rd'], batch_sample['channel'],batch_sample['name']
        # print(name)
        # print(target.shape)
        input, target, c = batch_sample['inputs'], batch_sample['rd'], batch_sample['channel']
        input1 = input.squeeze(0)
        input2 = input1.squeeze(0)
        target1 = target.squeeze(0)
        target2 = target1.squeeze(0)
        # print(c)
        for i in range(c // batch):
            if batch * i + batch <= c:  
                # print(batch*i + batch)
                main_inputs = input2[batch * i:batch * i + batch, :, :, :]
                main_target = target2[batch * i:batch * i + batch, :, :, :]
            else:
                # main_inputs = input2[batch*i:c,:,:,:]
                # main_target = target2[batch*i:c,:,:,:]
                break
            inputs, targets = Variable(main_inputs), Variable(main_target) 
            # print(type(inputs))
            # print(inputs.shape) #2 6 512 512
            # print(targets)
            # print(targets.max())




