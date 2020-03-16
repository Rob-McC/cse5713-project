"""
Model, training, test code for molecular dynamic structure prediction for t + tao
Modification: number of layers, number of neuron, data1 preprocessing, single model, plot
hyperparameters will adjust for best training loss

code citation:
@article{time-lagged-autoencoder,
	Author = {Christoph Wehmeyer and Frank No{\'{e}}},
	Doi = {10.1063/1.5011399},
	Journal = {J. Chem. Phys.},
	Month = {jun},
	Number = {24},
	Pages = {241703},
	Publisher = {{AIP} Publishing},
	Title = {Time-lagged autoencoders: Deep learning of slow collective variables for molecular kinetics},
	Volume = {148},
	Year = {2018}}
"""
import sys
import os
import natsort
import re
from torch import optim as optim1
from torch import cat as cat1
from torch import randn as randn1
from torch import sum as sum1
from torch import no_grad as no_grad1
import numpy as np
import torch as torch1
from torch import nn as nn1
from torch.utils.data import Dataset as Dataset1
from torch.utils.data import ConcatDataset as ConcatDataset1
from torch.utils.data import DataLoader as DataLoader1
import torch
import matplotlib.pyplot as plt

LR = 0.0001
HIDDEN = [280,180,100]
BETA = 1.0
DROPOUT = 0.2
ALPHA = 0.01
N_EPOCH = 5000
BATCH_SIZE = 100
TAO = 1
PATH = 'MDmodel/'

def loadData():
    xtrain,xtest =[],[]

    TRAINFILE = os.listdir(sys.argv[1])
    TESTFILE = os.listdir(sys.argv[2])
    TRAINFILE = natsort.natsorted(TRAINFILE)
    TESTFILE = natsort.natsorted(TESTFILE)
    print(TRAINFILE)
    # print(TESTFILE)
    for file in TRAINFILE :
        ff = sys.argv[1] + file
        temp = []
        with open(ff,'r') as f:
            count = 0
            for line in f:
                count+=1
                if count > 1:
                # print(line)
                    arr = re.split(" ",line)
                    temp.append(float(arr[0].replace("\n",'')))
                    temp.append(float(arr[1].replace("\n",'')))
                    temp.append(float(arr[2].replace("\n",'')))
        # print(file)
        # print(len(temp))
                # print(arr)
        xtrain.append(temp)
        # print(xtrain)
    for file in TESTFILE:
        ff = sys.argv[1] + file
        temp = []
        with open(ff, 'r') as f:
            count = 0
            for line in f:
                count += 1
                if count > 1:
                    # print(line)
                    arr = re.split(" ", line)
                    temp.append(float(arr[0].replace("\n", '')))
                    temp.append(float(arr[1].replace("\n", '')))
                    temp.append(float(arr[2].replace("\n", '')))
        # print(file)
        # print(len(temp))
        # print(arr)
        xtest.append(temp)

    return xtrain, xtest


class MatrixTransformStep(object):
#
    def __init__(self, mean=None, covariance=None):
        if mean is not None:
            self.sub = mean
        if covariance is not None:
            self.mul = matrixSqrt_Inverse(covariance)
    def __call__(self, x):
        try:
            x.sub_(self.sub[None, :])
        except AttributeError:
            pass
        try:
            x = x.mm(self.mul)
        except AttributeError:
            pass
        return x

class MatrixTransformClass(object):

    def __init__(
        self, x_mean=None, x_covariance=None, y_mean=None, y_covariance=None):
        self.x = MatrixTransformStep(mean=x_mean, covariance=x_covariance)
        self.y = MatrixTransformStep(mean=y_mean, covariance=y_covariance)
    def __call__(self, x, y):
        return self.x(x), self.y(y)

class DatasetTao(Dataset1):

    def __init__(self, data_tensor, tao=1):
        assert data_tensor.size(0) > tao, 'Please make sure your dataset has more than tao sample'
        assert tao >= 0, 'please make sure your tao is larger than 0'
        self.data_tensor = data_tensor
        self.tao = tao
    def __getitem__(self, index):
        return self.data_tensor[index], self.data_tensor[index + self.tao]
    def __len__(self):
        return self.data_tensor.size(0) - self.tao


def checkPositionFileFormat(data, dtype=np.float32):
    data = np.asarray(data, dtype=dtype)
    if data.ndim == 2:
        return data
    elif data.ndim == 1:
        return data.reshape(-1, 1)
    else:
        raise ValueError('Input data1 has incomplatible number of dimension: ' + str(data.ndim))


def matrixSqrt_Inverse(matrix, bias=1.0e-5):

    # matrix - symmetric real matrix, eValues, eVector = eigenvalues, eigenvectors
    eValues, eVector = torch1.symeig(matrix, eigenvectors=True)

    # dMatrix = diagnal matrix
    dMatrix = torch1.diag(1.0 / torch1.sqrt(torch1.abs(eValues) + bias))

    #(eVector x dMatrix) x eVector.t()
    return torch1.mm(torch1.mm(eVector, dMatrix), eVector.t())

def calculateMean(loader):

    xmean, ymean = None, None
    for x, y in loader:
        try:
            xmean.add_(x.sum(dim=0))
        except AttributeError:
            xmean = x.sum(dim=0)
        try:
            ymean.add_(y.sum(dim=0))
        except AttributeError:
            ymean = y.sum(dim=0)
    xmean.div_(float(len(loader.dataset)))
    ymean.div_(float(len(loader.dataset)))
    return xmean, ymean

def calculateCovariance(loader, xmean, ymean):

    #inital tensor with value zero
    covanriance_xx = torch1.zeros(len(xmean), len(xmean))
    covanriance_xy = torch1.zeros(len(xmean), len(ymean))
    covanriance_yy = torch1.zeros(len(ymean), len(ymean))

    for x, y in loader:
        x.sub_(xmean[None, :])
        y.sub_(ymean[None, :])
        #x.t() matrix multiply x
        covanriance_xx.add_(torch1.mm(x.t(), x))
        #x.t() matrix multipy y
        covanriance_xy.add_(torch1.mm(x.t(), y))
        #y.t() matrix multipy y
        covanriance_yy.add_(torch1.mm(y.t(), y))
    covanriance_xx.div_(float(len(loader.dataset)))
    covanriance_xy.div_(float(len(loader.dataset)))
    covanriance_yy.div_(float(len(loader.dataset)))
    return covanriance_xx, covanriance_xy, covanriance_yy

def whitenPCAdata(data_tensor, batch_size=100):

    loader = DataLoader1(
        DatasetTao(data_tensor, tao=0), batch_size=batch_size)
    xmean, ymean = calculateMean(loader)
    covariance_xx, covariance_xy, covariance_yy = calculateCovariance(loader, xmean, ymean)
    inverse_xx = matrixSqrt_Inverse(covariance_xx)
    whitenedPCAdataList = []
    for x, _ in loader:
        x.sub_(xmean[None, :])
        whitenedPCAdataList.append(x.mm(inverse_xx))
    return torch1.cat(whitenedPCAdataList)


def makeDatasetWithTao(data, tao=0, dtype=np.float32):
#create the dataset with time t and t + tao

    if isinstance(data, np.ndarray):
        return DatasetTao(
            torch1.from_numpy(checkPositionFileFormat(data, dtype=dtype)),
            tao=tao)
    elif isinstance(data, (list, tuple)):
        return ConcatDataset1([DatasetTao(
            torch1.from_numpy(checkPositionFileFormat(d, dtype=dtype)),
            tao=tao) for d in data])
    else:
        raise ValueError(
            'use a single or a list of numpy.ndarrays of dim 1 or 2')

##########model######################

class VariationalAutoEncoderModel(nn1.Module):

    def __init__(self, inputSize, lastSize, normalizeBatch=False, hiddenNodeSize=[], beta=1.0, dropout=0.5, alpha=0.01, prerelu=False, bias=True, lr=0.001, cuda=False, non_blocking=False):
        super(VariationalAutoEncoderModel, self).__init__()
        self.beta = beta
        self._mse_loss_function = nn1.MSELoss(reduction='sum')

        sizes = [inputSize] + list(hiddenNodeSize) + [lastSize]
        self.lastLayer = len(sizes) - 2
        if isinstance(dropout, float):
            dropout = nn1.Dropout(p=dropout)
        self.buildModelLayser(sizes, bias, alpha, prerelu, dropout)
        self.opt1 = optim1.Adam(self.parameters(), lr=lr)
        self.normalizeBatch = normalizeBatch
        self.non_blocking = non_blocking
        if cuda:
            self.use_cuda = True
            self.cuda()  # the non_blocking=... parameter is not accepted, here
        else:
            self.use_cuda = False

    def buildModelLayser(self, layerSizes, bias, alpha, prerelu, dropout):
    #build model layers
        # print("VAE setup function")
        for i, idex in enumerate(range(1, len(layerSizes) - 1)):
            setattr(self,'enc_prm_%d' % i, nn1.Linear(layerSizes[idex - 1], layerSizes[idex], bias=bias))
            self.activationFunction('enc', i, alpha, prerelu)
            if dropout is not None:
                setattr(self, 'enc_drp_%d' % i, dropout)
        setattr(self,'enc_prm_%d_mu' % self.lastLayer, nn1.Linear(layerSizes[-2], layerSizes[-1], bias=bias))
        self.activationFunction('enc', self.lastLayer, None, None, suffix='_mu')
        setattr(self,'enc_prm_%d_lv' % self.lastLayer, nn1.Linear(layerSizes[-2], layerSizes[-1], bias=bias))
        self.activationFunction('enc', self.lastLayer, None, None, suffix='_lv')
        for i, idex in enumerate(reversed(range(1, len(layerSizes)))):
            setattr(self,'dec_prm_%d' % i, nn1.Linear(layerSizes[idex], layerSizes[idex - 1], bias=bias))
            if i < self.lastLayer:
                self.activationFunction('dec', i, alpha, prerelu)
                if dropout is not None:
                    setattr(self, 'dec_drp_%d' % i, dropout)
            else:
                self.activationFunction('dec', i, None, None)

    def activationFunction(self, keyAtt, idex, alpha, prerelu, suffix=''):
    #apply activation function on layer
        # print("base activation function")

        if alpha is None:
            activationFunc = None
        elif alpha < 0.0:
            raise ValueError('alpha must be a non-negative number')
        elif alpha == 0.0:
            activationFunc = nn1.ReLU()
        elif prerelu:
            activationFunc = nn1.PReLU(num_parameters=1, init=alpha)
        else:
            activationFunc = nn1.LeakyReLU(negative_slope=alpha)
        if activationFunc is not None:
            setattr(self, keyAtt + '_act_%d%s' % (idex, suffix), activationFunc)
        layer = getattr(self, keyAtt + '_prm_%d%s' % (idex, suffix))
        nn1.init.kaiming_normal_(layer.weight.data, a=alpha, mode='fan_in')
        try:
            layer.bias.data.uniform_(0.0, 0.1)
        except AttributeError:
            pass
    def moduleForLayer(self, nameKey, inputValue):

        # print("base try to apply model function")
        try:
            return getattr(self, nameKey)(inputValue)
        except AttributeError:
            return inputValue
    def createLayer(self, nameKey, idex, inputValue):

        # print("base _apply layer function")
        return self.moduleForLayer(nameKey + '_drp_%d' % idex, self.moduleForLayer(nameKey + '_act_%d' % idex, self.moduleForLayer(
                    nameKey + '_prm_%d' % idex, inputValue)))

    def calculateKullbackLeiblerD(self,lv,mu):
        return -0.5 * sum1(1.0 + lv - mu.pow(2) - lv.exp())

    def trainForwardAppyLoss(self, x, y):
    #train network forward and apply loss function
        # print("VAE forward loss function")
        reconstrY, mu, lv = self(x)
        # print("VAE forward loss function yrecon mu lv : ", y_recon,mu,lv)
        mse = self._mse_loss_function(reconstrY, y)
        # print("mse =======", mse)
        kullbackld = -0.5 * sum1(1.0 + lv - mu.pow(2) - lv.exp())
        loss = mse + self.beta * kullbackld / float(y.size(1))
        return loss
    def encodeInput(self, input):
    #apply encode to input data1

        # print("VAE _encode function")
        inputy = input
        for layer in range(self.lastLayer):
            inputy = self.createLayer('enc', layer, inputy)
        latentmu = getattr(self, 'enc_prm_%d_mu' % self.lastLayer)(inputy)
        latentlv = getattr(self, 'enc_prm_%d_lv' % self.lastLayer)(inputy)
        return latentmu, latentlv
    def reparameterizeEncode(self, mu, lv):

        # print("VAE rep function")
        if self.training:
            std = lv.mul(0.5).exp_()
            randomeps = randn1(*std.size())
            if self.use_cuda:
                randomeps = randomeps.cuda()
            return randomeps.mul(std).add_(mu)
        else:
            return mu
    def encodeTransform(self, x):
        # print("VAE encode function")
        return self.reparameterizeEncode(*self.encodeInput(x))

    def decodeInput(self, reparamMuLv):
        '''Decode the given input.'''
        # print("VAE decode function")
        inputY = reparamMuLv
        for layer in range(self.lastLayer):
            inputY = self.createLayer('dec', layer, inputY)
        return getattr(self, 'dec_prm_%d' % self.lastLayer)(inputY)

    # def forward(self, x):
    #     '''Forward the given input through the network.'''
    #     # print("VAE forward function")
    #     mu, lv = self.encodeInput(x)
    #     return self.decodeInput(self.reparameterizeEncode(mu, lv)), mu, lv

    def trainEpoch(self, loader):
    #train one epoch
        # print("base train step function")
        self.train()
        trainLoss = 0
        # print("train load : ", loader)
        for input, target in loader:
            input, target = self.matrixTransfInputTarget(input, target)
            if self.use_cuda:
                input = input.cuda(non_blocking=self.non_blocking)
                target = target.cuda(non_blocking=self.non_blocking)
            self.opt1.zero_grad()
            # print('Train step x and y ===============',x,y)
            loss = self.trainForwardAppyLoss(input, target)
            # print("train step loss : ", loss)
            loss.backward()
            trainLoss += loss.item()
            self.opt1.step()
        if self.normalizeBatch:
            return trainLoss / float(len(loader))
        return trainLoss / float(len(loader.dataset))
    def testEpoch(self, loader):
        '''A single validation epoch'''
        self.eval()
        testLoss = 0
        if loader is None:
            return None
        for input, target in loader:
            input, target = self.matrixTransfInputTarget(input, target)
            if self.use_cuda:
                input = input.cuda(non_blocking=self.non_blocking)
                target = target.cuda(non_blocking=self.non_blocking)
            testLoss += self.trainForwardAppyLoss(input, target).item()
        if self.normalizeBatch:
            return testLoss / float(len(loader))
        return testLoss / float(len(loader.dataset))

    def fit(self, trainDataloader, NumEpochs, testDataloader=None):
    #training and testing multiple epoches
        # print("fit function")
        xMean, yMean = calculateMean(trainDataloader)
        # print("x mean, y mean ", x_mean,y_mean)
        covariance_xx, covariance_xy, covariance_yy = calculateCovariance(trainDataloader, xMean, yMean)

        self.matrixTransfInputTarget = MatrixTransformClass(x_mean=xMean, x_covariance=covariance_xx, y_mean=yMean, y_covariance=covariance_yy)
        trainLossList, testLossList = [], []
        # print("start epoch")
        for epoch in range(NumEpochs):
            print("epoch --------->", epoch)
            trainLossList.append(
                self.trainEpoch(
                    trainDataloader))
            with no_grad1():
                testLossList.append(
                    self.testEpoch(testDataloader))
            print(" loss : " , trainLossList[epoch])
        return trainLossList, testLossList

    def transformMatrix(self, loader):

        self.eval()
        latent = []
        for x, _ in loader:
            x = self.matrixTransfInputTarget.x(x)
            if self.use_cuda:
                x = x.cuda(non_blocking=self.non_blocking)
            y = self.encodeTransform(x)
            if self.cuda:
                y = y.cpu()
            latent.append(y)
        return cat1(latent).data

    def forward(self,x):
        mu, lv = self.encodeInput(x)
        return self.decodeInput(self.reparameterizeEncode(mu, lv)), mu, lv

def finialTransform(model, data, data_0, batchSize, whiten, pin_memory=False):
    loader = DataLoader1(data_0, batch_size=batchSize, pin_memory=pin_memory)
    if whiten:
        transformed_data = whitenPCAdata(model.transformMatrix(loader)).numpy()
    else:
        transformed_data = model.transformMatrix(loader).numpy()
    if isinstance(data, (list, tuple)):
        collectData = []
        counter = 0
        lengthList = [d.shape[0] for d in data]
        for length in lengthList:
            collectData.append(transformed_data[counter:counter+length, :])
            counter += length
        return collectData
    return transformed_data



def trainingVAE(
    trainData, validationData=None, dimension=None, tao=1, numEpochs=50,
    batchSize=100, whiten=False, pin_memory=False, **kwargs):
    vae_args = dict(
        hiddenNodeSize=HIDDEN,
        beta=BETA,
        dropout=DROPOUT,
        alpha=ALPHA,
        prerelu=False,
        bias=True,
        lr=LR,
        cuda=False,
        non_blocking=False)
    vae_args.update(kwargs)
    print("vae arg done")
    try:
        inputSize = trainData.shape[1]
    except AttributeError:
        inputSize = trainData[0].shape[1]
    dataNoTAO = makeDatasetWithTao(trainData, tao=0)
    trainDataTAO = makeDatasetWithTao(trainData, tao=tao)

    if validationData is None:
        print("load data1: Dataloader")
        trainLoader = DataLoader1(
            trainDataTAO, batch_size=batchSize, pin_memory=pin_memory)
        testLoader = None
        print("========== Done")
    else:
        valiDataTAO = makeDatasetWithTao(validationData)
        testLoader = DataLoader1(valiDataTAO, batch_size=batchSize, pin_memory=pin_memory)
        trainLoader = DataLoader1(
            trainDataTAO, batch_size=batchSize, pin_memory=pin_memory)

    print("start build model ")
    model = VariationalAutoEncoderModel(inputSize, dimension, **vae_args)
    print("model done, start training")
    trainLoss, testLoss = model.fit(
        trainLoader, numEpochs, testDataloader=testLoader)
    print("train done")

    # prediction =[]
    # for x,y in testLoader:
    #     prediction.append(model.forward(x))

    transformedData = finialTransform(model, testLoader, dataNoTAO, batchSize, whiten)
    # os.mkdir("MDmodel")
    # torch.save(model.state_dict(), PATH)
    return transformedData, trainLoss, testLoss

def plotTraining(trainloss):
    plt.plot(trainloss)
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.savefig("trainloss.png")

def plotTest(testloss):
    plt.plot(testloss)
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.savefig("testloss.png")


if __name__=="__main__":
    xtrain, xtest = loadData()
    odim = len(xtrain[0])

    print(odim)
    print()

    xtrain = np.reshape(xtrain, [-1, odim])
    # xtrain=np.array(xtrain)
    xtest = np.reshape(xtest,[-1,odim])
    print(xtrain.shape)
    # xtest = np.reshape(xtest, [-1, odim])
    print("call vae")
    Tdata, trainLoss, testLoss = trainingVAE(xtrain,validationData=xtest, dimension=odim,tao=TAO, numEpochs=N_EPOCH)

    plotTraining(trainLoss)
    plotTest(testLoss)

    os.mkdir("output/")
    out = "output/"
    for i in range(len(Tdata)):
        outPath = out+ str(i) + 'output.txt'
        with open(outPath,'w') as f:
            for j in range(int(len(Tdata[1])/3)):
                print(j)
                s = str(Tdata[i][0 + j*3]) + " " + str(Tdata[i][1 + j*3]) + " " + str(Tdata[i][2 + j*3])
                f.write(s + '\n')


    print(len(Tdata[1])/3-1)
