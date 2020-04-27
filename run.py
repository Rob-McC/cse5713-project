import shutil

from build_model import SAUCIE
from loader import Loader
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler



data1 = []
label = []
data =[]
tit = []
with open('input/lungCancerWithLabel.csv','r') as ff:
    i=0
    for line in ff:
        arr = line.split(',')
        # print(arr)
        if i>0 and len(arr)>2:
            temp= [(i.replace('""','')) for i in arr[1:]]
            label.append(int(arr[0]))
            data1=[float(i.replace('"','')) for i in temp]
                # print(j)
            data.append([math.log(a +1)for a in data1])
        elif i==0:
            tit = arr[1:]
        i+=1
# print(data[0])
print(tit)
x = np.array(data)
print(x.shape)
# print(x.shape[1])
x = sklearn.preprocessing.normalize(x, norm='l2', axis=1, copy=True, return_norm=False)
scaler = MinMaxScaler()
x1 = scaler.fit_transform(x)

print(label)

#x = np.concatenate([np.random.uniform(-3, -2, (1000, 40)), np.random.uniform(2, 3, (1000, 40))], axis=0)
load = Loader(x, shuffle=False)
import os
import tensorflow as tf

model_dir = os.path.join('models1', 'clustered')
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)
os.mkdir(model_dir)

saucie = SAUCIE(x.shape[1], lambda_c=.3, lambda_d=.6)



saucie.train(load,300)
saucie.save(folder=model_dir)
##############################################################################################################
from tensorflow.python import pywrap_tensorflow



reader=pywrap_tensorflow.NewCheckpointReader(r'models1/clustered/./SAUCIE')

all_var=reader.get_variable_to_shape_map()

for key in all_var:  # same as "for key in all_var.keys():"#
    if key == 'encoder0/kernel':
        encoder0W=reader.get_tensor(key)
        print(key)

        print(type(encoder0W))
    elif key =='encoder1/kernel':
        encoder1W = reader.get_tensor(key)
    elif key =='encoder2/kernel':
        encoder2W = reader.get_tensor(key)
    elif key =='embedding/kernel':
        embeddingW = reader.get_tensor(key)
    elif key =='decoder0/kernel':
        decoder0W = reader.get_tensor(key)
    elif key =='decoder1/kernel':
        decoder1W = reader.get_tensor(key)
    elif key =='decoder2/kernel':
        layer_cW = reader.get_tensor(key)

def rho(w,l):  return w + [None,0.1,0.0,0.0][l] * np.maximum(0,w)
def incr(z,l): return z + [None,0.0,0.1,0.0][l] * (z**2).mean()**.5+1e-9
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


encoder0L,encoder1L,encoder2L,embeddingsL,decoder0L,decoder1L,layer_cL = saucie.get_allLayer(load)
# print('encoder0',encoder0L)
# print(encoder0L.shape)




w1 = rho(layer_cW.T, 2)
z1 = incr(np.dot(decoder1L, w1.T), 2)
# s1 = layer_cL / z1
c1 = np.dot(z1, w1)
aj1 = decoder1L * c1





w2 = rho(decoder1W.T,1)
z2 = incr(np.dot(decoder0L,w2.T),1)
# s2 = aj1/z2
c2 = np.dot(z2,w2)
aj2 = decoder0L * c2



w3 = rho(decoder0W.T,1)
z3 = incr(np.dot(embeddingsL,w3.T),1)
# s3 = aj2/z3
c3 = np.dot(z3,w3)
aj3 = embeddingsL * c3

w4 = rho(embeddingW.T,1)
z4 = incr(np.dot(encoder2L,w4.T),1)
# s4 = aj3/z4
c4 = np.dot(z4,w4)
aj4 = encoder2L * c4

w5 = rho(encoder2W.T,1)
z5 = incr(np.dot(encoder1L,w5.T),1)
# s5 = aj4/z5
c5 = np.dot(z5,w5)
aj5 = encoder1L * c5


w6 = rho(encoder1W.T,1)
z6 = incr(np.dot(encoder0L,w6.T),1)
# s6 = aj5/z6
c6 = np.dot(z6,w6)
aj6 = encoder0L * c6



print(np.amax(x))
print(np.amin(x))
wp = np.maximum(0,encoder0W)
wm = np.minimum(0,encoder0W)
lb = x*0 + 1
hb = x*0 - 1
print(lb.shape)

z7 = np.dot(x,encoder0W)-np.dot(lb,wp)-np.dot(hb,wm)+1e-9
#  np.dot(lb,wp.T)
# zz = np.dot(hb,wm.T)
# z = A[0].dot(w)-lb.dot(wp)-hb.dot(wm)+1e-9        # step 1          -np.dot(lb,wp.T)-np.dot(hb,wm.T)+1e-9
# s7 = aj6/z7                                        # step 2
c,cp,cm  = z7.dot(encoder0W.T),z7.dot(wp.T),z7.dot(wm.T)     # step 3
aj7 = x*c-lb*cp-hb*cm                         # step 4

dicts = {}
normalDict = {}
LUADdict = {}
LUCAdict = {}

for i in range(len(aj7)):
    for index,element in enumerate(aj7[0]):
        dicts[index] = float(element)

    import operator

    if label[i] == 0:

        sorted_dict = dict(sorted(dicts.items(), key=operator.itemgetter(1)))
        a = 0
        for key in sorted_dict:
            a+=1
            if a > 20:
                break
            else:
                if tit[int(key)] in normalDict:
                    normalDict[tit[int(key)]]+=1
                else:
                    normalDict[tit[int(key)]] = 1
                # print(tit[int(key)])
    elif label[i] == 1:

        sorted_dict = dict(sorted(dicts.items(), key=operator.itemgetter(1)))
        a = 0
        for key in sorted_dict:
            a += 1
            if a > 20:
                break
            else:
                if tit[int(key)] in LUADdict:
                    LUADdict[tit[int(key)]] += 1
                else:
                    LUADdict[tit[int(key)]] = 1
                # print(tit[int(key)])
    elif label[i] == 2:

        sorted_dict = dict(sorted(dicts.items(), key=operator.itemgetter(1)))
        a = 0
        for key in sorted_dict:
            a += 1
            if a > 20:
                break
            else:
                if tit[int(key)] in LUCAdict:
                    LUCAdict[tit[int(key)]] += 1
                else:
                    LUCAdict[tit[int(key)]] = 1
                # print(tit[int(key)])

sorted_normal = dict(sorted(normalDict.items(), key=operator.itemgetter(1)))
sorted_nLUCA = dict(sorted(LUCAdict.items(), key=operator.itemgetter(1)))
sorted_LUAD = dict(sorted(LUCAdict.items(), key=operator.itemgetter(1)))

print(sorted_normal)
print(sorted_nLUCA)
print(sorted_LUAD)

import seaborn as sb

import matplotlib.pyplot as plt

heat_map = sb.heatmap(np.array(aj7))
plt.show()


embedding = saucie.get_embedding(load)
num_clusters, clusters = saucie.get_clusters(load)
reconstred =saucie.get_reconstruction(load)
#print(embedding.shape)
#print(saucie.get_reconstruction(load).shape)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(embedding[:, 0], embedding[:, 1], c=clusters)
fig.savefig('embedding_by_cluster11.png')
print(clusters)
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(x, clusters))


print("adjusted_rand_score",metrics.adjusted_rand_score(label, clusters))
print("adjusted_mutual_info_score",metrics.adjusted_mutual_info_score(label, clusters))
print("homogeneity_score",metrics.homogeneity_score(label, clusters))
print("v_measure_score",metrics.v_measure_score(label, clusters))
print("fowlkes_mallows_score",metrics.fowlkes_mallows_score(label, clusters))


def dic(arr):
    dics = {}

    dics['x'] = arr[0]
    dics['y'] = arr[1]
    return dics
import csv
with open('datavisiulization11.csv','w',newline='') as ff:
    filename = ['x','y']
    writer = csv.DictWriter(ff,fieldnames=filename)

    writer.writeheader()
    # print(embedding.shape)
    # print(embedding[0])
    for i in range(len(embedding)):
        # print(embedding[i])
        ss = dic(embedding[i])
        writer.writerow(ss)

def dic2(arr):
    dics ={}
    i=0
    for i in range(len(arr)):
        dics[tit[i]] = arr[i]
    return dics
with open('imputed11.csv','w') as ff:
    writer = csv.DictWriter(ff, fieldnames=tit)
    for i in range(len(reconstred)):

        ss = dic2(reconstred[i])
        writer.writerow(ss)






