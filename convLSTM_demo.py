#!/usr/bin/env python
import six
import cupy
#import numpy
import chainer
from chainer import computational_graph, Chain, Variable, utils, gradient_check, Function
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers

from PIL import Image

#read Kitti datas
str_dir1 = '65x65data/'
str_png = '.png'
str_03 = '500'
str_02 = '50'
str_01 = '5'

epochNum = 10 # the number of epoch
len_1data = 25
channelIn = 1 #channel number of input image
channelOut = 1 #channel number of output image
width = 65 #width of input image
height = 65 #height of imput image
ksize = 5
padsize = (ksize - 1) / 2

imgArrayTrain = cupy.zeros((126, height, width), dtype=cupy.float32)
imgArrayTest = cupy.zeros((26, height, width), dtype=cupy.float32)


#load to train array
for i in range(126):
    str_sum = str_dir1
    if i < 10:
        str_sum = str_dir1 + str_03 + str(i) + str_png

    elif i < 100:
        str_sum = str_dir1 + str_02 + str(i) + str_png

    else:
        str_sum = str_dir1 + str_01 + str(i) + str_png
    
    img_read = Image.open(str_sum)
    imgArrayPart = cupy.asarray(img_read).astype(dtype=cupy.float32)
    imgArrayTrain[i] = imgArrayPart

imgArrayTrain = imgArrayTrain / 255
imgArrayTrain2 = imgArrayTrain.reshape(len(imgArrayTrain), height, width).astype(dtype=cupy.float32)


#model class
class MyLSTM(chainer.Chain):
    def __init__(self):
        super(MyLSTM, self).__init__(
            Wz = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Wi = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Wf = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Wo = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Rz = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Ri = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Rf = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            Ro = L.Convolution2D(channelIn, channelOut, ksize, stride=1, pad=padsize),
            #W = L.Linear(k, k),
        )

    def __call__(self, s): #s is expected to cupyArray(num, height, width)
        accum_loss = None
        chan = channelIn
        hei = len(s[0])
        wid = len(s[0][0])
        h = Variable(cupy.zeros((1, chan, hei, wid), dtype=cupy.float32))
        c = Variable(cupy.zeros((1, chan, hei, wid), dtype=cupy.float32))
        
        for i in range(len(s) - 1): #len(s) is expected to 26

            tx = Variable(cupy.array(s[i + 1], dtype=cupy.float32).reshape(1, chan, hei, wid))
            x_k = Variable(cupy.array(s[i], dtype=cupy.float32).reshape(1, chan, hei, wid))
            z0 = self.Wz(x_k) + self.Rz(h)
            z1 = F.tanh(z0)
            i0 = self.Wi(x_k) + self.Ri(h)
            i1 = F.sigmoid(i0)
            f0 = self.Wf(x_k) + self.Rf(h)
            f1 = F.sigmoid(f0)
            c = z1 * i1 + f1 * c
            o0 = self.Wo(x_k) + self.Ro(h)
            o1 = F.sigmoid(o0)
            h = o1 * F.tanh(c)
            loss = F.mean_squared_error(h, tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss

        return accum_loss

#optimize
model = MyLSTM()
cuda.get_device(0).use() #for GPU
model.to_gpu() #for GPU
optimizer = optimizers.Adam()
optimizer.setup(model)

#learning phase
for epoch in range(epochNum):
    
    print "epoch =", epoch
    for j in range(5):
        
        print "now j is", j
        s = imgArrayTrain[j*25:(j+1)*25 + 1,:]
        model.zerograds()
        loss = model(s)
        loss.backward()    
        optimizer.update()

print 'learning is finished'





