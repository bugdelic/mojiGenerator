# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from six.moves import range

import time
import os
import sys
import glob
import cv2
from PIL import Image
from PIL import ImageFilter
from PIL import PngImagePlugin

import numpy as np

import nnabla as nn
import nnabla.logger as logger
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as save

from args import get_args
#from mnist_data import data_iterator_mnist

import os

import iterator

BN56 = False
BN56_DISDROP = False
BN56_DISDROP_NORANDOM = False
BN56_DISDROP_NORANDOM_DEEP = False
BN56_DISDROP_NORANDOM_DEEP2 = True

if BN56:
    from dcgan_kanji_3step_unlink_fin_bn56 import *

if BN56_DISDROP:
    from dcgan_kanji_3step_unlink_fin_bn56_disdrop import *
if BN56_DISDROP_NORANDOM:
    from dcgan_kanji_3step_unlink_fin_bn56_disdrop_norandom import *
if BN56_DISDROP_NORANDOM_DEEP:
    from dcgan_kanji_3step_unlink_fin_bn56_disdrop_norandom_deep import *
if BN56_DISDROP_NORANDOM_DEEP2:
    from dcgan_kanji_3step_unlink_fin_bn56_disdrop_norandom_deep2 import *

GEN_NET_NAME = "./3step_bn56_disdrop/generator_param_036000.h5"
VEC_NET_NAME = "./3step_bn56_disdrop/vectorizer_param_036000.h5"
GEN_NET_NAME = "./3step_bn56_disdrop_norandom/generator_param_113000.h5"
VEC_NET_NAME = "./3step_bn56_disdrop_norandom/vectorizer_param_113000.h5"
GEN_NET_NAME = "./3step_norandom_hiragana/generator_param_053000.h5"
VEC_NET_NAME = "./3step_norandom_hiragana/vectorizer_param_053000.h5"

GEN_NET_NAME = "./3step_norandom_hiragana/generator_param_158000.h5"
VEC_NET_NAME = "./3step_norandom_hiragana/vectorizer_param_158000.h5"
GEN_NET_NAME = "./3step_hiragana_norandom_deep/generator_param_201000.h5"
VEC_NET_NAME = "./3step_hiragana_norandom_deep/vectorizer_param_201000.h5"
GEN_NET_NAME = "./3step_hiragana_norandom_deep2/generator_param_016000.h5"
VEC_NET_NAME = "./3step_hiragana_norandom_deep2/vectorizer_param_016000.h5"
GEN_NET_NAME = "./3step_hiragana_norandom_deep2/generator_param_110000.h5"
VEC_NET_NAME = "./3step_hiragana_norandom_deep2/vectorizer_param_110000.h5"


JYOYO_LIST = "./jyoyo_ichiran.txt"
KANA_LIST = "./hiragana.txt"
JYOYO_IMG_DIR = "./fontImg56"
KANA_IMG_DIR = "./fontImgH56"


def help():
    import dcgan
    c = dcgan.BUGWORD()
    print "make morphing anime"
    c.morphTile(u" ",u" ",size=(56,56),vivid=True).show()
    
    print "make morphing video"
    out = c.morph(u" ",u" ",40)
    dcgan.makeVideo((out+out[::-1])*4,"./out61")


def jyoyoList():
    return open(JYOYO_LIST,'r').readlines()
def kanaList():
    return open(KANA_LIST,'r').readlines()

def utf2unicode(utf):
    if len(utf) == 6:
        return utf[:4].decode("utf8")
    if len(utf) == 5:
        return utf[:3].decode("utf8")
    return utf[:3].decode("utf8")

jyoyo = jyoyoList()
kana = kanaList()
mojidict = {}
for i,j in enumerate(jyoyo):
    mojidict[utf2unicode(j)] = os.path.join(JYOYO_IMG_DIR,str(i).zfill(4)+".png")
for i,j in enumerate(kana):
    mojidict[utf2unicode(j)] = os.path.join(KANA_IMG_DIR,str(3000+i).zfill(4)+".png")

def getCharImg(text):
    if type(text) == str:
        return mojidict[utf2unicode(text)]
    if type(text) == unicode:
        return mojidict[text]

class BUGWORD:
    def __init__(self,netLoad = True):
        self.x = nn.Variable([1, 1, 56, 56])
        TEST_MODE = True
        self.z = vectorizer(self.x,test=TEST_MODE)
        self.y = generator(self.z,test=TEST_MODE)
        
        self.z_ = self.z.unlinked()
        self.y_ = generator(self.z_,test=TEST_MODE)
        if netLoad:
            self.setGen()
            self.setVec()

    def setGen(self,filename = GEN_NET_NAME):
        with nn.parameter_scope("gen"):
            nn.load_parameters(GEN_NET_NAME)

    def setVec(self,filename = VEC_NET_NAME):
        with nn.parameter_scope("vec"):
            nn.load_parameters(VEC_NET_NAME)

    def vectorize(self,img): #imgary= uint8 (56,56) or filename or pilimg
        if type(img) == str:
            return self.vectorize(Image.open(img))
        elif type(img) == Image.Image or type(img) == PngImagePlugin.PngImageFile:
            return self.vectorize(np.asarray(img))
        else:
            self.x.d = u82d(img)
            self.z.forward()
            return self.z.d.copy()

    def generate(self,vecary,vivid = False,outImgFlag=False):
        self.z_.d = vecary
        self.y_.forward()
        if outImgFlag:
            return Image.fromarray(d2u8(self.y_.d[0][0],vivid))
        else:
            return d2u8(self.y_.d[0][0].copy(),vivid)

    def mix(self,char1,char2,ratio=0.5,vivid = False,outImgFlag=False):
        z1 = self.vectorize(getCharImg(char1))
        z2 = self.vectorize(getCharImg(char2))
        z3 = (1.0 - ratio) * z1 + ratio * z2
        return self.generate(z3,vivid,outImgFlag)
    
    def morphTile(self,char1,char2,size=(32,32),tile=(8,8),vivid = False):
        #out = []
        #for i in range(tile[0]*tile[1]):
        #    out.append(self.mix(char1,char2,i/(tile[0]*tile[1]*1.0 - 1.0)))
        out = self.morph(char1,char2,tile[0]*tile[1],vivid = vivid)
        return Image.fromarray(makeTile(out,size,tile))
    def morphLoopTile(self,char1,char2,size=(32,32),tile=(8,8),vivid = False):
        #out = []
        #for i in range(tile[0]*tile[1]/2):
        #    out.append(self.mix(char1,char2,i/(tile[0]*tile[1]*0.5 - 1.0)))
        out = self.morph(char1,char2,tile[0]*tile[1]/2,vivid = vivid)
        return Image.fromarray(makeTile(out+out[::-1],size,tile))
    def morph(self,char1,char2,step=100,vivid = False):
        out = []
        for i in range(step):
            out.append(self.mix(char1,char2,i/(step - 1.0),vivid))
        return out
    def morphChain(self,charList = ["","","","",""],step=100,vivid = False):
        out = []
        for j in range(len(charList)-1):
            for i in range(step):
                out.append(self.mix(charList[j],charList[j+1],i/(step - 1.0),vivid))
        return out

def makeVideo(arys,outDir):
    if os.path.isdir(outDir):
        return
    os.system("mkdir "+outDir)
    for i in range(len(arys)):
        img = Image.fromarray(arys[i])
        img.save(os.path.join(outDir,"img"+str(i).zfill(3)+".png"))
    os.system("ffmpeg -r 30 -i " + os.path.join(outDir,"img")+"%"+"03d.png -vcodec libx264 -pix_fmt yuv420p -r 60 "+os.path.join(outDir,"out.mp4"))


def makeTile(arys,size=(32,32),tile=(8,8)):
    outW = size[0]*tile[0]
    outH = size[1]*tile[1]
    out = np.zeros((outH,outW),dtype="uint8")
    for i in range(tile[1]):
        for j in range(tile[0]):
            if len(arys) > (i*tile[0]+j):
                out[i * size[1]:(i+1) * size[1], j * size[0]:(j+1) * size[0]] = cv2.resize( arys[i*tile[0]+j],size)
    return refine(out)

neiborhood4 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]],
                            np.uint8)

neiborhood8 = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                            np.uint8)

limEdgeAry = np.ones((28,28))
EDGE = 6
for i in range(EDGE):
    limEdgeAry[i,:] *= ((1.0 * (i/EDGE))**2)
for i in range(EDGE):
    limEdgeAry[:,i] *= ((1.0 * (i/EDGE))**2)
limEdgeAry = np.hstack(( limEdgeAry,limEdgeAry[:,::-1]))
limEdgeAry = np.vstack(( limEdgeAry,limEdgeAry[::-1,:]))

def limEdge(imgary):
    return imgary*limEdgeAry

def refine(imgary):
    return cv2.equalizeHist(imgary)

def refine4(imgary):
    img = Image.fromarray(imgary)
    img.show();time.sleep(1)
    img2 = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img2.show();time.sleep(1)
    imgary2 = np.asarray(img2)
    imgary3 = cv2.equalizeHist(imgary2)
    Image.fromarray(imgary3).show();time.sleep(1)
    return imgary3

def refine3(imgary):
    Image.fromarray(imgary).show();time.sleep(1)
    img = Image.fromarray(cv2.equalizeHist(imgary))
    img.show();time.sleep(1)
    img2 = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img2.show();time.sleep(1)
    imgary2 = np.asarray(img2)
    return imgary2

def refine2(imgary):
    img = Image.fromarray(imgary)
    img.show();time.sleep(1)
    img2 = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img2.show();time.sleep(1)
    imgary2 = np.asarray(img2)

    """
        imgary3 = cv2.dilate(imgary2,
                                neiborhood4,
                                iterations=1)
        Image.fromarray(imgary3).show();time.sleep(1)
        imgary4 = cv2.erode(imgary3,
                                neiborhood4,
                                iterations=1)
        Image.fromarray(imgary4).show();time.sleep(1)
        imgary4 = cv2.erode(imgary4,
                                neiborhood4,
                                iterations=1)
        Image.fromarray(imgary4).show();time.sleep(1)
        imgary5 = cv2.dilate(imgary4,
                                neiborhood4,
                                iterations=1)
        Image.fromarray(imgary5).show();time.sleep(1)
    """
    opening = cv2.morphologyEx(imgary2, cv2.MORPH_CLOSE, neiborhood4)
    Image.fromarray(opening).show();time.sleep(1)
    imgary5 = cv2.morphologyEx(opening, cv2.MORPH_OPEN, neiborhood4)
    Image.fromarray(imgary5).show();time.sleep(1)
    return imgary5

def u82d(d):
    return d / 255 * 2.0 - 1

def d2u8(d_, vivid = False):
    d = limEdge(d_ + 1) - 1
    if vivid:
        return np.uint8((d > 0) * 255)
    return np.uint8((d + 1) / 2.0 * 255)

if __name__ == '__main__':
    argvs=sys.argv
    print argvs
    main(argvs)