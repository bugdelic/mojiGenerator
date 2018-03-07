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

import os
import sys
import glob
from PIL import Image
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
BN56_DISDROP_NORANDOM = True

if BN56:
    from dcgan_kanji_3step_unlink_fin_bn56 import *

if BN56_DISDROP:
    from dcgan_kanji_3step_unlink_fin_bn56_disdrop import *
if BN56_DISDROP_NORANDOM:
    from dcgan_kanji_3step_unlink_fin_bn56_disdrop_norandom import *


VEC_SIZE = 1000

def main_simple(argvs):
    testId1 = int(argvs[1])
    testId2 = int(argvs[2])
    outDir = argvs[3]
    print testId1,"and",testId2," ->",outDir
    x = nn.Variable([1, 1, 28, 28])
    z = vectorizer(x)
    y = generator(z)
    with nn.parameter_scope("gen"):
        #nn.load_parameters("./tmp.monitor.dcgan.3step_unlink_fin_/generator_param_004000.h5")
        if BN56:
            nn.load_parameters("./tmp.monitor.dcgan.3step_unlink_fin_/generator_param_028000.h5")
        if BN56_DISDROP:
            nn.load_parameters("./3step_bn56_disdrop/generator_param_036000.h5")
            
    with nn.parameter_scope("vec"):
        #nn.load_parameters("./tmp.monitor.dcgan.3step_unlink_fin_/vectorizer_param_004000.h5")
        if BN56:
            nn.load_parameters("./tmp.monitor.dcgan.3step_unlink_fin_/vectorizer_param_028000.h5")
        if BN56_DISDROP:
            nn.load_parameters("./3step_bn56_disdrop/vectorizer_param_036000.h5")
    data = load_kanji_data()

    x.d = u82d( data[0][testId1:testId1+1] )
    y.forward()
    imgX = Image.fromarray(d2u8(x.d[0][0]))
    imgY = Image.fromarray(d2u8(y.d[0][0]))
    imgX.save(os.path.join(outDir,"imgX.png"))
    imgY.save(os.path.join(outDir,"imgY.png"))

def main(argvs):
    testId1 = int(argvs[1])
    testId2 = int(argvs[2])
    outDir = argvs[3]
    os.system("mkdir "+outDir)
    print testId1,"and",testId2," ->",outDir
    x = nn.Variable([1, 1, 56, 56])
    z = vectorizer(x,test=True)
    z_ = z.unlinked()
    y = generator(z,test=True)
    y_ = generator(z_,test=True)

    with nn.parameter_scope("gen"):
        #nn.load_parameters("./tmp.monitor.dcgan.3step_unlink_fin_/generator_param_004000.h5")
        #nn.load_parameters("./tmp.monitor.dcgan.3step_unlink_fin_/generator_param_028000.h5")
        if BN56:
            nn.load_parameters("./tmp.monitor.dcgan.3step_unlink_fin_bn56/generator_param_078000.h5")
        if BN56_DISDROP:
            nn.load_parameters("./3step_bn56_disdrop/generator_param_036000.h5")
        if BN56_DISDROP_NORANDOM:
            nn.load_parameters("./3step_norandom_hiragana/generator_param_053000.h5")
    with nn.parameter_scope("vec"):
        #nn.load_parameters("./tmp.monitor.dcgan.3step_unlink_fin_/vectorizer_param_004000.h5")
        #nn.load_parameters("./tmp.monitor.dcgan.3step_unlink_fin_/vectorizer_param_028000.h5")
        if BN56:
            nn.load_parameters("./tmp.monitor.dcgan.3step_unlink_fin_bn56/vectorizer_param_078000.h5")
        if BN56_DISDROP:
            nn.load_parameters("./3step_bn56_disdrop/vectorizer_param_036000.h5")
        if BN56_DISDROP_NORANDOM:
            nn.load_parameters("./3step_norandom_hiragana/vectorizer_param_053000.h5")
    data = load_kanji_data()

    x.d = u82d( data[0][testId1:testId1+1] )
    y.forward()
    img = Image.fromarray(d2u8(y.d[0][0]))
    img.save(os.path.join(outDir,"imgID1.png"))
    vec1 = z.d.copy()
    x.d = u82d( data[0][testId2:testId2+1] )
    y.forward()
    img = Image.fromarray(d2u8(y.d[0][0]))
    img.save(os.path.join(outDir,"imgID2.png"))
    vec2 = z.d.copy()

    if 0: #morphing vec1 -> vec2
        for i in range(100+1):
            vec = ( vec1 * i + vec2 * (100 - i) ) / 100.0
            z_.d = vec
            y_.forward()
            img = Image.fromarray(d2u8(y_.d[0][0],vivid = False))
            img.save(os.path.join(outDir,"img"+str(i).zfill(3)+".png"))
        os.system("ffmpeg -r 30 -i " + os.path.join(outDir,"img")+"%"+"03d.png -vcodec libx264 -pix_fmt yuv420p -r 60 "+os.path.join(outDir,"out.mp4"))
    if 1:
        for i in range(101):
            vec = ( vec1 + vec2 ) / 2.0
            vecBuf1 = ( vec * i + vec1 * (100 - i) ) / 100.0
            vecBuf2 = ( vec * i + vec2 * (100 - i) ) / 100.0
            z_.d = vecBuf1
            y_.forward()
            y_Buf = y_.d[0][0].copy()
            z_.d = vecBuf2
            y_.forward()
            y_Buf = np.hstack( ( y_Buf, y_.d[0][0].copy()))
            img = Image.fromarray(d2u8(y_Buf,vivid = False))
            img.save(os.path.join(outDir,"img"+str(i).zfill(3)+".png"))
        os.system("ffmpeg -r 30 -i " + os.path.join(outDir,"img")+"%"+"03d.png -vcodec libx264 -pix_fmt yuv420p -r 60 "+os.path.join(outDir,"out.mp4"))




def u82d(d):
    return d / 255 * 2.0 - 1

def d2u8(d, vivid = False):
    if vivid:
        return np.uint8((d > 0) * 255)
    return np.uint8((d + 1) / 2.0 * 255)

if __name__ == '__main__':
    argvs=sys.argv
    print argvs
    main(argvs)