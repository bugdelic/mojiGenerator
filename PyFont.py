#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import sys
import string

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

#generate_char_img(u'水',"font file path",(400,400)).show()
#sample path
sample_path="/Users/masarumizuochi/programming/of_v0.9.3/apps/Art2017/BaPABar/bin/data/Fonts/NotoSansCJKjp-Regular.otf"

def generate_char_img(char, fontname='Osaka', size=(64, 64)):
    img=Image.new('L', size, 'black')
    draw = ImageDraw.Draw(img)
    fontsize = int(size[0]*0.8)
    font = ImageFont.truetype(fontname, fontsize)

    # adjust charactor position.
    char_displaysize = font.getsize(char)
    offset = tuple((si-sc)//2 for si, sc in zip(size, char_displaysize))
    print "test",offset
    #assert all(o>=0 for o in offset)

    # adjust offset, half value is right size for height axis.
    print offset
    draw.text((offset[0], offset[1]//2), char, font=font, fill='#fff')
    return img

def save_img(img, filepath):
    img.save(filepath, 'png')

def centering(img):
    imgary = np.asarray(img)
    h,w = imgary.shape
    nonzeroh = np.nonzero(sum(imgary))[0]
    nonzerov = np.nonzero(sum(imgary.T))[0]
    marginL = nonzeroh[0]
    marginR = w - nonzeroh[-1] - 1
    marginT = nonzerov[0]
    marginB = h - nonzerov[-1] - 1
    print marginT, marginB, marginL, marginR
    marginV = int((marginT + marginB)/2.0 - marginT)
    marginH = int((marginL + marginR)/2.0 - marginL)
    
    ret=Image.new('L', (w,h), 'black')
    ret.paste(img,(marginH,marginV))
    return ret


def jyoyoList():
    return open("/Users/masarumizuochi/Dropbox/arthackday2018/jyoyo_ichiran.txt",'r').readlines()

def utf2unicode(utf):
    if len(utf) == 6:
        print 4
        return utf[:4].decode("utf8")
    if len(utf) == 5:
        print 3
        return utf[:3].decode("utf8")
    print 3
    return utf[:3].decode("utf8")


eng_char_list = list(string.digits+string.ascii_letters)

def main():
	pass

if __name__ == '__main__':
	argvs=sys.argv
	print argvs
	main()

