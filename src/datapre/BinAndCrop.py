import os
import sys
from PIL import Image
#from PIL.Image import core as image
import os, random, string, shutil

path = "."
src = "./src"

"""
    walk through a directory and get all the file in this directory.
"""


def subfilesName(directory):
    fl = []
    subdir=os.listdir(directory)
    """
    for i in os.walk(path, False):
        for f in i[2]:
            fl.append(f)
    """
    for i in subdir:
        path=os.path.join(directory,i)
        print(path)
        if os.path.isdir(path):
            fl.extend(subfilesName(path))
        elif os.path.isfile(path):
            fl.append(path)
            print("add"+path)    
    return fl


"""
    crop the image to fixed size
"""


def cropImage(fileName):
    print("Processing ... ", fileName)
    # open the image
    img = Image.open(fileName)
    img = img.resize((126, 126), Image.ANTIALIAS)
    img.save(fileName, "JPEG")


"""
    PreOperation of Image, include the binarization and crop
"""


def bin(fileName):
    state="covertL"
    print("Processing ... ", fileName)
    # open the image
    img = Image.open(fileName)
    img = img.convert("L")
    img.save(fileName, "JPEG")
    if state=="covertL":
        return 0;
    pixdata = img.load()
    rows = img.size[0]
    cols = img.size[1]
    # scan by cols

    # you could change the range function to xrange function
    for y in range(cols):
        for x in range(rows):
            if pixdata[x, y] > 127:
                pixdata[x, y] = 255
            else:
                pixdata[x, y] = 0
    img.save(fileName, "JPEG")


def binaryzationJpg(src):

    print("start binaryzationJpg()")
    files = subfilesName(src)    
    length = len(files)
    print("###########", src, "##", length, "###########")
    i = 1
    for f in files:
        print("this is"+f)
        if os.path.isfile(f):
            #bin(os.path.join(src , f))
            bin(f)
            i += 1
            #cropImage(os.path.join(src , f))
            cropImage(f)
        print(f)
    print("end binaryzationJpg()")

"""
    only process one directory.
"""
def binaryzations():
    binaryzationJpg("./train/face/")


# command=raw_input
binaryzations()

