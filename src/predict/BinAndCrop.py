from PIL import Image
#from PIL.Image import core as image
import os

"""
    walk through a directory and get all the file in this directory.
"""


class BinAndCropClass():
    def __init__(self, path='../data/img/test/data28/pic.jpg'):
        self.path = path
        print 'start'

    def bin(self):
        # open the image
        img = Image.open(self.path)
        img = img.convert("L")
        img = img.resize((126, 126), Image.ANTIALIAS)
        new_name = self.path[:24] + '2_2222'
        print(new_name)
        img.save(new_name, "JPEG")
        os.remove(self.path)

if __name__ == '__main__':
    f = '../data/img/test/data28/pic.jpg'
    start = BinAndCropClass(f)
    start.bin()


