from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd
import theano
class pcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    Blue='\34[95m'
    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


def dir_to_dataset(glob_files):
    print("Gonna process:\n\t %s"%glob_files)
    dataset = []
    clazz = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        image = Image.open(file_name)
	print file_name
        img = Image.open(file_name).convert('LA') #tograyscale
        pixels = [f[0] for f in list(img.getdata())]
        dataset.append(pixels)
        classLabel=file_name.split("_")[1];
	print file_name,"<--->",classLabel
	clazz.append(classLabel)
        if file_count % 1000 == 0:
            print("\t %s files processed"%file_count)
    # outfile = glob_files+"out"
    # np.save(outfile, dataset)

    return np.array(dataset), np.array(clazz)

def initDataSet():
	path="./train/data28/*";
	Data, y = dir_to_dataset(path)
	# Data and labels are read 
	print Data,y
	train_set_x =np.asarray(Data,dtype=theano.config.floatX);
	train_set_y=y;
      
	path="./verify/data28/*";
	Data, y = dir_to_dataset(path)
		
	val_set_x = np.asarray(Data,dtype=theano.config.floatX);
	val_set_y = y

	path="./test/data28/*";
	Data, y = dir_to_dataset(path)

	test_set_x = np.asarray(Data,dtype=theano.config.floatX);
	test_set_y = y

	train_set = train_set_x, train_set_y
	val_set = val_set_x, val_set_y
	test_set = test_set_x, test_set_y
	dataset = [train_set, val_set, test_set]
	f = gzip.open('file-28.pkl.gz','wb')
	cPickle.dump(dataset, f, protocol=2)
	f.close()

#this is to test the data
def testData(index):
        f = gzip.open('file-28.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        train_set_x=train_set[0];
        train_set_y=train_set[1];
        sample=np.asarray(train_set_x,dtype=theano.config.floatX)
        print len(sample[0])
        a=sample[index]
        j=0;
        for i in xrange(0,len(a)):
              if a[i]>128:
                   print pcolors.FAIL+"*",
              else:
                   print pcolors.HEADER+"0",
              if ((i+1)%28==0):
                   print ""
        print "label is:%s"%train_set_y[index]


# Divided dataset into 3 parts. I had 6281 images.
initDataSet();
#train_set = train_set_x, train_set_y
#val_set = val_set_x, val_set_y
#test_set = test_set_x, val_set_y

#dataset = [train_set, val_set, test_set]

#f = gzip.open('file.pkl.gz','wb')
#cPickle.dump(dataset, f, protocol=2)
#f.close()
