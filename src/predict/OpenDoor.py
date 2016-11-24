from __future__ import print_function
from createDataSet import initDataSet
from Predict import predict
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
from SaveCnn import CnnModel
from SaveCnn import HiddenLayer
from SaveCnn import LeNetConvPoolLayer
from SaveCnn import LogisticRegression
from SaveCnn import load_data
import copy
import BinAndCrop

if __name__ == '__main__':
    start = BinAndCrop.BinAndCropClass()
    start.bin()
    print('222')
    initDataSet()
    result = predict('file-126.pkl.gz')
    print (result)