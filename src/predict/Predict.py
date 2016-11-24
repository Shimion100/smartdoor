from __future__ import print_function
import six.moves.cPickle as pickle
import theano
import theano.tensor as T
from SaveCnn import CnnModel
from SaveCnn import HiddenLayer
from SaveCnn import LeNetConvPoolLayer
from SaveCnn import LogisticRegression
from SaveCnn import load_data
import copy


class DoorModel(object):

    def copyInit(self, fromModel, learning_rate=0.01, dataset='mnist.pkl.gz', nkerns=[32, 60, 90, 120], batch_size=1):
        # Init the params
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.nkerns = nkerns
        self.batch_size = batch_size
        datasets = load_data(dataset)
        # self.train_set_x, self.train_set_y = datasets[0]
        # self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        # Read the data
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')

        self.layer0_input = x.reshape((batch_size, 1, 126, 126))
        self.layer0 = LeNetConvPoolLayer()
        self.layer0.copyModel(
            input=self.layer0_input,
            image_shape=(batch_size, 1, 126, 126),
            filter_shape=(nkerns[0], 1, 7, 7),
            poolsize=(2, 2),
            w=copy.deepcopy(fromModel.layer0.W),
            b=copy.deepcopy(fromModel.layer0.b)

        )
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)

        self.layer1 = LeNetConvPoolLayer()
        self.layer1.copyModel(
            input=self.layer0.output,
            image_shape=(batch_size, nkerns[0], 60, 60),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2),
            w=copy.deepcopy(fromModel.layer1.W),
            b=copy.deepcopy(fromModel.layer1.b)
        )

        self.layer2 = LeNetConvPoolLayer()
        self.layer2.copyModel(
            input=self.layer1.output,
            image_shape=(batch_size, nkerns[1], 28, 28),
            filter_shape=(nkerns[2], nkerns[1], 5, 5),
            poolsize=(2, 2),
            w=copy.deepcopy(fromModel.layer2.W),
            b=copy.deepcopy(fromModel.layer2.b)
        )

        self.layer3 = LeNetConvPoolLayer()
        self.layer3.copyModel(
            input=self.layer2.output,
            image_shape=(batch_size, nkerns[2], 12, 12),
            filter_shape=(nkerns[3], nkerns[2], 5, 5),
            poolsize=(2, 2),
            w=copy.deepcopy(fromModel.layer3.W),
            b=copy.deepcopy(fromModel.layer3.b)
        )

        self.layer4_input = self.layer3.output.flatten(2)

        self.layer4 = HiddenLayer()

        self.layer4.copyInit(
            input=self.layer4_input,
            n_in=nkerns[3] * 4 * 4,
            n_out=480,
            W=copy.deepcopy(fromModel.layer4.W),
            b=copy.deepcopy(fromModel.layer4.b),
            activation=T.tanh
        )

        self.layer5 = LogisticRegression()
        self.layer5.copyInit(input=self.layer4.output, n_in=480, n_out=12,
                             W=copy.deepcopy(fromModel.layer5.W),
                             b=copy.deepcopy(fromModel.layer5.b))

        self.predict_model = theano.function(
            [index],
            self.layer5.y_pred,
            givens={
                x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
            }
        )

        # read the firstData every time
        aaIndex = 0

        self.predict_result = self.predict_model(aaIndex)


        # End of building model





def predict(my_data_set='file-126.pkl.gz'):

    cnnModel = pickle.load(open('best_model.pkl'))
    aDoorModel = DoorModel()
    aDoorModel.copyInit(cnnModel, 0.01, my_data_set, [16, 30, 40, 120], 1)
    aIndex = T.lscalar()
    predict_the_model = theano.function(
        inputs=[aIndex],
        outputs=theano.shared(aDoorModel.predict_result),
        on_unused_input='ignore'
    )

    print("Start----------------------------------")
    predicted_values = predict_the_model(20)
    print(predicted_values)
    print("End----------------------------------")
    name = ''
    if predicted_values == 0:
        name = 'cjr'
    elif predicted_values == 1:
        name = 'dbw'
    elif predicted_values == 2:
        name = 'gwd'
    elif predicted_values == 3:
        name = 'hzz'
    elif predicted_values == 4:
        name = 'jxd'
    elif predicted_values == 5:
        name = 'wy'
    elif predicted_values == 6:
        name = 'll'
    elif predicted_values == 7:
        name = 'qhx'
    elif predicted_values == 8:
        name = 'wwl'
    elif predicted_values == 9:
        name = 'xcl'
    elif predicted_values == 10:
        name = 'zsy'
    elif predicted_values == 11:
        name = 'zx'
    else:
        name = ''

    print(name)

    if predicted_values > 0:
        return True
    else:
        return False

"""
    Main method
"""
if __name__ == '__main__':
    predict('file-126.pkl.gz')
