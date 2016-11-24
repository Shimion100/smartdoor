
from __future__ import print_function
import gzip
import os
import sys
import timeit
import six.moves.cPickle as pickle
import numpy
from PIL import Image
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
import copy



"""
    Load data
"""
def load_data(dataset):

    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    def shared_dataset(data_xy, borrow=True):

        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

"""--------------------------------------------------------------------------------"""
"""
    HiddenLayer
"""

# start-snippet-1
class HiddenLayer(object):
    def __init__(self):
        print("Init hiddenLayer")
    def copyInit(self,input, n_in, n_out, W=None, b=None,activation=T.tanh):
        print("copyInit")
        self.input = input
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def oriInit(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

"""--------------------------------------------------------------------------------"""

class LogisticRegression(object):
    def __init__(self):
        print("init do nothing")
    def copyInit(self, input, n_in, n_out, W, b):
        self.W = W
        # initialize the biases b as a vector of n_out 0s
        self.b = b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input


    def oriInit(self, input, n_in, n_out):

        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.input = input

    def negative_log_likelihood(self, y):

        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

"""--------------------------------------------------------------------------------"""
"""
    ConvClass
"""
class LeNetConvPoolLayer (object):
    # to Copy a model
    def copyModel(self, input, filter_shape, image_shape, poolsize=(2, 2), w=T.dtensor4(), b=T.dvector()):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = w
        print(w)
        self.b = b
        print(b)
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = input

    # to init a new random model
    def CovDwonPoolingInit(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        assert image_shape[1] == filter_shape[1]
        self.input = input
        # the W only have relaship with the filter_shape
        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = input
    def __init__(self):
        print("void init")
"""--------------------------------------------------------------------------------"""

"""
    The model class
"""

class CnnModel(object):

    def copyInit(self, fromModel, learning_rate=0.1, dataset='mnist.pkl.gz', nkerns=[32, 60, 90,120], batch_size=1):
        print("copy from a model------------------");
        # Init the params
        self.learning_rate = learning_rate
        print("learning_rate----------------------%", learning_rate)
        self.dataset = dataset
        self.nkerns = nkerns
        self.batch_size = batch_size
        datasets = load_data(dataset)
        #self.train_set_x, self.train_set_y = datasets[0]
        #self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        # Read the data
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')

        # Building model
        print('... building the model')

        self.layer0_input = x.reshape((batch_size, 1, 126, 126))

        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)

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
                            W = copy.deepcopy(fromModel.layer5.W),
                            b = copy.deepcopy(fromModel.layer5.b))



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

    def __init__(self):
        print("The default Init----------------------")
        self.layer0 = LeNetConvPoolLayer()
        self.layer1 = LeNetConvPoolLayer()
        self.layer2 = LeNetConvPoolLayer()
        self.layer3 = LeNetConvPoolLayer()

    def oriInit(self, learning_rate, dataset='mnist.pkl.gz', nkerns=[32, 60, 90, 120], batch_size=1):
        #Init the params
        self.learning_rate = learning_rate
        print("learning_rate----------------------%",learning_rate)
        self.dataset = dataset
        self.nkerns = nkerns
        self.batch_size = batch_size
        datasets = load_data(dataset)
        self.train_set_x, self.train_set_y = datasets[0]
        self.valid_set_x, self.valid_set_y = datasets[1]
        self.test_set_x, self.test_set_y = datasets[2]

        #Read the data
        rng = numpy.random.RandomState(23455)
        index = T.lscalar()
        x = T.matrix('x')
        y = T.ivector('y')

        #Building model
        print('... building the model')

        self.layer0_input = x.reshape((batch_size, 1, 126, 126))

        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)

        self.layer0.CovDwonPoolingInit(
            rng,
            input=self.layer0_input,
            image_shape=(batch_size, 1, 126, 126),
            filter_shape=(nkerns[0], 1, 7, 7),
            poolsize=(2, 2)
        )

        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)

        self.layer1.CovDwonPoolingInit(
            rng,
            input=self.layer0.output,
            image_shape=(batch_size, nkerns[0], 60, 60),
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            poolsize=(2, 2)
        )

        self.layer2.CovDwonPoolingInit(
            rng,
            input=self.layer1.output,
            image_shape=(batch_size, nkerns[1], 28, 28),
            filter_shape=(nkerns[2], nkerns[1], 5, 5),
            poolsize=(2, 2)
        )
        self.layer3.CovDwonPoolingInit(
            rng,
            input=self.layer2.output,
            image_shape=(batch_size, nkerns[2], 12, 12),
            filter_shape=(nkerns[3], nkerns[2], 5, 5),
            poolsize=(2, 2)
        )

        self.layer4_input = self.layer3.output.flatten(2)

        self.layer4 = HiddenLayer()
        self.layer4.oriInit(
            rng,
            input=self.layer4_input,
            n_in=nkerns[3] * 4 * 4,
            n_out=480,
            activation=T.tanh
        )

        self.layer5 = LogisticRegression()
        self.layer5.oriInit(input=self.layer4.output, n_in=480, n_out=12)

        cost = self.layer5.negative_log_likelihood(y)

        self.test_model = theano.function(
            [index],
            self.layer5.errors(y),
            givens={
                x: self.test_set_x[index * batch_size: (index + 1) * batch_size],
                y: self.test_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        self.predict_model = theano.function(
            [index],
            self.layer5.y_pred,
            givens={
                x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
            }
        )

        self.validate_model = theano.function(
            [index],
            self.layer5.errors(y),
            givens={
                x: self.valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: self.valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        params = self.layer5.params + self.layer4.params + self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params
        grads = T.grad(cost, params)

        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)
            ]

        self.train_model = theano.function(
            [index],
            cost,
            updates=updates,
            givens={
                x: self.train_set_x[index * batch_size: (index + 1) * batch_size],
                y: self.train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )

        # End of building model

    """--------------------------------------------------------------------------------"""
    def trainModel(self, n_epochs):
        print("n_epochs-----------------------%",n_epochs)
        #Prepare the dataset

        n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        n_train_batches //= self.batch_size
        n_valid_batches //= self.batch_size
        n_test_batches //= self.batch_size

        print("---self.batch_size--------n_train_batches-------", self.batch_size, n_train_batches)

        ###############
        # TRAIN MODEL #
        ###############
        print('... training')
        # early-stopping parameters
        patience = 100  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
        # found
        improvement_threshold = 0.995  # a relative improvement of this much is
        # considered significant
        validation_frequency = min(n_train_batches, patience // 2)
        # go through this many
        # minibatche before checking the network
        # on the validation set; in this case we
        # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print('training @ iter = ', iter)
                cost_ij = self.train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in range(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss * \
                                improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in range(n_test_batches)
                            ]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print(('The code for file ' +
               os.path.split(__file__)[1] +
               ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    """--------------------------------------------------------------------------------"""

"""
    Main method
"""
if __name__ == '__main__':
    model = CnnModel()
    model.oriInit(0.01, 'mnist.pkl.gz', [16, 30, 40, 120], 120)
    model.trainModel(300)
    storedModel = CnnModel()
    storedModel.copyInit(model, 0.01, 'file-126.pkl.gz', [16, 30, 40, 120], 1);
    with open('best_model.pkl', 'wb') as f:
        pickle.dump(storedModel, f)
