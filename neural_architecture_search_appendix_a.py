import six
import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
import nutszebra_chainer
import functools
from collections import defaultdict


class Conv(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.conv(x)

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class Conv_ReLU_BN(nutszebra_chainer.Model):

    def __init__(self, in_channel, out_channel, filter_size=(3, 3), stride=(1, 1), pad=(1, 1)):
        super(Conv_ReLU_BN, self).__init__(
            conv=L.Convolution2D(in_channel, out_channel, filter_size, stride, pad),
            bn=L.BatchNormalization(out_channel),
        )

    def weight_initialization(self):
        self.conv.W.data = self.weight_relu_initialization(self.conv)
        self.conv.b.data = self.bias_initialization(self.conv, constant=0)

    def __call__(self, x, train=False):
        return self.bn(F.relu(self.conv(x)), test=not train)

    def count_parameters(self):
        return functools.reduce(lambda a, b: a * b, self.conv.W.data.shape)


class AppendixA(nutszebra_chainer.Model):

    def __init__(self, category_num):
        super(AppendixA, self).__init__()
        out_channels = [36, 48, 36, 36, 48, 48, 48, 36, 36, 36, 36, 48, 48, 48, 48]
        #                    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
        skip_connections = [[0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                            [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            ]
        filters = [(3, 3), (3, 3), (3, 3), (5, 5), (3, 7), (7, 7), (7, 7), (7, 3), (7, 1), (7, 7), (5, 7), (7, 7), (7, 5), (7, 5), (7, 5)]
        modules = []
        in_channel = 3
        for i in six.moves.range(len(out_channels)):
            modules += [('conv{}'.format(i), Conv_ReLU_BN(in_channel, out_channels[i], filters[i], 1, 0))]
            in_channel = int(np.sum([out_channels[ii] for ii, s in enumerate(skip_connections) if s[i] == 1])) + out_channels[i]
        modules += [('linear', Conv(out_channels[-1], category_num, 1, 1, 0))]
        # register layers
        [self.add_link(*link) for link in modules]
        self.modules = modules
        self.category_num = category_num
        self.out_channels = out_channels
        self.skip_connections = skip_connections
        self.filters = filters
        self.name = 'appndix_a_{}'.format(category_num)

    def weight_initialization(self):
        [link.weight_initialization() for _, link in self.modules]

    def count_parameters(self):
        return int(np.sum([link.count_parameters() for _, link in self.modules]))

    @staticmethod
    def _zero_pads(x, pad, axis):
        if type(x.data) is not np.ndarray:
            pad.to_gpu()
        return F.concat((x, pad), axis=axis)

    @staticmethod
    def zero_pads(x, sizes):
        batch, channel, height, width = x.data.shape
        diff_height = sizes[2] - height
        diff_width = sizes[3] - width
        # pad along with height
        if diff_height >= 1:
            pad = chainer.Variable(np.zeros((batch, channel, diff_height, width), dtype=x.dtype), volatile=x.volatile)
            x = AppendixA._zero_pads(x, pad, axis=2)
            _, _, height, _ = x.data.shape
        # pad along with width
        if diff_width >= 1:
            pad = chainer.Variable(np.zeros((batch, channel, height, diff_width), dtype=x.dtype), volatile=x.volatile)
            x = AppendixA._zero_pads(x, pad, axis=3)
        return x

    def _max(a, b):
        return (max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    @staticmethod
    def concatenate(X):
        sizes = (0, 0, 0, 0)
        for x in X:
            sizes = AppendixA._max(sizes, x.data.shape)
        X = [AppendixA.zero_pads(x, sizes) for x in X]
        return F.concat(X, axis=1)

    def __call__(self, x, train=False):
        x = [x]
        outputs = []
        for i in six.moves.range(len(self.out_channels)):
            x = self['conv{}'.format(i)](self.concatenate(x), train=train)
            outputs.append(x)
            x = [outputs[ii] for ii, s in enumerate(self.skip_connections) if s[i] == 1] + [outputs[i]]
        x = outputs[-1]
        batch, channels, height, width = x.data.shape
        x = F.reshape(F.average_pooling_2d(x, (height, width)), (batch, channels, 1, 1))
        return F.reshape(self.linear(x, train), (batch, self.category_num))

    def calc_loss(self, y, t):
        loss = F.softmax_cross_entropy(y, t)
        return loss

    def accuracy(self, y, t, xp=np):
        y.to_cpu()
        t.to_cpu()
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == True)[0]
        accuracy = defaultdict(int)
        for i in indices:
            accuracy[t.data[i]] += 1
        indices = np.where((t.data == np.argmax(y.data, axis=1)) == False)[0]
        false_accuracy = defaultdict(int)
        false_y = np.argmax(y.data, axis=1)
        for i in indices:
            false_accuracy[(t.data[i], false_y[i])] += 1
        return accuracy, false_accuracy
