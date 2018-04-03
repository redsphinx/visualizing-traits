import chainer
import numpy as np


class Generator(chainer.Chain):
    @staticmethod
    def make_deconv(multiplier, in_channels, s=(1,1)):
        base = 3
        deconv = chainer.links.Deconvolution2D(in_channels=in_channels,
                                               out_channels=multiplier * base,
                                               ksize=(4, 4),
                                               stride=s,
                                               pad=(1, 1),
                                               initialW=chainer.initializers.GlorotNormal())
        return deconv

    # def make_deconv(out_channels, in_channels, s=(2, 2), p=(1, 1)):
    #     deconv = chainer.links.Deconvolution2D(in_channels=in_channels,
    #                                            out_channels=out_channels,
    #                                            ksize=(4, 4),
    #                                            stride=s,
    #                                            pad=p,
    #                                            initialW=chainer.initializers.GlorotNormal())

    @staticmethod
    def make_batchnorm(multiplier):
        base = 3
        bn = chainer.links.BatchNormalization(multiplier * base)
        return bn

    @staticmethod
    def make_relu(inp):
        activations = chainer.functions.relu(inp)
        return activations

    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.deconv_1 = self.make_deconv(in_channels=1, multiplier=1, s=1)
            # self.deconv_1 = self.make_deconv(in_channels=1, multiplier=1, s=(2,2))
            self.bn_1 = self.make_batchnorm(1)
            self.deconv_2 = self.make_deconv(in_channels=3, multiplier=1, s=1)
            # self.deconv_2 = self.make_deconv(in_channels=3, multiplier=1, s=(2,2))
            self.bn_2 = self.make_batchnorm(1)
            self.deconv_3 = self.make_deconv(in_channels=3, multiplier=1)
            self.bn_3 = self.make_batchnorm(1)
            self.deconv_4 = self.make_deconv(in_channels=3, multiplier=1)
            self.bn_4 = self.make_batchnorm(1)
            self.deconv_5 = self.make_deconv(in_channels=3, multiplier=1)
            self.bn_5 = self.make_batchnorm(1)
            self.fc = chainer.links.Linear(32*32*3)

    def __call__(self, x):
        h = self.deconv_1(x)
        h = self.bn_1(h)
        h = chainer.functions.relu(h)
        h = self.deconv_2(h)
        h = self.bn_2(h)
        # h = chainer.functions.relu(h)
        # h = chainer.functions.tanh(h) * 127.5 + 127.5
        # h = self.deconv_3(h)
        # h = self.bn_3(h)
        # h = chainer.functions.relu(h)
        # h = self.deconv_4(h)
        # h = self.bn_4(h)
        # h = chainer.functions.relu(h)
        # h = self.deconv_5(h)
        # h = self.bn_5(h)
        # h = chainer.functions.relu(h)
        h = self.fc(h)
        # print('hdata:', type(h.data), np.shape(h.data))
        # for removing artifacts

        return h


class GeneratorPaper(chainer.Chain):
    @staticmethod
    def make_deconv(out_channels, in_channels, s=(2, 2), p=(1, 1)):
        deconv = chainer.links.Deconvolution2D(in_channels=in_channels,
                                               out_channels=out_channels,
                                               ksize=(4, 4),
                                               stride=s,
                                               pad=p,
                                               initialW=chainer.initializers.GlorotNormal())
        return deconv

    @staticmethod
    def make_batchnorm(out_channels):
        bn = chainer.links.BatchNormalization(out_channels)
        return bn

    @staticmethod
    def make_relu(inp):
        activations = chainer.functions.relu(inp)
        return activations

    @staticmethod
    def make_tanh(inp):
        activations = chainer.functions.tanh(inp)
        return activations

    def __init__(self):
        super(GeneratorPaper, self).__init__()
        with self.init_scope():
            self.deconv_1 = self.make_deconv(in_channels=1, out_channels=np.power(2, 9), s=(1, 1), p=0)
            self.bn_1 = self.make_batchnorm(np.power(2, 9))
            self.deconv_2 = self.make_deconv(in_channels=np.power(2, 9), out_channels=np.power(2, 8))
            self.bn_2 = self.make_batchnorm(np.power(2, 8))
            self.deconv_3 = self.make_deconv(in_channels=np.power(2, 8), out_channels=np.power(2, 7))
            self.bn_3 = self.make_batchnorm(np.power(2, 7))
            self.deconv_4 = self.make_deconv(in_channels=np.power(2, 7), out_channels=np.power(2, 6))
            self.bn_4 = self.make_batchnorm(np.power(2, 6))
            self.deconv_5 = self.make_deconv(in_channels=np.power(2, 6), out_channels=3)

    def __call__(self, x):
        h = self.deconv_1(x)
        h = self.bn_1(h)
        h = chainer.functions.relu(h)
        h = self.deconv_2(h)
        h = self.bn_2(h)
        h = chainer.functions.relu(h)
        h = self.deconv_3(h)
        h = self.bn_3(h)
        h = chainer.functions.relu(h)
        h = self.deconv_4(h)
        h = self.bn_4(h)
        h = chainer.functions.relu(h)
        h = self.deconv_5(h)
        h = chainer.functions.tanh(h)
        return h