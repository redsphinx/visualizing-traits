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
                                               initialW=chainer.initializers.HeNormal())
        return deconv

    @staticmethod
    def make_batchnorm(multiplier, initi=chainer.initializers.HeNormal()):
        base = 3
        bn = chainer.links.BatchNormalization(size=multiplier * base) #, initial_gamma=initi,
                                              # initial_beta=initi)
        return bn

    def __init__(self):
        super(Generator, self).__init__()
        with self.init_scope():
            self.deconv_1 = self.make_deconv(in_channels=1, multiplier=1)
            # self.deconv_1 = self.make_deconv(in_channels=1, multiplier=1, s=(2,2))
            self.bn_1 = self.make_batchnorm(1)
            self.deconv_2 = self.make_deconv(in_channels=3, multiplier=1)
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
        h = self.bn_1(h, )
        h = chainer.functions.relu(h)
        h = self.deconv_2(h)
        h = self.bn_2(h)
        h = chainer.functions.relu(h)
        h = self.deconv_3(h)
        h = self.bn_3(h)
        h = chainer.functions.relu(h)
        h = self.deconv_4(h)
        h = self.bn_4(h)
        h = self.fc(h)
        return h


class Discriminator(chainer.Chain):
    @staticmethod
    def make_conv(out, in_channels, s=(1, 1)):
        conv = chainer.links.Convolution2D(in_channels=in_channels,
                                           out_channels=out,
                                           ksize=(4, 4),
                                           stride=s,
                                           pad=(1, 1),
                                           initialW=chainer.initializers.HeNormal())
        return conv

    @staticmethod
    def make_batchnorm(in_channel):
        # base = 3
        bn = chainer.links.BatchNormalization(in_channel)#, initial_gamma=chainer.initializers.HeNormal(),
                                              # initial_beta=chainer.initializers.HeNormal())
        return bn

    def __init__(self):
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.conv_1 = self.make_conv(out=8, in_channels=3, s=(2,2))
            self.conv_2 = self.make_conv(out=16, in_channels=8, s=(2,2))
            self.conv_3 = self.make_conv(out=32, in_channels=16, s=(2,2))
            self.conv_4 = self.make_conv(out=64, in_channels=32, s=(2,2))
            self.conv_5 = self.make_conv(out=128, in_channels=64, s=(2,2))
            self.bn_1 = self.make_batchnorm(8)
            self.bn_2 = self.make_batchnorm(16)
            self.bn_3 = self.make_batchnorm(32)
            self.bn_4 = self.make_batchnorm(64)
            self.bn_5 = self.make_batchnorm(128)
            self.fc = chainer.links.Linear(in_size=128, initialW=chainer.initializers.HeNormal(), out_size=1)

    def __call__(self, x):
        slope = 0.2
        h = self.conv_1(x)
        h = self.bn_1(h)
        h = chainer.functions.leaky_relu(h, slope=slope)
        h = self.conv_2(h)
        h = self.bn_2(h)
        h = chainer.functions.leaky_relu(h, slope=slope)
        h = self.conv_3(h)
        h = self.bn_3(h)
        h = chainer.functions.leaky_relu(h, slope=slope)
        h = self.conv_4(h)
        h = self.bn_4(h)
        h = chainer.functions.leaky_relu(h, slope=slope)
        h = self.conv_5(h)
        h = self.bn_5(h)
        h = chainer.functions.leaky_relu(h, slope=slope)
        h = self.fc(h)
        h = chainer.functions.sigmoid(h)
        return h


class GeneratorPaper(chainer.Chain):
    @staticmethod
    def make_deconv(out_channels, in_channels, s=(2, 2), p=(1, 1)):
        deconv = chainer.links.Deconvolution2D(in_channels=in_channels,
                                               out_channels=out_channels,
                                               ksize=(4, 4),
                                               stride=s,
                                               pad=p,
                                               initialW=chainer.initializers.HeNormal())
        return deconv

    @staticmethod
    def make_batchnorm(out_channels):
        bn = chainer.links.BatchNormalization(out_channels)
        return bn

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
            print(self.deconv_5.outsize)
            self.fc = chainer.links.Linear(32 * 32 * 3)

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
        h = self.fc(h)
        h = chainer.functions.tanh(h) * 127.5 + 127.5
        return h


class DiscriminatorPaper(chainer.Chain):
    @staticmethod
    def make_conv(out, in_channels, p=(1,1), s=(1, 1)):
        conv = chainer.links.Convolution2D(in_channels=in_channels,
                                           out_channels=out,
                                           ksize=(4, 4),
                                           stride=s,
                                           pad=p,
                                           initialW=chainer.initializers.HeNormal())
        return conv

    @staticmethod
    def make_batchnorm(in_channel):
        # base = 3
        bn = chainer.links.BatchNormalization(in_channel)
        return bn

    def __init__(self):
        super(DiscriminatorPaper, self).__init__()
        with self.init_scope():
            self.conv_1 = self.make_conv(out=2**(5+1), in_channels=3, s=(2, 2))
            self.conv_2 = self.make_conv(out=2**(5+2), in_channels=2**(5+1), s=(2, 2))
            self.conv_3 = self.make_conv(out=2**(5+3), in_channels=2**(5+2), s=(2, 2))
            self.conv_4 = self.make_conv(out=2**(5+4), in_channels=2**(5+3), s=(2, 2))
            self.conv_5 = self.make_conv(out=2**(5+5), in_channels=2**(5+4), p=0)
            self.bn_1 = self.make_batchnorm(2**(5+1))
            self.bn_2 = self.make_batchnorm(2**(5+2))
            self.bn_3 = self.make_batchnorm(2**(5+3))
            self.bn_4 = self.make_batchnorm(2**(5+4))
            # self.bn_5 = self.make_batchnorm(128)
            self.fc = chainer.links.Linear(in_size=2**(5+5), initialW=chainer.initializers.HeNormal(), out_size=1)

    def __call__(self, x):
        slope = 0.2
        h = self.conv_1(x)
        h = self.bn_1(h)
        h = chainer.functions.leaky_relu(h, slope=slope)
        h = self.conv_2(h)
        h = self.bn_2(h)
        h = chainer.functions.leaky_relu(h, slope=slope)
        h = self.conv_3(h)
        h = self.bn_3(h)
        h = chainer.functions.leaky_relu(h, slope=slope)
        h = self.conv_4(h)
        h = self.bn_4(h)
        h = chainer.functions.leaky_relu(h, slope=slope)
        h = self.conv_5(h)
        # h = self.bn_5(h)
        h = chainer.functions.leaky_relu(h, slope=slope)
        h = self.fc(h)
        # h = chainer.functions.sigmoid(h)
        return h