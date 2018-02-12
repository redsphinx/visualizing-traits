import auditory_stream
import chainer
import visual_stream
from project_paths2 import ON_GPU
import numpy as np
from project_constants import DEVICE


### MODEL ###
class ResNet18(chainer.Chain):
    def __init__(self):
        super(ResNet18, self).__init__(
            aud=auditory_stream.ResNet18(),
            vis=visual_stream.ResNet18(),
            fc=chainer.links.Linear(512, 6, initialW=chainer.initializers.HeNormal())
        )
        self._validation = False

    @property
    def validation(self):
        return self._validation

    @validation.setter
    def validation(self, value):
        self._validation = value

    def __call__(self, x):

        if self._validation:
            # for testing
            if ON_GPU:
                h = [self.aud(chainer.cuda.to_gpu(x[0], device='0')), chainer.functions.expand_dims(
                    chainer.functions.sum(self.vis(chainer.cuda.to_gpu(x[1][:256], device='0')), 0), 0)]
            else:
                h = [self.aud(x[0]), chainer.functions.expand_dims(
                    chainer.functions.sum(self.vis(x[1][:256]), 0), 0)]

            for i in xrange(256, x[1].shape[0], 256):
                if ON_GPU:
                    h[1] += chainer.functions.expand_dims(
                        chainer.functions.sum(self.vis(chainer.cuda.to_gpu(x[1][i: i + 256], device='0')), 0), 0)
                else:
                    h[1] += chainer.functions.expand_dims(
                        chainer.functions.sum(self.vis(x[1][i: i + 256]), 0), 0)

            h[1] /= x[1].shape[0]

            return chainer.cuda.to_cpu(
                ((chainer.functions.tanh(self.fc(chainer.functions.concat(h))) + 1) / 2).data[0])


            # ch = chainer.functions.concat(h)
            # fch = self.fc(ch)
            # cfch = chainer.functions.tanh(fch)
            # print('tanh: ', cfch)
            # # scale between 0-1
            # cfch_1 = cfch + 1
            # cfch_1_half = cfch_1 / 2
            # d = cfch_1_half.data[0]
            # dd = chainer.cuda.to_cpu(d)
            # print('dd: ', dd)
            # return dd
        else:
            # for training
            if ON_GPU:
                a = self.aud(chainer.cuda.to_gpu(x[0], device='0'))
                v = self.vis(chainer.cuda.to_gpu(x[1], device='0'))
                h = [a, v]
            else:
                a = self.aud(x[0])
                v = self.vis(x[1])
                h = [a, v]

            ch = chainer.functions.concat(h)
            fch = self.fc(ch)
            cfch = chainer.functions.tanh(fch)
            # scale between 0-1
            cfch_1 = cfch + 1
            cfch_1_half = cfch_1 / 2

            return cfch_1_half

### MODEL ###
