import auditory_stream
import chainer
import visual_stream
from project_paths2 import ON_GPU
import numpy as np

### MODEL ###
class ResNet18(chainer.Chain):
    def __init__(self):
        super(ResNet18, self).__init__(
            aud=auditory_stream.ResNet18(),
            vis=visual_stream.ResNet18(),
            fc=chainer.links.Linear(512, 6, initialW=chainer.initializers.HeNormal())
        )


    def __call__(self, x):
        # doesn't work
        # h = [self.aud(True, chainer.Variable(chainer.cuda.to_gpu(x[0]), True)), chainer.functions.expand_dims(
        #     chainer.functions.sum(self.vis(True, chainer.Variable(chainer.cuda.to_gpu(x[1][:256]), True)), 0), 0)]
        #
        # for i in xrange(256, x[1].shape[0], 256):
        #     h[1] += chainer.functions.expand_dims(
        #         chainer.functions.sum(self.vis(True, chainer.Variable(chainer.cuda.to_gpu(x[1][i: i + 256]), True)), 0),
        #         0)

        #

        # avg over channels instead of spatial dim
        if ON_GPU:
            h = [self.aud(chainer.cuda.to_gpu(x[0], device='0')), chainer.functions.expand_dims(
                chainer.functions.sum(self.vis(chainer.cuda.to_gpu(x[1][:256], device='0')), 0), 0)]
        else:
            a = self.aud(x[0])
            v = self.vis(x[1])
            h = [a, v]
            # take avg of 256 frames
            # v = self.vis(x[1][:256])
            # s = chainer.functions.sum(v, 0)
            # e = chainer.functions.expand_dims(s, 0)
            # h = [a, e]

        # x1_shape = x[1].shape
        # #
        # for i in xrange(256, x1_shape[0], 256):
        #     if ON_GPU:
        #         h[1] += chainer.functions.expand_dims(
        #             chainer.functions.sum(self.vis(chainer.cuda.to_gpu(x[1][i: i + 256], device='0')), 0), 0)
        #     else:
        #         v = self.vis(x[1][i: i + 256])
        #         s = chainer.functions.sum(v, 0)
        #         e = chainer.functions.expand_dims(s, 0)
        #         h[1] += e

        # h[1] /= x[1].shape[0]
        ch = chainer.functions.concat(h)
        fch = self.fc(ch)
        cfch = chainer.functions.tanh(fch)
        # scale between 0-1
        cfch_1 = cfch + 1
        cfch_1_half = cfch_1 / 2
        return cfch_1_half
        # get rid of first dimension
        # d = cfch_1_half.data[0]
        # ret = chainer.cuda.to_cpu(d)
        # return ret

### MODEL ###
