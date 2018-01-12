import auditory_stream
import chainer
import visual_stream
from project_paths2 import ON_GPU


### MODEL ###
class ResNet18(chainer.Chain):
    def __init__(self):
        super(ResNet18, self).__init__(
            aud=auditory_stream.ResNet18(),
            vis=visual_stream.ResNet18(),
            fc=chainer.links.Linear(512, 5, initialW=chainer.initializers.HeNormal())
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

        return chainer.cuda.to_cpu(((chainer.functions.tanh(self.fc(chainer.functions.concat(h))) + 1) / 2).data[0])

### MODEL ###
