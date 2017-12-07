import chainer
import audiovisual_stream
from preprocessing import RandomIterator
import numpy as np
import tqdm
import chainer.functions as F
import matplotlib.pyplot as plt

"""
training procedure in Gucluturk et al. 2016 
http://arxiv.org/abs/1609.05119%0Ahttp://dx.doi.org/10.1007/978-3-319-49409-8_28

- The audio data and the visual data of the video clip are extracted. 
- A random 50176 sample temporal crop of the audio data is fed into the auditory stream. The activities of the 
penultimate layer of the auditory stream are temporally pooled.
- A random 224 pixels x 224 pixels spatial crop of a random frame of the visual data is randomly flipped in the 
left/right direction and fed into the visual stream. The activities of the penultimate layer of the visual stream 
are spatially pooled.
- Pooled activities of the auditory stream and the visual stream are concatenated and fed into the fully-connected layer
- The fully-connected layer outputs five continuous prediction values between the range [0, 1] corresponding to each 
trait for the video clip.
"""


batchsize = 32
epochs = 900


def main():
    # TODO: define below
    # TODO: create a pipelining tool to get videos and labels and feed to iterator
    # ----------------
    train_data = None
    train_labels = None
    test_data = None
    test_labels = None
    # ----------------

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    model = audiovisual_stream.ResNet18().to_gpu(device='0')
    optimizer = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
    optimizer.setup(model)

    train_iter = RandomIterator(train, batchsize)
    test_iter = RandomIterator(test, batchsize)

    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    for epoch in tqdm.trange(epochs):

        with chainer.using_config('train', True):

            for data in train_iter:
                # train step
                model.cleargrads()  # zero the gradient buffer
                loss = F.softmax_cross_entropy(model(data[0]), data[1])
                loss.backward()
                optimizer.update()

                train_loss[epoch] += loss.data

        train_loss[epoch] /= train_iter.data._length

        # validation
        with chainer.using_config('train', False):
            for data in test_iter:
                test_loss[epoch] += F.softmax_cross_entropy(model(data[0]), data[1]).data

        test_loss[epoch] /= test_iter.data._length

    plt.plot(np.vstack([train_loss, test_loss]).T)
    plt.legend(['train loss', 'validation loss'])
    plt.show()


main()