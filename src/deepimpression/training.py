import chainer
import audiovisual_stream
from preprocessing import RandomIterator
import numpy as np
import tqdm
import chainer.functions as F
import matplotlib.pyplot as plt
from random import randint
import subprocess
import training_util
import project_paths2 as pp
import project_constants as pc
import os
"""
training procedure in Gucluturk et al. 2016 
https://arxiv.org/pdf/1609.05119.pdf

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

# TODO: get metadata on the dataset


def make_training_set(get_audio=True):
    # TODO: fix the shape

    shape_frames = (pc.BATCH_SIZE, 3, pc.SIDE, pc.SIDE)
    sample_length = 50176
    shape_audio = (pc.BATCH_SIZE, 1, 1, sample_length)
    # shape_audio = (pc.BATCH_SIZE, sample_length)

    batch_frames = np.zeros(shape=shape_frames, dtype='float32')
    batch_audio = None

    if get_audio:
        batch_audio = np.zeros(shape=shape_audio, dtype='float32')

    video_names, labels = training_util.get_names()

    labels = np.asarray(labels, dtype='float32')
    if pc.BATCH_SIZE == 1:
        labels = np.reshape(labels, 6)
    else:
        labels = np.reshape(labels, (pc.BATCH_SIZE, 6))

    for i in range(pc.BATCH_SIZE):
        name = video_names[i]
        if get_audio:
            frame, batch_audio[i] = training_util.extract_frame_and_audio(name, get_audio=get_audio)
            # reshape frame to channels first
            frame = np.reshape(frame, (3, 192, 192))
            batch_frames[i] = frame
        else:
            # batch_audio will stay None
            frame, batch_audio = training_util.extract_frame_and_audio(name, get_audio=get_audio)
            # reshape frame to channels first
            frame = np.reshape(frame, (3, 192, 192))
            batch_frames[i] = frame

    # print('shape frames: %s, shape audio: %s, shape labels: %s' % (str(np.shape(batch_frames)),
    #                                                                str(np.shape(batch_audio)), str(np.shape(labels))))

    return batch_frames, batch_audio, labels


def main():
    # data specifications: array(shape=(1, batch, height, width, channels))
    # example:
    # def get_data(folder):
    #     list_items = sorted(os.listdir(folder))
    #     data = np.zeros((20, 128, 48, 3), int)
    #     for ind in range(len(list_items)):
    #         item = list_items[ind]
    #         item_path = os.path.join(folder, item)
    #
    #         tmp = ndimage.imread(item_path)
    #
    #         data[ind] = np.array(tmp, 'float32')
    #
    #     data = np.reshape(data, (1, 20, 128, 48, 3))
    #     return data

    # train = chainer.datasets.TupleDataset(train_data, train_labels)
    # test = chainer.datasets.TupleDataset(test_data, test_labels)
    # TODO: add selection for only visual stream
    if pp.ON_GPU:
        model = audiovisual_stream.ResNet18().to_gpu(device='0')
    else:
        model = audiovisual_stream.ResNet18()

    optimizer = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
    optimizer.setup(model)

    train_loss = np.zeros(pc.EPOCHS)
    # test_loss = np.zeros(pc.EPOCHS)

    # for epoch in tqdm.trange(pc.EPOCHS):
    for epoch in range(pc.EPOCHS):
        print('EPOCH: %d out of %d' % (epoch, pc.EPOCHS))

        with chainer.using_config('train', True):

            num_steps = 6000 / pc.BATCH_SIZE

            for s in tqdm.tqdm(range(num_steps)):

                frames, audios, labels = make_training_set()
                # check length
                # video_capture = np.reshape(video_capture, (frames, video_shape[-1], video_shape[1], video_shape[2]), 'float32')

                # train = chainer.datasets.TupleDataset(audios, frames, labels)
                # train = RandomIterator(train)

            # for data in train_iter:
                # train step
                model.cleargrads()  # zero the gradient buffer
                # loss = F.softmax_cross_entropy(model(data[0]), data[1])
                # frames, audios, labels = make_training_set(side=192)
                # loss = F.softmax_cross_entropy(model(train[0]), train[1])
                # loss = F.softmax_cross_entropy(model([train[0], train[1]]), train[2])
                # try:
                prediction = model([audios, frames])
                # print(prediction)
                # prediction = np.asarray(prediction, dtype='float32')
                # prediction = model([train[0], train[1]])
                # loss = F.softmax_cross_entropy(prediction, labels)
                loss = F.mean_absolute_error(prediction, labels)
                # print(loss)

                loss.backward()
                optimizer.update()

                train_loss[epoch] += loss.data

        # calculate average loss per epoch
        # train_loss[epoch] /= train_iter.data._length
        train_loss[epoch] /= pc.BATCH_SIZE * num_steps
        print(train_loss)

        log_file = pp.LOG

        if not os.path.exists(log_file):
            os.mkdir(log_file)

        with open(log_file, 'a') as my_file:
            line = 'epoch: %d loss: %f\n' % (epoch, train_loss[epoch])
            my_file.write(line)

        # validation
        # with chainer.using_config('train', False):
        #     for data in test_iter:
        #         test_loss[epoch] += F.softmax_cross_entropy(model(data[0]), data[1]).data
        #
        # test_loss[epoch] /= test_iter.data._length

    # plt.plot(np.vstack([train_loss, test_loss]).T)
    # plt.plot(np.vstack([train_loss]).T)
    # plt.legend(['train loss', 'validation loss'])
    # plt.legend(['train loss'])
    # plt.show()


main()
