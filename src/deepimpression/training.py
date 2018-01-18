import chainer
import audiovisual_stream
# from preprocessing import RandomIterator
import numpy as np
import tqdm
import chainer.functions as F
# import matplotlib.pyplot as plt
from random import randint
# import subprocess
import training_util
import project_paths2 as pp
import project_constants as pc
import os
from scipy import ndimage
# from scipy.io.wavfile import read
import time
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


# def make_training_set(get_audio=True):
#     t = time.time()
#     shape_frames = (pc.BATCH_SIZE, 3, pc.SIDE, pc.SIDE)
#     sample_length = 50176
#     shape_audio = (pc.BATCH_SIZE, 1, 1, sample_length)
#
#     batch_frames = np.zeros(shape=shape_frames, dtype='float32')
#     batch_audio = None
#
#     if get_audio:
#         batch_audio = np.zeros(shape=shape_audio, dtype='float32')
#
#     video_names, labels = training_util.get_names()
#
#     labels = np.asarray(labels, dtype='float32')
#     if pc.BATCH_SIZE == 1:
#         labels = np.reshape(labels, 6)
#     else:
#         labels = np.reshape(labels, (pc.BATCH_SIZE, 6))
#
#     for i in range(pc.BATCH_SIZE):
#         name = video_names[i]
#         if get_audio:
#             frame, batch_audio[i] = training_util.extract_frame_and_audio(name, get_audio=get_audio)
#             # reshape frame to channels first
#             frame = np.reshape(frame, (3, 192, 192))
#             batch_frames[i] = frame
#         else:
#             # batch_audio will stay None
#             frame, batch_audio = training_util.extract_frame_and_audio(name, get_audio=get_audio)
#             # reshape frame to channels first
#             frame = np.reshape(frame, (3, 192, 192))
#             batch_frames[i] = frame
#
#     # print('shape frames: %s, shape audio: %s, shape labels: %s' % (str(np.shape(batch_frames)),
#     #                                                                str(np.shape(batch_audio)), str(np.shape(labels))))
#     print(time.time() - t, 'seconds')
#     return batch_frames, batch_audio, labels


def make_training_set(get_audio=False):
    t1 = time.time()
    shape_frames = (pc.BATCH_SIZE, 3, pc.SIDE, pc.SIDE)
    sample_length = 50176
    shape_audio = (pc.BATCH_SIZE, 1, 1, sample_length)
    batch_frames = np.zeros(shape=shape_frames, dtype='float32')
    batch_audio = np.zeros(shape=shape_audio, dtype='float32')

    video_names, labels = training_util.get_names()

    labels = np.asarray(labels, dtype='float32')
    if pc.BATCH_SIZE == 1:
        labels = np.reshape(labels, 6)
    else:
        labels = np.reshape(labels, (pc.BATCH_SIZE, 6))
    print('setup ', time.time() - t1, 'seconds')

    t2 = time.time()
    tv = 0
    ta = 0
    for i in range(pc.BATCH_SIZE):
        t3 = time.time()
        name = video_names[i]

        # get a random frame
        num_frames = len(os.listdir(name))
        rn = randint(0, num_frames - 1)
        random_frame = os.path.join(name, '%03d.jpg' % rn)
        arr_frame = ndimage.imread(random_frame)
        arr_frame = np.reshape(arr_frame, (3, 192, 192))
        batch_frames[i] = arr_frame
        tv += time.time() - t3

        t4 = time.time()
        if get_audio:
            audio_path = os.path.join(name, 'audio.wav')
            aud = training_util.get_random_audio_clip(audio_path)
            batch_audio[i] = aud
        # else:
        #     batch_audio = None
        ta += time.time() - t4

    print('for loop ', time.time() - t2, 'seconds. of which ', ta, 'for audio and ', tv, ' for video')
    print('entire: ', time.time() - t1)
    return batch_frames, batch_audio, labels


def main():
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
                model.cleargrads()  # zero the gradient buffer
                prediction = model([audios, frames])

                loss = F.mean_absolute_error(prediction, labels)
                print(loss)

                loss.backward()
                optimizer.update()

                train_loss[epoch] += loss.data

                # delete things we don't need
                del prediction
                del loss
                del frames, audios, labels

        # calculate average loss per epoch
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
