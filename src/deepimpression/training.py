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
from util import save_model, predict_trait, find_video_val, track_prediction, get_accuracy, load_model
import pickle as pkl
import random
from project_constants import DEVICE


def make_training_set(get_audio=False):
    shape_frames = (pc.BATCH_SIZE, 3, pc.SIDE, pc.SIDE)
    sample_length = 50176
    shape_audio = (pc.BATCH_SIZE, 1, 1, sample_length)
    batch_frames = np.zeros(shape=shape_frames, dtype='float32')
    batch_audio = np.zeros(shape=shape_audio, dtype='float32')

    video_names, labels = training_util.get_names(labels=pp.TRAIN_LABELS,
                                                  data=pp.CHALEARN_JPGS,
                                                  batch_size=pc.BATCH_SIZE,
                                                  number_folders=pc.NUMBER_TRAINING_FOLDERS)

    labels = np.asarray(labels, dtype='float32')
    for i in range(pc.BATCH_SIZE):
        name = video_names[i]

        # get a random frame
        num_frames = len(os.listdir(name))
        rn = randint(0, num_frames - 1)
        random_frame = os.path.join(name, '%03d.jpg' % rn)
        arr_frame = ndimage.imread(random_frame)
        arr_frame = np.reshape(arr_frame, (3, 192, 192))
        batch_frames[i] = arr_frame

        if get_audio:
            audio_path = os.path.join(name, 'audio.wav')
            aud = training_util.get_random_audio_clip(audio_path)
            batch_audio[i] = aud
    return batch_frames, batch_audio, labels


def validation(model, epoch):
    with chainer.using_config('train', False):
        # get ground truth from the pkl file
        # ----------------------------------------------------------------------------
        pkl_path = pp.VALIDATION_LABELS
        # ----------------------------------------------------------------------------

        f = open(pkl_path, 'r')
        annotation_val = pkl.load(f)
        # ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness']
        annotation_val_keys = annotation_val.keys()
        all_video_names = annotation_val[annotation_val_keys[0]].keys()

        # log = os.path.join(pp.VALIDATION_LOG, 'epoch_%d.txt' % epoch)

        # make a list of 200 random numbers between 0 and 1999
        random_list = [random.randrange(0, 1999, 1) for _ in range(pc.VAL_BATCH_SIZE)]

        y_tmp = np.zeros((pc.VAL_BATCH_SIZE, len(annotation_val_keys)), dtype=np.float32)
        target_tmp = np.zeros((pc.VAL_BATCH_SIZE, len(annotation_val_keys)), dtype=np.float32)

        cnt = 0

        for ind in random_list:
            print('ind: ', ind)
            video_id = all_video_names[ind]
            target_labels = [annotation_val['extraversion'][video_id],
                             annotation_val['agreeableness'][video_id],
                             annotation_val['conscientiousness'][video_id],
                             annotation_val['neuroticism'][video_id],
                             annotation_val['interview'][video_id],
                             annotation_val['openness'][video_id]]
            video = find_video_val(video_id)
            y = predict_trait(video, model)

            y_tmp[cnt] = y
            target_tmp[cnt] = target_labels
            cnt += 1
            # print(video_id)
            # print('ValueExtraversion, ValueAgreeableness, ValueConscientiousness, ValueNeurotisicm, ValueInterview,'
            #       ' ValueOpenness')
            # print(y)
            # print(target_labels)

            # track_prediction(video_id, y, target_labels, write_file=log)

        # calculate validation loss
        y_tmp.astype(np.float32)
        target_tmp.astype(np.float32)
        loss = F.mean_absolute_error(y_tmp, target_tmp)
        print(loss)

        log_file = pp.VALIDATION_LOG

        # make log file
        if not os.path.exists(log_file):
            log_file = open(log_file, 'w')
            log_file.close()

        # write to log file
        with open(log_file, 'a') as my_file:
            my_file.write('%s,epoch=%d' % (str(loss), epoch))


def main(pretrained=False):
    # TODO: add selection for only visual stream

    if pretrained:
        model = load_model(this_model=pp.TRAIN_PRETRAINED)
    else:
        if pp.ON_GPU:
            model = audiovisual_stream.ResNet18().to_gpu(device=DEVICE)
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

            # for s in tqdm.tqdm(range(num_steps)):
            for s in range(int(num_steps)):
                # set validation to false
                model.validation = False

                frames, audios, labels = make_training_set()
                model.cleargrads()  # zero the gradient buffer
                prediction = model([audios, frames])

                loss = F.mean_absolute_error(prediction, labels)
                print(loss, loss.data)

                loss.backward()
                optimizer.update()

                train_loss[epoch] += loss.data

                # delete things we don't need
                del prediction
                del loss
                del frames, audios, labels

        # calculate average loss per epoch
        # train_loss[epoch] /= pc.BATCH_SIZE * num_steps
        train_loss[epoch] /= num_steps
        print(train_loss[epoch])

        log_file = pp.TRAIN_LOG

        if not os.path.exists(log_file):
            f = open(log_file, 'w')
            f.close()

        with open(log_file, 'a') as my_file:
            line = 'epoch: %d loss: %f\n' % (epoch, train_loss[epoch])
            my_file.write(line)

        # validation on 200 random videos
        # set validation to True
        model.validation = True
        validation(model, epoch)

        save_every = 50
        if epoch + 1 % save_every == 0:
            save_model(model, epoch)


    # TODO: run model on test data
    # plt.plot(np.vstack([train_loss, test_loss]).T)
    # plt.plot(np.vstack([train_loss]).T)
    # plt.legend(['train loss', 'validation loss'])
    # plt.legend(['train loss'])
    # plt.show()


main()
