import numpy as np
import project_paths as pp
from generator import Generator, GeneratorPaper
import chainer
import os
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import util
import project_constants as pc
import random
import time
import subprocess
import h5py as h5


def training():
    print('setting up...')
    # num_features = util.get_number_of_features(pp.CELEB_FACES_FC6_TRAIN)
    num_features = util.get_number_of_features(pp.CELEB_FACES_FC6_TEST)
    total_steps = num_features / pc.BATCH_SIZE
    # all_names = np.array(util.get_names_h5_file(pp.FC6_TRAIN_H5))
    all_names = np.array(util.get_names_h5_file(pp.FC6_TEST_H5))
    generator = Generator()
    train_loss = np.zeros(pc.EPOCHS)
    # generator = GeneratorPaper()
    optimizer = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5, beta2=0.999, eps=10e-8)
    optimizer.setup(generator)
    save_list = random.sample(xrange(num_features), 20)
    save_list_names = [''] * 20
    cnt = 0

    for i in save_list:
        # save_list_names[cnt] = util.sed_line(pp.CELEB_FACES_FC6_TRAIN, i).strip().split(',')[0]
        save_list_names[cnt] = util.sed_line(pp.CELEB_FACES_FC6_TEST, i).strip().split(',')[0]
        cnt += 1

    print('training...')
    for epoch in range(pc.EPOCHS):

        # shuffle training instances
        order = range(num_features)
        random.shuffle(order)

        names_order = all_names[order]

        print('epoch %d' % epoch)
        for step in range(total_steps):
            # names, features = util.get_features_in_batches(step, train=True)
            names = names_order[step * pc.BATCH_SIZE:(step + 1) * pc.BATCH_SIZE]
            # features = util.get_features_h5_in_batches(names, train=True)
            features = util.get_features_h5_in_batches(names, train=False)
            features = util.to_correct_input(features)
            labels = util.get_labels(names)
            labels = np.asarray(labels, dtype=np.float32)

            with chainer.using_config('train', True):
                generator.cleargrads()
                prediction = generator(features)
                loss = chainer.functions.mean_absolute_error(prediction, labels)
                # print('loss', loss.data)
                print('%d/%d %d/%d loss: %f' % (epoch, pc.EPOCHS, step, total_steps, float(loss.data)))
                loss.backward()
                optimizer.update()
                train_loss[epoch] += loss.data

            with chainer.using_config('train', False):
                for i in range(len(names)):
                    if names[i] in save_list_names:
                        f = np.expand_dims(features[i], 0)
                        prediction = generator(f)
                        util.save_image(prediction, names[i], epoch)
                        print("image '%s' saved" % names[i])

        if (epoch+1) % pc.SAVE_EVERY_N_STEPS == 0:
            util.save_model(generator, epoch)

        train_loss[epoch] /= total_steps
        print(train_loss[epoch])

        log_file = pp.TRAIN_LOG

        if not os.path.exists(log_file):
            f = open(log_file, 'w')
            f.close()

        with open(log_file, 'a') as my_file:
            line = 'epoch: %d loss: %f\n' % (epoch, train_loss[epoch])
            my_file.write(line)


# def plot_results():
#     f, axarr = plt.subplots(10, 10)
#     p = '/home/gabi/Documents/temp_datasets/celeba_sample_10'
#     p2 = '/home/gabi/Documents/temp_datasets/celeba_reconstruction'
#     names = os.listdir(p)
#     for i in range(10):
#
#         n_all = names[i]
#         n = n_all.split('.')[0]
#         print(n)
#
#         for j in range(10):
#             if j == 9:
#                 np = os.path.join(p, n_all)
#                 img = ndimage.imread(np)
#                 axarr[i, j].imshow(img)
#             else:
#                 n_ = '%s_%d.jpg' % (n, j)
#                 np = os.path.join(p2, n_)
#                 img = ndimage.imread(np)
#                 axarr[i, j].imshow(img)
#     plt.show()


# plot_results()
# training()

# DONE 1: load from large file
# names, features = util.get_features_in_batches(6, train=False)
# DONE 2: save generated images at intervals
# DONE 3: save model at intervals
# TODO 4: write code for testing the data
# TODO 5: rewrite plot results

training()
