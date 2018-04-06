import numpy as np
import project_paths as pp
from generator import Generator, Discriminator
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
import chainer.functions as F


def training():
    print('setting up...')

    if pc.TRAIN:
        num_features = util.get_number_of_features(pp.CELEB_FACES_FC6_TRAIN)
        all_names = np.array(util.get_names_h5_file(pp.FC6_TRAIN_H5))
        path_images = pp.CELEB_FACES_FC6_TRAIN
    else:
        num_features = util.get_number_of_features(pp.CELEB_FACES_FC6_TEST)
        all_names = np.array(util.get_names_h5_file(pp.FC6_TEST_H5))
        path_images = pp.CELEB_FACES_FC6_TEST

    total_steps = num_features / pc.BATCH_SIZE

    # ----------------------------------------------------------------
    # GENERATOR
    generator = Generator()
    generator_train_loss = np.zeros(pc.EPOCHS)
    generator_optimizer = chainer.optimizers.Adam(alpha=0.0002, beta1=0.9, beta2=0.999, eps=10**-8)
    generator_optimizer.setup(generator)
    # ----------------------------------------------------------------
    # DISCRIMINATOR
    discriminator = Discriminator()
    discriminator_train_loss = np.zeros(pc.EPOCHS)
    discriminator_optimizer = chainer.optimizers.Adam(alpha=0.0002, beta1=0.9, beta2=0.999, eps=10**-8)
    discriminator_optimizer.setup(discriminator)
    # ----------------------------------------------------------------
    # OPTIMIZER
    # optimizer = chainer.optimizers.Adam(alpha=0.0002, beta1=0.9, beta2=0.999, eps=10**-8)
    # optimizer.setup(generator)
    # optimizer.setup(discriminator)
    # ----------------------------------------------------------------

    save_list = random.sample(xrange(num_features), 20)
    save_list_names = [''] * 20
    cnt = 0

    for i in save_list:
        save_list_names[cnt] = util.sed_line(path_images, i).strip().split(',')[0]
        cnt += 1

    print('training...')
    for epoch in range(pc.EPOCHS):

        # shuffle training instances
        order = range(num_features)
        random.shuffle(order)

        names_order = all_names[order]

        print('epoch %d' % epoch)
        for step in range(total_steps):
            names = names_order[step * pc.BATCH_SIZE:(step + 1) * pc.BATCH_SIZE]
            features = util.get_features_h5_in_batches(names, train=pc.TRAIN)
            features = util.to_correct_input(features)
            labels = util.get_labels(names)
            labels = np.asarray(labels, dtype=np.float32)

            with chainer.using_config('train', True):
                generator.cleargrads()
                prediction = generator(chainer.Variable(features))

                discriminator.cleargrads()
                data = np.reshape(prediction.data, (32, 32, 32, 3))
                data = np.transpose(data, (0, 3, 1, 2))
                fake_prob = discriminator(chainer.Variable(data))
                
                other_data = np.reshape(labels, (32, 32, 32, 3))
                other_data = np.transpose(other_data, (0, 3, 1, 2))
                real_prob = discriminator(chainer.Variable(other_data))

                # ----------------------------------------------------------------
                # CALCULATE LOSS
                ones1 = util.make_ones(generator)
                lambda_adv = 1#- 10 ** 2
                lambda_sti = 1#2 * 10 ** -6
                generator_loss = lambda_adv * F.sigmoid_cross_entropy(fake_prob, ones1) + \
                        lambda_sti * F.mean_squared_error(labels, prediction)
                # L_gen = F.sigmoid_cross_entropy(fake_prob, ones) + \
                #         F.mean_squared_error(labels, prediction)
                # L_gen = F.sigmoid_cross_entropy(discriminator(generator(features)), generator.xp.ones(pc.BATCH_SIZE)) + \
                #         F.mean_squared_error(labels, generator(features))

                # umut note: (3) = sigm x entr
                zeros = util.make_zeros(generator)
                ones2 = util.make_ones(generator)
                lambda_dis = 1#-10 ** 2
                discriminator_loss = lambda_dis * (F.sigmoid_cross_entropy(real_prob, ones2) +
                                                   F.sigmoid_cross_entropy(fake_prob, zeros))
                # L_dis = F.sigmoid_cross_entropy(real_prob, ones) + \
                #         F.sigmoid_cross_entropy(fake_prob, zeros)
                # L_dis = F.sigmoid_cross_entropy(discriminator(labels), generator.xp.ones(pc.BATCH_SIZE)) + \
                #         F.sigmoid_cross_entropy(discriminator(generator(features)), generator.xp.zeros(pc.BATCH_SIZE))

                generator_loss.backward()
                generator_optimizer.update()
                generator_train_loss[epoch] += generator_loss.data
                discriminator_loss.backward()
                discriminator_optimizer.update()
                discriminator_train_loss[epoch] += discriminator_loss.data

                print('%d/%d %d/%d  generator: %f   discriminator: %f' % (
                epoch, pc.EPOCHS, step, total_steps, generator_loss.data, discriminator_loss.data))

            with chainer.using_config('train', False):
                for i in range(len(names)):
                    if names[i] in save_list_names:
                        f = np.expand_dims(features[i], 0)
                        prediction = generator(f)
                        util.save_image(prediction, names[i], epoch, pp.RECONSTRUCTION_FOLDER)
                        print("image '%s' saved" % names[i])

        if (epoch+1) % pc.SAVE_EVERY_N_STEPS == 0:
            util.save_model(generator, epoch)

        generator_train_loss[epoch] /= total_steps
        print(generator_train_loss[epoch])
        
        discriminator_train_loss[epoch] /= total_steps
        print(discriminator_train_loss[epoch])

        # log_file = pp.TRAIN_LOG
        #
        # if not os.path.exists(log_file):
        #     f = open(log_file, 'w')
        #     f.close()
        #
        # with open(log_file, 'a') as my_file:
        #     line = 'epoch: %d loss: %f\n' % (epoch, train_loss[epoch])
        #     my_file.write(line)


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

training()
