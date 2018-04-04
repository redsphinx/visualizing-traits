import numpy as np
import chainer
import os
import project_paths as pp
import project_constants as pc
from generator import Generator
import util
import random


def main():
    model_name = 'generator_e_59'
    model_name = os.path.join(pp.MODEL_SAVES, model_name)
    model = Generator()
    chainer.serializers.load_npz(model_name, model)

    num_features = util.get_number_of_features(pp.CELEB_FACES_FC6_TEST)
    all_names = np.array(util.get_names_h5_file(pp.FC6_TEST_H5))

    y_tmp = np.zeros((num_features, 32 * 32 * 3), dtype=np.float32)
    target_tmp = np.zeros((num_features, 32 * 32 * 3), dtype=np.float32)

    save_list_names = os.listdir('/home/gabi/Documents/temp_datasets/test_celeba_reconstruction_m99')
    save_list_names = [i.split('_')[0]+'.jpg' for i in save_list_names]
    # save_list = random.sample(xrange(num_features), 100)
    # save_list_names = [''] * 100
    cnt = 0



    # for i in save_list:
    #     save_list_names[cnt] = util.sed_line(pp.CELEB_FACES_FC6_TEST, i).strip().split(',')[0]
    #     cnt += 1

    cnt = 0
    for i in all_names:
        features = util.get_features_h5_in_batches([i], train=False)
        features = util.to_correct_input(features)
        labels = util.get_labels([i])
        labels = np.asarray(labels, dtype=np.float32)
        target_tmp[cnt] = labels

        with chainer.using_config('train', False):
            f = np.expand_dims(features[0], 0)
            prediction = model(f)
            y_tmp[cnt] = prediction.data[0]
            if i in save_list_names:
                util.save_image(prediction, i, epoch=0)
                print("image '%s' saved" % i)

        cnt += 1

    # calculate validation loss
    y_tmp.astype(np.float32)
    target_tmp.astype(np.float32)
    loss = chainer.functions.mean_absolute_error(y_tmp, target_tmp)
    print('model: ', model_name, ' loss model: ', loss)


main()

# ('model: ', '/home/gabi/Documents/temp_datasets/generator_models/generator_e_19', ' loss model: ', variable(30.824942))
# ('model: ', '/home/gabi/Documents/temp_datasets/generator_models/generator_e_99', ' loss model: ', variable(30.748234))
