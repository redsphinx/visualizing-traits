from util import load_model, predict_trait, find_video_test, track_prediction, get_accuracy
from training_util import get_random_frame
import pickle as pkl
import os
import project_paths2 as pp
import project_constants as pc
import numpy as np
import chainer
# from scipy.stats import linregress
from sklearn import linear_model
import statsmodels.api as sm

def main():
    model = load_model()
    model.validation = False

    # get ground truth from the pkl file
    # ----------------------------------------------------------------------------
    pkl_path = pp.TEST_LABELS
    # ----------------------------------------------------------------------------

    f = open(pkl_path, 'r')
    annotation_test = pkl.load(f)
    # ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness']
    annotation_test_keys = annotation_test.keys()
    all_video_names = annotation_test[annotation_test_keys[0]].keys()
    print('len annotation', len(annotation_test_keys))

    # ---
    y_tmp = np.zeros((pc.NUM_VIDEO_FRAMES, len(annotation_test_keys)), dtype=np.float32)
    target_tmp = np.zeros((pc.NUM_VIDEO_FRAMES, len(annotation_test_keys)), dtype=np.float32)

    # make zero audios
    sample_length = 50176
    shape_audio = (1, 1, 1, sample_length)
    audios = np.zeros(shape=shape_audio, dtype='float32')

    for ind in range(pc.NUM_VIDEO_FRAMES):
        print('ind: ', ind)
        video_id = all_video_names[ind]
        target_labels = [annotation_test['extraversion'][video_id],
                         annotation_test['agreeableness'][video_id],
                         annotation_test['conscientiousness'][video_id],
                         annotation_test['neuroticism'][video_id],
                         annotation_test['interview'][video_id],
                         annotation_test['openness'][video_id]]
        video = find_video_test(video_id)

        # for videos
        # y = predict_trait(video, model)
        # print(video_id)
        # print('ValueExtraversion, ValueAgreeableness, ValueConscientiousness, ValueNeurotisicm, ValueInterview,'
        #       ' ValueOpenness')
        # print(y)
        # print(target_labels)
        # track_prediction(video_id, y, target_labels, write_file=pp.TEST_LOG)

    # calculate and print mean performance
    # get_accuracy(pp.TEST_LOG, num_keys=len(annotation_test_keys))
    # get_accuracy(pp.TEST_LOG, num_keys=len(annotation_test_keys)-1)

        # for single random frame
        # grab frame
        frame = get_random_frame(video)
        frame_shape = np.shape(frame)
        # reshape
        frame = np.reshape(frame, (3, frame_shape[0], frame_shape[1]))
        frame = np.expand_dims(frame, 0)
        # prediction
        with chainer.using_config('train', False):
            prediction = model([audios, frame])

        # prediction = chainer.cuda.to_cpu(prediction.data)

        y_tmp[ind] = prediction.data
        target_tmp[ind] = target_labels

    # calculate validation loss
    y_tmp.astype(np.float32)
    target_tmp.astype(np.float32)
    loss = chainer.functions.mean_absolute_error(y_tmp, target_tmp)
    print('model: ', pp.PRE_TRAINED, ' loss model: ', loss)

    # check if log file exists
    if not os.path.exists(pp.VALIDATION_LOG):
        _ = open(pp.VALIDATION_LOG, 'w')
        _.close()

    # save loss
    try:
        with open(pp.VALIDATION_LOG, 'a') as my_file:
            line = 'model: %s, loss model: %s\n' % (pp.PRE_TRAINED, str(loss))
            # line = 'model: ', pp.PRE_TRAINED,' loss model: ', loss, '\n'
            my_file.write(line)
    except:
        pass


# main()

def predict_interview():
    pkl_path = pp.TRAIN_LABELS
    num = 6000
    f = open(pkl_path, 'r')
    annotation_test = pkl.load(f)

    interview = np.zeros(num)
    # order: extraversion, agreeableness, conscientiousness, neuroticism, openness
    b5_traits = np.zeros((num, 5))

    video_names = annotation_test['interview'].keys()

    for i in range(num):
        name = video_names[i]
        interview[i] = annotation_test['interview'][name]
        b5_traits[i][0] = annotation_test['extraversion'][name]
        b5_traits[i][1] = annotation_test['agreeableness'][name]
        b5_traits[i][2] = annotation_test['conscientiousness'][name]
        b5_traits[i][3] = annotation_test['neuroticism'][name]
        b5_traits[i][4] = annotation_test['openness'][name]

    clf = linear_model.LinearRegression()
    # clf.fit([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
    clf.fit(b5_traits, interview)
    print('extraversion, agreeableness, conscientiousness, neuroticism, openness:')
    print(clf.coef_)
    print(clf.score(b5_traits, interview))

    b5_traits = sm.add_constant(b5_traits)
    model = sm.OLS(interview, b5_traits).fit()
    predictions = model.predict(b5_traits)
    print(model.summary())

    # slope, intercept, r_value, p_value, std_err = linregress(b5_traits, interview)
    # print(slope, intercept, r_value, p_value, std_err)

predict_interview()