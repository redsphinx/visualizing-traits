from util import load_model, predict_trait, find_video_test, track_prediction, get_accuracy
import pickle as pkl
import os
import project_paths2 as pp


def main():
    model = load_model()
    model.validation = True

    # get ground truth from the pkl file
    # ----------------------------------------------------------------------------
    pkl_path = pp.TEST_LABELS
    # ----------------------------------------------------------------------------

    f = open(pkl_path, 'r')
    annotation_test = pkl.load(f)
    # ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness']
    annotation_test_keys = annotation_test.keys()
    all_video_names = annotation_test[annotation_test_keys[0]].keys()
    print(len(annotation_test_keys))

    for ind in range(10):
        print('ind: ', ind)
        video_id = all_video_names[ind]
        target_labels = [annotation_test['extraversion'][video_id],
                         annotation_test['agreeableness'][video_id],
                         annotation_test['conscientiousness'][video_id],
                         annotation_test['neuroticism'][video_id],
                         annotation_test['openness'][video_id]]
        video = find_video_test(video_id)

        y = predict_trait(video, model)
        print(video_id)
        print('ValueExtraversion, ValueAgreeableness, ValueConscientiousness, ValueNeurotisicm, ValueOpenness')
        print(y)
        print(target_labels)
        track_prediction(video_id, y, target_labels, write_file=pp.TEST_LOG)

    # calculate and print mean performance
    # get_accuracy(pp.TEST_LOG, num_keys=len(annotation_test_keys))
    get_accuracy(pp.TEST_LOG, num_keys=len(annotation_test_keys)-1)


main()
