from util import load_model, predict_trait, find_video, track_prediction, get_accuracy
import pickle as pkl
import os


def main():
    model = load_model()

    # get ground truth from the pkl file
    # ----------------------------------------------------------------------------
    pkl_path = 'chalearn_fi_17_compressed/annotation_test.pkl'
    # ----------------------------------------------------------------------------

    f = open(pkl_path, 'r')
    annotation_test = pkl.load(f)
    # ['extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'interview', 'openness']
    annotation_test_keys = annotation_test.keys()
    all_video_names = annotation_test[annotation_test_keys[0]].keys()

    for ind in range(2000):
        print('ind: ', ind)
        video_id = all_video_names[ind]
        target_labels = [annotation_test['extraversion'][video_id],
                         annotation_test['agreeableness'][video_id],
                         annotation_test['conscientiousness'][video_id],
                         annotation_test['neuroticism'][video_id],
                         annotation_test['openness'][video_id]]
        video = find_video(video_id)
        y = predict_trait(video, model)
        print(video_id)
        print('ValueExtraversion, ValueAgreeableness, ValueConscientiousness, ValueNeurotisicm, ValueOpenness')
        print(y)
        print(target_labels)
        track_prediction(video_id, y, target_labels)

        # calculate and print mean performance
        get_accuracy('performance_chalearn.txt')


main()
