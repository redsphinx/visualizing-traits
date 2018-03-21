import numpy as np
import project_paths as pp


def get_features():
    features = list(np.genfromtxt(pp.CELEB_FACES_FC6, dtype=str))
    return features


def main():
    f = get_features()
    print('asdf')


main()
