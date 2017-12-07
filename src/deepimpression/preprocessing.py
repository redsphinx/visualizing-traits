import numpy as np


# TODO: create thing that randomly samples a 50176 sample temporal crop of the audio data

# TODO: create thing that randomly gets a 224 pixels x 224 pixels spatial crop of a random frame of the
# TODO: visual data is randomly flipped in the left/right direction and fed into the visual stream

# TODO: create a pipelining tool to get videos and labels and feed to iterator

class RandomIterator(object):
    """
    Generates random subsets of data
    """

    def __init__(self, data, batch_size=1):
        """

        Args:
            data (TupleDataset):
            batch_size (int):

        Returns:
            list of batches consisting of (input, output) pairs
        """

        self.data = data

        self.batch_size = batch_size
        self.n_batches = len(self.data) // batch_size

    def __iter__(self):

        self.idx = -1
        self._order = np.random.permutation(len(self.data))[:(self.n_batches * self.batch_size)]

        return self

    def next(self):

        self.idx += 1

        if self.idx == self.n_batches:
            raise StopIteration

        i = self.idx * self.batch_size

        # handles unlabeled and labeled data
        if isinstance(self.data, np.ndarray):
            return self.data[self._order[i:(i + self.batch_size)]]
        else:
            return list(self.data[self._order[i:(i + self.batch_size)]])