from multiprocessing import Pool
import os


def f(x):
    os.mkdir('asf_%s' % str(x))
    s = x + '_OMG'
    return s


if __name__ == '__main__':
    pool = Pool(processes=4)              # start 4 worker processes
    some_list = ['asdf', 'wert', 'qweirtywuer', 'asdfgasdf', '2345']
    pool.apply_async(f)    # evaluate "f(10)" asynchronously
    # print result.get(timeout=1)           # prints "100" unless your computer is *very* slow
    pool.map(f, some_list)          # prints "[0, 1, 4,..., 81]"