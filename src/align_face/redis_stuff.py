from main_thingy import align_faces_in_video
# from util import redis_stuff
import util
from redis import Redis
from rq import Queue


which_folder = 'test-1'
list_path_all_videos = util.get_path_videos(which_folder)[0:5]
util.make_folder_dirs(which_folder)
q = Queue(connection=Redis())


l1 = '/media/gabi/DATADRIVE1/datasets/chalearn_fi_17_compressed/test-1/test80_01/2Z8Xi_DTlpI.000.mp4'

q.enqueue(align_faces_in_video, l1)
