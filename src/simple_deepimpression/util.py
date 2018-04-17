import audiovisual_stream
import chainer.serializers
import librosa
import numpy as np
import skvideo.io

def load_audio(data):
    aud = librosa.load(data, 16000)[0][None, None, None, :]
    # aud = np.expand_dims(aud, 0)
    print('audio shape: ', aud.shape)
    return aud

def load_model():
    model = audiovisual_stream.ResNet18()
    
    chainer.serializers.load_npz('./model', model)
    
    return model

def load_video(data):
    print('loading data')
    video_capture = skvideo.io.vread(data)

    frames = 1
    video_capture = video_capture[:frames]
    video_shape = np.shape(video_capture)
    video_capture = np.reshape(video_capture, (frames, video_shape[-1], video_shape[1], video_shape[2]), 'float32')
    video = np.array(video_capture, 'float32')
    # video = np.expand_dims(video, 0)

    print('video shape: ', video.shape)
    return video

def predict_trait(data, model):
    x = [load_audio(data), load_video(data)]
    
    return model(x)
