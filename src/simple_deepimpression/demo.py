from util import load_model, predict_trait
model = load_model()

# data can be the path of any video that can be loaded by librosa and skvideo.
data = '1Gn4GX8miWQ.005.mp4'
y = predict_trait(data, model)

print(data)
print('ValueExtraversion, ValueAgreeableness, ValueConscientiousness, ValueNeurotisicm, ValueOpenness')
print(y)
