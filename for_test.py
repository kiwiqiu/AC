import numpy as np

data = np.load('dataset/features/0/1737031675060.npy')

print("----shape-----")
print(data.shape)
print("----data-------")
print(data)

# import librosa
#
# audio_path = "dataset/14772-7-0-0.wav"  # 示例路径
# y, sr = librosa.load(audio_path, sr=None)  # sr=None保留原始采样率
# print("采样率:", sr)  # 输出应为44100