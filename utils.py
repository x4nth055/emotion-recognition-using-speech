"""
Defines various functions for processing the data.
"""
import numpy as np
import soundfile
# import speechpy
# import scipy.io.wavfile as wav
# import glob
import librosa
# from speechpy.feature import mfcc

# analyzing emotions using text:
# common: sad, neutral, surprise, happy, angry
# sadness: 5165
# happiness: 5209
# neutral: 8638
# surprise: 2187
# anger: 110

categories = {
    0: "neutral",
    1: "angry",
    2: "happy",
    3: "ps", # pleasant surprised
    4: "sad"
}

categories_reversed = {v:k for k,v in categories.items() }


def get_label(audio_config):
    features = ["mfcc", "chroma", "mel", "contrast", "tonnetz"]
    label = ""
    for feature in features:
        if audio_config[feature]:
            label += f"{feature}-"
    return label.rstrip("-")


def get_label_from_config(audio_config):
    features = {0: "mfcc", 1: "chroma", 2: "mel", 3: "contrast", 4: "tonnetz"}
    label = ""
    for i, feature in features.items():
        if audio_config[i]:
            label += f"{feature}-"
    return label.rstrip("-")


# code from http://aqibsaeed.github.io/2016-09-03-urban-sound-classification-part-1/
# edited
def extract_feature(file_name, **kwargs):
    """Extract feature from audio file `file_name`
        Features supported:
            * MFCC (mfcc)
            * Chroma (chroma)
            * MEL Frequency (mel)
            * Contrast (contrast)
            * Tonnetz (tonnetz)
        Example:
        `features = extract_feature(path, mel=True, mfcc=True)`"""
    # X, sample_rate = librosa.load(file_name, sr=None)
    # mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=True
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
    # sample_rate, X = wav.read(file_name)
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result


# class SpeechAugmentation:

#     def __init__(self, emo=False):
#         self.emo = emo
#         self.sample_rate = 16000

#     def read_audio_file(self, file_path, sr=16000):
#         data, self.sample_rate = librosa.core.load(file_path)
#         return data

#     def add_noise(self, data, n=0.1):
#         # noise = np.random.randn(len(data))
#         data_noise = data + n
#         return data_noise

#     def shift(self, data, roll_rate):
#         return np.roll(data, roll_rate) + 0.035

#     def stretch(self, data, rate=1.1):
#         return librosa.effects.time_stretch(data, rate)

#     def write_audio_file(self, file, data, sr=None):
#         if sr is None:
#             librosa.output.write_wav(file, data, self.sample_rate)
#         else:
#             librosa.output.write_wav(file, data, sr)

#     def augment(self, file):
#         signal = self.read_audio_file(file)
#         # noised = self.add_noise(signal)
#         # shifted = self.shift(signal)
#         # stretched = self.stretch(signal)
#         abs_path, basename = os.path.split(file)
#         if self.emo:
#             name, ext = basename.split(".")
#             noisename = name + "_noised." + ext
#             shiftname = name + "_shifted." + ext
#             stretchname = name + "_stretched." + ext     
#         else:
#             noisename = "noised_" + basename
#             shiftname = "shifted_" + basename
#             stretchname = "stretched_" + basename

#         stretches = [0.9, 1, 1.1, 1.2]
#         noises = [0.01, 0.05, 0.1, 0.2, 0.15]
#         shifts = [200, 500, 1000, 2000, 3000]
#         monster = [14000, 16000, 18000, 22000, 25000, 26000]
#         for i, noise in enumerate(noises):
#             name, ext = basename.split(".")
#             noisename = name + f"_noised{i}." + ext
#             noised = self.add_noise(signal, n=noise)
#             self.write_audio_file(os.path.join(abs_path, noisename), noised)
#         for i, shift in enumerate(shifts):
#             shifted = self.shift(signal, shift)
#             name, ext = basename.split(".")
#             shiftname = name + f"_shifted{i}." + ext
#             self.write_audio_file(os.path.join(abs_path, shiftname), shifted)
#         for i, stretch in enumerate(stretches):
#             stretched = self.stretch(signal, stretch)
#             name, ext = basename.split(".")
#             stretchname = name + f"_stretched{i}." + ext
#             self.write_audio_file(os.path.join(abs_path, stretchname), stretched)

#         for i, m in enumerate(monster):
#             monstered = self.read_audio_file(file, sr=m)
#             name, ext = basename.split(".")
#             monstername = name + f"_monstered{i}." + ext
#             self.write_audio_file(os.path.join(abs_path, monstername), monstered, sr=m)

# if __name__ == "__main__":
#     aug = SpeechAugmentation(emo=True)
#     path = r"E:\datasets\speech\emotion_by_speech\emo_db\wav\03a01Fa.wav"
#     for file in tqdm.tqdm(glob.glob(r"E:\datasets\speech\emotion_by_speech\emo_db\wav\*.wav"), "Augmenting data"):
#         aug.augment(file)