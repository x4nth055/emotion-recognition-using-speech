"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""

import numpy as np
import pandas as pd
import random
import tqdm
import os

from keras.utils import to_categorical
from utils import extract_feature, get_label, categories_reversed

class AudioGenerator():
    def __init__(self, minibatch_size=20, desc_file=None, audio_config=None):
        """
        Params:
            desc_file (str, optional): Path to a CSV file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.cur_train_index = 0
        self.cur_valid_index = 0
        self.cur_test_index = 0
        self.minibatch_size = minibatch_size
        self.audio_config = audio_config

    def get_batch(self, partition):
        """ Obtain a batch of train, validation, or test data
        """
        if partition == 'train':
            audio_paths = self.train_audio_paths
            cur_index = self.cur_train_index
            emotions = self.train_emotions
            features = self.train_features
        elif partition == 'valid':
            audio_paths = self.valid_audio_paths
            cur_index = self.cur_valid_index
            emotions = self.valid_emotions
            features = self.valid_features
        elif partition == 'test':
            audio_paths = self.test_audio_paths
            cur_index = self.cur_test_index
            emotions = self.test_emotions
            features = self.test_features
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")

        features = [ feature for feature in features[cur_index: cur_index+self.minibatch_size] ]
        
        # in case some features aren't fixed size ( won't happen )
        max_length = max([features[i].shape[0] 
            for i in range(0, self.minibatch_size)])
        
        # initialize the arrays
        X_data = np.zeros([self.minibatch_size, max_length])
        
        y_data = np.zeros([self.minibatch_size, 5])
        for i in range(0, self.minibatch_size):
            # calculate X_data & input_length
            feat = features[i]
            X_data[i, :feat.shape[0]] = feat
            
            emotion = emotions[cur_index+i]

            y_data[i] = to_categorical(categories_reversed[emotion]-1, 5, dtype="int32")
        # return the arrays
        X_data = np.expand_dims(X_data, axis=1)
        return X_data, y_data

    def shuffle_data_by_partition(self, partition):
        """ Shuffle the training or validation data
        """
        if partition == "train":
            self.train_audio_paths, self.train_emotions, self.train_features = shuffle_data(self.train_audio_paths,
             self.train_emotions, self.train_features)
        elif partition == "valid":
            self.valid_audio_paths, self.valid_emotions, self.valid_features = shuffle_data(self.valid_audio_paths,
             self.valid_emotions, self.valid_features)
        elif partition == "test":
            self.test_audio_paths, self.test_emotions, self.test_features = shuffle_data(self.test_audio_paths,
             self.test_emotions, self.test_features)
        else:
            raise Exception("Invalid partition. "
                "Must be train/validation")

    def next_train(self):
        """ Obtain a batch of training data
        """
        while True:
            ret = self.get_batch('train')
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index >= len(self.train_emotions) - self.minibatch_size:
                self.cur_train_index = 0
                self.shuffle_data_by_partition('train')
            yield ret    

    def next_valid(self):
        """ Obtain a batch of validation data
        """
        while True:
            ret = self.get_batch('valid')
            self.cur_valid_index += self.minibatch_size
            if self.cur_valid_index >= len(self.valid_emotions) - self.minibatch_size:
                self.cur_valid_index = 0
                self.shuffle_data_by_partition('valid')
            yield ret

    def next_test(self):
        """ Obtain a batch of test data
        """
        while True:
            ret = self.get_batch('test')
            self.cur_test_index += self.minibatch_size
            if self.cur_test_index >= len(self.test_emotions) - self.minibatch_size:
                self.cur_test_index = 0
            yield ret

    def load_train_data(self, desc_file='train_speech.csv', shuffle=False):
        self.load_metadata_from_desc_file(desc_file, 'train')
        if shuffle:
            self.shuffle_data_by_partition("train")

    def load_validation_data(self, desc_file='valid_speech.csv', shuffle=False):
        self.load_metadata_from_desc_file(desc_file, 'validation')
        if shuffle:
            self.shuffle_data_by_partition("valid")

    def load_test_data(self, desc_file='test_speech.csv', shuffle=False):
        self.load_metadata_from_desc_file(desc_file, 'test')
        if shuffle:
            self.shuffle_data_by_partition("test")
    
    def load_metadata_from_desc_file(self, desc_file, partition):
        """ Read metadata from a CSV file
        Params:
            desc_file (str):  Path to a CSV file that contains labels and
                paths to the audio files
            partition (str): One of 'train', 'validation' or 'test'
        """
        df = pd.read_csv(desc_file)
        print("Loading audio file paths and its corresponding labels...")
        audio_paths, emotions = list(df['path']), list(df['emotion'])

        if not os.path.isdir("features"):
            os.mkdir("features")

        label = get_label(self.audio_config)
        name = f"features/{partition}_{label}"
        if os.path.isfile(name + ".npy"):
            features = np.load(f"{name}.npy")
        else:
            features = np.array([ extract_feature(a, **self.audio_config) for a in tqdm.tqdm(audio_paths, "Extracting features for {}".format(partition)) ])
            np.save(name, features)
        if partition == "train":
            try:
                self.train_audio_paths
            except AttributeError:
                self.train_audio_paths = audio_paths
                self.train_emotions = emotions
                self.train_features = features
            else:
                print("Adding train")
                self.train_audio_paths += audio_paths
                self.train_emotions += emotions
                self.train_features = np.vstack((self.train_features, features))
        elif partition == "validation":
            try:
                self.valid_audio_paths
            except AttributeError:
                self.valid_audio_paths = audio_paths
                self.valid_emotions = emotions
                self.valid_features = features
            else:
                print("Adding train")
                self.valid_audio_paths += audio_paths
                self.valid_emotions += emotions
                self.valid_features = np.vstack((self.valid_features, features))
        elif partition == "test":
            try:
                self.test_audio_paths
            except AttributeError:
                self.test_audio_paths = audio_paths
                self.test_emotions = emotions
                self.test_features = features
            else:
                print("Adding train")
                self.test_audio_paths += audio_paths
                self.test_emotions += emotions
                self.test_features = np.vstack((self.test_features, features))

    def get_input_dim(self):
        if not self.train_audio_paths:
            raise TypeError("Please load training data before trying to get the input dimension.")
        return extract_feature(self.train_audio_paths[0], **self.audio_config).shape[0]


# def shuffle_data(audio_paths, durations, emotion):
def shuffle_data(audio_paths, emotions, features):
    """ Shuffle the data (called after making a complete pass through 
        training or validation data during the training process)
    Params:
        audio_paths (list): Paths to audio clips
        durations (list): Durations of utterances for each audio clip
        emotions (list): Emotions in each audio clip
    """
    p = np.random.permutation(len(audio_paths))
    audio_paths = [audio_paths[i] for i in p] 
    # durations = [durations[i] for i in p] 
    emotions = [emotions[i] for i in p]
    features = [features[i] for i in p]
    # return audio_paths, durations, emotions
    return audio_paths, emotions, features