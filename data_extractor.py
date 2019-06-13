
import numpy as np
import pandas as pd
import pickle
import tqdm
import os

from utils import get_label, extract_feature, get_first_letters


class AudioExtractor:
    """A class that is used to featurize audio clips, and provide
    them to the network for training and testing"""
    def __init__(self, desc_file=None, audio_config=None, verbose=1, features_folder_name="features", classification=True,
                    emotions=['sad', 'neutral', 'happy']):
        """Params:
            desc_file (str, optional): Path to a csv file that contains labels and paths to the audio files.
            If provided, then load metadata right away."""
        self.desc_file = desc_file
        self.audio_config = audio_config if audio_config else {'mfcc': True, 'chroma': True, 'mel': True, 'contrast': False, 'tonnetz': False}
        self.verbose = verbose
        self.features_folder_name = features_folder_name
        self.classification = classification
        self.emotions = emotions
        # input dimension
        self.input_dimension = None

    def _load_data(self, desc_files, partition, shuffle):
        self.load_metadata_from_desc_file(desc_files, partition)
        if shuffle:
            self.shuffle_data_by_partition(partition)

    def load_train_data(self, desc_files=["train_speech.csv"], shuffle=False):
        """Loads training data from the metadata files `desc_files`"""
        self._load_data(desc_files, "train", shuffle)
        
    def load_test_data(self, desc_files=["test_speech.csv"], shuffle=False):
        """Loads testing data from the metadata files `desc_files`"""
        self._load_data(desc_files, "test", shuffle)

    def shuffle_data_by_partition(self, partition):
        if partition == "train":
            self.train_audio_paths, self.train_emotions, self.train_features = shuffle_data(self.train_audio_paths,
            self.train_emotions, self.train_features)
        elif partition == "test":
            self.test_audio_paths, self.test_emotions, self.test_features = shuffle_data(self.test_audio_paths,
            self.test_emotions, self.test_features)
        else:
            raise TypeError("Invalid partition, must be either train/test")

    def load_metadata_from_desc_file(self, desc_files, partition):
        """Read metadata from a CSV file
        Params:
            desc_files"""
        # empty dataframe
        df = pd.DataFrame({'path': [], 'emotion': []})
        for desc_file in desc_files:
            # concat dataframes
            df = pd.concat((df, pd.read_csv(desc_file)), sort=False)
        if self.verbose:
            print("[*] Loading audio file paths and its corresponding labels...")
        # get columns
        audio_paths, emotions = list(df['path']), list(df['emotion'])
        # if not classification, convert emotions to numbers
        if not self.classification:
            # so naive and need to be implemented
            # in a better way
            if len(self.emotions) == 3:
                categories = {'sad': 0, 'neutral': 1, 'happy': 2}
            elif len(self.emotions) == 5:
                categories = {'angry': 1, 'sad': 2, 'neutral': 3, 'ps': 4, 'happy': 5}
            else:
                raise TypeError("Regression is only for either ['sad', 'neutral', 'happy'] or ['angry', 'sad', 'neutral', 'ps', 'happy']")
            emotions = [ categories[e] for e in emotions ]
        # make features folder if does not exist
        if not os.path.isdir(self.features_folder_name):
            os.mkdir(self.features_folder_name)
        # get label for features
        label = get_label(self.audio_config)
        # construct features file name
        n_samples = len(audio_paths)
        first_letters = get_first_letters(self.emotions)
        name = os.path.join(self.features_folder_name, f"{partition}_{label}_{first_letters}_{n_samples}.npy")
        if os.path.isfile(name):
            # if file already exists, just load then
            if self.verbose:
                print("[+] Feature file already exists, loading...")
            features = np.load(name)
        else:
            # file does not exist, extract those features and dump them into the file
            features = []
            append = features.append
            for audio_file in tqdm.tqdm(audio_paths, f"Extracting features for {partition}"):
                feature = extract_feature(audio_file, **self.audio_config)
                if self.input_dimension is None:
                    self.input_dimension = feature.shape[0]
                append(feature)
            # convert to numpy array
            features = np.array(features)
            # save it
            np.save(name, features)
        if partition == "train":
            try:
                self.train_audio_paths
            except AttributeError:
                self.train_audio_paths = audio_paths
                self.train_emotions = emotions
                self.train_features = features
            else:
                if self.verbose:
                    print("[*] Adding additional training samples")
                self.train_audio_paths += audio_paths
                self.train_emotions += emotions
                self.train_features = np.vstack((self.train_features, features))
        elif partition == "test":
            try:
                self.test_audio_paths
            except AttributeError:
                self.test_audio_paths = audio_paths
                self.test_emotions = emotions
                self.test_features = features
            else:
                if self.verbose:
                    print("[*] Adding additional testing samples")
                self.test_audio_paths += audio_paths
                self.test_emotions += emotions
                self.test_features = np.vstack((self.test_features, features))
        else:
            raise TypeError("Invalid partition, must be either train/test")

    

def shuffle_data(audio_paths, emotions, features):
    """ Shuffle the data (called after making a complete pass through 
        training or validation data during the training process)
    Params:
        audio_paths (list): Paths to audio clips
        emotions (list): Emotions in each audio clip
        features (list): features audio clips
    """
    p = np.random.permutation(len(audio_paths))
    audio_paths = [audio_paths[i] for i in p] 
    emotions = [emotions[i] for i in p]
    features = [features[i] for i in p]
    return audio_paths, emotions, features


def load_data(train_desc_files, test_desc_files, audio_config=None, classification=True, shuffle=True,
                emotions=['sad', 'neutral', 'happy']):
    audiogen = AudioExtractor(audio_config=audio_config, classification=classification, emotions=emotions, verbose=0)
    # Loads training data
    audiogen.load_train_data(train_desc_files, shuffle=shuffle)
    # Loads testing data
    audiogen.load_test_data(test_desc_files, shuffle=shuffle)
    # X_train, X_test, y_train, y_test
    return np.array(audiogen.train_features), np.array(audiogen.test_features), np.array(audiogen.train_emotions), np.array(audiogen.test_emotions)