import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf
import numpy as np

# config = tf.ConfigProto(intra_op_parallelism_threads=5,
#                         inter_op_parallelism_threads=5, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )
from keras.layers import LSTM, GRU, Dense, Activation, LeakyReLU, Dropout
from keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical

from sklearn.metrics import accuracy_score, mean_absolute_error, confusion_matrix

from data_extractor import load_data
from create_csv import write_custom_csv, write_emodb_csv, write_tess_ravdess_csv
from emotion_recognition import EmotionRecognizer
from utils import get_first_letters, AVAILABLE_EMOTIONS, extract_feature

import pandas as pd


class DeepEmotionRecognizer(EmotionRecognizer):
    def __init__(self, n_cnn_layers=0, n_rnn_layers=2, n_dense_layers=1, cnn_units=128, kernel_size=11, rnn_units=128,
                dense_units=128, tess_ravdess=True, emodb=True, custom_db=True, tess_ravdess_name="tess_ravdess.csv",
                emodb_name="emodb.csv", audio_config=None, classification=True, overrite_csv=True,
                cell=LSTM, emotions=["sad", "neutral", "happy"], dropout=None, optimizer="adam",
                loss="categorical_crossentropy", batch_size=128, epochs=1000, verbose=1):
        # init EmotionRecognizer
        super().__init__(None, emotions=emotions, tess_ravdess=tess_ravdess, emodb=emodb, custom_db=custom_db,
                        tess_ravdess_name=tess_ravdess_name, emodb_name=emodb_name, audio_config=audio_config,
                        classification=classification, overrite_csv=overrite_csv)
        self.emotions = emotions

        self.n_cnn_layers = n_cnn_layers
        self.n_rnn_layers = n_rnn_layers
        self.n_dense_layers = n_dense_layers
        self.cnn_units = cnn_units
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.cell = cell

        # list of dropouts of each layer
        # must be len(dropouts) = n_rnn_layers + n_dense_layers
        self.dropout = dropout if dropout else [0.1] * ( self.n_rnn_layers + self.n_dense_layers )
        # number of classes ( emotions )
        self.output_dim = len(self.emotions)

        # optimization attributes
        self.optimizer = optimizer
        self.loss = loss

        # training attributes
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.verbose = verbose
        # the name of the model
        self.model_name = ""
        self._update_model_name()

        # init the model
        self.model = None

        # compute the input length
        self._compute_input_length()

        # boolean attributes
        self.model_created = False

    def _update_model_name(self):
        emotions_str = get_first_letters(self.emotions)
        # 'c' for classification & 'r' for regression
        problem_type = 'c' if self.classification else 'r'
        self.model_name = f"{emotions_str}-{problem_type}-{self.cell.__name__}-layers-{self.n_rnn_layers}-{self.n_dense_layers}-units-{self.rnn_units}-{self.dense_units}-dropout-{'_'.join([ str(d) for d in self.dropout])}.h5"

    def _get_model_filename(self):
        """Returns the relative path of this model name"""
        return f"results/{self.model_name}"

    def _model_exists(self):
        """Checks if model already exists in disk, returns the filename,
        returns `None` otherwise"""
        filename = self._get_model_filename()
        return filename if os.path.isfile(filename) else None

    def _compute_input_length(self):
        if not self.data_loaded:
            self.load_data()

        self.input_length = self.X_train[0].shape[1]

    def _verify_emotions(self):
        super()._verify_emotions()
        self.int2emotions = {i: e for i, e in enumerate(self.emotions)}
        self.emotions2int = {v: k for k, v in self.int2emotions.items()}

    def create_model(self):
        """
        Constructs the neural network
        """

        if self.model_created:
            # model already created, why call twice
            return

        if not self.data_loaded:
            # if data isn't loaded yet, load it
            self.load_data()
        
        model = Sequential()

        # # cnn layers
        # for i in range(self.n_cnn_layers):
        #     if i == 0:
        #         # first layer
        #         model.add(Conv1D(self.cnn_units))
        #         model.add(Dropout(self.dropout[0]))
        #     if i % 2 == 0:
        #         model.add(MaxPool1D())
        
        # model.add()

        # rnn layers
        for i in range(self.n_rnn_layers):
            if i == 0:
                # first layer
                model.add(self.cell(self.rnn_units, return_sequences=True, input_shape=(None, self.input_length)))
                model.add(Dropout(self.dropout[i]))
            else:
                # middle layers
                model.add(self.cell(self.rnn_units, return_sequences=True))
                model.add(Dropout(self.dropout[i]))

        if self.n_rnn_layers == 0:
            i = 0

        # dense layers
        for j in range(self.n_dense_layers):
            # if n_rnn_layers = 0, only dense
            if self.n_rnn_layers == 0 and j == 0:
                model.add(Dense(self.dense_units, input_shape=(None, self.input_length)))
                model.add(Dropout(self.dropout[i+j]))
            else:
                model.add(Dense(self.dense_units))
                model.add(Dropout(self.dropout[i+j]))
                
        if self.classification:
            model.add(Dense(self.output_dim, activation="softmax"))
            model.compile(loss=self.loss, metrics=["accuracy"], optimizer=self.optimizer)
        else:
            model.add(Dense(1, activation="linear"))
            model.compile(loss="mean_absolute_error", metrics=["mean_absolute_error"], optimizer=self.optimizer)
        
        self.model = model
        self.model_created = True
        if self.verbose > 0:
            print("[+] Model created")

    def load_data(self):
        super().load_data()
        # reshape to 3 dims
        X_train_shape = self.X_train.shape
        X_test_shape = self.X_test.shape
        self.X_train = self.X_train.reshape((1, X_train_shape[0], X_train_shape[1]))
        self.X_test = self.X_test.reshape((1, X_test_shape[0], X_test_shape[1]))

        if self.classification:
            self.y_train = to_categorical([ self.emotions2int[str(e)] for e in self.y_train ])
            self.y_test = to_categorical([ self.emotions2int[str(e)] for e in self.y_test ])
        
        y_train_shape = self.y_train.shape
        y_test_shape = self.y_test.shape
        if self.classification:
            self.y_train = self.y_train.reshape((1, y_train_shape[0], y_train_shape[1]))    
            self.y_test = self.y_test.reshape((1, y_test_shape[0], y_test_shape[1]))
        else:
            self.y_train = self.y_train.reshape((1, y_train_shape[0], 1))
            self.y_test = self.y_test.reshape((1, y_test_shape[0], 1))

    def train(self, override=True):
        
        # if model isn't created yet, create it
        if not self.model_created:
            self.create_model()

        # if the model already exists and trained, just load the weights and return
        # but if override is True, then just skip loading weights
        if not override:
            model_name = self._model_exists()
            if model_name:
                self.model.load_weights(model_name)
                self.model_trained = True
                if self.verbose > 0:
                    print("[*] Model weights loaded")
                return
        
        if not os.path.isdir("results"):
            os.mkdir("results")

        if not os.path.isdir("logs"):
            os.mkdir("logs")

        model_filename = self._get_model_filename()

        self.checkpointer = ModelCheckpoint(model_filename, save_best_only=True, verbose=1)
        self.tensorboard = TensorBoard(log_dir=f"logs/{self.model_name}")

        self.history = self.model.fit(self.X_train, self.y_train,
                        batch_size=self.batch_size,
                        epochs=self.epochs,
                        validation_data=(self.X_test, self.y_test),
                        callbacks=[self.checkpointer, self.tensorboard],
                        verbose=self.verbose)
        
        self.model_trained = True
        if self.verbose > 0:
            print("[+] Model trained")

    def predict(self, audio_path):
        feature = extract_feature(audio_path, **self.audio_config).reshape((1, 1, self.input_length))
        if self.classification:
            return self.int2emotions[self.model.predict_classes(feature)[0][0]]
        else:
            return self.model.predict(feature)[0][0][0]

    def test_score(self):
        y_test = self.y_test[0]
        if self.classification:
            y_pred = self.model.predict_classes(self.X_test)[0]
            return accuracy_score(y_true=y_test, y_pred=y_pred)
        else:
            y_pred = self.model.predict(self.X_test)[0]
            return mean_absolute_error(y_true=y_test, y_pred=y_pred)

    def confusion_matrix(self, percentage=True, labeled=True):
        """Compute confusion matrix to evaluate the test accuracy of the classification"""
        if not self.classification:
            raise NotImplementedError("Confusion matrix works only when it is a classification problem")
        y_pred = self.model.predict_classes(self.X_test)[0]
        matrix = confusion_matrix(self.y_test, y_pred, labels=self.emotions).astype(np.float32)
        if percentage:
            for i in range(len(matrix)):
                matrix[i] = matrix[i] / np.sum(matrix[i])
            # make it percentage
            matrix *= 100
        if labeled:
            matrix = pd.DataFrame(matrix, index=[ f"true_{e}" for e in self.emotions ],
                                    columns=[ f"predicted_{e}" for e in self.emotions ])
        return matrix


if __name__ == "__main__":
    rec = DeepEmotionRecognizer("", n_rnn_layers=2, n_dense_layers=1, dropout=[0.35, 0.35, 0.35],
                                classification=True, verbose=2)
    rec.train()