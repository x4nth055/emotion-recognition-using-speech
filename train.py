# to use CPU
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf

# config = tf.ConfigProto(intra_op_parallelism_threads=6,
#                         inter_op_parallelism_threads=6, 
#                         allow_soft_placement=True,
#                         device_count = {'CPU' : 1,
#                                         'GPU' : 0}
#                        )

from data_generator import AudioGenerator, categories_reversed
from keras.callbacks import ModelCheckpoint
from itertools import product

import pickle
import numpy as np

TESS_OTHER_TRAIN = "train_speech.csv"
TESS_OTHER_VALID = "valid_speech.csv"

if not os.path.isdir("benchmark"):
    os.makedirs("benchmark")


def train_model(name, model, batch_size=32, epochs=10, loss="categorical_crossentropy",
                optimizer="adam", metrics=["accuracy"]):

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.summary()
    # chroma-mel-contrast
    audio_config = {
        "mfcc": True,
        "chroma": False,
        "mel": True,
        "contrast": False,
        "tonnetz": False
    }
    audio_gen = AudioGenerator(minibatch_size=batch_size, audio_config=audio_config)
    audio_gen.load_train_data(TESS_OTHER_TRAIN, shuffle=True)
    audio_gen.load_validation_data(TESS_OTHER_VALID, shuffle=True)

    steps_per_epoch = len(audio_gen.train_audio_paths) // batch_size

    if not os.path.isdir("results"):
        os.mkdir("results")

    checkpointer = ModelCheckpoint("results/" + name + "_{val_loss:.2f}.h5", save_best_only=True, verbose=1)

    history = model.fit_generator(audio_gen.next_train(), verbose=1, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        validation_data=audio_gen.next_valid(), validation_steps=steps_per_epoch, callbacks=[checkpointer])

    return history



if __name__ == "__main__":
    from models import first_model

    model = first_model(168, len(categories_reversed))
    train_model("first_model_v2", model, batch_size=64, epochs=100, optimizer="rmsprop")

