import glob
import pandas as pd
import os


def write_emodb_csv(emotions=["sad", "neutral", "happy"], train_name="train_emo.csv",
                    test_name="test_emo.csv", train_size=0.8, verbose=1):
    """
    Reads speech emodb dataset from directory and write it to a metadata CSV file.
    """
    target = {"path": [], "emotion": []}
    categories = {
        "W": "angry",
        "L": "boredom",
        "E": "disgust",
        "A": "fear",
        "F": "happy",
        "T": "sad",
        "N": "neutral"
    }
    # delete not specified emotions
    categories_reversed = { v: k for k, v in categories.items() }
    for emotion, code in categories_reversed.items():
        if emotion not in emotions:
            del categories[code]
    for file in glob.glob("data/emodb/wav/*.wav"):
        try:
            emotion = categories[os.path.basename(file)[5]]
        except KeyError:
            continue
        target['emotion'].append(emotion)
        target['path'].append(file)
    if verbose:
        print("[EMO-DB] Total files to write:", len(target['path']))
    n_samples = len(target['path'])
    test_size = int((1-train_size) * n_samples)
    train_size = int(train_size * n_samples)
    if verbose:
        print("[EMO-DB] Training samples:", train_size)
        print("[EMO-DB] Testing samples:", test_size)   
    X_train = target['path'][:train_size]
    X_test = target['path'][train_size:]
    y_train = target['emotion'][:train_size]
    y_test = target['emotion'][train_size:]
    pd.DataFrame({"path": X_train, "emotion": y_train}).to_csv(train_name)
    pd.DataFrame({"path": X_test, "emotion": y_test}).to_csv(test_name)


def write_tess_ravdess_csv(emotions=["sad", "neutral", "happy"], train_name="train_tess_ravdess.csv",
                            test_name="test_tess_ravdess.csv", verbose=1):
    """
    Reads speech TESS & RAVDESS datasets from directory and write it to a metadata CSV file.
    """
    target = {"path": [], "emotion": []}
    categories = {
        1: "neutral",
        2: "calm",
        3: "happy",
        4: "sad",
        5: "angry",
        6: "fear",
        7: "disgust",
        8: "ps"
    }
    # delete not specified emotions
    categories_reversed = { v: k for k, v in categories.items() }
    for emotion, code in categories_reversed.items():
        if emotion not in emotions:
            del categories[code]
    # for training speech directory
    for _, category in categories.items():
        for i, path in enumerate(glob.glob(f"data/training/Actor_*/*_{category}.wav")):
            target["path"].append(path)
            target["emotion"].append(category)
        if verbose:
            print(f"[TESS&RAVDESS] There are {i} training audio files for category:{category}")
    pd.DataFrame(target).to_csv(train_name)
    target = {"path": [], "emotion": []}
    # for validation speech directory
    for _, category in categories.items():
        for i, path in enumerate(glob.glob(f"data/validation/Actor_*/*_{category}.wav")):
            target["path"].append(path)
            target["emotion"].append(category)
        if verbose:
            print(f"[TESS&RAVDESS] There are {i} testing audio files for category:{category}")
    pd.DataFrame(target).to_csv(test_name)