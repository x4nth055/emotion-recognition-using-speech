import glob
import pandas as pd
from utils import categories


target = {"path": [], "emotion": []}

# for training speech directory

for index, category in categories.items():
    for i, path in enumerate(glob.glob(f"data/training/Actor_*/*_{category}.wav")):
        target["path"].append(path)
        target["emotion"].append(category)
    print(f"There are {i} training audio files for category:{category}")


pd.DataFrame(target).to_csv("train_speech.csv")

target = {"path": [], "emotion": []}

# for validation speech directory

for index, category in categories.items():
    for i, path in enumerate(glob.glob(f"data/validation/Actor_*/*_{category}.wav")):
        target["path"].append(path)
        target["emotion"].append(category)
    print(f"There are {i} validation audio files for category:{category}")


pd.DataFrame(target).to_csv("valid_speech.csv")