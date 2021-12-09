import json
import os
import tempfile
from pathlib import Path

import cog
from emotion_recognition import EmotionRecognizer


class EmoPredictor(cog.Predictor):
    def setup(self):
        """Load the emotion recognition model and (quickly) train it"""
        # self.rec = EmotionRecognizer(None, emotions=["boredom", "neutral"], features=["mfcc"])
        self.rec = EmotionRecognizer(
            None,
            emotions=["sad", "neutral", "happy"],
            features=["mfcc"],
            probability=True,
        )
        # evaluate all models in `grid` folder and determine the best one in terms of test accuracy
        self.rec.determine_best_model()

    @cog.input("input", type=Path, help="Speech audio file")
    def predict(self, input):
        """Compute emotion prediction"""
        prediction = self.rec.predict_proba(str(input))

        return prediction
