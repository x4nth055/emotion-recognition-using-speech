# Speech Emotion Recognition
## Introduction
- The basic idea behind this tool is to build and train a suited machine learning algorithm that could recognize and detects human emotions from speech.
- This is useful for many industry fields such as making product recommendations, affective computing, etc.
## Requirements
- **Python 3**
### Python Packages
- **librosa==0.6.3**
- **numpy**
- **pandas**
- **soundfile==0.9.0**
- **wave**
- **sklearn**
- **tqdm==4.28.1**
- **matplotlib==2.2.3**
- **pyaudio==0.2.11**

Install these libraries by the following command:
```
pip3 install -r requirements.txt
```

### Dataset
This repository used 4 datasets (including this repo's custom dataset) which are downloaded and formatted already in `data` folder:
- [**RAVDESS**](https://zenodo.org/record/1188976) : The **R**yson **A**udio-**V**isual **D**atabase of **E**motional **S**peech and **S**ong that contains 24 actors (12 male, 12 female), vocalizing two lexically-matched statements in a neutral North American accent.
- [**TESS**](https://tspace.library.utoronto.ca/handle/1807/24487) : **T**oronto **E**motional **S**peech **S**et that was modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966). A set of 200 target words were spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years).
- [**EMO-DB**](http://emodb.bilderbar.info/docu/) : As a part of the DFG funded research project SE462/3-1 in 1997 and 1999 we recorded a database of emotional utterances spoken by actors. The recordings took place in the anechoic chamber of the Technical University Berlin, department of Technical Acoustics. Director of the project was Prof. Dr. W. Sendlmeier, Technical University of Berlin, Institute of Speech and Communication, department of communication science. Members of the project were mainly Felix Burkhardt, Miriam Kienast, Astrid Paeschke and Benjamin Weiss.
- **Custom** : Some unbalanced noisy dataset in which you can add/remove recording samples easily by converting the raw audio to 16000 sample rate, and mono channel (this is provided in `create_wavs.py` script in ``convert_audio(audio_path)`` method ).

### Emotions available
So the total emotions available are "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise) and "boredom".
## Feature Extraction
Feature extraction is the main part of the speech emotion recognition system. It is basically accomplished by changing the speech waveform to a form of parametric representation at a relatively lesser data rate.

In this repository, we have used the most used features that are available in [librosa](https://github.com/librosa/librosa) library including:
- [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- Chromagram 
- MEL Spectrogram Frequency (mel)
- Contrast
- Tonnetz (tonal centroid features)

## Algorithms Used
This repository can be used to build ( using sklearn ) machine learning classifiers as well as regressors for the case of 3 emotions {'sad': 0, 'neutral': 1, 'happy': 2} and the case of 5 emotions {'angry': 1, 'sad': 2, 'neutral': 3, 'ps': 4, 'happy': 5}
### Classifiers
- SVC
- RandomForestClassifier
- GradientBoostingClassifier
- KNeighborsClassifier
- MLPClassifier
- BaggingClassifier
### Regressors
- SVR
- RandomForestRegressor
- GradientBoostingRegressor
- KNeighborsRegressor
- MLPRegressor
- BaggingRegressor
## Grid Search
Grid search results are already provided in `grid` folder, but if you want to tune various grid search parameters in `parameters.py`, you can run the script `grid_search.py` by:
```
python grid_search.py
```
This may take several hours to complete execution, once it is finished, results are stored in `grid`.

## Example: Using 3 Emotions
The way to build and train a model for classifying 3 emotions is as shown below:
```python
from emotion_recognition import EmotionRecognizer
from sklearn.svm import SVC
# init a model, let's use SVC
my_model = SVC()
# pass my model to EmotionRecognizer instance
rec = EmotionRecognizer(model=my_model, emotions=['sad', 'neutral', 'happy'], verbose=0)
# train the model
rec.train()
# check the test accuracy for that model
print("Test score:", rec.test_score())
# check the train accuracy for that model
print("Train score:", rec.train_score())
```
**Output:**
```
Test score: 0.8148148148148148
Train score: 1.0
```

In order to determine the best model, you can so by retrieving the results of the GridSearchCV ( that is stored in `grid` folder ):

```python
# loads the best estimators that was retrieved from GridSearchCV,
# and set the model to the best in terms of test score, and then train it
rec.determine_best_model(train=True)
print(rec.model)
print("Test score:", rec.test_score())
```
**Output:**
```
MLPClassifier(activation='relu', alpha=0.001, batch_size=1024, beta_1=0.9,
       beta_2=0.999, early_stopping=False, epsilon=1e-08,
       hidden_layer_sizes=(300,), learning_rate='adaptive',
       learning_rate_init=0.001, max_iter=500, momentum=0.9,
       n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
       random_state=None, shuffle=True, solver='adam', tol=0.0001,
       validation_fraction=0.1, verbose=False, warm_start=False)
    
Test Score: 0.8958333333333334
```
### Predicting
Just pass an audio path to the `rec.predict()` method as shown below:
```python
# this is a neutral speech from emo-db
print("Prediction:", rec.predict("data/emodb/wav/15a04Nc.wav"))
# this is a sad speech from TESS
print("Prediction:", rec.predict("data/tess_ravdess/validation/Actor_25/25_01_01_01_mob_sad.wav"))
```
**Output:**
```
Prediction: neutral
Prediction: sad
```
### Testing
You can test your own voice by executing the following command:
```
python test.py
```
Wait until "Please talk" prompt is appeared, then you can start talking, and the model will automatically detects your emotion when you stop (talking).
### Plotting Histograms
This will only work if grid search is performed.
```python
from emotion_recognition import plot_histograms
# plot histograms on different classifiers
plot_histograms(classifiers=True)
```
**Output:**

<img src="images/Figure.png">
<p align="center">A Histogram shows different algorithms metric results on different data sizes as well as time consumed to train/predict.</p>