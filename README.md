# Music-De-Composer

## IronHack Final Project ##

Audio feature extraction script for song decomposition

-> Features Returned:

- Genre
- Tempo
- Duration of song
- Key

-> Graphs Returned:

- Mono Audiowave
- Harmonic + Percussive Content
- MFCC´s 
- Mel-Spectrogram
- Chromagram cqt
- Chromagram stft

## You input a song and it returns a folder with all the info and features of the song and a little audio player 

By using Machine Learning to predict the genre of the song and extracting the rest of the features using Librosa,
I was able to decompose a song to return the raw features that make up that song.

Using Keras and Tensorflow I have trained a neural network to predict a song´s genre with a 90% accuracy.

## Files and Directories in Github Repo

-- Demo.py

In the demo.py file you can see the full functioning code for demoing the script using a song in your system.

-- Main.py

In the Main.py file ...

-- Feature_Extraction.py

In the feature_Extraction.py file you ca see the full code for the manipulation and return of a fully functional
dataset from a directory of audio files (another dataset) if you want to train the model using a different set of 
data.

-- Jupyter Notebooks

In the jupyter notebook files you can see the proccess of feature extraction and dataframe manipulation for training the model and
setting the dataset.

-- DATA

In the DATA directory you will find two subdirectories:

- genres -> Contains the initial dataset used to train the model
- FadsoMusicDemos -> Contains the demos used to test the jupyter notebooks and demo.py
