# Music-De-Composer

## IronHack Final Project ##

Audio feature extraction script for song decomposition

-> Features Returned:                

- Genre
- Tempo
- Duration of song
- Key
- Chosen Song Name

-> Graphs Returned:

- Mono Audiowave
- Harmonic + Percussive Content
- MFCC´s 
- Mel-Spectrogram
- Chromagram cqt
- Chromagram stft
- Genre Percentage

## Input A Song And Retrieve All Of It´s Features Hassle Free

By using Machine Learning (Neural Networks) I was able to decompose a song in order to return the raw features that make up that song as well as the percentages of each genre in that song.

Using Keras and Tensorflow I have trained a model to predict a song´s genre with over a 90% accuracy.

## Files And Directories In Github Repo

-> DEMO_PIPELINE Folder:

- Demo.py

In the demo.py file you can see the full functioning code you can use to demo a song in your DATA directory.

- Main.py

In the Main.py file you can find the final pipeline with which you can demo the script from the terminal.

-> JUPYTER_NOTEBOOKS Folder:

In the jupyter notebook folder you can find the jupyter notebooks for the full and uncut proccesses of feature extraction, data manipulation, dataset acquisition and model training.

- Dataset Creation And Wrangling:

Full proccess for data acquisition and manipulation. In this notebook you can see how the original dataset was nourished by splitting each song into 10 chunks. The chunks are then stored in your /DATA/SongSplits directory.

- Create Final DataFrame Pickle:

In this notebook you can see a step by step construcction of the final dataframe used as the main data for the project. At the end the dataframe is exported as a pickle to your DATA/PICKLES directory for easy and quick use as the dataframe takes about 40 minutes to be created.

- Training Neural Networks Model:

Contains a simple step-by-step of the construction of the model using Keras and Tensorflow. The model is the exported to your /DATA/MODELS directory as a .h5 file.

- Demo:

This notebook contains the fully functional demo of the project. It returns a directory in your DATA/GRAPHS directory with all the graphs and information. It also gives you an audio player for song review.

-> DATA

You will need to create a DATA directory in your Project directory with the following subdirectories: 
/n(The directory names must be exactly the same in order for the code to work!)

- genres -> Contains the initial dataset 
- SongSplits -> Contains the full dataset after manipulation
- PICKLES -> Contains the exported data pickles
- MODELS -> Contains the saved models 
- MusicDemos -> Contains the demos used to test the jupyter notebooks and demo.py
- GRAPHS -> Contains the returned graphs for your songs

