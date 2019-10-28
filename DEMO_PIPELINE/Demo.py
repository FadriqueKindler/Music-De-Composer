import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as ms
import librosa
import librosa.feature
import librosa.display
import IPython.display as ipd
from pydub import AudioSegment
import soundfile as sf
import sys
import os
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, scale
from keras import models
from keras import layers
from keras.models import load_model
import tensorflow as tf
from scipy.stats import kurtosis
from scipy.stats import skew


print("Enter Song Name:")

inputSong_name = input()
inputSong_path = "./DATA/FadsoMusicDemos/"
input_song = inputSong_path + inputSong_name + ".wav"
exported_song = input_song[23:4]
inputPickle_path = "./DATA/DEMO_SONGS/"

TEST = inputPickle_path + exported_song + ".pkl"
TRAIN = "./DATA.Model_Features_Dataset.pkl"
MODEL = "./DATA/MODELS/NeuralNetworks_Model.h5"


genres = {0: 'Blues', 1: 'Classical', 2: 'Country', 3: 'Disco', 4: 'Hiphop',
          5: 'Jazz', 6: 'Metal', 7: 'Pop', 8: 'Reggae', 9: 'Rock'}



def import_song(input_song, sr = 44100, n_fft = 1024, hop_length = 512):
    
    
             
    # import song
    filename = input_song
    y, sr = librosa.load(filename)
    
    # create directory to save all graphs
    dirName = "GRAPHS_FOR_{}".format(inputSong_name)
    
    try:
    # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")


    # print song name
    print("Your Chosen Song Is : {}".format(inputSong_name))
    
    # print duration 
    duration = librosa.core.get_duration(y=y,sr=sr)
    print("duration: " + str("{:0.2f}".format(duration)) + " seconds")
    
    # Get song Tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    print("Song tempo: "+ str("{:0.2f}".format(tempo)) + " BPM")

    # print audio player

    #display(ipd.Audio(data = filename, rate =sr))

    # DISPLAY AUDIO WAVE
    plt.figure(figsize=(24,6))

    librosa.display.waveplot(y, sr=sr)
    plt.title('Mono');
    plt.savefig('./GRAPHS_FOR_{}/Audiowave.png'.format(inputSong_name))
    
    ## DISPLAY HARMONIC + PERCUSSIVE (MONO)
    
    y_harm, y_perc = librosa.effects.hpss(y)
    plt.figure(figsize=(24,6))
    librosa.display.waveplot(y_harm, sr=sr, alpha=0.35)
    librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.25)
    plt.title('Harmonic + Percussive')
    plt.savefig('./GRAPHS_FOR_{}/Harmonic_Percussive_AudioWave.png'.format(inputSong_name))

    
    #Plot mfcc (normalized)
    mfcc = librosa.feature.mfcc(y = y , n_fft=1024, hop_length=512)
    S = librosa.feature.melspectrogram(y = y, sr=sr)
    S_dB = librosa.power_to_db(S, ref = np.max)
    
    plt.figure(figsize=(22, 6))
    librosa.display.specshow(S_dB, x_axis='time',
                          y_axis='mel', sr=sr,
                          fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.savefig('./GRAPHS_FOR_{}/Mel-Spectrogram.png'.format(inputSong_name))

    
    mfcc = librosa.feature.mfcc(S = S_dB, n_mfcc= 15, n_fft=n_fft, hop_length=hop_length)
    mfcc_nzd = sklearn.preprocessing.scale(mfcc, axis=1)

    plt.figure(figsize=(22, 6))
    librosa.display.specshow(mfcc_nzd, sr=sr, x_axis='time');
    plt.savefig('./GRAPHS_FOR_{}/MFCC(Normalized).png'.format(inputSong_name))

    
    # Plot Chromagrams
    
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    plt.figure(figsize=(22,6))
    librosa.display.specshow(chroma_stft, y_axis='chroma', x_axis='time')
    plt.title('chroma_stft')
    plt.colorbar()
    plt.savefig('./GRAPHS_FOR_{}/Chroma(STFT).png'.format(inputSong_name))

    plt.figure(figsize=(22,6))
    librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time')
    plt.title('chroma_cqt')
    plt.colorbar()
    plt.savefig('./GRAPHS_FOR_{}/Chroma(CQT).png'.format(inputSong_name))



def create_primaryDf(input_song, sr = 44100, n_fft = 1024, hop_length = 512):
    
    ### DEFINE VARIABLES ###
    dictKeySong_ValueArray = {}
    s_c = []
    s_r = []
    s_f = []
    rms = []
    zcr = []
    chgrm = []
    mfcc = []
    
    
    y, sr = librosa.load(input_song, sr=sr)
    dictKeySong_ValueArray[input_song] = y
            
    s_c.append(librosa.feature.spectral_centroid(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel())
    s_r.append(librosa.feature.spectral_rolloff(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel())
    s_f.append(librosa.onset.onset_strength(y=y, sr=sr).ravel())
    rms.append(librosa.feature.rms(y, frame_length=n_fft, hop_length=hop_length).ravel())
    zcr.append(librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length).ravel())
    chgrm.append(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel())

    ## APPEND MFCC ##
    mfcc_array = librosa.feature.mfcc(y, n_fft=n_fft, hop_length=hop_length, n_mfcc=15)
    mfcc.append(mfcc_array.ravel())
        
        
    primaryDf = pd.DataFrame(list(dictKeySong_ValueArray.items()))
    
    primaryDf['Spectral_Centroid'] = s_c
    primaryDf['Spectral_Rolloff'] = s_r
    primaryDf['Spectral_flux'] = s_f
    primaryDf['Rms'] = rms
    primaryDf['Zcr'] = zcr
    primaryDf['Chromagram'] = chgrm
    primaryDf["Mfcc"] = mfcc
    #primaryDf["Tempo"] = tempo_lst
    
    primaryDf = primaryDf.rename(columns = {0: "Song_Name" , 1: "Song_Array_values"})
    
    primaryDf.set_index('Song_Name')
        
    return primaryDf



def create_finalDf(primaryDf, sr = 44100 , n_fft = 1024, hop_length = 512):
    
    mean_lst = []
    std_lst = []
    kurt_lst = []
    skew_lst = []
    
    useless_cols = ["Song_Name", "Song_Array_values", 
                    "Spectral_Centroid", "Spectral_Rolloff", 
                    "Spectral_flux", "Rms", "Zcr", 
                    "Chromagram", "Mfcc"]
    
    columns = list(primaryDf)[2:]
    #return columns
    
    for column in columns:
         
        data_mean = primaryDf[column].loc[0].mean()
        data_std = primaryDf[column].loc[0].std()
        data_kurt = kurtosis(primaryDf[column].loc[0])
        data_skew = skew(primaryDf[column].loc[0])

        #mean_lst.append(data_mean)
        #std_lst.append(data_std)
        #kurt_lst.append(data_kurt)
        #skew_lst.append(data_skew)

        primaryDf['{}_Mean'.format(column)] = data_mean
        primaryDf['{}_Std'.format(column)] = data_std
        primaryDf['{}_Kurt'.format(column)] = data_kurt
        primaryDf['{}_Skew'.format(column)] = data_skew

    finalDf = primaryDf.drop(columns = useless_cols)

    exportedSong_pickle = finalDf.to_pickle("./DATA/DEMO_SONGS/{}.pkl".format(exported_song))


def model_load(MODEL, show=False):
    """
    Load previously trained model
    """
    modelo = load_model(MODEL)
    if show:
        print(modelo.summary())
    return modelo



def normalize(TEST):
    """
    Normalize test data for prediction
    """
    # Training data:
    data = pd.read_pickle(TRAIN)
    X = data.drop('Genres', axis=1)
    # Test data:
    Xt = pd.read_pickle(TEST)
    rows = Xt.shape[0]
    # Append original data and test data for normalization:
    Xt = X.append(Xt, ignore_index=True)
    # Normalize:
    sc = StandardScaler().fit_transform(Xt.values)
    Xt = pd.DataFrame(sc[-rows:], index=Xt[-rows:].index, columns=Xt.columns)
    return Xt


def predict(modelo, Xt):

    """
    Predicts labels for each song
    """
    preds = modelo.predict_classes(Xt)
    predicted = genres.get(preds[0])
    probs = modelo.predict(Xt)[0] 
    if preds == 0:
        preds = 'Blues'
    elif preds == 1:
        preds = "Classical"
    elif preds == 2:
        preds = "Country"
    elif preds == 3:
        preds = "Disco"
    elif preds == 4:
        preds = "Hiphop"
    elif preds == 5:
        preds = "Jazz"
    elif preds == 6:
        preds = "Metal"
    elif preds == 7:
        preds = "Pop"
    elif preds == 8:
        preds = "Reggae"
    elif preds == 9:
        preds = "Rock"

    return preds, probs


def present_results(preds, probs):

    """
    Present results of prediction
    """
    plt.figure(figsize=(10,10))
    plt.title(f'Predicted genre: {preds}')
    plt.bar(genres.values(), probs, color='r')
    for j in range(len(probs)):
        plt.text(x=j - 0.1, y=probs[j], s='{:.2f} %'.format((probs[j]) * 100), size=10)
    plt.savefig('./GRAPHS_FOR_{}/Predicted_Genre.png'.format(inputSong_name))
    #plt.show()

    print("Your Results Are In The {} Directory".format(inputSong_name))