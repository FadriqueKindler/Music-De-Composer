import argparse
import threading
import warnings
import numpy as np
import pandas as pd 
import warnings
warnings.filterwarnings("ignore")

#from MainFunctions import create_primaryDf, create_finalDf, test_train_split
from Demo import import_song, create_primaryDf, create_finalDf, model_load, normalize, predict, present_results

#print("Enter Song Name:")

#inputSong_name = input()
#inputSong_path = "./DATA/FadsoMusicDemos/"
#input_song = inputSong_path + inputSong_name + ".wav"
#exported_song = input_song[23:4]
#inputPickle_path = "./DATA/DEMO_SONGS/"

#TEST = inputPickle_path + exported_song + ".pkl"
#TRAIN = "./DATA.Model_Features_Dataset.pkl"
#MODEL = "./DATA/MODELS/NeuralNetworks_Model.h5"

def main():

    graphs = import_song(input_song)
    primaryDf = create_primaryDf(input_song)
    finalDf = create_finalDf(primaryDf)
    modelo = model_load(MODEL)
    Xt = normalize(TEST)
    prediction, probability = predict(modelo, Xt)
    results = present_results(prediction, probability)



if __name__ == "__main__":
    
    main()