#imports
import numpy as np
import os
import pandas as pd

#class for processing music labels
class labelDataset():

    def __init__(self, path):
        '''
        Class that contains methods for loading and preprocessing collaborative data that will be
        used as labels for training ML model. Class is currently setup to take an 'apple music play data.csv'
        as input, but any similar data file could be used with some tweaking.

        :param path: Full filepath to .csv file containing collaborative data (String or os.path object).

        :returns None
        '''

        self.path = path

        #load .csv into pandas dataframe
        df = pd.read_csv(self.path)

        #drop unnecessary columns
        columns_to_remove = ['Apple Id Number','Apple Music Subscription','Build Version','Client IP Address',
                             'Content Provider', 'Content Specific Type', 'Device Identifier','End Reason Type',
                             'Event Reason Hint Type','Feature Name', 'Genre', 'Item Type','Media Type',
                             'Metrics Bucket Id', 'Metrics Client Id','Offline', 'Original Title',
                             'Provided Audio Bit Depth','Provided Audio Channel', 'Provided Audio Sample Rate',
                             'Provided Bit Rate', 'Provided Codec', 'Provided Playback Format','Source Type',
                             'Store Country Name','Targeted Audio Bit Depth', 'Targeted Audio Channel',
                             'Targeted Audio Sample Rate', 'Targeted Bit Rate', 'Targeted Codec',
                             'Targeted Playback Format', 'User’s Audio Quality','User’s Playback Format']
        df = df.drop(columns_to_remove, axis = 1)

        #add / rename columns useful for calculating song labels
        print(df.head())
        print(df.columns)
        print(df.shape[0])

        #create a new df representing total playtime of each track, the raw feature used for labels

        #iterr through rows
        '''
        music_data = {}
        for i,row in self.df.iterrows():
            uID = f'{row["Artist Name"]}, {row["Content Name"]}'
            if uID in music_data:
                music_data[uID] += 1
            else:
                music_data[uID] = 1
        print(len(music_data.keys()))
        sorted_data = {k: v for k, v in sorted(music_data.items(), key=lambda item: item[1])}
        print(sorted_data)
        '''


trialDataset = labelDataset('/Users/seancondon/content-based-music-recommender/apple_music_data.csv')