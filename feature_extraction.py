#imports
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import json

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
        print(f'Ingesting data at {self.path}...')

        #load .csv into pandas dataframe
        all_plays = pd.read_csv(self.path)

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
        all_plays = all_plays.drop(columns_to_remove, axis = 1)

        #add / rename columns useful for calculating song labels
        all_plays = all_plays.rename(columns={'Content Name':'Title', 'Artist Name':'Artist'})
        #calculate the fraction of song listened to for each play
        all_plays['Play Datetime'] = pd.to_datetime(all_plays['Event Start Timestamp'])
        tmp = np.divide(all_plays['Play Duration Milliseconds'],all_plays['Media Duration In Milliseconds'])
        all_plays['Play Fraction'] = np.where(tmp > 1, 1, tmp)
        all_plays['Play Fraction'] = np.abs(all_plays['Play Fraction'])
        all_plays = all_plays.drop(['End Position In Milliseconds','Event End Timestamp', 'Event Received Timestamp',
                      'Milliseconds Since Play', 'Play Duration Milliseconds','Event Start Timestamp',
                      'Event Type','Start Position In Milliseconds', 'UTC Offset In Seconds'],axis=1)


        #merge all play data by same song and artist
        print('Merging play data by same song+artist...')
        self.features = self.merge_by_artist_title(all_plays)
        # normalize the data in total_time column by calling normalize_labels method
        print('Normalizing labels...')
        normalized_labels = self.normalize_labels()
        self.features['normalized_label'] = normalized_labels
        #remove rows with nan values
        self.features = self.features.dropna()
        #resolve album names
        print('resolving album names...')
        self.resolve_albums('./song_to_album.json')
        print('Song labels done loading / processing!\n\n')


    def merge_by_artist_title(self, all_plays):
        '''
        Takes the self.all_plays dataframe created during init, which contains all plays, and merges plays
        by uID = 'Artist + Title'. Essentially converts the dataframe of all plays into a dataframe of number
        of plays by song.

        :param all_plays: Pandas dataframe of all songs played, created during init.

        :return: None, adds self.song_plays
        '''

        #use a dictionary to keep track of number plays, total time played, most recent play
        n_plays = {}
        total_time = {}
        recent_play = {}

        for i, row in all_plays.iterrows():
            #extract relevant data from each row in all_plays
            uID = f'{row["Artist"]}\t{row["Title"]}'
            playtime = row['Play Fraction'] * row['Media Duration In Milliseconds']
            if np.isnan(playtime): playtime = 0
            datetime = row['Play Datetime']

            #update total plays, total time, and most recent play
            if uID in n_plays:
                n_plays[uID] += 1
                total_time[uID] += playtime
                if str(recent_play[uID]) == 'NaT':
                    recent_play[uID] = datetime
                elif datetime > recent_play[uID] and str(datetime) != 'NaT':
                    recent_play[uID] = datetime
            else:
                n_plays[uID] = 1
                total_time[uID] = playtime
                recent_play[uID] = datetime


        #create a new dataframe with the plays merged by Artist + Title
        data_rows = []
        for key in n_plays.keys():
            artist, title = key.split('\t')[0], key.split('\t')[1]
            row_i = {'artist':artist, 'title':title, 'n_plays':n_plays[key],'total_time':total_time[key],'most_recent':recent_play[key],'album':'None'}
            if artist != 'nan' and title != 'nan':
                data_rows.append(row_i)
        merged_df = pd.DataFrame(data_rows)
        return merged_df

    def resolve_albums(self, album_dat_path):
        '''
        Optional method that takes the path to an album data csv and resolves all artist_song pairs in self.features
        to their corresponding album. Not very useful for training a content based recommender, but fun to rank
        all albums for least listens to most listens.

        :param album_dat_path: filepath to .json file resolving song names to albums
        :return: None, sets 'album' column in self.features
        '''

        #read album_dat_path
        album_data = {}
        with open(album_dat_path, 'r') as f:
            song_list = json.load(f)

        #loop through all songs, saving their album into album_data dict
        for song in song_list:
            try:
                title, artist, album = song['Title'], song['Artist'], song['Album']
                uID = f'{artist},{title}'
                album_data[uID] = album
            except: pass

        #loop through features, resolving songs to albums
        numResolved = 0
        for i,row in self.features.iterrows():
            row_uID = f'{row["artist"]},{row["title"]}'
            if row_uID in album_data:
                row['album'] = album_data[row_uID]
                numResolved += 1
        print(f'Resolved album for {numResolved} of {i} songs')

    def normalize_labels(self):
        '''
        Takes self.features and converts the raw listening times for each songs into a normalized feature for
        use in a neural network.

        :return: y, array containing the normalized total_time values for each song
        '''

        #grab the total_time column from raw data
        y = self.features['total_time']

        #filter out nan and zero values
        y = y[~np.isnan(y)]
        y = y[y > 0]

        #convert to log scale, transform to zero mean, and put through sigmoid activation
        y = np.log(y)
        mean = np.mean(y)
        y = np.subtract(y, np.mean(y))
        y = 1 / (1 + np.exp(np.multiply(y, -1)))
        return y



class Song():

    def __init__(self,mp3_path):
        pass


trialDataset = labelDataset('/Users/seancondon/content-based-music-recommender/apple_music_data.csv')