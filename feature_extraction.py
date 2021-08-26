#imports
import numpy as np
import os
import pandas as pd
import datetime as DT

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

        self.raw_data = self.merge_by_artist_title(all_plays)
        print(self.raw_data.columns)
        sorted_data = self.raw_data.sort_values('most_recent', ascending=False)
        for i,row in sorted_data.iterrows():
            print(f'{row["artist"]}, {row["title"]}')
            print(f'plays: {row["n_plays"]}, time: {row["total_time"]}')
            print(f'most recent: {row["most_recent"]}')
            print('\n')


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



trialDataset = labelDataset('/Users/seancondon/content-based-music-recommender/apple_music_data.csv')