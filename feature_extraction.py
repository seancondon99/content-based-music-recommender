#imports
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import pydub
import scipy.signal
import librosa

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
                self.features.at[i,'album'] = album_data[row_uID]
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

    GLOBAL_SAMPLE_RATE = 20500 #allows frequencies of up to 10,250 Hz, which should be fine for this task
    SGRAM_FRAMES = 599

    def __init__(self,mp3_path):

        #read timeseries from mp3 file
        self.mp3_path = mp3_path
        self.mp3_to_timeseries()

        #take the mel spectrogram of the timeseries
        self.timeseries_to_mel_spectrogram()

        #chop full spectrogram into chunks of SGRAM_FRAMES number of frames
        n_bins, n_frames = self.mel_sgram.shape
        init_frame = int(0.1 * n_frames)
        end_frame = int(0.9 * n_frames)
        n_chunks = (end_frame - init_frame) // self.SGRAM_FRAMES
        self.chunks = []
        chunk_start = init_frame
        for i in range(n_chunks):
            chunk_range = (chunk_start, chunk_start+ self.SGRAM_FRAMES)
            chunk_start += self.SGRAM_FRAMES
            self.chunks.append(chunk_range)



    def mp3_to_timeseries(self):
        '''
        Takes the self.mp3_path of a song object and coverts that mp3 file into a time series that is usable for
        calculating mel-spectrograms. Also record the sampling frequency of the song.

        :return: None, set self.timeseries
        '''
        print(self.mp3_path)
        #open the audio and extract audio info using pydub
        audio = pydub.AudioSegment.from_mp3(self.mp3_path)
        info = pydub.utils.mediainfo(self.mp3_path)
        sample_rate = info['sample_rate']
        n_channels = info['channels']

        #convert audio timeseries to np array, reshape to num_channels
        y_arr = np.array(audio.get_array_of_samples())
        y_arr = y_arr.reshape(audio.channels, -1, order='F')
        #if stereo, average to convert to mono
        if audio.channels > 1:
            y_arr = np.mean(y_arr, axis = 0)

        #downsample (or upsample) so that all mp3's have the same sampling rate
        n_samples = len(y_arr) * (self.GLOBAL_SAMPLE_RATE / int(sample_rate))
        n_samples = int(n_samples)
        y = scipy.signal.resample(x = y_arr, num = n_samples)
        self.timeseries = y

    def timeseries_to_mel_spectrogram(self):
        '''
        Converts self.timeseries into a mel_spectrogram with the parameters defined below
        Uses librosa to take fourier transform and build mel spectrogram

        :return: None, adds mel_spectrogram to self.mel_sgram
        '''

        #declare mel_spectrogram hyperparameters
        n_frames = 599
        freq_bins = 128

        #first, take the short time fourier transform of the timeseries
        sgram = librosa.stft(self.timeseries, n_fft = 1024, hop_length = 512) #each sgram frame is ~50 ms

        #then shift frequency axis to mel-scale, and shift power axis to decibel scale
        sgram_mag, _ = librosa.magphase(sgram)
        mel_scale_sgram = librosa.feature.melspectrogram(S=sgram_mag, sr=self.GLOBAL_SAMPLE_RATE)
        mel_sgram = librosa.amplitude_to_db(mel_scale_sgram, ref=np.min)
        self.mel_sgram = mel_sgram


    def plot_timeseries(self):
        '''
        Helper function that uses matplotlib and librosa to plot all or part of a audio timeseries

        :return: None
        '''
        import librosa.display
        #only plot start_s -> end_s seconds
        start_s, end_s = 10, 20
        ts = self.timeseries[start_s*self.GLOBAL_SAMPLE_RATE : end_s*self.GLOBAL_SAMPLE_RATE]

        #set up a new figure and use librosa.display to plot waveform
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(ts, sr=self.GLOBAL_SAMPLE_RATE)
        plt.show()

    def plot_spectrogram(self):
        '''
        Helper function that uses matplotlib and librosa to plot all or part of a mel spectrogram.

        :return: None
        '''
        import librosa.display
        #set the start and end frame of display
        start_frame, end_frame = 1000, 1600
        #choose the chunk number to display
        chunk_num = 1
        start_index, end_index = self.chunks[chunk_num]
        mel_sgram = self.mel_sgram[:, start_index:end_index]
        print(mel_sgram.shape)

        #plot mel sgram using librosa and matplotlib
        librosa.display.specshow(mel_sgram, sr=self.GLOBAL_SAMPLE_RATE, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.show()

ex_song = Song("/Users/seancondon/content-based-music-recommender/song_data/3OH!3-Don't Trust Me.mp3")
print(ex_song.mel_sgram.shape)
print(ex_song.chunks)
ex_song.plot_spectrogram()



