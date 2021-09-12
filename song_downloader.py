#imports
import urllib.request
import re
import youtube_dl
import subprocess
import glob

#local imports
from feature_extraction import *


def song_download_helper():
    '''
    I'll be manually downloading songs to train the music recommender, so this is a helper function to aid in that
    downloading processes. It will look through all the currently downloaded songs in ./song_data and parse
    ./apple_music_data.csv to find all the songs that still need to be downloaded.

    :return: None
    '''

    #load dataframe with all input features
    label_dataframe = labelDataset('/Users/seancondon/content-based-music-recommender/apple_music_data.csv')

    #iter through dataframe, checking if each song is in ./song_data
    to_download = []
    downloaded = [f.strip('.mp3') for f in os.listdir('./song_data') if f != '.DS_Store']

    for i,row in label_dataframe.features.iterrows():
        uID = f'{row["artist"]}-{row["title"]}'
        if uID not in downloaded:
            to_download.append(uID)

    #print the list of songs that still need to be downloaded
    to_download = sorted(to_download)
    return to_download


def uID_to_search_result(uID):
    '''
    Takes the uID of a song and returns the youtube html of the first video search result when that uID
    is put into a youtube search. Used to quickly download a large dataset of songs from youtube.

    :param uID: string of artist+title representing the unique identifier for a song
    :return: first_result, string of url for first video result upon searching for uID
    '''
    #replace bad characters, get youtube search query
    searchID = uID.replace(' ', '+')
    searchID = searchID.replace("'","%27")
    search_q = f'https://www.youtube.com/results?search_query={searchID}'

    #use urllib to query youtube with search_q
    html_out = urllib.request.urlopen(search_q)

    #parse html_out to get url's of top video results, return top result
    video_ids = re.findall(r"watch\?v=(\S{11})", html_out.read().decode())
    youtube_stem = ''
    first_result = f'https://www.youtube.com/watch?v={video_ids[0]}'
    return first_result

def download_url(url, outputname):
    '''
    Downloads .mp3 associated with a url (should be a link to a youtube video).

    :param url: string, the url to download .mp3 from
    :return: TODO
    '''
    print(url)
    #get the video info from the url
    video_info = youtube_dl.YoutubeDL().extract_info(url = url, download = False)

    #set the download options and then download
    options = { 'format': 'bestaudio/best', 'keepvideo': False,
                'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192' }]}
    with youtube_dl.YoutubeDL(options) as ydl:
        ydl.download([video_info['webpage_url']])

    #rename that .mp3 to outputname
    file_to_move = glob.glob('*.mp3')
    subprocess.run(['mv', file_to_move[0], outputname])

def uID_to_mp3(uID):
    '''
    Handles the entire song download process from uID through downloading the mp3. Handles exceptions
    when bad strings are passed to be downloaded, and moves mp3 files to where they should be in
    the working directory

    :param uID: string of artist+title representing the unique identifier for a song
    :return: None
    '''

    #get the correct url to download
    url_to_download = uID_to_search_result(uID)

    #download that url
    outname = f'./song_data/{uID}.mp3'
    download_url(url_to_download, outname)


if __name__ == '__main__':

    #get a list of uID's that need to be downloaded
    to_dowload = song_download_helper()
    print(f'{len(to_dowload)} songs still need to be downloaded!')

    #download the first N of these
    N = 2000
    d_queue = to_dowload[:N]
    for uID in d_queue:
        print(f'downloading {uID}...')
        try:
            uID_to_mp3(uID)
        except UnicodeEncodeError:
            print('this song probably has some super weird characters in it')

        except:
            print('this song is probably age-gated')

        print('\n')
    #print(to_dowload)

