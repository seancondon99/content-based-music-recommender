# content-based-music-recommender

This repo contains the code to train and analyze a content-based music recommendation algorithm that listens to 15 second snippets of songs and predicts how much you will enjoy them. The recommender model is a CNN implemented in PyTorch that is trained on your personal music listening history, and predicts a song likeability metric from 15 second mel spectrograms.

You can read more about this project on my personal wesbite at https://seancondon99.github.io/cbr.html.

## Description of Files

#### feature_extraction.py
Contains classes for music listening history csv data and song mp3 data, with methods to extract song likeability metrics from music listening history and mel spectrograms from mp3 data. Also has functions for viewing and debugging this data.

#### model.py 
The model architecture for the recommender CNN that takes song spectrograms and input and predicts song likeability as output. Implemented in PyTorch.

#### song_downloader.py
Examines the music listening history csv and song data directories to determine what songs still need to have their mp3 data downloaded. This file will then find mp3s for those missing songs on youtube, download them, and store them in the correct directory.

## How to Run
This part is still under developement, please check back later!

