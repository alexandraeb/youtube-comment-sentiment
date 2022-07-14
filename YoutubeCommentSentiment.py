# -*- coding: utf-8 -*-
"""
Description:
    Takes a youtube videoID inputted by the user and categorizes the comments using sentiment analysis
    Output is summary stats of comments written to the command line and a csv file of categorized comments

"""

import os
import demoji
import argparse
import pandas as pd
import numpy as np

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import detect
from langdetect import DetectorFactory
from textblob import TextBlob #library that contains NLP & text analysis functions


# Ensure DEVELOPER_KEY is set to registered API key value from https://cloud.google.com/console
DEVELOPER_KEY = ""
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
#This is max limit imposed by the youtube API commentThreads.list
results_cap = 100


def get_youtube_comment_threads(youtube, video_id):
    """
    Calls the youtube API commentThreads.list method to create list of text for comments of a video.
    Parameters
    ----------
    youtube : Str
        class indicating youtube API service, version and developer key
    video_id : Str
        string indicating ID of the video

    Returns
    -------
    comments_text : list of strings
        list of comments for the given video
    """
    results = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=results_cap
        ).execute()
    
    comments_text = []
    for item in results["items"]:
        comment = item["snippet"]["topLevelComment"]
        text = comment["snippet"]["textDisplay"]
        comments_text.append(text)
    
    return comments_text


def remove_emoji(text):
    """
    Uses the demoji library to find and remove emojis from a single comment

    Parameters
    ----------
    text : Str
        String to remove emojis from

    Returns
    -------
    demoji_text : Str
        String with emojis replaced with empty string
    """
    emoji_idxs = demoji.findall(text)
    demoji_text = text
    for emoji in emoji_idxs.keys():
        demoji_text = demoji_text.replace(emoji, '')
    return demoji_text


def remove_non_english(text):
    """
    Uses the langDetect library to detect the language of a comment. Sets non-english comments to empty string.    

    Parameters
    ----------
    text : Str
        String to determine whether english or not

    Returns
    -------
    str
        The unchanged english input string, or empty string for all other languages
    """
    DetectorFactory.seed = 0    
    try:
        language = detect(text)
        if language == 'en': return text
    except:
        # No features for lang_detect to use for identification, remove comment from analysis
        return ''


def clean_comments(comments):
    """
    Iterates over all comments and calls functions to clean comments for analysis

    Parameters
    ----------
    comments : list of strings
        list of comments of interest

    Returns
    -------
    comments : list of strings
        list of comments with unusable comments/words set to empty strings
    """
    for index, comment in enumerate(comments):
        # Remove emojis
        comments[index] = remove_emoji(comment)
        # Extract english comments
        comments[index] = remove_non_english(comments[index])   
    return comments



def get_sentiment(text):
    """
    Uses the Text Blob sentimant polarity function to get sentiment from text
    
    Parameters
    ----------
    text : Str
        String of text to get sentiment from

    Returns
    -------
    Result : Str
        String indicating Pos, Neg or Neutral sentiment based on sentiment polarity value
    """
    text_blob = TextBlob(text)
    sentiment = text_blob.sentiment.polarity
    if sentiment > 0:
        result = 'Positive'
    elif sentiment < 0:
        result = 'Negative'
    else:
        result = "Neutral"


def build_sentiment_df(comments):
    """
    Builds & returns pandas dataframe of sentiment for each comment.
    Empty comments are removed and Summary stats printed to user    

    Parameters
    ----------
    comments : list of strings
        list of strings containing comments of interest

    Returns
    -------
    df : pandas dataframe
        dataframe containing all usable comments for analysis & their associated sentiment
    """
    df = pd.DataFrame(comments)
    df.columns = ['Comment']
    # remove empty comments that have been cleaned
    df['Comment'].replace('', np.nan, inplace = True)
    count_nan = df['Comment'].isna().sum()
    df.dropna(subset=['Comment'], inplace = True)

    df['Sentiment'] = df['Comment'].apply(get_sentiment)
    print("Summary of sentiment in all Comments \n(%d comments removed from analysis): " %count_nan)
    print(df['Sentiment'].value_counts())
    return df


if __name__ == "__main__":
    # Ensure user inputs videoID value
    parser = argparse.ArgumentParser()
    parser.add_argument("--videoid", help="Required; ID for video for which comments will be found")
    parser.add_argument("--csv", choices=['y', 'n'], help="Optional; Y/N to export results as csv")
    args = parser.parse_args()
    
    if not args.videoid:
        exit("Please specify videoid using the --videoid= parameter.")
    video_id = args.videoid
    
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)
    
    # Try to grab data through youtube data API - catch exception if error occurs and notify user
    video_comments = []
    try:
        video_comments = get_youtube_comment_threads(youtube, video_id)
        print ("Comments successfully accessed.")
    except HttpError:
        print("An HTTP error occurred")
    
    # Remove comments that are unsuitable for analysis
    cleaned_comments = clean_comments(video_comments)
    result = build_sentiment_df(cleaned_comments)        
    # Save result dataframe to csv file
    if args.csv == 'y':
        result.to_csv(os.getcwd() + r'\SentimentComments_' + video_id + r'.csv')

    
    