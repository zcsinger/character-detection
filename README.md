# Character Detection using Machine Learning in Super Smash Bros. Ultimate

This project allows for the automatic detection of characters and stage used in Super Smash Bros. Ultimate. The detector has been trained using a Convolutional Neural Network. 

The immediate applications of this project include a streamlined process for uploading individual matches from tournament streams, which have hundreds of matches continuously streamed with no break. An identification of the characters and stages would allow for easy labeling and uploading to archives on YouTube or VODS.co.

## Gathering Data

Due to the prevalence of streaming, several Twitch clips from prominent channels such as twitch.tv/vgbootcamp have been gathered and turned into staggered images for training. 

## Usage

Example clips have been placed in the clips directory. To generate clips one can use `clips/clips.py` and specify the frame delay and resolution scaling. 

It is suggested to create a `data` subdirectory in `clips` to store the clips as they can take up a lot of space visually.

Training and testing sets can then be specified in CSV files in locations such as `clips/data/train_data.csv` and `clips/data/test_data.csv`. 

An example image before cropping and downsampling:

![example data](https://raw.githubusercontent.com/zcsinger/character-detection/master/clips/example_data/lucina_palutena_kl_1_frame0.jpg)

## Training

With the usage format as detailed above, one can train with 
```
python train.py
```

## Current Scope

The initial stage of this project is for the identification of tournament legal stages and the current best characters in the game. Future updates will incorporate all legal stages and characters (which requires classification of pairs of over 70 characters). 
