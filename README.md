# M-AI-AN
## Glyph-Translator
Currently WIP

## Information
M-AI-AN is an AI system which can be used to identify Mayan glyphs. 
It is trained on glyphs from The Mayan Epigraphic Database1. 


## Rationale

M-AI-AN is a Combination of CNN and RNN neural networks, chained together, to provide an English translation of 

## Methodology:

M-AI-AN is to be trained to do the following:

1. Identify the sequence of glyphs in a writing block [CNN]
2. Identify each glyph with affixes within the block [CNN]
3. Translate the resulting text into English [RNN]

# PROGRESS

## Main Glyph Identifier:

1. Data Formatter [COMPLETED]
- Using Mayan Epigraphic Database to create test data for the convolutional network
    - Lack of large scale “real world” data
     - Work around by using graphical representation of glyphs with edits 
    - Performs mild translations and rotations to provide input data relevant to input photographs
    - Creates a specified number of training samples based on specific glyph indicies.
2. M-AI-AN Classifier [COMPLETED]
    - Train model on training and test data
     - Split data into subgroups for training and testing
     - Train for finite number of epochs
    - Test multiple convolutional network designs
     - Tested 16 separate designs
     - Kept accuracy logs for each design with log of changes made and relevant calculations for the design of the next
    - Implemented testing data image identification
    - Implemented custom image data identification
     - Uses Pillow library to modify image to similar black and white “graphic” style as training data
     - Feeds into network to produce output glyph index


## Future Progress:
    • Usage of multi-head convolutional model to decypher entire Mayan glyphs:
        ◦ Glyphs can have up to six different parts
        ◦ At least six different heads to be created
            ▪ Potential to double up heads for most common patterns (ie. 3 seperate affix/main combinations)

