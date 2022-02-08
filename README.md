## Understanding geological reports based on knowledge graphs using a deep learning approach



## Description

This paper proposes a  method for understanding  geological report content using deep learning . GeoERE-Net is a triplet extraction network proposed for text knowledge mining in the geological field. We are currently submitting the paper to the journal 《Computers & Geosciences》. If you refer to this repository, please cite it.

## Requirements

- pytorch 1.7
- pytorch_pretrained_bert
- numpy
- einops
- tqdm

## Dateset

*The format of the dataset is Chinese text, and the following content is the translation content. The format of each statement is as follows, which are stored in the . json file. The dataset is divided into training set, validation set and test set.*

{
      "text": "The volcanic rocks are mainly exposed in the Tuojiqubuqu area.",
      "triple_list": [
       [
             "The volcanic rocks",
             "Exposed in",
             "The Tuojiqubuqu area"
      ] 
   ]
 }

## Usage

1. ```
   Training
       python train.py
   ```

   

2. ```
   Inference
       python test.py
   ```

