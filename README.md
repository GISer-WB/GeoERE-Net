## Understanding geological reports based on knowledge graphs using a deep learning approach



## Description

This paper proposes a  method for understanding  geological report content using deep learning . GeoERE-Net is a triplet extraction network proposed for text knowledge mining in the geological field. This paper has been accepted by 《Computers & Geosciences》. If you refer to this repository, please cite it. The link of the paper: [GeoERE-Net](https://www.sciencedirect.com/science/article/pii/S0098300422001789)

## Requirements

- pytorch 1.7
- pytorch_pretrained_bert
- numpy
- einops
- tqdm

## Dataset

*The format of the dataset is Chinese text, and the following content is the translation content. The format of each statement is as follows, which are stored in the . json file. The dataset is divided into training set, validation set and test set.*

```
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
```

## Usage

1. ```
   Training
       python train.py
   ```

   

2. ```
   Inference
       python test.py
   ```

## Citation

Wang B, Wu L, Xie Z, et al. Understanding geological reports based on knowledge graphs using a deep learning approach[J]. Computers & Geosciences, 2022: 105229.
