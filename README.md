# Project Title

Classification of Brain MR Images in parkinson desease

## Getting Started

This project consists of two phases:
1- skull stripping in MR images to extract brain
2- classification of parkinson deseas

### Dataset

PPMI dataset is used in this project:  

```
https://www.ppmi-info.org/access-data-specimens/download-data/
```

## Running the tests

First of all, all images have to be skull stripped. To do this, you have execute "skullstrip.py". All of the result of this part are saved in .npy format.

Finally for classification, you have to execute "mri.py". This piece of code will load the result of previous section and process them.
