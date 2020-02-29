# Tidying Deep Saliency Prediction Architectures

This repository contains Pytorch Implementation of SimpleNet and MDNSal. 

## Abstract

## Architecture
SimpleNet Architecture
![](./extras/SimpleNet.png)

MDNSal Architecture
![](./extras/MDNSal.png)
## Testing
Clone this repository and download the pretrained weights of SimpleNet, for multiple encoders, trained on SALICON dataset from this [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/samyak_j_research_iiit_ac_in/Eddzj58KvrhFtb6XinOFkhMBn8uFapCnOM6Ia0K1jFJhqw).

Then just run the code using 
```bash
$ python3 test.py --val_img_dir path/to/test/images --results_dir path/to/results --model_val_path path/to/saved/models
```
This will generate saliency maps for all images in the images directory and dump these maps into results directory

## Training
For training the model from scratch, download the pretrained weights of PNASNet from [here](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/samyak_j_research_iiit_ac_in/ERpsc82shFJNhtn-xfRr69ABCHtJNUlSHkSc99srArDtQQ?e=VIabCg) and place these weights in the PNAS/ folder. Run the following command to train 

```bash
$ python3 train.py --dataset_dir path/to/dataset 
```
The dataset directory structure should be 
```
└── Dataset  
    ├── fixations  
    │   ├── train  
    │   └── val  
    ├── images  
    │   ├── train  
    │   └── val  
    ├── maps  
        ├── train  
        └── val  
```

## Experiments

## Visual Results

## Contact 
If any question, please contact samyak.j@research.iiit.ac.in , or use public issues section of this repository

## License 
This code is distributed under MIT LICENSE.