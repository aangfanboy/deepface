# liyana - face recognition

This is a sub-field of liyana face analysis project. Under this title, i intend to reach state-of-art face recognition accuracies and use my methods on real life applications.



![](https://i.ibb.co/26ZRX8q/sv2.png)



## [Classifier Trainer](train_classifier.py)

### General

Model can be trained with both _ArcFace_ head and _Softmax_ head. Parameters supported with comments in Python file. 



## [Test With LFW](test_with_lfw.py)

### General

Get evaluation scores for LFW, AgeDB and CFP. 



## Model Informations



| Model                                                        | Architecture                                                 | Epochs | LFW Acc | AgeDB Acc | CFP Acc |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------ | ------- | --------- | ------- |
| [A](https://drive.google.com/open?id=1gUuir8ul-_RFCtnavUwSWSeqy9BVN9wF) | [InceptionResNetV1](model_scripts/inception_resnet_v1.py)    | 9      | %99.53  | %95.11    | %93.97  |
| [B](https://drive.google.com/open?id=1bTV279ZIs6p7kdARqyrqU4Vujtf5_cCF) | [ResNet50V2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2) | 11     | %99.51  | %94.53    | %93.60  |
| [C](https://drive.google.com/open?id=1qUtjhrYDMRqc2lRPNBV1ftkbyIPmTJYp) | [L_Resnet50_E_IR](model_scripts/LResNetIR.py)                | 7      | %99.70  | %96.75    | %97.34  |



PS: I train models on Google Colab



## To Do



- [x] Train with ResNet50V2

- [x] Train with ResNet101V2(Results are even worse than ResNet50V2 Model, not gonna share this one)

- [x] Train with L_Resnet50_E_IR(I could try to train more epochs but i will focus on Resnet100 for now)

- [ ] Train with L_Resnet100_E_IR

- [ ] Train with VarGFaceNet

- [ ] Re-train with lower weight decay 

- [ ] Train with Dataset V4

  

## How to train?

### Prepare Dataset

First download data to"dataset" folder, then use [this](/face_recognition/data_manager/turn_idx2tfrecord.py) script to turn it into tfrecord.



### Training

Run [this](/face_recognition/train_classifier.py) script. Parameters supported in Python file. Script will test on LFW at every 10k step.

PS: Default model architecture set to [InceptionResNetV1](model_scripts/inception_resnet_v1.py), check [this](/face_recognition/model_scripts/main_model_architect.py) file for other architectures such ResNet.



## Face Recognition Basic App

Go to [script](apps/photo_app/main.py). Parameters supported with comments in Python file. 



![match](/images-and-figures/match.png)

*color is green because faces belong to same person*



![no match](/images-and-figures/no_match.png)

*color is red because faces belong to different persons*

