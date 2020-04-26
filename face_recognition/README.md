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



| Model                                                        | ArcFace | Architecture                                              | Epochs | LFW Acc | AgeDB Acc | CFP Acc |
| ------------------------------------------------------------ | ------- | --------------------------------------------------------- | ------ | ------- | --------- | ------- |
| [A](https://drive.google.com/open?id=1V8rz_gabP2qnPrGJYwWTJX4U6vw0pxE-) | True    | [InceptionResNetV1](model_scripts/inception_resnet_v1.py) | 6      | %99.53  | %95.11    | %93.97  |



## To Do



- [ ] Train with ResNet50

- [ ] Train with ResNet101

- [ ] Train with VarGFaceNet

- [ ] Re-train with lower weight decay 

- [ ] Train with Dataset V4

  