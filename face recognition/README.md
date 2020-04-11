# liyana - face recognition

This is a sub-field of liyana face analysis project. Under this title, we intend to reach state-of-art face recognition accuracies and use our methods on real life applications.



![](https://i.ibb.co/26ZRX8q/sv2.png)



## [Classifier Trainer](train_classifier.py)

### General

Model can be trained with both _ArcFace_ head and _Softmax_ head. Parameters supported with comments in Python file. 



## [Triplet Loss Trainer](train_triplet.py)

### General

Model which trained with _ArcFace_ or _Softmax_, can be re-trained with triplet loss. Parameters supported with comments in Python file. 





## Model Informations



| Model                                                        | ArcFace | Softmax | Center Loss | Triplet Loss | Architecture                                      | Regularization | LR   |
| ------------------------------------------------------------ | ------- | ------- | ----------- | ------------ | ------------------------------------------------- | -------------- | ---- |
| [A](https://drive.google.com/open?id=1yU4upF0_-0aSKd9J3-EBW4vw16B5N32y) | False   | True    | False       | False        | [ResNet50](https://keras.io/applications/#resnet) | **l2**(5e-4)   |      |





## Model Results

| Model | Flip Loss |
| ----- | --------- |
|       |           |

