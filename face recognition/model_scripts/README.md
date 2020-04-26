# Model Training Scripts

## [ArcFaceLayer](ArcFaceLayer.py)

### General

ArcFace as layer for Keras model.

### Thanks To

https://github.com/peteryuX/arcface-tf2/tree/master/modules



## [Inception ResNet V1](inception_resnet_v1.py)

### General

Inception ResNet V1 model architecture for Keras model.

### Thanks To:

https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py



## [Main Model Architect](main_model_architect.py)

### General

Functions for creating and training Keras models



### Valid Model Structres

* ResNet50
* ResNet101
* EfficientNetFamily || [0, 7]
* Xception
* InceptionResNetV1



### Features

* Train and Test step functions  _with/without_ regression loss
* SGD optimizer
* Sparse Categorical Crossentropy loss
* Triplet Loss
* _ArcFace_ head and _Normal_ head are both available



### TODO

* Make it available for triplet loss training from scratch



> Imagenet weights are **not** available for _Inception ResNet V1_



## [TensorBoard Helper](tensorboard_helper.py)

### General

Class to create graphs through TensorBoard while training



## Attention

> Referred ArcFace paper can be found in [here](../papers/ArcFace.pdf)