# liyana: face analysis project

[<img src="https://cdn2.iconfinder.com/data/icons/minimalism/128/twitter.png" alt="Twitter" title="follow my twitter to hear more about AI"/>](https://twitter.com/tyburakk)                            [<img src="https://i7.uihere.com/icons/381/52/851/logo-media-social-twitch-icon-ac3b3fceb6a1933f5769a3bb147b2963.png" alt="Twitch" title="i sometimes make live coding, follow to get notificated when i go live"/>](https://www.twitch.tv/tyburak)                              [<img src="https://icons.iconarchive.com/icons/dakirby309/simply-styled/128/YouTube-icon.png" alt="YouTube" title="i will make instructive videos about both liyana and AI, stay tuned"/>](https://www.youtube.com/channel/UC_ruNwRztPY_P4aUxbIFuOQ)



__For supporters, My Bitcoin/coin.space: 1LUFWnzrGVLdsZ7gnfee87iX6QqSn24Tvr__

__Thank you for donations, i am grateful.__



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/github/aangfanboy/deepface/blob/master/Face_Recognition_Model_Training_ArcFace.ipynb)



## Features

- Facial recognition with database
- Facial recognition on webcam
- Facial recognition on video
- Displaying 2D space which database lies on
- Age, sex and ethnicity detection
- DeepFake detection



## Features that expected to come in next versions

- Online database that every user can add face to common pool
- Face re-generation with extracted features
- Advanced video analysis



![liyana 1.1.0](images-and-figures/liyana_1-2-0.png)

_User gives the face at the right side of page to program, program extract features through machine learning model, features compares with those already saved in database and result prints on the screen. Face on the left side of page is Emma Watson's face in the database. Values such age, sex and ethnicity are loaded from database too, you can re-analyze those values by clicking **re-analyze with models**_



# Usage

## Python Side

- Install libraries with ``` pip install -r requirements.txt ``` 
- Download a model from [face recognition folder](/face_recognition/), extract it, copy *arcface_final.h5* to *main_app/python_server/arcface_final.h5*
- Download Age-Sex-Ethnicity classification models, details and last models can be found [here](age_sex_ethnicity_detection/)
- Download DeepFake detection models, details and last model can be found [here](deepfake_detection)
- Run *main_app/python_server/server.py*
- If you don't need a GUI, run *main_app/python_server/client.py*. Commands are listed in [here](/main_app/python_server/server.py#L11-L57).

I highly recommend to use a GPU, you can follow [those](https://www.tensorflow.org/install/gpu) steps if you have one.



## Electron Side

- Install electron with npm. Check [this page](https://www.electronjs.org/docs/tutorial/installation) for help.
- You may need to install dependencies for electron, all can be found in [here](/main_app/electron_scripts/package.json)
- Go */main_app/electron_scripts* and run ```electron .```  while python server is running. 



I will add a video to YouTube to explain how you can install and how it works, stay tuned.



I will make a version of app that works with TensorFlow JS so it can be work with just electron, stay tuned.