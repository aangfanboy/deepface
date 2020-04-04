# All Data Management

[dataset_manager.py](dataset_manager.py)

###   		General

   - image data type: *float32*

   - label data type: *int64*

   - random left-right flip: _True_

   - normalization technique: _(x - 127.5) / 128.0_

### Classes

- _MainEngineClass_
- _DataEngineTypical_
- _DataEngineTFRecord_

### TODO

* reshuffle_each_iteration set to be False in order to create a  appropriate test set, fix it

* make a better _get_triplet_examples_from_batch_ in order to better triplet selection

* fix the _test_batch_ and make it a lambda function so it could change with batch size

  



[turn_idx2tfrecord.py](turn_idx2tfrecord.py)

> This script modified for TensorFlow >= 2.0.0 from [here](https://github.com/auroua/InsightFace_TF/blob/master/data/mx2tfrecords.py).  

### TODO

* find a way to do it without mxnet