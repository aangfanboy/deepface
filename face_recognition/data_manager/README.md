# All Data Management

## [Dataset_Manager](dataset_manager.py)

###   		General

   - image data type: *float32*

   - label data type: *int64*

   - random left-right flip: _True_

   - normalization technique: _(x - 127.5) / 128.0_

### Classes

- _DataEngineTypical_
- _DataEngineTFRecord_

### To Do

- [ ] reshuffle_each_iteration set to be False in order to create a  appropriate test set, fix it
- [ ] fix the _test_batch_ and make it a lambda function so it could change with batch size
- [ ] find a way to do it without mxnet

