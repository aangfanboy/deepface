import tensorflow as tf

from dataset_manager import DataEngineTypical as DET


class Engine:
	def _bytes_feature(self, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def _int64_feature(self, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def __init__(self, data_engine_dataset: tf.data.Dataset, path: str = "../../datasets/tran.tfrecords"):
		self.dataset = data_engine_dataset
		self.path = path

		self.writer = tf.io.TFRecordWriter(self.path)

	def __call__(self):
		print(f"processing to {self.path}")
		for i, (x, y) in enumerate(self.dataset):
			feature = {
				'label': self._int64_feature(y),
				'image_raw': self._bytes_feature(bytes(tf.io.read_file(x[0]).numpy())),
			}

			tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
			self.writer.write(tf_example.SerializeToString())

			if i % 10000 == 0:
				print(f"{i} images processed")

		print(f"Done! TFRecord created in {self.path} with {i} images")


if __name__ == '__main__':
	"""
	Examples usage can be found below

	In this example, script will read images from '../../datasets/dataset_V4'
	and save them to TFRecord file located in '../../datasets/dataset_V4/tran_(test/train).tfrecords'
	"""

	data_engine_for_images = DET(
		main_path="../../datasets/dataset_V4/",
		batch_size=1,
		buffer_size=0,
		epochs=1,
		map_to=False
	)

	engine_for_writing_tfrecord_train = Engine(
		data_engine_for_images.dataset,
		"../../datasets/dataset_V4/tran_train.tfrecords"
	)
	engine_for_writing_tfrecord_train()

	engine_for_writing_tfrecord_test = Engine(
		data_engine_for_images.dataset_test,
		"../../datasets/dataset_V4/tran_test.tfrecords"
	)
	engine_for_writing_tfrecord_test()
