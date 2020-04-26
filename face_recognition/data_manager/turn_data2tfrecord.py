import tensorflow as tf

from dataset_manager import DataEngineTypical as DET


class Engine:
	def _bytes_feature(self, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def _int64_feature(self, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def __init__(self, data_engine: DET, path: str = "../../datasets/tran.tfrecords"):
		self.data_engine = data_engine
		self.path = path

		self.writer = tf.io.TFRecordWriter(self.path)

	def __call__(self):
		print(f"processing to {self.path}")
		for i, (x, y) in enumerate(self.data_engine.dataset):
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

	In this example, script will read images from '../../datasets/105_classes_pins_dataset_aligned'
	and save them to TFRecord file located in '../../datasets/105_classes_pins_dataset_aligned/tran.tfrecords'
	"""

	data_engine_for_images = DET(
		main_path="../../datasets/105_classes_pins_dataset_aligned",
		batch_size=1,
		buffer_size=0,
		epochs=1,
		test_batch=0,
		map_to=False
	)

	engine_for_writing_tfrecord = Engine(
		data_engine_for_images,
		"../../datasets/105_classes_pins_dataset_aligned/tran.tfrecords"
	)

	engine_for_writing_tfrecord()
