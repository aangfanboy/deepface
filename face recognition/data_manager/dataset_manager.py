import tensorflow as tf


class DataEngineTypical:
	def make_label_map(self):
		self.label_map = {}

		for i, class_name in enumerate(tf.io.gfile.listdir(self.main_path)):
			self.label_map[class_name] = i

		self.reverse_label_map = {v: k for k, v in self.label_map.items()}

	def path_yielder(self):
		for class_name in tf.io.gfile.listdir(self.main_path):
			if not "tfrecords" in class_name:
				for path_only in tf.io.gfile.listdir(self.main_path + class_name):
					yield (self.main_path + class_name + "/" + path_only, self.label_map[class_name])

	def image_loader(self, image):
		image = tf.io.read_file(image)
		image = tf.io.decode_jpeg(image, channels=3)
		image = tf.image.resize(image, (112, 112), method="nearest")
		image = tf.image.random_flip_left_right(image)

		return (tf.cast(image, tf.float32) - 127.5) / 128.

	def mapper(self, path, label):
		return (self.image_loader(path), label)

	def __init__(self, main_path: str, batch_size: int = 16, buffer_size: int = 10000, epochs: int = 1,
	             reshuffle_each_iteration: bool = False, test_batch=64,
	             map_to: bool = True):
		self.main_path = main_path.rstrip("/") + "/"
		self.make_label_map()

		self.dataset_test = None
		if test_batch > 0:
			reshuffle_each_iteration = False
			print(f"[*] reshuffle_each_iteration set to False to create a appropriate test set, this may cancelled if tf.data will fixed.")

		self.dataset = tf.data.Dataset.from_generator(self.path_yielder, (tf.string, tf.int64))
		if buffer_size > 0:
			self.dataset = self.dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)

		if map_to:
			self.dataset = self.dataset.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		self.dataset = self.dataset.batch(batch_size, drop_remainder=True)

		if test_batch > 0:
			self.dataset_test = self.dataset.take(int(test_batch))
			self.dataset = self.dataset.skip(int(test_batch))

		self.dataset = self.dataset.repeat(epochs)


class DataEngineTFRecord:
	def image_loader(self, image_raw):
		image = tf.image.decode_jpeg(image_raw, channels=3)
		image = tf.image.resize(image, (112, 112), method="nearest")
		image = tf.image.random_flip_left_right(image)

		return (tf.cast(image, tf.float32) - 127.5) / 128.

	def mapper(self, tfrecord_data):
		features = {'image_raw': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.int64)}
		features = tf.io.parse_single_example(tfrecord_data, features)

		return self.image_loader(features['image_raw']), tf.cast(features['label'], tf.int64)

	def __init__(self, tf_record_path: str, batch_size: int = 16, epochs: int = 10, buffer_size: int = 50000,
	             reshuffle_each_iteration: bool = True,
	             test_batch=64, map_to: bool = True):
		self.dataset_test = None
		if test_batch > 0:
			reshuffle_each_iteration = False
			print(
				f"[*] reshuffle_each_iteration set to False to create a appropriate test set, this may cancelled if tf.data will fixed.")
		self.tf_record_path = tf_record_path

		self.dataset = tf.data.TFRecordDataset(self.tf_record_path)
		if buffer_size > 0:
			self.dataset = self.dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)

		if map_to:
			self.dataset = self.dataset.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		self.dataset = self.dataset.batch(batch_size, drop_remainder=True)

		if test_batch > 0:
			self.dataset_test = self.dataset.take(int(test_batch))
			self.dataset = self.dataset.skip(int(test_batch))

		self.dataset = self.dataset.repeat(epochs)


if __name__ == '__main__':
	print("go check README.md")
