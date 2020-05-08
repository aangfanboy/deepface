import tensorflow as tf


def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def fashion_mnist_main():
	writer = tf.io.TFRecordWriter("fashion_mnist_data.tfrecords")

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

	x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, axis=-1))
	x_test = tf.image.grayscale_to_rgb(tf.expand_dims(x_test, axis=-1))

	i = 0
	for x, y in zip(x_train, y_train):
		x = tf.image.resize(x, (112, 112), method="nearest")
		x = bytes(tf.io.encode_jpeg(x).numpy())

		feature = {
			'label': _int64_feature(y),
			'image_raw': _bytes_feature(x),
		}

		tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(tf_example.SerializeToString())

		if i % 10000 == 0:
			print(f"{i} images processed")

		i += 1

	for x, y in zip(x_test, y_test):
		x = tf.image.resize(x, (112, 112), method="nearest")
		x = bytes(tf.io.encode_jpeg(x).numpy())

		feature = {
			'label': _int64_feature(y),
			'image_raw': _bytes_feature(x),
		}

		tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(tf_example.SerializeToString())

		if i % 10000 == 0:
			print(f"{i} images processed")

		i += 1

	print(f"Done! TFRecord created in \"fashion_mnist_data.tfrecords\" with {i} images")


def mnist_main():
	writer = tf.io.TFRecordWriter("mnist_data.tfrecords")

	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, axis=-1))
	x_test = tf.image.grayscale_to_rgb(tf.expand_dims(x_test, axis=-1))

	i = 0
	for x, y in zip(x_train, y_train):
		x = tf.image.resize(x, (112, 112), method="nearest")
		x = bytes(tf.io.encode_jpeg(x).numpy())

		feature = {
			'label': _int64_feature(y),
			'image_raw': _bytes_feature(x),
		}

		tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(tf_example.SerializeToString())

		if i % 10000 == 0:
			print(f"{i} images processed")

		i += 1

	for x, y in zip(x_test, y_test):
		x = tf.image.resize(x, (112, 112), method="nearest")
		x = bytes(tf.io.encode_jpeg(x).numpy())

		feature = {
			'label': _int64_feature(y),
			'image_raw': _bytes_feature(x),
		}

		tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(tf_example.SerializeToString())

		if i % 10000 == 0:
			print(f"{i} images processed")

		i += 1

	print(f"Done! TFRecord created in \"mnist_data.tfrecords\" with {i} images")


if __name__ == '__main__':
	fashion_mnist_main()
	mnist_main()
