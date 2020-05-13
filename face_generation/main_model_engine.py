import sys
sys.path.append("../")
import tensorflow as tf

from face_recognition.data_manager.dataset_manager import DataEngineTFRecord as DET


class MainEngine:
	def __init__(self):
		pass


if __name__ == '__main__':
	TDOM = DET(
		"../datasets/fashion_mnist_data.tfrecords",
		batch_size=16,
		epochs=-1,
		buffer_size=70000,
		reshuffle_each_iteration=True,
		test_batch=0,
		map_to=True
	)
