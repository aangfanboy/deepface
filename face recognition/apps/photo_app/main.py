import sys
sys.path.append("../../../")
import tensorflow as tf

from face_detection import mtcnn_detector


class Engine:
	def __init__(self, model_path: str):
		self.model = tf.keras.models.load_model(model_path)


if __name__ == '__main__':
	pass
