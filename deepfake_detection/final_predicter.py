import tensorflow as tf


class Tester:
	def __init__(self, model_deepfake_path: str):
		self.model_deepfake = tf.keras.models.load_model(model_deepfake_path)

	def predict_deepfake(self, outputs, **kwargs):
		pred = self.model_deepfake(outputs, training=False)
		pred = tf.nn.softmax(pred)[:, 1]

		return pred
