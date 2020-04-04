import tensorflow as tf


class ArcFaceLayer(tf.keras.layers.Layer):
	def __init__(self, num_classes, arc_m=0.5, arc_s=64, **kwargs):  # has been set to it's defaults according to arcface paper
		super(ArcFaceLayer, self).__init__(**kwargs)

		self.num_classes = num_classes
		self.arc_m = arc_m
		self.arc_s = arc_s

		self.w = self.add_weight("kernel", shape=[512, self.num_classes], initializer='uniform', trainable=True)

		self.cos_m = tf.math.cos(self.arc_m)
		self.sin_m = tf.math.sin(self.arc_m)
		self.th = -self.cos_m
		self.mm = tf.multiply(self.sin_m, self.arc_m)

	def __call__(self, features, labels):
		features_norm = tf.nn.l2_normalize(features, axis=1)
		weights_norm = tf.nn.l2_normalize(self.w, axis=0)

		cos_t = tf.matmul(features_norm, weights_norm)
		sin_t = tf.sqrt(1. - (cos_t ** 2))

		cos_mt = tf.where(cos_t > self.th, tf.subtract(cos_t*self.cos_m, sin_t*self.sin_m), cos_t - self.mm)
		mask = tf.one_hot(tf.cast(labels, tf.int64), self.num_classes)

		logits = tf.identity(tf.where(mask == 1., cos_mt, cos_t) * self.arc_s)

		return logits


if __name__ == '__main__':
	print("go check README.md")