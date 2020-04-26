# modified from https://github.com/auroua/InsightFace_TF/blob/master/losses/face_losses.py

import math
import tensorflow as tf


class ArcFaceLayer(tf.keras.layers.Layer):
	def __init__(self, num_classes, arc_m=0.5, arc_s=64., regularizer_l: float = 5e-4, **kwargs):  # has been set to it's defaults according to arcface paper
		super(ArcFaceLayer, self).__init__(**kwargs)
		self.num_classes = num_classes
		self.regularizer_l = regularizer_l
		self.arc_m = arc_m
		self.arc_s = arc_s

		self.cos_m = tf.identity(math.cos(self.arc_m))
		self.sin_m = tf.identity(math.sin(self.arc_m))
		self.th = tf.identity(math.cos(math.pi - self.arc_m))
		self.mm = tf.multiply(self.sin_m, self.arc_m)

	def build(self, input_shape):
		self.kernel = self.add_weight(name="kernel", shape=[512, self.num_classes], initializer=tf.keras.initializers.glorot_normal(),
		                              trainable=True, regularizer=tf.keras.regularizers.l2(self.regularizer_l))

		super(ArcFaceLayer, self).build(input_shape)

	def call(self, features, labels):
		embedding_norm = tf.norm(features, axis=1, keepdims=True)
		embedding = tf.divide(features, embedding_norm, name='norm_embedding')
		weights_norm = tf.norm(self.kernel, axis=0, keepdims=True)
		weights = tf.divide(self.kernel, weights_norm, name='norm_weights')

		cos_t = tf.matmul(embedding, weights, name='cos_t')
		cos_t2 = tf.square(cos_t, name='cos_2')
		sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
		sin_t = tf.sqrt(sin_t2, name='sin_t')
		cos_mt = self.arc_s * tf.subtract(tf.multiply(cos_t, self.cos_m), tf.multiply(sin_t, self.sin_m), name='cos_mt')

		cond_v = cos_t - self.th
		cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

		keep_val = self.arc_s*(cos_t - self.mm)
		cos_mt_temp = tf.where(cond, cos_mt, keep_val)

		mask = tf.one_hot(labels, depth=self.num_classes, name='one_hot_mask')
		inv_mask = tf.subtract(1., mask, name='inverse_mask')

		s_cos_t = tf.multiply(self.arc_s, cos_t, name='scalar_cos_t')

		output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')

		return output


if __name__ == '__main__':
	print("go check README.md")
