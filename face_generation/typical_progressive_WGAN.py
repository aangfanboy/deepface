import sys
import numpy as np
sys.path.append("../")
import tensorflow as tf

from face_recognition.data_manager.dataset_manager import DataEngineTFRecord as DET


class PixelNorm(tf.keras.layers.Layer):
	def call(self, x, epsilon=1e-8):
		return tf.multiply(x, tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon))


class MiniBatchStddev(tf.keras.layers.Layer):
	def call(self, x, group_size=4):
		group_size = tf.minimum(group_size, tf.shape(x)[0])     # Minibatch must be divisible by (or smaller than) group_size.
		s = x.shape                                             # [NHWC]  Input shape.
		y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])   # [GMHWC] Split minibatch into M groups of size G.
		y = tf.cast(y, tf.float32)                              # [GMHWC] Cast to FP32.
		y -= tf.reduce_mean(y, axis=0, keepdims=True)           # [GMHWC] Subtract mean over group.
		y = tf.reduce_mean(tf.square(y), axis=0)                # [MHWC]  Calc variance over group.
		y = tf.sqrt(y + 1e-8)                                   # [MHWC]  Calc stddev over group.
		y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)    # [M111]  Take average over fmaps and pixels.
		y = tf.cast(y, x.dtype)                                 # [M111]  Cast back to original data type.
		y = tf.tile(y, [group_size, s[1], s[2], 1])             # [N1HW]  Replicate over group and pixels.
		return tf.concat([x, y], axis=-1)                        # [NHWC]  Append as new fmap.


class Generator:
	def add_block(self, q: int):
		filter_dict = {2: 512, 3: 512, 4: 512, 5: 512, 6: 256, 7: 128, 8: 64, 9: 32, 10: 16}

		last_usable = self.model.layers[-2].output
		x = tf.keras.layers.UpSampling2D()(last_usable)
		x = tf.keras.layers.Conv2D(filter_dict[q], (3, 3), padding="SAME", kernel_initializer="he_normal", activation=None)(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = tf.keras.layers.Conv2D(filter_dict[q], (3, 3), padding="SAME", kernel_initializer="he_normal", activation=None)(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer="he_normal")(x)

		w_before = self.model.layers[-1].get_weights()
		if filter_dict[q] != filter_dict[q - 1]:
			a, b = tf.split(w_before[0], 2, axis=2)

			if self.weight_selection_mode == "first":
				w_before[0] = a
			elif self.weight_selection_mode == "second":
				w_before[0] = b
			else:
				w_before[0] = (a + b) / 2

		self.model = tf.keras.models.Model(self.model.layers[0].input, x, name="generator")
		if self.weight_selection_mode is not False:
			self.model.layers[-1].set_weights(w_before)
		self.model.summary()

	def __init__(self, weight_selection_mode: str = "mean"):
		self.weight_selection_mode = weight_selection_mode  # "mean", "first", "second" or False

	def __call__(self, *args, **kwargs):
		input_size = 512

		input_layer = tf.keras.layers.Input((input_size, ))
		x = tf.keras.layers.Dense(input_size*16, activation=None, kernel_initializer="he_normal")(input_layer)
		x = tf.keras.layers.Reshape((4, 4, input_size))(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = PixelNorm()(x)
		x = tf.keras.layers.Conv2D(512, (3, 3), padding="SAME", kernel_initializer="he_normal", activation=None)(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = PixelNorm()(x)

		x = tf.keras.layers.Conv2D(3, (1, 1), kernel_initializer="he_normal", activation=None)(x)

		self.model = tf.keras.models.Model(input_layer, x, name="generator")
		self.model.summary()


class Discriminator:
	def add_block(self, q: int):
		filter_dict = {1: 512, 2: 512, 3: 512, 4: 512, 5: 512, 6: 256, 7: 128, 8: 64, 9: 32, 10: 16}
		input_layer = tf.keras.layers.Input((int(2**q), int(2**q), 3))
		x = tf.keras.layers.Conv2D(filter_dict[q], (1, 1), kernel_initializer="he_normal", activation=None, name="base_1x1")(input_layer)
		x = tf.keras.layers.LeakyReLU(0.2, name="base_1x1_leaky")(x)

		a = list(range(q - 2))
		a.reverse()
		for i in a:
			if i == a[0]:
				x = tf.keras.layers.Lambda(lambda x_i: x_i, name="receiver")(x)

			x = tf.keras.layers.Conv2D(filter_dict[i+3], (3, 3), padding='same', kernel_initializer="he_normal", activation=None, name=f"{filter_dict[i+3]}_{q}-{i}_{int(2**q)}x{int(2**q)}")(x)
			x = tf.keras.layers.LeakyReLU(0.2, name=f"{filter_dict[i+3]}_{q}-{i}_{int(2**q)}x{int(2**q)}_leaky")(x)
			x = tf.keras.layers.Conv2D(filter_dict[i+2], (3, 3), padding='same', kernel_initializer="he_normal", activation=None, name=f"{filter_dict[i+2]}_{q}-{i}_sub_{int(2**q)}x{int(2**q)}")(x)
			x = tf.keras.layers.LeakyReLU(0.2, name=f"{filter_dict[i+2]}_{q}-{i}_{int(2**q)}x{int(2**q)}_sub_leaky")(x)
			x = tf.keras.layers.AveragePooling2D(name=f"{i}_pooling")(x)

		x = MiniBatchStddev()(x)
		x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer="he_normal", activation=None, name="bottom_3x3")(x)
		x = tf.keras.layers.LeakyReLU(0.2, name="bottom_3x3_leaky")(x)
		x = tf.keras.layers.Conv2D(512, (4, 4), kernel_initializer="he_normal", activation=None, name="bottom_4x4")(x)
		x = tf.keras.layers.LeakyReLU(0.2, name="bottom_4x4_leaky")(x)
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(1, kernel_initializer="he_normal", name="last_dense")(x)

		found = False
		weights_saved = []
		for layer in self.model.layers:
			if found:
				weights_saved.append(layer.get_weights())

			if layer.name == "receiver":
				found = True

		self.model = tf.keras.models.Model(input_layer, x)
		weights_saved.reverse()
		ll = len(weights_saved)

		found = False
		wait4 = 0
		for layer in self.model.layers:
			if found:
				wait4 += 1
				if wait4 >= 6:
					w = weights_saved.pop()
					layer.set_weights(w)

			if layer.name == "receiver":
				found = True

		self.model.summary()
		print(f"[**] {ll} weights re-placed")

	def __init__(self):
		pass

	def __call__(self, *args, **kwargs):
		input_layer = tf.keras.layers.Input((4, 4, 3))

		x = tf.keras.layers.Conv2D(512, (1, 1), kernel_initializer="he_normal", activation=None)(input_layer)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = tf.keras.layers.Lambda(lambda x_i: x_i, name="receiver")(x)
		x = MiniBatchStddev()(x)
		x = tf.keras.layers.Conv2D(512, (3, 3), padding='same', kernel_initializer="he_normal", activation=None)(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = tf.keras.layers.Conv2D(512, (4, 4), kernel_initializer="he_normal", activation=None)(x)
		x = tf.keras.layers.LeakyReLU(0.2)(x)
		x = tf.keras.layers.Flatten()(x)
		x = tf.keras.layers.Dense(1, activation=None, kernel_initializer="he_normal")(x)

		self.model = tf.keras.models.Model(input_layer, x, name="discriminator")
		self.model.summary()


class GeneratorBasic:
	def add_block(self, q):
		pass

	def __init__(self):
		pass

	def __call__(self, *args, **kwargs):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(512,)))
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.LeakyReLU())

		model.add(tf.keras.layers.Reshape((7, 7, 256)))
		assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

		model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
		assert model.output_shape == (None, 7, 7, 128)
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.LeakyReLU())

		model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
		assert model.output_shape == (None, 14, 14, 64)
		model.add(tf.keras.layers.BatchNormalization())
		model.add(tf.keras.layers.LeakyReLU())

		model.add(tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
		assert model.output_shape == (None, 28, 28, 3)

		self.model = model


class DiscriminatorBasic:
	def add_block(self, q):
		pass

	def __init__(self):
		pass

	def __call__(self, *args, **kwargs):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 3]))
		model.add(tf.keras.layers.LeakyReLU())
		model.add(tf.keras.layers.Dropout(0.3))

		model.add(tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
		model.add(tf.keras.layers.LeakyReLU())
		model.add(tf.keras.layers.Dropout(0.3))

		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(1))

		self.model = model


class Engine:
	@tf.function
	def train_step(self, x, z):
		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
			generated_images = self.generator.model(z, training=True)

			real_output = self.discriminator.model(x, training=True)
			fake_output = self.discriminator.model(generated_images, training=True)

			gen_loss = -fake_output
			disc_loss = -real_output
			disc_loss += fake_output

			def lerp(a, b, t): return t * a + (1 - t) * b
			coefficients = tf.keras.backend.random_uniform([tf.shape(x)[0], 1, 1, 1])
			interpolated_images = lerp(x, generated_images, coefficients)
			interpolated_adversarial_logits = self.discriminator.model(interpolated_images)
			interpolated_gradients = tf.gradients(interpolated_adversarial_logits, [interpolated_images])[0]
			interpolated_gradient_penalties = tf.square(
				1 - tf.sqrt(tf.reduce_sum(tf.square(interpolated_gradients), axis=[1, 2, 3]) + 1e-8))
			disc_loss += interpolated_gradient_penalties * 10.0

			gen_loss = tf.reduce_mean(gen_loss)
			disc_loss = tf.reduce_mean(disc_loss)

			"""
			real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
			fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
			disc_loss = real_loss + fake_loss

			gen_loss = self.cross_entropy(tf.ones_like(fake_output), fake_output)
			"""

		gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.model.trainable_variables)
		gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.model.trainable_variables)
		gradients_of_discriminator = [tf.clip_by_value(p, -0.01, 0.01) for p in gradients_of_discriminator]

		self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.model.trainable_variables))
		self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.model.trainable_variables))

		return gen_loss, disc_loss, generated_images

	def add_block(self, q: int):
		self.generator.add_block(q)
		self.discriminator.add_block(q)

	def __init__(self, dataset_manager):
		self.dataset_manager = dataset_manager
		self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

		self.generator = Generator()
		self.discriminator = Discriminator()

		self.generator()
		self.discriminator()

		self.add_block(3)
		self.add_block(4)
		self.add_block(5)

		self.generator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0., beta_2=0.99, epsilon=10e-8)
		self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0., beta_2=0.99, epsilon=10e-8)
		import cv2

		for x, y in self.dataset_manager.dataset:
			z = tf.keras.backend.random_normal((16, 512))
			x = tf.convert_to_tensor([tf.image.resize(n, (32, 32), method="nearest") for n in x])

			gen_loss, disc_loss, generated_images = self.train_step(x, z)
			print((gen_loss, disc_loss))

			cv2.imshow("a", cv2.resize(tf.cast((generated_images[0] * 128.) + 127.5, tf.uint8).numpy(), (64, 64)))
			cv2.waitKey(60)


if __name__ == '__main__':
	TDOM = DET(
		"../datasets/mnist_data.tfrecords",
		batch_size=16,
		epochs=-1,
		buffer_size=70000,
		reshuffle_each_iteration=True,
		test_batch=0,
		map_to=True
	)

	e = Engine(TDOM)
