import tensorflow as tf
import efficientnet.tfkeras as efn

from model_scripts import inception_resnet_v1
from model_scripts.ArcFaceLayer import ArcFaceLayer


class BatchNormalization(tf.keras.layers.BatchNormalization):
	"""Make trainable=False freeze BN for real (the og version is sad).
	   ref: https://github.com/zzh8829/yolov3-tf2
	"""

	def call(self, x, training=False):
		if training is None:
			training = tf.constant(False)
		training = tf.logical_and(training, self.trainable)
		return super().call(x, training)


class MainModel:
	@tf.function
	def test_step_reg(self, x, y):
		logits, features = self.model([x, y], training=False)
		loss = self.loss_function(y, logits)

		reg_loss = tf.add_n(self.model.losses)

		return logits, features, loss, reg_loss

	@tf.function
	def train_step_reg(self, x, y):
		with tf.GradientTape() as tape:
			logits, features = self.model([x, y], training=True)

			loss = self.loss_function(y, logits)
			reg_loss = tf.add_n(self.model.losses)

			loss_all = tf.add(loss, reg_loss)

		gradients = tape.gradient(loss_all, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		return logits, features, loss, reg_loss

	def change_learning_rate_of_optimizer(self, new_lr: float):
		self.optimizer.learning_rate = new_lr
		self.last_lr = new_lr

		assert self.optimizer.learning_rate == self.optimizer.lr

		return True

	def __init__(self):
		self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.last_lr = None

	@tf.function
	def train_step(self, x, y):
		with tf.GradientTape() as tape:
			logits, features = self.model([x, y], training=True)
			loss = self.loss_function(y, logits)

		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		return logits, features, loss

	@tf.function
	def test_step(self, x, y):
		logits, features = self.model([x, y], training=False)
		loss = self.loss_function(y, logits)

		return logits, features, loss

	def turn_softmax_into_arcface(self, num_classes: int):
		label_input_layer = tf.keras.layers.Input((None,), dtype=tf.int64)

		x = ArcFaceLayer(num_classes=num_classes, name="arcfaceLayer")(self.model.layers[-3].output, label_input_layer)

		self.model = tf.keras.models.Model([self.model.layers[0].input, label_input_layer], [x, self.model.layers[-3].output])
		self.model.summary()

	def change_regularizer_l(self, new_value: float = 5e-4):
		for layer in self.model.layers:
			if "Conv" in str(layer):
				layer.kernel_regularizer = tf.keras.regularizers.l2(new_value)

			elif "BatchNorm" in str(layer):
				layer.gamma_regularizer = tf.keras.regularizers.l2(new_value)
				layer.momentum = 0.9
				layer.epsilon = 2e-5

			elif "PReLU" in str(layer):
				layer.alpha_regularizer = tf.keras.regularizers.l2(new_value)

			elif "Dense" in str(layer):
				layer.kernel_regularizer = tf.keras.regularizers.l2(new_value)

			elif "arcfaceLayer" in str(layer):
				layer.kernel_regularizer = tf.keras.regularizers.l2(new_value)

		self.model = tf.keras.models.model_from_json(self.model.to_json())  # To apply regularizers
		print(f"[*] Kernel regularizer value set to --> {new_value}")

	def __call__(self, input_shape, weights: str = None, num_classes: int = 10, learning_rate: float = 0.1,
	             regularizer_l: float = 5e-4, weight_path: str = None,
	             pooling_layer: tf.keras.layers.Layer = tf.keras.layers.GlobalAveragePooling2D,
	             create_model: bool = True, use_arcface: bool = True,
	             optimizer="ADAM"):

		self.last_lr = learning_rate

		if optimizer == "ADAM":
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=0.1)
			print("[*] ADAM chosen as optimizer")
		elif optimizer == "SGD":
			self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
			print("[*] SGD chosen as optimizer")
		elif optimizer == "MOMENTUM":
			self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
			# MomentumOptimizer is not recommended, it is from TF 1.x makes problem at learning rate change, i will update if TF 2.x version comes out
			print("[*] MomentumOptimizer chosen as optimizer")
		else:
			raise Exception(f"{optimizer} is not a valid name! Go with either ADAM, SGD or MOMENTUM")

		if create_model:
			label_input_layer = tf.keras.layers.Input((None,), dtype=tf.int64)
			self.model = self.get_model(input_shape=input_shape, weights=weights)
			self.model.trainable = True

			self.change_regularizer_l(regularizer_l)
			# ACCORDING TO ARCFACE PAPER
			x = pooling_layer()(self.model.layers[-1].output)
			x = BatchNormalization(momentum=0.9, epsilon=2e-5)(x)
			x = tf.keras.layers.Dropout(0.4)(x)
			x1 = tf.keras.layers.Dense(512, activation=None, name="features_without_bn", use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)
			x = BatchNormalization(momentum=0.9, scale=False, epsilon=2e-5)(x1)

			if use_arcface:
				x = ArcFaceLayer(num_classes=num_classes, arc_m=0.5, arc_s=64., regularizer_l=regularizer_l, name="arcfaceLayer")(x, label_input_layer)
			else:
				x = tf.keras.layers.Dense(num_classes, activation=None, name="classificationLayer", kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)

			self.model = tf.keras.models.Model([self.model.layers[0].input, label_input_layer], [x, x1], name=f"{self.__name__}-ArcFace")
			self.model.summary()

			try:
				self.model.load_weights(weight_path)
				print("[*] WEIGHTS FOUND FOR MODEL, LOADING...")
			except Exception as e:
				print(e)
				print("[*] THERE IS NO WEIGHT FILE FOR MODEL, INITIALIZING...")


class ResNet50(MainModel):
	@property
	def __name__(self):
		return "ResNet50"

	def __init__(self, **kwargs):
		super(ResNet50, self).__init__(**kwargs)

	def get_model(self, input_shape, weights: str = None, **kwargs):
		return tf.keras.applications.ResNet50V2(input_shape=input_shape, weights=weights, include_top=False)


class ResNet101(MainModel):
	@property
	def __name__(self):
		return "ResNet101"

	def __init__(self, **kwargs):
		super(ResNet101, self).__init__(**kwargs)

	def get_model(self, input_shape, weights: str = None, **kwargs):
		return tf.keras.applications.ResNet101V2(input_shape=input_shape, weights=weights, include_top=False)


class ResNet152(MainModel):
	@property
	def __name__(self):
		return "ResNet101"

	def __init__(self, **kwargs):
		super(ResNet101, self).__init__(**kwargs)

	def get_model(self, input_shape, weights: str = None, **kwargs):
		return tf.keras.applications.ResNet152V2(input_shape=input_shape, weights=weights, include_top=False)


class EfficientNetFamily(MainModel):
	all_models = [
		efn.EfficientNetB0,
		efn.EfficientNetB1,
		efn.EfficientNetB2,
		efn.EfficientNetB3,
		efn.EfficientNetB4,
		efn.EfficientNetB5,
		efn.EfficientNetB6,
		efn.EfficientNetB7,
	]

	@property
	def __name__(self):
		return f"EfficientNetB{self.model_id}"

	def __init__(self, model_id: int, **kwargs):
		self.model_id = model_id
		if not 0 <= self.model_id <= 7:
			raise ValueError(f"model_id must be \"0 <= model_id <=7\", yours({self.model_id}) is not valid!")

		super(EfficientNetFamily, self).__init__(**kwargs)

	def get_model(self, input_shape, weights: str = None, **kwargs):
		return self.all_models[self.model_id](input_shape=input_shape, weights=weights, include_top=False)


class Xception(MainModel):
	@property
	def __name__(self):
		return "Xception"

	def __init__(self, **kwargs):
		super(Xception, self).__init__(**kwargs)

	def get_model(self, input_shape, weights: str = None, **kwargs):
		return tf.keras.applications.Xception(input_shape=input_shape, weights=weights, include_top=False)


class InceptionResNetV1(MainModel):
	@property
	def __name__(self):
		return "InceptionResNetV1"

	def __init__(self, **kwargs):
		super(InceptionResNetV1, self).__init__(**kwargs)

	def get_model(self, input_shape, **kwargs):
		return inception_resnet_v1.InceptionResNetV1(input_shape=input_shape)


if __name__ == '__main__':
	print("go check README.md")
