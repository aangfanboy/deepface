import os
import sys

sys.path.append("../")
import tensorflow as tf

from UTKdata_engine import DataEngineTypical as DET
from face_recognition.model_scripts import main_model_architect as MMA
from face_recognition.model_scripts import tensorboard_helper as TBH


class MainModel(MMA.MainModel):
	@property
	def __name__(self):
		return "FromArcFace"

	def __init__(self):
		super(MainModel, self).__init__()

	def __call__(self, input_shape, weights: str = None, arcface_model_path: str = None,
	             ASE_model_path: str = None, num_classes: int = 10,
	             regularizer_l: float = 5e-4, use_arcface: bool = True, optimizer="ADAM", learning_rate: float = 0.1,
	             pooling_layer: tf.keras.layers.Layer = tf.keras.layers.GlobalAveragePooling2D):

		self.last_lr = learning_rate

		if optimizer == "ADAM":
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
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

		if ASE_model_path is not None and os.path.exists(ASE_model_path):
			self.model = tf.keras.models.load_model(ASE_model_path, custom_objects={"ArcFaceLayer": MMA.ArcFaceLayer})
			# self.change_regularizer_l(regularizer_l)

			self.model.summary()
			print(f"[*] ASE model loaded from {ASE_model_path}")

		elif arcface_model_path is not None and os.path.exists(arcface_model_path):
			self.model = tf.keras.models.load_model(arcface_model_path)

			x1 = self.model.layers[-1].output
			label_input_layer = tf.keras.layers.Input((None,), dtype=tf.int64)
			x = MMA.BatchNormalization(momentum=0.9, scale=False, epsilon=2e-5, name="sub_bn_for_ase")(x1)
			if use_arcface:
				x = MMA.ArcFaceLayer(num_classes=num_classes, arc_m=0.5, arc_s=10., regularizer_l=regularizer_l,
				                     name="ArcFaceLayer")(x, label_input_layer)
			else:
				x = tf.keras.layers.Dense(num_classes, activation=None, name="ClassificationLayer",
				                          kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)

			self.model = tf.keras.models.Model([self.model.layers[0].input, label_input_layer], [x, x1],
			                                   name=f"{self.__name__}-ASE-{num_classes}")
			self.model.summary()

			print(f"[*] Model created from ArcFace Model that placed in {arcface_model_path}")

		else:
			label_input_layer = tf.keras.layers.Input((None,), dtype=tf.int64)
			self.model = self.get_model(input_shape=input_shape, weights=weights)
			self.model.trainable = True

			self.change_regularizer_l(regularizer_l)
			# ACCORDING TO ARCFACE PAPER
			x = pooling_layer()(self.model.layers[-1].output)
			x = MMA.BatchNormalization(momentum=0.9, epsilon=2e-5)(x)
			x = tf.keras.layers.Dropout(0.4)(x)
			x1 = tf.keras.layers.Dense(512, activation=None, name="features_without_bn", use_bias=True,
			                           kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)

			x = MMA.BatchNormalization(momentum=0.9, scale=False, epsilon=2e-5)(x1)
			if use_arcface:
				x = MMA.ArcFaceLayer(num_classes=num_classes, arc_m=0.5, arc_s=10., regularizer_l=regularizer_l,
				                     name="ArcFaceLayer")(x, label_input_layer)
			else:
				x = tf.keras.layers.Dense(num_classes, activation=None, name="ClassificationLayer",
				                          kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)

			self.model = tf.keras.models.Model([self.model.layers[0].input, label_input_layer], [x, x1],
			                                   name=f"{self.__name__}-ASE-{num_classes}")
			self.model.summary()

			print("[*] Model structure created from scratch")


class ResNet50(MainModel, MMA.ResNet50):
	pass


class Trainer:
	@staticmethod
	def get_wrong(y_real, y_pred):
		return tf.where(
			tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), -1), tf.cast(y_real, tf.int64)), tf.float32) == 00)

	@staticmethod
	def calculate_accuracy(y_real, y_pred):
		return tf.reduce_mean(
			tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), axis=1), tf.cast(y_real, tf.int64)), dtype=tf.float32))

	def only_test(self, dataset_test=None, display_wrong_images: bool = False):
		if dataset_test is None:
			if self.dataset_engine.dataset_test is None:
				raise Exception("there is no defined test dataset")

			dataset_test = self.dataset_engine.dataset_test

		acc_mean = tf.keras.metrics.Mean()
		loss_mean = tf.keras.metrics.Mean()
		wrong_images = []

		for i, (x, y) in enumerate(dataset_test):
			logits, features, loss, reg_loss = self.model_engine.test_step_reg(x, y)
			accuracy = self.calculate_accuracy(y, logits)
			if accuracy < 1.0:
				images = x.numpy()[self.get_wrong(y, logits).numpy()][0]
				[wrong_images.append(image) for image in images]

			acc_mean(accuracy)
			loss_mean(loss)

			print(f"[*] Step {i}, Accuracy --> %{accuracy} || Loss --> {loss} || Reg Loss --> {reg_loss}")

		if display_wrong_images and len(wrong_images) > 0:
			self.tensorboard_engine.initialize(delete_if_exists=False)
			print(f"[*] TensorBoard initialized on {self.tensorboard_engine.logdir}")

			self.tensorboard_engine.add_images(f"wrong images from 'only_test' function",
			                                   tf.convert_to_tensor(wrong_images), 0)
			print(f"[*] Wrong images({len(wrong_images)}) added to TensorBoard")

		print(f"\n\n[*] Accuracy Mean --> %{acc_mean.result().numpy()} || Loss Mean --> {loss_mean.result().numpy()}")

		return acc_mean, loss_mean, wrong_images

	def only_test_last(self, dataset_test=None, display_wrong_images: bool = False):
		if dataset_test is None:
			if self.dataset_engine.dataset_test is None:
				raise Exception("there is no defined test dataset")

			dataset_test = self.dataset_engine.dataset_test

		acc_mean = tf.keras.metrics.Mean()
		wrong_images = []

		for i, (x, y) in enumerate(dataset_test):
			logits = self.model_engine.model(x, training=False)
			print(tf.argmax(tf.nn.softmax(logits), axis=-1))
			accuracy = self.calculate_accuracy(y, logits)
			if accuracy < 1.0:
				images = x.numpy()[self.get_wrong(y, logits).numpy()][0]
				[wrong_images.append(image) for image in images]

			acc_mean(accuracy)

			print(f"[*] Step {i}, Accuracy --> %{accuracy}")

		if display_wrong_images and len(wrong_images) > 0:
			self.tensorboard_engine.initialize(delete_if_exists=False)
			print(f"[*] TensorBoard initialized on {self.tensorboard_engine.logdir}")

			self.tensorboard_engine.add_images(f"wrong images from 'only_test' function",
			                                   tf.convert_to_tensor(wrong_images), 0)
			print(f"[*] Wrong images({len(wrong_images)}) added to TensorBoard")

		print(f"\n\n[*] Accuracy Mean --> %{acc_mean.result().numpy()}")

		return acc_mean, wrong_images

	def __init__(self, model_engine: MMA, dataset_engine: DET, tensorboard_engine: TBH, use_arcface: bool,
	             learning_rate: float = 0.01,
	             model_path: str = "classifier_model.tf", arcface_model_path: str = "arcface_final.h5",
	             pooling_layer: tf.keras.layers.Layer = tf.keras.layers.GlobalAveragePooling2D,
	             lr_step_dict: dict = None, num_classes: int = 2,
	             optimizer: str = "ADAM", regularizer_l: float = 5e-4):
		self.model_path = model_path
		self.model_engine = model_engine
		self.dataset_engine = dataset_engine
		self.tensorboard_engine = tensorboard_engine
		self.use_arcface = use_arcface
		self.lr_step_dict = lr_step_dict

		self.num_classes = num_classes
		tf.io.gfile.makedirs("/".join(self.model_path.split("/")[:-1]))

		self.tb_delete_if_exists = True
		if self.lr_step_dict is not None:
			print("[*] LEARNING RATE WILL BE CHECKED WHEN step\\alfa_divided_ten == 0")
			learning_rate = list(self.lr_step_dict.values())[0]

		self.model_engine(
			input_shape=(112, 112, 3),
			num_classes=self.num_classes,
			learning_rate=learning_rate,
			regularizer_l=regularizer_l,  # weight decay, train once with 5e-4 and then try something lower such 1e-5
			pooling_layer=pooling_layer,  # Recommended: GlobalAveragePooling
			use_arcface=self.use_arcface,  # set False if you want to train it as regular classification
			ASE_model_path=self.model_path,  # path of keras model(h5) , it is okay if doesn't exists
			arcface_model_path=arcface_model_path,  # path of arcface keras model(h5) , it is okay if doesn't exists
			optimizer=optimizer  # Recommended: SGD
		)

	def __call__(self, max_iteration: int = None, alfa_step=1000, qin: float or int = 10.):
		if max_iteration is not None and max_iteration <= 0:
			max_iteration = None

		alfa_divided_ten = int(alfa_step / 10)
		alfa_multiplied_qin = int(alfa_step * qin)

		print(f"[*] Possible maximum step: {tf.data.experimental.cardinality(self.dataset_engine.dataset)}\n")

		acc_mean = tf.keras.metrics.Mean()
		loss_mean = tf.keras.metrics.Mean()

		self.tensorboard_engine.initialize(
			delete_if_exists=self.tb_delete_if_exists
		)
		print(f"[*] TensorBoard initialized on {self.tensorboard_engine.logdir}")

		for i, (x, y) in enumerate(self.dataset_engine.dataset):
			logits, features, loss, reg_loss = self.model_engine.train_step_reg(x, y)
			accuracy = self.calculate_accuracy(y, logits)
			acc_mean(accuracy)
			loss_mean(loss)

			self.tensorboard_engine({"loss": loss, "reg_loss": reg_loss, "accuracy": accuracy})

			if i % alfa_divided_ten == 0:
				if i % alfa_step == 0 and i > 10:
					self.model_engine.model.save(self.model_path)
					print(f"[{i}] Model saved to {self.model_path}")

				print(
					f"[{i}] Loss: {loss_mean.result().numpy()} || Reg Loss: {reg_loss.numpy()} || Accuracy: %{acc_mean.result().numpy()} || LR: {self.model_engine.optimizer.learning_rate.numpy()}")
				acc_mean.reset_states()
				loss_mean.reset_states()
				if self.lr_step_dict is not None:
					lower_found = False
					for key in self.lr_step_dict:
						if i < int(key):
							lower_found = True
							lr_should_be = self.lr_step_dict[key]
							if lr_should_be != self.model_engine.last_lr:
								self.model_engine.change_learning_rate_of_optimizer(lr_should_be)
								print(f"[{i}] Learning Rate set to --> {lr_should_be}")

							break

					if not lower_found:
						print(
							f"[{i}] Reached to given maximum steps in 'lr_step_dict'({list(self.lr_step_dict.keys())[-1]})")
						self.model_engine.model.save(self.model_path)
						print(f"[{i}] Model saved to {self.model_path}, end of training.")
						self.save_final_model(path=f"models_all/{self.model_path}")
						break

				if i % alfa_multiplied_qin == 0 and self.dataset_engine.dataset_test is not None and i > 10:
					print("[*] Calculating validation loss and accuracy, this may take some time")
					acc_mean.reset_states()
					loss_mean.reset_states()
					for x_test, y_test in self.dataset_engine.dataset_test:
						logits, features, loss, reg_loss = self.model_engine.test_step_reg(x_test, y_test)
						accuracy = self.calculate_accuracy(y_test, logits)

						self.tensorboard_engine({"val. loss": loss, "val. accuracy": accuracy})

						acc_mean(accuracy)
						loss_mean(loss)

					print(
						f"[{i}] Val. Loss --> {loss_mean.result().numpy()} || Val. Accuracy --> %{acc_mean.result().numpy()}")
					acc_mean.reset_states()
					loss_mean.reset_states()

				if max_iteration is not None and i >= max_iteration:
					print(f"[{i}] Reached to given maximum iteration({max_iteration})")
					self.model_engine.model.save(self.model_path)
					print(f"[{i}] Model saved to {self.model_path}, end of training.")
					self.save_final_model(path=f"models_all/{self.model_path}")
					break

		if max_iteration is None:
			print(f"[*] Reached to end of dataset")
			self.model_engine.model.save(self.model_path)
			print(f"[*] Model saved to {self.model_path}, end of training.")
			self.save_final_model(path=f"models_all/{self.model_path}")

	def save_final_model(self, path: str = "deepfake_final.h5", n: int = -1, sum_it: bool = True):
		m = tf.keras.models.Model(self.model_engine.model.layers[0].input, self.model_engine.model.layers[n].output)
		if sum_it:
			m.summary()

		m.save(path)
		print(f"[*] Final feature extractor saved to {path}")

		return True

	def destroy_the_leftovers(self):
		self.tensorboard_engine.delete_graphs()
		self.save_final_model(path=f"models_all/{self.model_path}")

		try:
			os.remove(self.model_path)
		except FileNotFoundError:
			pass

		return True


if __name__ == '__main__':
	by_value = "eth"  # change this
	by_num_classes = {"sex": 2, "age": 24, "eth": 5}  # don't change those values

	TDOM = DET(
		"../datasets/UTKFace/",  # tfrecord path
		batch_size=64,
		epochs=-1,  # set to "-1" so it can stream forever
		buffer_size=100000,
		reshuffle_each_iteration=True,  # set True if you set test_batch to 0
		by=by_value
	)  # TDOM for "Tensorflow Dataset Object Manager"

	TBE = TBH.TensorBoardCallback(
		logdir="classifier_tensorboard"  # folder to write TensorBoard
	)  # TBE for "TensorBoard Engine"

	ME = ResNet50()  # ME for "Model Engine"

	k_value: float = 0.125  # k_value = float(8/TDOM.batch_size)
	trainer = Trainer(
		model_engine=ME,
		dataset_engine=TDOM,
		tensorboard_engine=TBE,
		use_arcface=False,  # set False if you want to train a normal classification model, and u should set False
		learning_rate=0.0001,  # it doesn't matter if you set lr_step_dict to anything but None
		model_path=f"{by_value}_model.h5",  # it will save only weights, you can chose "h5" as extension too
		arcface_model_path="models_all/arcface_final.h5",
		optimizer="SGD",  # SGD, ADAM or MOMENTUM. MOMENTUM is not recommended
		regularizer_l=5e-4,  # "l" parameter for l2 regularizer,
		num_classes=by_num_classes[by_value],  # 2 for sex, 24 for age, 5 for eth
	)

	trainer(max_iteration=3000, alfa_step=5000, qin=0.2)
	trainer.destroy_the_leftovers()
