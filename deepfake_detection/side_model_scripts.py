import tensorflow as tf

from shutil import rmtree


class TensorBoardCallback:
	def delete_graphs(self):
		if tf.io.gfile.exists(self.logdir):
			rmtree(self.logdir)
			print(f"[*] {self.logdir} has deleted with shutil's rmtree")

	def initialize(self, delete_if_exists: bool = False):
		if delete_if_exists:
			self.delete_graphs()

		self.file_writer = tf.summary.create_file_writer(logdir=self.logdir)

	def __init__(self, logdir: str = "graphs/"):
		self.logdir = logdir
		self.file_writer = None

		self.initial_step = 0

	def __call__(self, data_json: dict, description: str = None, **kwargs):
		with self.file_writer.as_default():
			for key in data_json:
				tf.summary.scalar(key, data_json[key], step=self.initial_step, description=description)

		self.initial_step += 1

	def add_with_step(self, data_json: dict, description: str = None, step: int = 0):
		with self.file_writer.as_default():
			for key in data_json:
				tf.summary.scalar(key, data_json[key], step=step, description=description)

	def add_text(self, name: str, data: str, step: int, **kwargs):
		with self.file_writer.as_default():
			tf.summary.text(name, data, step=step)

	def add_images(self, name: str, data, step: int, max_outputs: int = None, **kwargs):
		if max_outputs is None:
			max_outputs = data.shape[0]

		with self.file_writer.as_default():
			tf.summary.image(name, data, max_outputs=max_outputs, step=step)


class ModelEngine:
	def change_learning_rate_of_optimizer(self, new_lr: float):
		self.optimizer.learning_rate = new_lr
		self.last_lr = new_lr

		assert self.optimizer.learning_rate == self.optimizer.lr

		return True

	def create_model_from_fr(self, fr_model_path: str, base_trainable: bool = True, bn_before_cl: bool = True):
		base_model = tf.keras.models.load_model(fr_model_path)
		base_model.trainable = base_trainable

		x = base_model.layers[-1].output
		x = tf.keras.layers.ReLU()(x)
		if bn_before_cl:
			x = tf.keras.layers.BatchNormalization(momentum=0.9, name="bn_before_df_classifier")(x)

		x = tf.keras.layers.Dense(self.num_classes, activation=None, kernel_regularizer=tf.keras.regularizers.l2(5e-4), name="last_cl_layer")(x)

		self.model = tf.keras.models.Model(base_model.layers[0].input, x, name=f"DeepFake-Model-{self.num_classes}")
		self.model.summary()

	def load_model_from_df(self, df_model_path: str):
		self.model = tf.keras.models.load_model(df_model_path)
		self.model.summary()

	def __init__(self, optimizer: str = "SGD", learning_rate: float = 0.01, num_classes: int = 2):
		self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.num_classes = num_classes
		self.last_lr = None
		self.model = None

		if optimizer == "ADAM":
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=0.1)
			print("[*] ADAM chosen as optimizer")
		elif optimizer == "SGD":
			self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
			print("[*] SGD chosen as optimizer")
		elif optimizer == "MOMENTUM":
			self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
			print("[*] MomentumOptimizer chosen as optimizer")
		else:
			raise Exception(f"{optimizer} is not a valid name! Go with either ADAM, SGD or MOMENTUM")

	@tf.function
	def test_step_reg(self, x, y):
		logits = self.model(x, training=False)
		loss = self.loss_function(y, logits)

		reg_loss = tf.add_n(self.model.losses)

		return logits, loss, reg_loss

	@tf.function
	def train_step_reg(self, x, y):
		with tf.GradientTape() as tape:
			logits = self.model(x, training=True)
			loss = self.loss_function(y, logits)

			reg_loss = tf.add_n(self.model.losses)
			loss_all = tf.add(loss, reg_loss)

		gradients = tape.gradient(loss_all, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		return logits, loss, reg_loss


if __name__ == '__main__':
	print("go check README.md")
