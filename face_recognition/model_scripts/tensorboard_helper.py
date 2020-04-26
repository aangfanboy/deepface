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


if __name__ == '__main__':
	print("go check README.md")
