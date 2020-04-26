import os
import sys
sys.path.append("../../../")
import tensorflow as tf

from tqdm import tqdm
from face_detection import mtcnn_detector


class Engine:
	@staticmethod
	def turn_rgb(images):
		b, g, r = tf.split(images, 3, axis=-1)
		images = tf.concat([r, g, b], -1)

		return images

	@staticmethod
	def set_face(face):
		face = tf.image.resize(face, (112, 112), method="nearest")

		return (tf.cast(face, tf.float32) - 127.5) / 128.

	def __init__(self, model_path: str):
		self.model = tf.keras.models.load_model(model_path)
		self.data = {}

		self.cos_dis = lambda x, y: tf.norm(x-y)  # tf.keras.losses.CosineSimilarity()

		self.detector = mtcnn_detector.Engine()

	def get_output(self, images):
		return tf.nn.l2_normalize(self.model(images, training=False))

	def __call__(self, main_path1: str, main_path2: str, th: float = 1.0):
		image = self.detector.load_image(main_path1)
		faces = self.detector.get_faces_from_image(image)
		boxes1 = self.detector.get_boxes_from_faces(faces)
		face_frames = self.turn_rgb(self.detector.take_faces_from_boxes(image, boxes1))
		face_frames = tf.convert_to_tensor([self.set_face(n) for n in face_frames])
		image1 = image.copy()
		output1 = self.get_output(face_frames)[0]

		image = self.detector.load_image(main_path2)
		faces = self.detector.get_faces_from_image(image)
		boxes2 = self.detector.get_boxes_from_faces(faces)
		face_frames = self.turn_rgb(self.detector.take_faces_from_boxes(image, boxes2))
		face_frames = tf.convert_to_tensor([self.set_face(n) for n in face_frames])
		image2 = image.copy()
		output2 = self.get_output(face_frames)[0]

		dist = self.cos_dis(output1, output2)
		color = (0, 0, 255)
		if dist < th:
			color = (0, 255, 0)

		image1 = self.detector.draw_faces_on_image(image1, boxes1, color=color)
		image2 = self.detector.draw_faces_on_image(image2, boxes2, color=color)

		self.detector.display_image(image1, name="image1", destroy_after=False, wait=False)
		self.detector.display_image(image2, name="image2", destroy_after=True, n=0)


if __name__ == '__main__':
	e = Engine("../../arcface_final.h5")
	e("t2.jpg", "t3.jpg")

