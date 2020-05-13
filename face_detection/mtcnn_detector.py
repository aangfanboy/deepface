import cv2
import math
import numpy as np

from PIL import Image
from mtcnn.mtcnn import MTCNN
from face_detection.detector_main import MainHelper


class Engine(MainHelper):
	def draw_faces_and_labels_on_image(self, image, boxes, labels, color=(255, 0, 0), thickness: int = 5):
		if type(color) is not list:
			cl = color
			color = [cl for _ in range(len(boxes))]

		for box, clr, label in zip(boxes, color, labels):
			x1, y1, x2, y2 = box
			color2g = clr
			if clr == "different":
				color2g = self.generate_color()

			cv2.rectangle(image, (x1, y1), (x1+x2, y1+y2), color2g, thickness)
			cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color2g, 2, cv2.LINE_AA)

		return image

	def draw_faces_on_image(self, image, boxes, color=(255, 0, 0), thickness: int = 5):
		if type(color) is not list:
			cl = color
			color = [cl for _ in range(len(boxes))]

		for box, clr in zip(boxes, color):
			x1, y1, x2, y2 = box
			color2g = clr
			if clr == "different":
				color2g = self.generate_color()

			cv2.rectangle(image, (x1, y1), (x1+x2, y1+y2), color2g, thickness)

		return image

	def __init__(self, **kwargs):
		super(Engine, self).__init__(**kwargs)

		self.detector = MTCNN()

	def get_faces_from_image(self, image):
		return self.detector.detect_faces(image)

	def take_faces_from_boxes(self, image, boxes):
		frames = []
		for box in boxes:
			x1, y1, x2, y2 = box
			diff = int(int(112/abs(y2 - x2))*10)

			if y2 > x2:
				x_e, y_e = 2*diff, diff
			elif y2 == x2:
				x_e, y_e = 0, 0
			else:
				x_e, y_e = diff, 2*diff

			x_e = int(x2/5)
			y_e = int(y2/10)
			frames.append(image[y1-y_e:y1+y2+y_e, x1-x_e:x1+x2+x_e])

		return frames		

	@staticmethod
	def euclidean_distance(a, b):
		x1 = a[0]
		y1 = a[1]

		x2 = b[0]
		y2 = b[1]
		return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

	def align_image_from_eyes(self, image, eyes):
		left_eye, right_eye = eyes[0]
		left_x, left_y = left_eye
		right_x, right_y = right_eye

		x_diff = float(abs(right_x - left_x))
		y_diff = float(abs(right_y - left_y))
		c_dist = math.sqrt(x_diff**2 + y_diff**2)

		alpha_ = np.arccos((c_dist**2 + x_diff**2 - y_diff ** 2)/(2*x_diff*c_dist))
		alpha = (alpha_ * 180) / math.pi

		if left_y < right_y:
			direction = 1
		else:
			direction = -1
			# alpha = 90 - alpha

		new_img = Image.fromarray(image)
		new_img = np.array(new_img.rotate(direction * alpha))
		return new_img

	def get_boxes_from_faces_with_eyes(self, faces, th: float = None):
		boxes = []
		eyes = []
		for face in faces:
			if th is not None and face["confidence"] < th:
				continue

			boxes.append(face["box"])
			eyes.append((face["keypoints"]["mouth_left"], face["keypoints"]["mouth_right"]))

		return boxes, eyes

	def get_boxes_from_faces(self, faces, th: float = None):
		boxes = []
		for face in faces:
			if th is not None and face["confidence"] < th:
				continue

			boxes.append(face["box"])

		return boxes


if __name__ == '__main__':
	e = Engine()

	image = e.load_image("test.jpg")
	faces = e.get_faces_from_image(image)
	boxes, eyes = e.get_boxes_from_faces_with_eyes(faces)
	image = e.align_image_from_eyes(image, eyes)
	faces = e.get_faces_from_image(image)
	boxes, eyes = e.get_boxes_from_faces_with_eyes(faces)
	image = e.draw_faces_on_image(image, boxes, color="different")

	e.display_image(image, destroy_after=False, n=0)
