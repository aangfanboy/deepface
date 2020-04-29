import cv2

from face_detection.detector_main import MainHelper
from mtcnn.mtcnn import MTCNN


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
	boxes = e.get_boxes_from_faces(faces)
	image = e.draw_faces_on_image(image, boxes, color="different")

	e.display_image(image, destroy_after=False, n=0)
