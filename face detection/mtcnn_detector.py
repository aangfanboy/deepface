import cv2

from detector_main import MainHelper
from mtcnn.mtcnn import MTCNN


class Engine(MainHelper):
    def draw_faces_on_image(self, image, boxes, color=(255, 0, 0), thickness: int = 5):
        for box in boxes:
            x1, y1, x2, y2 = box
            color2g = color
            if color == "different":
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
			diff = int(112/abs(y2 - x2))*2

			if y2 > x2:
				x_e, y_e = 2*diff, diff
			elif y2 == x2:
				x_e, y_e = 0, 0
			else:
				x_e, y_e = diff, 2*diff

			frames.append(image[y1-y_e:y1+y2+y_e, x1-x_e:x1+x2+x_e])

		return frames		

	def get_boxes_from_faces(self, faces, th: float = None):
		boxes = []
		for face in faces:
			if th is not None and face["confidance"] < th:
				continue

			boxes.append(face["box"])

		return boxes


if __name__ == '__main__':
	e = Engine()

	image = e.load_image("test1.jpg")

	for image in e.yield_video("test2.gif"):
		faces = e.get_faces_from_image(image)
		boxes = e.get_boxes_from_faces(faces)
		image = e.draw_faces_on_image(image, boxes)

		# face_frames = e.take_faces_from_boxes(image, boxes)

		# for face in face_frames:
		# 	e.display_image(face)

		e.display_image(image, destroy_after=False, n=60)
