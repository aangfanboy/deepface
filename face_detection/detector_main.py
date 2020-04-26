import cv2
import numpy as np


class MainHelper:
	@staticmethod
	def generate_color():
		color = tuple(np.random.choice(range(256), size=3))
		color = (int(color[0]), int(color[1]), int(color[2]))

		return color

	def yield_video(self, path):
		cap = cv2.VideoCapture(path)

		while True:
			try:
				result, frame = cap.read()
				if not result:
					break

				yield frame

			except Exception as e:
				print(e)
				continue

	def __init__(self):
		pass

	def display_video(self, path):
		for frame in e.yield_video(path):
			e.display_image(frame, destroy_after=False, n=60)

	def load_image(self, path):
		return cv2.imread(path)

	def display_image(self, image, name: str = "image", wait: bool = True, destroy_after: bool = True, n: int = 0):
		cv2.imshow(name, image)
		if wait:
			cv2.waitKey(n)

		if destroy_after:
			cv2.destroyWindow(name)


if __name__ == '__main__':
	e = MainHelper()
	e.display_video("test2.gif")
