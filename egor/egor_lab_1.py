import numpy as np
import imutils
import cv2

puzzle = cv2.imread("example_sad.jpg")
face = cv2.imread("example_face.png")
faceHeight, faceWidth = face.shape[:2]

result = cv2.matchTemplate(puzzle, face, cv2.TM_CCOEFF)
_, _, minLoc, maxLoc = cv2.minMaxLoc(result)
topLeft = maxLoc
botRight = (topLeft[0] + faceWidth, topLeft[1] + faceHeight)
roi = puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]]

mask = np.zeros(puzzle.shape, dtype="uint8")
puzzle = cv2.addWeighted(puzzle, 0.25, mask, 0.75, 0)

puzzle[topLeft[1]:botRight[1], topLeft[0]:botRight[0]] = roi

cv2.imshow("Puzzle", imutils.resize(puzzle, height=650))
cv2.imshow("SAD", face)
cv2.waitKey(0)
