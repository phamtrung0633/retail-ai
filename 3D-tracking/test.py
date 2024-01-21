from embeddings.embedder import Embedder
import cv2
e = Embedder()
e.initialise()
m = cv2.imread("images/environmentRight/2.png")
e.search("shelf_1", m)
