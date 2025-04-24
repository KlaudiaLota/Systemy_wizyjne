import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('bocian.jpg', cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(image, 100, 200)

_, thresholded = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

kernel = np.ones((5, 5), np.uint8)
opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title("Konturowanie")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Progowanie")
plt.imshow(thresholded, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Otwarcie")
plt.imshow(opened, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Zamknięcie")
plt.imshow(closed, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
