import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = 'termowizja.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

x, y, w, h = 100, 50, 300, 200
roi = image[y:y+h, x:x+w]

# Progowanie proste
_, thresh_simple = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY)

# Progowanie adaptacyjne
thresh_adaptive = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Progowanie Otsu
_, thresh_otsu = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(roi, cmap='gray')
axs[0].set_title('Oryginał')
axs[1].imshow(thresh_simple, cmap='gray')
axs[1].set_title('Progowanie proste')
axs[2].imshow(thresh_adaptive, cmap='gray')
axs[2].set_title('Progowanie adaptacyjne')
axs[3].imshow(thresh_otsu, cmap='gray')
axs[3].set_title('Progowanie Otsu')

for ax in axs:
    ax.axis('off')

plt.show()