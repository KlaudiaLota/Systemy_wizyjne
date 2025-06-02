import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj obraz
image = cv2.imread('kwiat.jpg')  # Podmień na własną nazwę pliku
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Konwersja do przestrzeni HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# Progowanie kanału H
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)

# Opcjonalne oczyszczanie maski
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Nałożenie maski na oryginalny obraz
result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_cleaned)

# --- Wizualizacja ---
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title("Oryginalny obraz")
axs[0, 0].axis('off')

axs[0, 1].imshow(h, cmap='hsv')
axs[0, 1].set_title("Kanał H")
axs[0, 1].axis('off')

axs[0, 2].imshow(mask, cmap='gray')
axs[0, 2].set_title("Maska binarna (przed czyszczeniem)")
axs[0, 2].axis('off')

axs[1, 0].imshow(mask_cleaned, cmap='gray')
axs[1, 0].set_title("Maska binarna (po czyszczeniu)")
axs[1, 0].axis('off')

axs[1, 1].imshow(result)
axs[1, 1].set_title("Wyizolowany kwiat")
axs[1, 1].axis('off')

axs[1, 2].axis('off')  # pusta komórka

plt.tight_layout()
plt.show()
