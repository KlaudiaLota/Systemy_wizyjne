import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj obraz
image = cv2.imread("DSC00012.jpg")

# Konwersja na odcienie szarości
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("szary.jpg", gray)

# Oblicz histogram przed wyrównaniem
hist_before = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Wyrównanie histogramu
equalized = cv2.equalizeHist(gray)
cv2.imwrite("wyrownany.jpg", equalized)

# Oblicz histogram po wyrównaniu
hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256])

# Zapis histogramów
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(hist_before, color='gray')
plt.title("Histogram przed wyrównaniem")
plt.xlabel("Poziom jasności")
plt.ylabel("Liczba pikseli")

plt.subplot(1, 2, 2)
plt.plot(hist_after, color='black')
plt.title("Histogram po wyrównaniu")
plt.xlabel("Poziom jasności")
plt.ylabel("Liczba pikseli")

plt.tight_layout()
plt.savefig("histogramy.png")
plt.show()
