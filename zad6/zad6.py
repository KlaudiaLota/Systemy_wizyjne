import cv2
import numpy as np
import matplotlib.pyplot as plt

# Wczytaj obraz
image = cv2.imread("rynek_frag.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Wykryj rogi
corners1 = cv2.goodFeaturesToTrack(gray, maxCorners=2700, qualityLevel=0.01, minDistance=10)
corners1 = corners1.reshape(-1, 2)

# Obrót obrazu
def rotate_image_and_matrix(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated, M

angle = 45
rotated_image, rot_matrix = rotate_image_and_matrix(image, angle)
rotated_gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

# Wykryj rogi po obrocie
corners2 = cv2.goodFeaturesToTrack(rotated_gray, maxCorners=2700, qualityLevel=0.01, minDistance=10)
corners2 = corners2.reshape(-1, 2)

# Transformuj punkty
def transform_points(points, matrix):
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    return np.dot(matrix, points.T).T

transformed_corners1 = transform_points(corners1, rot_matrix)

# Dopasowanie punktów
def match_points(transformed, detected, tolerance=5.0):
    matches = []
    for pt in transformed:
        distances = np.linalg.norm(detected - pt, axis=1)
        min_idx = np.argmin(distances)
        if distances[min_idx] < tolerance:
            matches.append((tuple(pt), tuple(detected[min_idx])))
    return matches

matches = match_points(transformed_corners1, corners2)

# Wizualizacja
image_display = cv2.cvtColor(rotated_image.copy(), cv2.COLOR_BGR2RGB)

for x, y in corners2:
    cv2.circle(image_display, (int(x), int(y)), 3, (0, 255, 0), -1) 

for (pt1, pt2) in matches:
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    cv2.circle(image_display, (x2, y2), 4, (0, 255, 0), 1) 

# Pokaż oba obrazy
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

image_with_corners = image.copy()
for x, y in corners1:
    cv2.circle(image_with_corners, (int(x), int(y)), 3, (0, 255, 0), -1)

axs[0].imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
axs[0].set_title("Oryginalny obraz z rogami")
axs[0].axis("off")

axs[1].imshow(image_display)
axs[1].set_title("Obrócony obraz z rogami i dopasowaniami")
axs[1].axis("off")

plt.tight_layout()
plt.show()

print(f"Liczba wykrytych rogów w oryginale: {len(corners1)}")
print(f"Liczba rogów po obrocie: {len(corners2)}")
print(f"Liczba dopasowanych rogów: {len(matches)}")
