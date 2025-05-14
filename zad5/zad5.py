import cv2
import numpy as np
import matplotlib.pyplot as plt

def show(image, title='', cmap='gray'):
    plt.figure(figsize=(8,6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def find_extreme_vertical_lines(lines, width):
    left_x = width
    right_x = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if 80 < abs(angle) < 100:  # pionowe
            left_x = min(left_x, x1, x2)
            right_x = max(right_x, x1, x2)
    return left_x, right_x

def find_average_horizontal_angle(lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 10:  # poziome linie
            angles.append(angle)
    return np.mean(angles) if angles else 0

# 1. Wczytanie
image = cv2.imread("DSCN0091.jpg")
resized = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5), 0)

# 2. Detekcja pionowych krawędzi
edges = cv2.Canny(blurred, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

if lines is not None:
    h, w = gray.shape
    left_x, right_x = find_extreme_vertical_lines(lines, w)
    
    # Przycięcie
    cropped = resized[:, right_x:]
    show(cropped, "Po przycięciu boków", cmap=None)
    
    # Detekcja poziomych linii
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blurred_cropped = cv2.GaussianBlur(gray_cropped, (3, 3), 0)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(gray_cropped, kernel, iterations=1)
    edges_cropped = cv2.Canny(dilated, 30, 100)
    
    horizontal_lines = cv2.HoughLinesP(edges_cropped, 1, np.pi/180, 50, minLineLength=30, maxLineGap=20)
    lines_image = cropped.copy()

    if horizontal_lines is not None:
        for line in horizontal_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # zielone linie
        show(lines_image, "Wykryte linie poziome", cmap=None)

        avg_angle = find_average_horizontal_angle(horizontal_lines)
        rotated = rotate_image(cropped, avg_angle)
        show(rotated, f"Po obrocie o {avg_angle:.2f} stopni", cmap=None)
        cv2.imwrite("rotated_result.jpg", rotated)
    else:
        print("Nie znaleziono poziomych linii.")
else:
    print("Nie znaleziono pionowych linii.")
