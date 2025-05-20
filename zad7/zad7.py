import zipfile
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO

zip_path = 'test_img_1.zip'
output_dir = 'obrazy'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

image_files = sorted([
    os.path.join(output_dir, fname)
    for fname in os.listdir(output_dir)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
])

positions = []
frames = []

for file in image_files:
    img = cv2.imread(file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = -1, -1

    positions.append((cX, cY))

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    if cX != -1:
        draw.ellipse([cX - 5, cY - 5, cX + 5, cY + 5], outline='red', width=2)
    frames.append(img_pil)

frames[0].save("animacja.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)

with open("sprawozdanie.txt", "w") as f:
    f.write("Nr_klatki\tX\tY\tdX\tdY\n")
    x0, y0 = positions[0]
    for i, (x, y) in enumerate(positions):
        dx = x - x0
        dy = y - y0
        f.write(f"{i}\t{x}\t{y}\t{dx}\t{dy}\n")

print("Zakończono. Wygenerowano pliki: animacja.gif, sprawozdanie.txt")
