import cv2

image = cv2.imread("face-of-male-mandrill-baboon-mandrillus-animal-images.jpg")

cv2.circle(image, (250, 200), 50, (0, 0, 255), -1)

cv2.rectangle(image, (100, 300), (300, 450), (0, 255, 0), -1)

cv2.ellipse(image, (400, 400), (100, 50), 30, 0, 360, (0, 255, 255), -1)

output_filename = "mandrill_with_shapes.jpg"
cv2.imwrite(output_filename, image)