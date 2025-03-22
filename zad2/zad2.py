import cv2
import numpy as np
import sys
import os

def get_kernel():
    while True:
        try:
            n = int(input("Podaj n (wymiar macierzy 2n+1): "))
            if n < 0:
                raise ValueError("n musi być liczbą nieujemną")
            size = 2 * n + 1
            print(f"Podaj macierz {size}x{size}, wpisując kolejne wiersze oddzielone spacjami:")
            kernel = []
            for i in range(size):
                row = list(map(float, input().split()))
                if len(row) != size:
                    raise ValueError("Niepoprawna liczba elementów w wierszu.")
                kernel.append(row)
            return np.array(kernel, dtype=np.float32)
        except ValueError as e:
            print(f"Błąd: {e}, spróbuj ponownie.")

def main():
    image_path = "mandrill.jpg"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if image is None:
        print("Błąd: Nie udało się wczytać obrazu.")
        sys.exit(1)
    
    kernel = get_kernel()
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    combined_image = np.hstack((image, filtered_image))
    
    cv2.imshow("Oryginal i przefiltrowany obraz", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
