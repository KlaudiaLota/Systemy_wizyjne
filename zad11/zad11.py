from PIL import Image
import numpy as np

def white_balance_gray_world(image_path, save_path=None):
    # Wczytaj obraz
    image = Image.open(image_path)
    image_np = np.array(image).astype(np.float32)

    # Oblicz średnie wartości R, G, B
    avg_r = np.mean(image_np[:, :, 0])
    avg_g = np.mean(image_np[:, :, 1])
    avg_b = np.mean(image_np[:, :, 2])

    # Oblicz średnią ze wszystkich kanałów
    avg_gray = (avg_r + avg_g + avg_b) / 3

    # Oblicz współczynniki korekcji
    gain_r = avg_gray / avg_r
    gain_g = avg_gray / avg_g
    gain_b = avg_gray / avg_b

    # Zastosuj korekcję do każdego kanału
    image_np[:, :, 0] *= gain_r
    image_np[:, :, 1] *= gain_g
    image_np[:, :, 2] *= gain_b

    # Ogranicz wartości do zakresu 0-255 i konwertuj na uint8
    image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    # Zamień z powrotem na obraz PIL
    balanced_image = Image.fromarray(image_np)

    # Zapisz jeśli podano ścieżkę
    if save_path:
        balanced_image.save(save_path)

    return balanced_image

# Przykładowe użycie
input_path = "DSC00636crop.jpg"
output_path = "DSC00636crop_white_balanced.jpg"
balanced_img = white_balance_gray_world(input_path, output_path)
balanced_img.show()
