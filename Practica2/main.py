import cv2
import functions as fx

if __name__ == '__main__':
    tablero = cv2.imread('imagenes/Tablero1.jpg', flags=cv2.IMREAD_COLOR)
    piramide = fx.generate_gaussian_piramide(img_src, subsample_factor, n_levels)