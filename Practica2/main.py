import cv2
import functions as fx

if __name__ == '__main__':
    tablero = cv2.imread('imagenes/Tablero1.jpg', flags=cv2.IMREAD_GRAYSCALE)
    # fx.show_img(tablero,                'Piramide gaussiana')
    fx.extract_harris_points(cv2.imread('imagenes/Tablero1.jpg', flags=cv2.IMREAD_GRAYSCALE), blockS = 3, kSize = 5)