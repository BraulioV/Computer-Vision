import cv2
import functions as fx

if __name__ == '__main__':
    tablero = cv2.imread('imagenes/Tablero1.jpg', flags=cv2.IMREAD_GRAYSCALE)
    fx.extract_harris_points(tablero, blockS = 3, kSize = 5)