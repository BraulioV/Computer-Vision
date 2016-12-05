import cv2
import math
import numpy as np
import functions as fx
############################################################
#                       Ejercicio 1
# Estimación de la matriz de una cámara a partir
# del conjunto de puntos en correspondencias
############################################################

def generate_Pcamera():
    # La matriz cámara tiene la siguiente estructura:
    #              P =  K[R | T]
    # donde det(R) != 0, al igual que det(K) != 0, por
    # lo que podemos hacer que P = [KR | KT] = [M | M_4]

    # Generamos una matriz con valores aleatorios en el
    # intervalo [0,1)
    P_cam = np.random.rand(3,4)

    # Comprobamos si det(M) != 0. En caso de que no lo
    # sea, volvemos a generar una nueva matriz cámara.
    while not np.linalg.det(P[0:3,0:3]):
        P_cam = np.random.rand(3,4)        
    
    return P_cam


    
