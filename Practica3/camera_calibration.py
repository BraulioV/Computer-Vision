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
    while not np.linalg.det(P_cam[0:3,0:3]):
        P_cam = np.random.rand(3,4)        
    
    return P_cam


def generate_points():
    # Generamos los valores de x1 y x2
    x1 = np.arange(start = 0, stop = 1,
                   step = 0.1, dtype=np.float32)
    x2 = np.arange(start = 0, stop = 1,
                   step = 0.1, dtype=np.float32)
    # Obtenemos una combinación de los puntos que hemos obtenido
    points2D = np.concatenate(np.array(np.meshgrid(x1,x2)).T)

    set1 = np.hstack((np.zeros(points2D.shape[0])[..., None], points2D))
    set2 = np.hstack((points2D, np.zeros(points2D.shape[0])[..., None]))

    return np.concatenate((set1, set2))


def project_points(points, camera):
    projection = np.zeros(shape=points.shape, dtype=np.float32)
    homogeneus = np.hstack((points, (np.ones(points.shape[0]))[...,None]))
    for i in range(points.shape[0]):
        projection[i] = (camera*homogeneus_points[i].T)

    return projection

#camera = generate_Pcamera()
#points = generate_points()

#projection = project_points(points, camera)
