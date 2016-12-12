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
    # Añadimos un cero por la izquierda y uno por la derecha respectivamente
    set1 = np.hstack((np.zeros(points2D.shape[0])[..., None], points2D))
    set2 = np.hstack((points2D, np.zeros(points2D.shape[0])[..., None]))
    # Y devolvemos un único conjunto de puntos
    return np.concatenate((set1, set2))


def project_points(points, camera):
    # Pasamos las coordenadas de los puntos a coordenadas homogéneas
    homogeneus_points = np.hstack((points, (np.ones(points.shape[0]))[...,None]))
    # Obtenemos una matriz vacía que serán las proyecciones
    # de los puntos al pasar por la cámara.
    projection = np.zeros(shape=(points.shape[0],2), dtype=np.float32)
    # Realizamos la multiplicación
    #    xy' = P * xy
    for i in range(homogeneus_points.shape[0]):
        point = np.dot(camera,homogeneus_points[i].T)
        projection[i,0] = point[0]/point[2]
        projection[i,1] = point[1]/point[2]

    # Devolvemos las proyecciones de los puntos
    return homogeneus_points, projection


#def DLT_algorithm(3D_points, projected_points):
    # Recorremos todos los puntos 3D que tenemos

    #for i in range(3D_points.shape[0]):
        #pass

    
