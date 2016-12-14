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


# Los puntos han de ser en coordenadas homogéneas
def normalize(points):
    # Obtenemos la media de los puntos y su desviación
    # típica para normalizar los datos
    x_mean = np.mean(points, 0)
    y_mean = np.mean(points, 1)
    print(x_mean)
    s = np.std(points)

    # Creamos la matriz N para normalizar los puntos, esta
    # matriz tiene la forma:
    if len(points[0]) == 2:
        N = np.matrix([ [s, 0., -s*x_mean], [0., s, -s*y_mean], [0., 0., 1.] ])
    else:
        N = np.matrix([ [s, 0., 0., -s*x_mean], [0., s, 0., -s*y_mean], [0., 0., 0., 1.] ])

    N = np.linalg.inv(N)
    normalized_points = np.dot(N, points)
    normalized_points = normalized_points[0:normalized_points.shape[1]-1,:].T

    return normalized_points, N


def DLT_algorithm(real_points, projected_points, camera):
    # Normalizamos los puntos para mejorar el resultado
    # del algoritmo DLT
    N_matrix, normalized_points = normalize(real_points)
    # Recorremos todos los puntos 3D que tenemos
    # y generamos una matriz M con todos los puntos
    aux = []
    for i in range(normalized_points.shape[0]):
        x_i, y_i, z_i = normalized_points[i,0], normalized_points[i,1], normalized_points[i,2]
        u, v = projected_points[i,0], projected_points[i,1]
        aux.append([x_i, y_i, z_i, 1, 0, 0, 0, 0, -u*x_i, -u*y_i, -u*z_i, -u])
        aux.append([0, 0, 0, 1, x_i, y_i, z_i, 1, -v*x_i, -v*y_i, -v*z_i, -v])

    # Descomponemos la matriz
    U, s, V = np.linalg.svd(np.matrix(aux, dtype=np.float32))
    # Obtenemos los parámetros
    camera_estimated = V[-1,:]/V[-1,-1]
    camera_estimated.reshape(3,normalized_points.shape[1]+1)
    # Desnormalizamos
    camera_estimated = np.dot( np.dot( np.linalg.pinv(N_matrix), camera_estimated), N_matrix)
    camera_estimated = camera_estimated/camera_estimated[-1,-1]
    
    error = ((camera - camera_estimated)**2).mean(axis=None)

    return camera_estimated, error

