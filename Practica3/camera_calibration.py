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

    P_cam = P_cam / P_cam[-1,-1]
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
    set1 = np.hstack((np.zeros(points2D.shape[0])[...,None], points2D))
    set2 = np.hstack((points2D, np.zeros(points2D.shape[0])[...,None]))
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
def normalize(points, dim):
    # Obtenemos la media de los puntos y su desviación
    # típica para normalizar los datos
    points_mean = np.mean(points, 0)
    s = np.std(points[:,0:points.shape[1]-1])

    # Creamos la matriz N para normalizar los puntos, esta
    # matriz tiene la forma:
    if dim == 2:
        N = np.matrix([ [s, 0, -s*points_mean[0]], [0, s, -s*points_mean[1]], [0, 0, 1] ])
    else:
        N = np.matrix([[s, 0, 0, -s*points_mean[0]], [0, s, 0, -s*points_mean[1]], [0, 0, s, -s*points_mean[2]], [0, 0, 0, 1]])

    N = np.linalg.inv(N)
    normalized_points = np.dot(N, points.T)
    normalized_points = normalized_points[0:dim,:].T
    
    return N, normalized_points


# Algoritmo DLT para obtener una cámara estimada a partir
# de los puntos en el mundo y los puntos de la retina
def DLT_algorithm(real_points, projected_points, camera):
    # Normalizamos los puntos para mejorar el resultado
    # del algoritmo DLT
    N_matrix, normalized_points = normalize(real_points, 3)
    homogeneus_proj_pt = np.hstack((projected_points, (np.ones(projected_points.shape[0]))[...,None]))
    N_matrix2d, norm_points_2d = normalize(homogeneus_proj_pt, 2)
    # Recorremos todos los puntos 3D que tenemos
    # y generamos una matriz M con todos los puntos
    aux = []
    for i in range(normalized_points.shape[0]):
        x_i, y_i, z_i = normalized_points[i,0], normalized_points[i,1], normalized_points[i,2]
        u, v = norm_points_2d[i,0], norm_points_2d[i,1]
        aux.append([x_i, y_i, z_i, 1, 0, 0, 0, 0, -u*x_i, -u*y_i, -u*z_i, -u])
        aux.append([0, 0, 0, 0, x_i, y_i, z_i, 1, -v*x_i, -v*y_i, -v*z_i, -v])

    # Descomponemos la matriz
    U, s, V = np.linalg.svd(np.array(aux, dtype=np.float64))
    # Obtenemos los parámetros
    camera_estimated = V[-1,:]/V[-1,-1]
    camera_estimated = np.matrix(camera_estimated).reshape(3,4)
    # Desnormalizamos
    camera_estimated = np.dot(np.dot(np.linalg.pinv(N_matrix2d), camera_estimated), N_matrix)
    camera_estimated = camera_estimated/camera_estimated[-1,-1]
    # Calculamos el error de la cámara estimada
    error = np.linalg.norm(x=(camera - camera_estimated), ord=None)
    
    return camera_estimated, error


def calibrate_camera_from(images, use_lenss = False, alpha = 1):
    valids = []
    size = (13, 12)
    # Seleccionamos los flags que vamos a usar
    cv2_flags =  cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_FILTER_QUADS
    # Creamos nosotros las coordenadas del mundo para poder
    # compararlas con los puntos de las imágenes 
    # y generar la cámara
    world_points =  np.zeros((13*12,3), np.float32)
    world_points[:,:2] = np.mgrid[0:13,0:12].T.reshape(-1,2)
    
    for img in images:
        valids.append(cv2.findChessboardCorners(img, size, flags=cv2_flags))
        if valids[-1][0]:
        # Si la imagen es válida, procedemos a refinar
        # los puntos con cornerSubPix
            cv2.cornerSubPix(image=img.astype(np.float32), corners=valids[-1][1],
                             winSize=(5, 5), zeroZone=(-1, -1),
                             criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_COUNT, 30, 0.001))
    # Coordenadas de las imágenes seleccionadas
    coordinates = []
    worldP = []
    valid_images = []
    for i in range(0,len(valids)):
        # Si es un punto válido:
        if valids[i][0]:
            # Mostramos el patrón de puntos encontrado
            fx.show_img(cv2.drawChessboardCorners(image = images[i], 
                                                  patternSize = size,
                                                  corners = valids[i][1], 
                                                  patternWasFound = valids[i][0]),
                        "imagen "+str(i))
            # Almacenamos las coordenadas de los puntos que forman el
            # patrón para calibrar la cámara
            coordinates.append(valids[i][1])
            # Las coordenadas del mundo para formar las correspondencias
            worldP.append(world_points)
            # Y guardamos las imágenes válidas
            valid_images.append(images[i])
    
    # Tras esto, llamamos a calibrateCamera para calibrar la
    # cámara a partir de las coordenadas del mundo y las del 
    # patrón
    reprojection_error, camera, distorsion_coefs, rotation_vecs, translation_vecs = cv2.calibrateCamera(worldP, 
                                                           coordinates, 
                                                           valid_images[-1].shape[::-1],
                                                           None,None)
    print("reprojection_error = ", reprojection_error)
    print("camera = \n", camera)
    print("distorsion coeffs = ", distorsion_coefs)
    print("rotation vecs = \n", rotation_vecs)
    print("tvecs = \n", translation_vecs)
    
    if use_lenss:
        # si el parámetro use_lenss está activo, vamos a proceder
        # a quitar la distorsión producida por las lentes. Primero
        # refinamos la cámara obtenida anteriormente
        height, width = valid_images[-1].shape[:2]
        # Devolvemos la cámara y el rectángulo óptimo de píxeles para
        # evitar la distorsión
        ref_cam, valid_rectangle = cv2.getOptimalNewCameraMatrix(camera, distorsion_coefs, 
                                                                 (width, height), alpha, 
                                                                 (width, height))
        # Una vez que hemos obtenido la cámara refinada
        # pasamos a rectificar la distorsión.
        correct_image = cv2.undistort(src=valid_images[-1], cameraMatrix=camera, 
                                   distCoeffs=distorsion_coefs, dst=None,
                                   newCameraMatrix=ref_cam)
        # Obtenemos por separado los valores de la cuaterna
        # para trabajar más fácilmente.
        print("refined camera = \n", ref_cam)
        fx.show_img(correct_image, 'Resultado con lentes')    
