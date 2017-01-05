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

        
def get_matches(image1, image2):
     # Vamos a inicializar un dectector ORB 
    # y un detector BRISK, y dejaremos aquel que obtenga
    # más puntos
    orb_detector = cv2.ORB_create()
    brisk_detector = cv2.BRISK_create()
    # Buscamos los keypoints y los descriptores de ambas
    # imágenes haciendo uso de ORB
    keyP1_orb, des1_orb = orb_detector.detectAndCompute(image1,None)
    keyP2_orb, des2_orb = orb_detector.detectAndCompute(image2,None)
    # Buscamos los keypoints y los descriptores de ambas
    # imágenes haciendo uso de BRISK
    keyP1_brisk, des1_brisk = brisk_detector.detectAndCompute(image1,None)
    keyP2_brisk, des2_brisk = brisk_detector.detectAndCompute(image2,None)
    # Inicializamos el BFMatcher con la norma Hamming para ORB y BRISK
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Hacemos matching entre los descriptores de ORB
    matches_orb = bf.match(des1_orb, des2_orb)
    # y para los de BRISK
    matches_brisk = bf.match(des1_brisk, des2_brisk)
    # Calculamos cuál ha obtenido más puntos en
    # correspondencias y nos quedamos con sus matches
    if len(matches_orb) > len(matches_brisk):
        matches = matches_orb
        kp1, kp2 = keyP1_orb, keyP2_orb
        des1, des2 = des1_orb, des2_orb
        print("Puntos en corresponencia usando ORB")
        print("Total de puntos: ", len(matches))
        img_match = cv2.drawMatches(img1 = image1, keypoints1 = keyP1_orb, 
                                    img2 = image2, keypoints2 = keyP2_orb, 
                                    matches1to2 = matches, outImg = None, flags=2)
        sorted_kp_img_match = cv2.drawMatches(img1 = image1, keypoints1 = keyP1_orb, 
                                    img2 = image2, keypoints2 = keyP2_orb, 
                                    matches1to2 = sorted(matches, key = lambda x:x.distance)[0:int(len(matches)*0.15)], 
                                    outImg = None, flags=2)
    else:
        matches = matches_brisk
        kp1, kp2 = keyP1_brisk, keyP2_brisk
        des1, des2 = des1_brisk, des2_brisk
        print("Puntos en corresponencia usando BRISK")
        print("Total de puntos: ", len(matches))
        img_match = cv2.drawMatches(img1 = image1, keypoints1 = keyP1_brisk, 
                                    img2 = image2, keypoints2 = keyP2_brisk, 
                                    matches1to2 = matches, outImg = None, flags=2)
        sorted_kp_img_match = cv2.drawMatches(img1 = image1, keypoints1 = keyP1_brisk, 
                                    img2 = image2, keypoints2 = keyP2_brisk, 
                                    matches1to2 = sorted(matches, key = lambda x:x.distance)[0:int(len(matches)*0.15)], 
                                    outImg = None, flags=2)
    
    fx.show_img(img_match, 'Todos los puntos en corresponencias')    
    fx.show_img(sorted_kp_img_match, 'El 15% de mejores puntos en corresponencias')  
    
    return matches, kp1, des1, kp2, des2
        
def estimate_fundamental_matrix_from(image1, image2):
    # Obtenemos los puntos en correspondencias
    matches, kp1, des1, kp2, des2 = get_matches(image1, image2)
    img_points1 = []
    img_points2 = []
    # Recuperamos las coordenadas de los puntos en correspondencias:
    for match in matches:
        img_points1.append(kp1[match.queryIdx].pt)
        img_points2.append(kp2[match.trainIdx].pt)
        
    img_points1 = np.array(img_points1, dtype=np.int32)
    img_points2 = np.array(img_points2, dtype=np.int32)
    # Psamos a obtener la matriz fundamental con el 
    # algoritmo de los 8 puntos usando RANSAC
    fundamental_mat, mask = cv2.findFundamentalMat(points1 = img_points1, 
                                             points2 = img_points2,
                                             method = cv2.FM_8POINT + cv2.FM_RANSAC, 
                                             param1 = 10**-2,
                                             param2 = 0.9999999)
    
    # Descartamos los puntos que son outliers
    img_points1 = img_points1[mask.ravel()==1]
    img_points2 = img_points2[mask.ravel()==1]
    print("Matriz fundamental F:\n",fundamental_mat)
    
    return fundamental_mat, img_points1, img_points2
    
    
def draw_epilines(image1, img_points1, image2, img_points2, epilines):
    # Pasamos las imágenes de escala de grises a color para
    # poder representar las líneas epipolares de una manera
    # más clara
    aux_img1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    aux_img2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    for i in range(len(epilines)):
        # Generamos un color aleatorio
        line_color = tuple(np.random.randint(0,255,3).tolist())
        init_point = (0, int(-epilines[i][2]/epilines[i][1]))
        end_point = (image1.shape[1], int(-(epilines[i][2]+epilines[i][0]*image1.shape[1])/epilines[i][1]))
        # al tener definidos los dos puntos, podemos 
        # crear la línea epipolar que pasa por esos 
        # dos puntos
        aux_img1 = cv2.line(img = aux_img1, pt1 = init_point, 
                            pt2 = end_point, color = line_color,
                            thickness = 2)
        aux_img1 = cv2.circle(img = aux_img1, 
                              center=tuple(img_points1[i].astype(np.int64)), 
                              radius = 3, color = line_color)
        aux_img2 = cv2.circle(img = aux_img2, 
                              center=tuple(img_points2[i].astype(np.int64)), 
                              radius = 3, color = line_color)
    
    return aux_img1, aux_img2
    
def show_epilines(image1, img_points1, image2, img_points2, fundamental_mat):
    # Obtenemos las epilineas de ambas imágenes
    epipolarline_img1 = cv2.computeCorrespondEpilines(img_points1, 1, fundamental_mat).reshape(-1,3)
    epipolarline_img2 = cv2.computeCorrespondEpilines(img_points2, 2, fundamental_mat).reshape(-1,3)
    # Dibujamos las líneas epipolares
    # Lineas epipolares de la primera imagen sobre la segunda
    epip1, epip2 = draw_epilines(image1, img_points1, image2, img_points2, epipolarline_img2)
    canvas1 = np.zeros((epip1.shape[0],epip1.shape[1]+epip2.shape[1], 3), dtype = np.uint8)
    fx.insert_img_into_other(img_src=epip2, img_dest=canvas1,
                          pixel_left_top_row=0, pixel_left_top_col=0,
                          substitute=True)
    fx.insert_img_into_other(img_src=epip1, img_dest=canvas1,
                          pixel_left_top_row=0, pixel_left_top_col=epip1.shape[1],
                          substitute=True)
    
    # Lineas epipolares de la segunda imagen sobre la primera
    epip3, epip4 = draw_epilines(image2, img_points2, image1, img_points1, epipolarline_img1)
    canvas2 = np.zeros((epip3.shape[0],epip3.shape[1]+epip4.shape[1], 3), dtype = np.uint8)
    fx.insert_img_into_other(img_src=epip3, img_dest=canvas2,
                          pixel_left_top_row=0, pixel_left_top_col=0,
                          substitute=True)
    fx.insert_img_into_other(img_src=epip4, img_dest=canvas2,
                          pixel_left_top_row=0, pixel_left_top_col=epip1.shape[1],
                          substitute=True)
    # Mostramos ambas imágenes
    fx.show_img(canvas1, 'Todos los puntos en corresponencias')
    fx.show_img(canvas2, 'Todos los puntos en corresponencias')
    
    return epipolarline_img1, epipolarline_img2
    
# Para calcular la bondad de F, usaremos el error
# epipolar simétrico
def epipolar_line_error(pts_im1, pts_im2, line_1, line_2):
    
    abs_value = math.fabs
    sqrt = math.sqrt
    # Función que calcula la distancia de un punto a una recta
    dst = lambda line, point: abs_value((line[0]*point[0] + line[1]*point[1] 
                                         + line[2])/sqrt(line[0]**2 + line[1]**2))
    
    dst_pt1_to_line1 = []
    dst_pt2_to_line2 = []
    # Recorremos los puntos calculando las distancias 
    # del punto a la línea
    for i in range(len(pts_im1)):
        dst_pt1_to_line1.append(dst(line_1[i], pts_im2[i]))
        dst_pt2_to_line2.append(dst(line_2[i], pts_im1[i]))
    
    # Calculamos el error:
    F_error = (np.mean(dst_pt1_to_line1) + np.mean(dst_pt2_to_line2))/2
    print("Error de F: ", F_error)
    return F_error
