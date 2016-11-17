import math
import cv2
import random
import numpy as np
from functions import *

harrisCriterio = lambda det, tr, k=0.04: det - k*(tr**2)

# Esta función comprueba si el centro del entorno es
# un máximo local o no
def local_maximun(environment):
    floor = math.floor
    height, width = environment.shape[:2]
    center = environment[floor(width/2),floor(height/2)]
    return center == np.max(environment)


def get_local_maximun(imgs, index_mask, mask_size):
    # obtenemos los índices de los puntos que han sobrepasado
    # el umbral mínimo
    img = 0
    xy_bests_points = []
    harrisV_bests_points = []
    for i in index_mask:
        # Arrays para almacenar las coordenadas en X y en Y
        coord_x = []
        coord_y = []
        rows = i[0]
        cols = i[1]
        # Extendemos la imagen para poder captar fácilmente
        # los máximos locales de los bordes
        imgaux = extend_image_n_pixels(img_src=imgs[img],
                                       border_type=4,
                                       n_pixels=mask_size)
        # Array para almacenar los máximos locales almacenados
        maxHs = []
        for k in range(len(rows)):
            # Obtenemos las cuatro esquinas de la región a analizar
            # La región la declaramos con las misma coordenadas que
            # obtuvimos de la imagen original. Esto se debe a que si
            # ensanchamos la imagen, estas coordenadas estarán
            # desplazadas, floor(-n/2) píxeles en el eje X y en el eje Y
            left = rows[k]
            right = (rows[k]+mask_size)
            top = cols[k]
            down = (cols[k] + mask_size)
            # Y comprobamos si la región contiene en su centro un máximo local
            if local_maximun(imgaux[left:right, top:down]):
                # si lo es, almacenamos su posición y su valor harris
                coord_x.append(rows[k])
                coord_y.append(cols[k])
                maxHs.append(imgs[img][rows[k],cols[k]])

        # Insertamos los puntos máximos y su valor en la lista
        xy_bests_points.append(np.array([np.array(coord_x), np.array(coord_y)]))
        # xy_bests_points.append(maxls_xy)
        harrisV_bests_points.append(maxHs)
        img += 1

    return xy_bests_points, harrisV_bests_points


def prepare_img_to_harris_points(img):
    # Extraemos las dimensiones de la imagen, para que,
    # en caso de que sean de tamaño impar, añadirle las filas
    # o columnas que correspondan, para que, al reducir,
    # podamos recuperar fácilmente las coordenadas de los puntos
    # harris.
    alt, anch = img.shape[:2]

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aux=[]
    # Ensanchamos una fila de la imagen
    if alt % 4 != 0 and anch % 4 == 0:
        aux = np.ones(shape=(alt + alt % 4, anch), dtype=np.uint8)
        insert_img_into_other(img_src=gray_img, img_dest=aux, pixel_left_top_col=0,
                              pixel_left_top_row=0, substitute=True)

    # Ensanchamos una columna de la imagen
    elif alt % 4 == 0 and anch % 4 != 0:
        aux = np.ones(shape=(alt, anch + anch % 4), dtype=np.uint8)
        insert_img_into_other(img_src=gray_img, img_dest=aux, pixel_left_top_col=0,
                              pixel_left_top_row=0, substitute=True)

    # Ensanchamos una fila y una columna de la imagen
    elif alt % 4 != 0 and anch % 4 != 0:
        aux = np.ones(shape=(alt + alt % 4, anch + anch % 4), dtype=np.uint8)
        insert_img_into_other(img_src=gray_img, img_dest=aux, pixel_left_top_col=0,
                              pixel_left_top_row=0, substitute=True)
    # se queda igual que la original
    else:
        aux = np.copy(gray_img)

    return aux

# Obtiene los valores de harris y los indices
# de los puntos que superan el umbral
def get_eigenVals_and_eigenVecs(pyramide, thresdhold, blockS, kSize):
    eingen_vals_and_vecs = []
    strong_values = []

    for im in pyramide:
        # Obtenemos la matriz de con los autovalores de la matriz
        # y los respectivos autovectores para cada uno de los autovalores
        result = cv2.split(cv2.cornerEigenValsAndVecs(src=im.astype(np.uint8),
                                                      blockSize=blockS, ksize=kSize))
        # Calculamos el determinante como el producto de los autovalores
        det = cv2.mulSpectrums(result[0], result[1], flags=cv2.DFT_ROWS)
        # Calculamos la traza como la suma de los autovalores
        trace = result[0] + result[1]
        # Realizamos la función de valoración de Harris
        eingen_vals_and_vecs.append(harrisCriterio(det, trace))
        # Y obtenemos los índices de los píxeles que sobrepasan el umbral mínimo
        strong_values.append(np.where(eingen_vals_and_vecs[-1] > thresdhold))

    return eingen_vals_and_vecs, strong_values

def get_best_points(img_points, xy_points, harrisV, n_points):
    # Pasamos a poner a 1 los puntos con máximos locales
    it = 0
    floor = math.floor
    # Esto representa el porcentaje de puntos que tomaremos de cada escala
    # tomando del primer nivel el 70%, del segundo el 20% y del último el 10%
    percentages = [.7, .2, .1]
    escala = 1
    selected_points = []
    # Empezamos a recorrer los puntos que hemos extraído como máximos locales
    for points in harrisV:
        # ordenamos los puntos. Como argsort los da ordenados
        # de menor a mayor, invertimos el vector para obtenerlos
        # de mayor a menor.
        index = np.argsort(points)[::-1]
        # tomamos las coordenadas del % de puntos mejores
        coord_xy = [xy_points[it][0][index[0:floor(n_points * percentages[it])]],
                    xy_points[it][1][index[0:floor(n_points * percentages[it])]]]

        # Almacenamos los mejores puntos de cada escala
        # ya ordenados, y sin
        selected_points.append(np.array(coord_xy).T)

        # Almacenamos los puntos en una lista para poder
        # dibujar los círculos
        coordinates_for_circles = [xy_points[it][1][index[0:floor(n_points * percentages[it])]] * escala,
                                        xy_points[it][0][index[0:floor(n_points * percentages[it])]] * escala,]
        # coordinates_for_circles.append(coord_xy*escala)
        # y los ponemos a 1
        img_points[coordinates_for_circles[::-1]] = 255
        it += 1
        escala *= 2

    show_img(img_points, "Primeros puntos seleccionados")

    return selected_points

def show_binary_points_on_img(img, coordinates_for_circles):
    # Pasamos a pintar los puntos seleccionados en la imagen original
    # colocando un círculo de color rojo sobre esta, con radio proporcional
    # a la escala en la que se ha obtenido este punto
    circle_radius = [30, 15, 5]
    it = 0
    aux = np.copy(img)
    for coordinates in coordinates_for_circles:
        for i in range(len(coordinates[0])):
            cv2.circle(img = aux, center = (coordinates[0][i],coordinates[1][i]),
                       radius = circle_radius[it], color=0)
        it += 1
    show_img(aux, "Mejores puntos seleccionados")

def refine_points(pyramide, selected_points):
    refined_points = []
    it = 0

    for img in pyramide:
        float_esquinas = np.array(selected_points[it], dtype=np.float32).copy()
        cv2.cornerSubPix(image=img.astype(np.float32), corners=float_esquinas,
                         winSize=(5, 5), zeroZone=(-1, -1),
                         criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
        it += 1
        refined_points.append(float_esquinas)

    return refined_points


def detect_angles(pyramide, refined_points):
    # inicializamos la lista con los ángulos
    angles = []
    # y recorremos las tres escalas
    for scale in range(3):
        dx_img, dy_img = get_derivates_of(img=pyramide[scale], sigma=4.5)
        # Obtenemos los indices para poder
        indices = np.array(refined_points[scale].T, dtype=int)
        # Calculamos los ángulos de forma vectorizada, donde
        # el ángulo es:
        #           ang = atan( dy/dx )
        angles.append((np.arctan2(dy_img[indices[0], indices[1]],
                                  dx_img[indices[0], indices[1]])))

    # Devolvemos los ángulos
    return angles


def show_result(img, refined_points, angles):
    aux2 = np.copy(img)
    floor = math.floor
    sin = np.sin
    cos = np.cos
    radio = [5, 10, 20]
    colors = [(175, 0, 0), (0, 175, 0), (0, 0, 175)]
    size = 1
    for scale in range(3):
        # for i in random.sample(range(len(refined_points[scale])), 100):
        for i in range(len(refined_points[scale])):
            punto = refined_points[scale][i].astype(np.int) * size
            angle = angles[scale][i] * 180/np.pi
            cv2.circle(img=aux2, center=(punto[1], punto[0]),
                       radius=radio[scale], color=colors[scale],
                       thickness=1)
            cv2.arrowedLine(img=aux2, pt1=(punto[1], punto[0]),
                            pt2=(punto[1] + floor(cos(angle) * radio[scale]),
                                 punto[0] + floor(sin(angle) * radio[scale])),
                            color=colors[scale])
        size *= 2

    show_img(aux2, 'Puntos y ángulos')

def extract_harris_points(img, blockS, kSize, thresdhold, n_points = 1500,
                          show_selected_points = True,
                          show_best_points = True):
    #######################################
    # Apartado a: extrare lista potencial
    # de puntos Harris
    #######################################
    alt, anch = img.shape[:2]
    aux = prepare_img_to_harris_points(img)
    # obtenemos la pirámide gaussiana
    pyramide = generate_gaussian_pyramide(img_src=aux, subsample_factor=2, n_levels=3)

    # Obtenemos los autovalores y autovectores, junto
    # con los puntos que superan el umbral
    eingen_vals_and_vecs, strong_values = \
        get_eigenVals_and_eigenVecs(pyramide, thresdhold, blockS, kSize)

    # pasamos a eliminar los no máximos
    xy_points, harrisV = get_local_maximun(imgs=eingen_vals_and_vecs,
                                           index_mask=strong_values,
                                           mask_size=3)
    # inicializamos una imagen binaria (0,255) para
    # representar los máximos locales de la imagen
    img_points = np.zeros(shape=img.shape,dtype=np.uint8)
    # Obtenemos los mejores puntos para cada uno de
    # los niveles
    selected_points = get_best_points(img_points, xy_points, harrisV, n_points)

    # if show_selected_points:
    #     show_binary_points_on_img(img_points, coordinates_for_circles)

    #######################################
    # Apartado b, refinar las coordenadas
    #######################################
    refined_points = refine_points(pyramide, selected_points)

    ####################################
    # Apartado c, detectar orientacion
    ####################################
    # Eliminamos aquellos puntos que puedan haberse excedido del
    # tamaño de la imagen
    refined_points[0] = np.delete(refined_points[0],
                                  np.where(refined_points[0][:, 0] > alt), 0)
    refined_points[0] = np.delete(refined_points[0],
                                  np.where(refined_points[0][0, :] > anch), 0)

    # Obtenemos los ángulos de los gradientes
    angles =  detect_angles(pyramide, refined_points)
    # y mostramos el resultado si procede
    if show_best_points:
        show_result(img, refined_points, angles)

    return refined_points, angles

######################################################################
def AKAZE_descriptor_matcher(img1, img2, use_KAZE_detector = False,
                             show_matches = True):
    # KAZE detector
    if not use_KAZE_detector:
        detector = cv2.AKAZE_create()
    else:
        detector = cv2.KAZE_create()
    # Obtenemos las imágenes en blanco y negro
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # detectamos y computamos los keypoints y los descriptores
    # los almacenamos. Estos descriptres serán descriptores SIFT
    keypoints1, descriptors1 = detector.detectAndCompute(image=img1_gray, mask=None)
    keypoints2, descriptors2 = detector.detectAndCompute(image=img2_gray, mask=None)
    # Creamos el BFmatcher (Brute Force) que usará validación cruzada
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
    # Detectamos las correspondencias o matches
    matches = bf.match(descriptors1,descriptors2)
    # Y las ordenamos según la distancia
    # matches = sorted(matches, key=lambda x: x.distance)
    if show_matches:
        match_img = cv2.drawMatches(img1 = img1, keypoints1=keypoints1,
                                    img2 = img2, keypoints2=keypoints2,
                                    matches1to2=matches[:50], outImg=None, flags=4)

        show_img(match_img, "Correspondencias")

    return [keypoints1, descriptors1], [keypoints2, descriptors2], matches


########################################################################
# Ejercicio 3
########################################################################

def create_mosaico(imgs_list):
    length, height = 0, 0
    max_of = max
    for i in imgs_list:
        alt, anch = i.shape[:2]
        length += anch
        height = max_of(height, alt)

        if len(i.shape) == 3:
            color_imgs = True

    mosaico = np.zeros(shape=(length, height), dtype=np.uint8)


def create_two_mosaico(img1, img2, error=5.0):
    kp_dsp1, kp_dsp2, matches = AKAZE_descriptor_matcher(img1, img2,show_matches=False)

    src_points = np.float32([kp_dsp2[0][point.trainIdx].pt for point in matches])
    dest_points = np.float32([ kp_dsp1[0][point.queryIdx].pt for point in matches])



    H, boolean = cv2.findHomography(srcPoints=src_points,dstPoints=dest_points, method=cv2.RANSAC,
                              ransacReprojThreshold=error)

