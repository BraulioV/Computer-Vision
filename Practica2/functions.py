import math
import cv2
import numpy as np

###############################################################################
#   EJERCICIO 1
###############################################################################
# Esta función realiza la exponencial
#                 x^2
#      (-0.5*-----------)
#   e^         sigma^2
#
from numpy.core.multiarray import einsum

fx = lambda x, sigma: math.exp(-0.5 * (x ** 2 / sigma ** 2))


# Devuelve el vector gaussiano de la máscara
def get_mask_vector(sigma):
    # Obtenemos el valor "límite" que puede tener un
    # píxel para que sea significativo
    limit_3sigma = fx(3 * sigma, sigma)
    # Obtenemos un array de valores discretos
    # para realizar la gaussiana
    aux = np.arange(math.floor(-3 * sigma), math.ceil(3 * sigma) + 1)
    # Rellenamos la máscara aplicando la exponencial
    mask = np.array([fx(i, sigma) for i in aux])
    # Despreciamos los valores menores a 3 sigma
    mask[mask < limit_3sigma] = 0
    # Normalizamos la máscara a 1
    mask = np.divide(mask, np.sum(mask))
    # Devolvemos la máscara
    return mask


def convolution(mask, img_array):
    # Es lo mismo que realizar esto:
    # np.convolve(img_array, mask, 'valid')
    return np.sum(mask * img_array)


def insert_img_into_other(img_src, pixel_left_top_row,
                          pixel_left_top_col,
                          img_dest, substitute=False):
    alt, anch = img_src.shape[:2]
    if not substitute:
        img_dest[pixel_left_top_row:alt+pixel_left_top_row,
                 pixel_left_top_col:anch+pixel_left_top_col] += img_src
    else:
        img_dest[pixel_left_top_row:alt + pixel_left_top_row,
                 pixel_left_top_col:anch + pixel_left_top_col] = img_src


# Border replicate => coge el último píxel y lo replica
# n_pixels veces
def border_replicate(img, n_pixels, alt, anch):
    img[0:n_pixels, ] = img[n_pixels,]
    img[n_pixels + alt:(alt + 2 * n_pixels), ] = img[-n_pixels-1,]
    for i in range(n_pixels):
        img[:, i] = img[:, n_pixels]
    for i in range(n_pixels + anch, img.shape[1]):
        img[:, i] = img[:, -n_pixels-1]

# Border reflect => refleja n_pixels del borde y los añade
# al borde por ambos lados
def border_reflect(img, n_pixels, alt, anch):
    # Borde superior
    img[0:n_pixels, ] = np.matrix(img[n_pixels:n_pixels * 2, ])[::-1]
    # Borde inferior
    img[n_pixels + alt:(alt + 2 * n_pixels), ] = \
        np.matrix(img[(-2 * n_pixels):(-n_pixels), ])[::-1]
    # Para invertir los bordes de los laterales, no podemos usar el "atajo" de [::-1] para
    # invertir la matriz. Para ello, usaremos la función fliplr de numpy que nos devuelve
    # la matriz con las columnas en orden inverso
    # Borde izquierdo
    img[:, 0:n_pixels] = np.fliplr(img[:, n_pixels:2 * n_pixels])
    # Borde derecho
    img[:, n_pixels + anch:img.shape[1]] = \
        np.fliplr(img[:, (-2 * n_pixels):(-n_pixels)])

# Border reflect 101 => similar al anterior, pero, refleja
# el borde en el borde contrario de la imagen
def border_reflect_101(img, n_pixels, alt, anch):
    # Borde superior
    img[0:n_pixels, ] = np.matrix(img[(-2 * n_pixels):(-n_pixels), ])[::-1]
    # Borde inferior
    img[n_pixels + alt:(alt + 2 * n_pixels), ] = \
        np.matrix(img[n_pixels:n_pixels * 2, ])[::-1]
    # Borde izquierdo
    img[:, 0:n_pixels] = np.fliplr(img[:, (-2 * n_pixels):(-n_pixels)])
    # Borde derecho
    img[:, n_pixels + anch:img.shape[1]] = \
        np.fliplr(img[:, n_pixels:2 * n_pixels])

# Border wrap => toma los últimos n_pixels de un borde de la
# imagen, y los añade tal cual al otro borde de la imagen
def border_wrap(img, n_pixels, alt, anch):
    # Borde superior
    img[0:n_pixels, ] = np.matrix(img[(-2 * n_pixels):(-n_pixels), ])
    # Borde inferior
    img[n_pixels + alt:(alt + 2 * n_pixels), ] = \
        np.matrix(img[n_pixels:n_pixels * 2, ])
    # Borde izquierdo
    img[:, 0:n_pixels] = img[:, (-2 * n_pixels):(-n_pixels)]
    # Borde derecho
    img[:, n_pixels + anch:img.shape[1]] = \
        img[:, n_pixels:2 * n_pixels]


# Border constant => a partir de una constante k, añade a los bordes
    # píxeles con valor k
def border_constant(img, n_pixels, alt, anch, k):
    # Borde superior
    img[0:n_pixels, ] = np.ones((n_pixels, img.shape[1]), dtype=np.float64) * k
    # Borde inferior
    img[n_pixels + alt:(alt + 2 * n_pixels), ] = \
        np.ones((n_pixels, img.shape[1]), dtype=np.float64) * k
    # Borde izquierdo
    img[:, 0:n_pixels] = np.ones((img.shape[0], n_pixels), dtype=np.float64) * k
    # Borde derecho
    img[:, n_pixels + anch:img.shape[1]] = \
        np.ones((img.shape[0], n_pixels), dtype=np.float64) * k


def extend(img_src, n_pixels, border_type, k):
    # Tomamos la altura y el ancho de la imagen original
    alt, anch = img_src.shape[:2]
    # Generamos una nueva matriz expandida a partir del
    # alto y ancho de la anterior
    new_extended_img = np.zeros((alt + 2 * n_pixels, anch + 2 * n_pixels), dtype=np.float64)

    insert_img_into_other(img_src=img_src, pixel_left_top_row=n_pixels,
                          pixel_left_top_col=n_pixels, img_dest=new_extended_img)

    if border_type == 0:    # Border Replicate
        border_replicate(new_extended_img, n_pixels, alt, anch)
    elif border_type == 1:  # Border Reflect
        border_reflect(new_extended_img, n_pixels, alt, anch)
    elif border_type == 2:  # Border Reflect 101
        border_reflect_101(new_extended_img, n_pixels, alt, anch)
    elif border_type == 3:  # Border wrap
        border_wrap(new_extended_img, n_pixels, alt, anch)
    elif border_type == 4:  # Border Constant
        border_constant(new_extended_img, n_pixels, alt, anch, k)

    return new_extended_img


def extend_image_n_pixels(img_src, n_pixels, border_type, k=0):

    if len(img_src.shape) != 3: # Imagen en escala de grises
        return extend(img_src, n_pixels, border_type, k)

    else:   # Imagen en color
        b_channel, g_channel, r_channel = cv2.split(img_src)

        b_channel_ext = extend(b_channel, n_pixels, border_type, k)
        g_channel_ext = extend(g_channel, n_pixels, border_type, k)
        r_channel_ext = extend(r_channel, n_pixels, border_type, k)

        return cv2.merge((b_channel_ext, g_channel_ext, r_channel_ext))

def derivate_img_on_x(mask, img_src, n_pixels):
    alt_ext, anch_ext = img_src.shape[:2]

    cv = convolution
    copy = np.array

    # Obtenemos una matriz auxiliar
    dx = copy(img_src, copy=True, dtype=np.float64)

    for i in range(n_pixels, alt_ext - n_pixels):
        for j in range(n_pixels, anch_ext - n_pixels):
            dx[i, j] = cv(mask, img_src[i, j - n_pixels:1 + j + n_pixels])

    return dx

def derivate_img_on_y(mask, img_src, n_pixels):
    alt_ext, anch_ext = img_src.shape[:2]

    cv = convolution
    copy = np.array

    # Obtenemos una matriz auxiliar
    dy = copy(img_src, copy=True, dtype=np.float64)

    for j in range(n_pixels, anch_ext - n_pixels):
        for i in range(n_pixels, alt_ext - n_pixels):
            dy[i, j] = cv(mask, img_src[i - n_pixels:1 + i + n_pixels, j])

    return dy

def get_derivates_of(img, sigma=1):
    # Extendemos los bordes de la imagen para hacer posible el uso de la
    # mascara sobre esta
    gaussian_mask = get_mask_vector(sigma)

    n_pixels = math.floor(gaussian_mask.size / 2)

    extended_image = extend_image_n_pixels(img, n_pixels, border_type = 0)

    return (derivate_img_on_x(gaussian_mask, extended_image, n_pixels),
            derivate_img_on_y(gaussian_mask, extended_image, n_pixels))


def convolution_grey_scale_img(mask, img, n_pixels):
    # Obtenemos el alto y ancho de la imagen
    alt_ext, anch_ext = img.shape[:2]

    cv = convolution
    copy = np.array

    # Obtenemos una matriz auxiliar
    aux = copy(img, copy=True,dtype=np.float64)

    for i in range(n_pixels, alt_ext - n_pixels):
        for j in range(n_pixels, anch_ext - n_pixels):
            aux[i, j] = cv(mask, img[i, j - n_pixels:1 + j + n_pixels])

    aux_2 = copy(aux, copy=True, dtype=np.float64)

    for j in range(n_pixels, anch_ext - n_pixels):
        for i in range(n_pixels, alt_ext - n_pixels):
            aux_2[i, j] = cv(mask, aux[i - n_pixels:1 + i + n_pixels, j])

    return aux_2[n_pixels:-n_pixels, n_pixels:-n_pixels]


def convolution_color_img(mask, img, n_pixels):
    # Obtenemos el alto y ancho de la imagen
    alt_ext, anch_ext = img.shape[:2]

    # Separamos los colores de la imagen en tres canales
    b_channel, g_channel, r_channel = cv2.split(img)

    cv = convolution
    copy_img = np.copy
    zeros = np.zeros

    # Obtenemos una matriz auxiliar por cada uno de
    # los canales de color de la imagen
    aux_r = zeros((alt_ext, anch_ext), dtype=np.float64)
    aux_g = zeros((alt_ext, anch_ext), dtype=np.float64)
    aux_b = zeros((alt_ext, anch_ext), dtype=np.float64)
    # Empezamos
    for i in range(n_pixels, alt_ext - n_pixels):
        for j in range(n_pixels, anch_ext - n_pixels):
            aux_b[i, j] = cv(mask, b_channel[i, j - n_pixels:1 + j + n_pixels])
            aux_g[i, j] = cv(mask, g_channel[i, j - n_pixels:1 + j + n_pixels])
            aux_r[i, j] = cv(mask, r_channel[i, j - n_pixels:1 + j + n_pixels])

    # Volvemos a tomar los valores de los bordes
    r_channel, g_channel, b_channel = copy_img(aux_r), copy_img(aux_g), copy_img(aux_b)
    # Y obtenemos nuevas matrices auxilares para terminar la convolucion
    aux2_r, aux2_g, aux2_b = copy_img(r_channel), copy_img(g_channel), copy_img(g_channel)

    for j in range(n_pixels, anch_ext - n_pixels):
        for i in range(n_pixels, alt_ext - n_pixels):
            aux2_b[i, j] = cv(mask, b_channel[i - n_pixels:1 + i + n_pixels, j])
            aux2_g[i, j] = cv(mask, g_channel[i - n_pixels:1 + i + n_pixels, j])
            aux2_r[i, j] = cv(mask, r_channel[i - n_pixels:1 + i + n_pixels, j])

    # Recomponemos la imagen
    result = cv2.merge((aux2_b, aux2_g, aux2_r))

    # Y la devolvemos
    return result[n_pixels:-n_pixels, n_pixels:-n_pixels]


####################################################################
# Parámetros de la función:
#  - im: imagen de entrada
#   - maskCovol: vector de la máscara
#   - out: imagen de salida es la que se devuelve en la función
#   - border_type: tipo de borde que se aplica, por defecto,
#      replicar el borde
def my_im_gauss_convolution(im, mask_convolution,
                            border_type=0):
    # Extendemos los bordes de la imagen para hacer posible el uso de la
    # mascara sobre esta
    n_pixels = math.floor(mask_convolution.size / 2)

    extended_image = extend_image_n_pixels(im, n_pixels, border_type)
    # Si la imagen está en blanco y negro,
    if len(im.shape) != 3:
        return convolution_grey_scale_img(mask_convolution,
                                          extended_image, n_pixels)
    else:
        return convolution_color_img(mask_convolution,
                                     extended_image, n_pixels)


def make_hybrid_image(img_lowF, img_highF,
                      smoothing_mask_sigma,
                      sharpering_mask_sigma,
                      show_images = False):
    # Obtenemos las máscaras gaussianas para trabajar
    # con las imágenes
    smoothing_mask = get_mask_vector(smoothing_mask_sigma)
    sharpering_mask = get_mask_vector(sharpering_mask_sigma)
    # Suavizamos la imagen que usaremos de base
    low_frecuencies = my_im_gauss_convolution(img_lowF, smoothing_mask, 0)
    # Guardamos una copia de la imagen para mostrarla más
    # adelante por pantalla
    low_frecuencies_img1 = np.copy(low_frecuencies)
    # Obtenemos las frecuencias bajas de la imagen que superpondremos a la base
    low_HF_aux = my_im_gauss_convolution(img_highF, sharpering_mask, 0)
    # Restamos la imagen original menos sus frecuencias bajas para obtener las frecuencias altas
    sharper = img_highF - low_HF_aux
    # y superponemos las imágenes
    insert_img_into_other(img_src=sharper, img_dest=low_frecuencies,
                          pixel_left_top_row=0, pixel_left_top_col=0)

    if show_images:
        show_img(generate_continous_canvas([low_frecuencies_img1, sharper, low_frecuencies]),
                 'Bajas frecuencias, altas frecuencias, imagen hibrida')

    return low_frecuencies


def subsample_image(img_src, subsample_factor = 2):
    alt, anch = img_src.shape[:2]
    # Obtenemos una máscara de suavizado
    smoothing_mask = get_mask_vector(1)
    # Suavizamos la máscara
    smoothed_img = my_im_gauss_convolution(im=img_src,mask_convolution=smoothing_mask)
    # Eliminamos las filas que queramos

    if len(img_src.shape) != 3:
        aux = smoothed_img[range(0, alt, subsample_factor)]
    else:
        # b_channel, g_channel, r_channel = cv2.split(aux)
        aux = smoothed_img[range(0, alt, subsample_factor),:]

    # Y devolvemos de la matriz anterior, todas las filas
    # y las columnas que queramos
    return aux[:, range(0,anch, subsample_factor)]


def generate_gaussian_pyramide(img_src, subsample_factor, n_levels):

    gaussian_pyramid = []

    gaussian_pyramid.append(img_src)

    for i in range(1, n_levels):
        # Realizamos el subsample
        gaussian_pyramid.append(subsample_image(img_src=gaussian_pyramid[-1],
                                     subsample_factor=subsample_factor))

    return gaussian_pyramid

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


def max_images_on_pyramidal_canvas(img_src, canvas_rows, subsample_factor):

    floor = math.floor
    # Obtenemos el número de píxeles disponibles en filas para
    # poder crear la imagen
    rows_available = canvas_rows
    # Obtenemos la altura de la imagen para operar con ella
    # y obtener el máximo de imágenes que pueden entrar en el canvas
    height = canvas_rows
    length = img_src.shape[1]
    # Anotamos la altura a la que se encuentra el último
    # píxel de la imagen añadido
    n_imgs = 0
    # Al final de la imagen, se deja un margen prudencial
    # de 10 píxeles
    while rows_available >= height and \
            (height > 10 or length > 10):

        height = floor(height/subsample_factor)
        length = floor(length/subsample_factor)
        rows_available -= height

        n_imgs += 1

    return n_imgs


def generate_new_pyramidal_canvas(img_src, times_to_show, subsample_factor = 2):

    alt, anch = img_src.shape[:2]
    # Generamos un canvas vacío dependiendo de si la imagen es
    # en color o en escala de grises
    if len(img_src.shape) != 3:
        canvas = np.zeros((alt, anch + math.ceil(anch / subsample_factor)),
                          dtype=np.float64) + 255
    else:
        canvas = np.zeros((alt, anch + math.ceil(anch / subsample_factor),3),
                          dtype=np.float64) + 255

    # insertamos la imagen original a la izquierda del canvas
    insert_img_into_other(img_src=img_src, img_dest=canvas,
                          pixel_left_top_row=0, pixel_left_top_col=0,
                          substitute=True)
    # fila en la que se insertará la siguiente imagen
    # a la que se le aplicará el subsampling
    row = 0
    # imagen copia de la original con la que se trabajará
    im = np.copy(img_src)
    # máximo número de imágenes que se pueden incrustar en pirámide
    max = max_images_on_pyramidal_canvas(img_src, canvas.shape[1], subsample_factor)
    if times_to_show > max:
        times_to_show = max

    pyramide = generate_gaussian_pyramide(img_src,subsample_factor,times_to_show)

    for i in range(1, times_to_show):
        # insertamos la imagen en el canvas
        insert_img_into_other(img_src=pyramide[i], img_dest=canvas,
                              pixel_left_top_row=row, pixel_left_top_col=anch,
                              substitute=True)
        # y anotamos en qué fila insertar la siguiente imagen
        row += pyramide[i].shape[0]

    return canvas

# Esta función genera un canvas en el que va insertando
# de forma continua las imágenes que se encuentran en
# la lista que le pasamos como parámetro
def generate_continous_canvas(list_imgs):
    length, height = 0, 0

    max_of = max
    # Obtenemos la anchura máxima del canvas
    # y la altura máxima que debe tener para que
    # pueda a
    color_imgs = False
    for i in list_imgs:
        alt, anch = i.shape[:2]
        length += anch
        height = max_of(height, alt)

        if len(i.shape) == 3:
            color_imgs = True

    # Diferenciamos entre imágenes en color o en escala
    # de grises, para crear un canvas u otro
    if not color_imgs:
        canvas = np.ones((height, length), dtype=np.float64)*255
    else:
        canvas = np.ones((height, length, 3), dtype=np.float64) * 255

    length = 0
    # Añadimos
    for i in list_imgs:
        insert_img_into_other(i, 0, length, canvas, substitute=True)
        length+=i.shape[1]


    return canvas

# Esta función muestra la imagen por pantalla, y asocia
# a la ventana el nombre que entra por parámetro.
# Esta imagen, es muy probable que tenga datos de tipo float64
# por lo que se hace un "cast" a unsigned int 8, que son
# valores que van de 0 a 255
def show_img(im, name):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

########################################################################################################################
########################################################################################################################
########################################################################################################################

harrisCriterio = lambda det, tr, k=0.04: det - k*(tr**2)

# Esta función comprueba si el centro del entorno es
# un máximo local o no
def local_maximun(environment):
    floor = math.floor
    height, width = environment.shape[:2]
    center = environment[floor(width/2),floor(height/2)]
    return center == np.min(environment)

def set_down_to_zero(environment):
    # Ponemos todos los elementos de la matriz e
    environment[:,] = 0


def get_local_maximun(imgs, index_mask, mask_size):
    # inicializamos una imagen binaria (0,255) para
    # representar los máximos locales de la imagen
    matrix = np.zeros(shape=imgs[0].shape,dtype=np.uint8)
    half_mask_size = math.floor(mask_size/2)
    # obtenemos los índices de los puntos que han sobrepasado
    # el umbral mínimo
    escala = 1
    img = 0
    for i in index_mask:
        rows = i[0]
        cols = i[1]
        imgaux = extend_image_n_pixels(img_src=imgs[img], border_type=4, n_pixels=mask_size)
        for k in range(len(rows)):
            left = rows[k]
            right = (rows[k]+mask_size)
            top = cols[k]
            down = (cols[k] + mask_size)
            if local_maximun(imgaux[left:right, top:down]):
                matrix[rows[k]*escala,cols[k]*escala] = 255

        img += 1
        escala *= 2

    return matrix


def extract_harris_points(img, blockS, kSize, thresdhold):
    # Extraemos las dimensiones de la imagen, para que,
    # en caso de que sean de tamaño impar, añadirle una fila
    # o columna según corresponda, para que, al reducir,
    # podamos recuperar fácilmente las coordenadas de los puntos
    # harris.
    alt, anch = img.shape[:2]
    # Ensanchamos una fila de la imagen
    if alt % 2 != 0 and anch % 2 == 0:
        aux = np.ones(shape=(alt+1, anch), dtype=np.uint8)
        insert_img_into_other(img_src=img, img_dest=aux, pixel_left_top_col=0,
                              pixel_left_top_row=0, substitute=True)

    # Ensanchamos una columna de la imagen
    elif alt % 2 == 0 and anch % 2 != 0:
        aux = np.ones(shape=(alt, anch+1), dtype=np.uint8)
        insert_img_into_other(img_src=img, img_dest=aux, pixel_left_top_col=0,
                              pixel_left_top_row=0, substitute=True)

    # Ensanchamos una fila y una columna de la imagen
    elif alt % 2 != 0 and anch % 2 != 0:
        aux = np.ones(shape=(alt+1, anch+1), dtype=np.uint8)
        insert_img_into_other(img_src=img, img_dest=aux, pixel_left_top_col=0,
                              pixel_left_top_row=0, substitute=True)
    # se queda igual que la original
    else:
        aux = np.copy(img)
    
    # obtenemos la pirámide gaussiana
    pyramide = generate_gaussian_pyramide(img_src=aux, subsample_factor=2, n_levels=3)

    eingen_vals_and_vecs = []
    strong_values = []

    for im in pyramide:
        # Obtenemos la matriz de con los autovalores de la matriz
        # y los respectivos autovectores para cada uno de los autovalores
        result =cv2.split(cv2.cornerEigenValsAndVecs(src=im.astype(np.uint8), blockSize=blockS, ksize=kSize))
        # Calculamos el determinante como el producto de los autovalores
        det = cv2.mulSpectrums(result[0], result[1], flags=cv2.DFT_ROWS)
        # Calculamos la traza como la suma de los autovalores
        trace = result[0] + result[1]
        # Realizamos la función de valoración de Harris
        eingen_vals_and_vecs.append(harrisCriterio(det, trace))
        # Y obtenemos los índices de los píxeles que sobrepasan el umbral mínimo
        strong_values.append(np.where(eingen_vals_and_vecs[-1] > thresdhold))

    # pasamos a eliminar los no máximos

    best_points = get_local_maximun(imgs=eingen_vals_and_vecs, index_mask=strong_values, mask_size=3)

    show_img(best_points, 'a')