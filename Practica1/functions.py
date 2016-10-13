import cv2
import math
import numpy as np

###############################################################################
#   EJERCICIO 1
###############################################################################
# Esta función realiza la exponencial
#                 x^2
#      (-0.5*-----------)
#   e^         sigma^2
#
fx = lambda x, sigma: math.exp(-0.5 * (x ** 2 / sigma ** 2))


# Devuelve el vector gaussiano de la máscara
def get_mask_vector(sigma):
    # Obtenemos el valor "límite" que puede tener un
    # píxel para que sea significativo
    limit_3sigma = fx(3 * sigma, sigma)
    # Obtenemos un array de valores discretos
    # para realizar la gaussiana
    aux = np.arange(math.floor(-3 * sigma), math.ceil(3 * sigma )+ 1)
    # Rellenamos la máscara aplicando la exponencial
    mask = np.array([fx(i, sigma) for i in aux])
    # Despreciamos los valores menores a 3 sigma
    mask[mask < limit_3sigma] = 0
    # Normalizamos la máscara a 1
    mask = np.divide(mask, np.sum(mask))
    # Devolvemos la máscara
    return mask


def convolution(mask, img_array):
    return np.sum(img_array * mask)


def insert_img_into_other(img_src, pixel_left_top_row, pixel_left_top_col, img_dest):
    alt, anch = img_src.shape[:2]
    img_dest[pixel_left_top_row:alt+pixel_left_top_row,
             pixel_left_top_col:anch+pixel_left_top_col] += img_src

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
    img[0:n_pixels, ] = np.ones((n_pixels, img.shape[1]), dtype=np.uint8) * k
    # Borde inferior
    img[n_pixels + alt:(alt + 2 * n_pixels), ] = \
        np.ones((n_pixels, img.shape[1]), dtype=np.uint8) * k
    # Borde izquierdo
    img[:, 0:n_pixels] = np.ones((img.shape[0], n_pixels), dtype=np.uint8) * k
    # Borde derecho
    img[:, n_pixels + anch:img.shape[1]] = \
        np.ones((img.shape[0], n_pixels), dtype=np.uint8) * k


def extend(img_src, n_pixels, border_type, k):
    # Tomamos la altura y el ancho de la imagen original
    alt, anch = img_src.shape[:2]
    # Generamos una nueva matriz expandida a partir del
    # alto y ancho de la anterior
    new_extended_img = np.zeros((alt + 2 * n_pixels, anch + 2 * n_pixels), dtype=np.uint8)

    insert_img_into_other(img_src=img_src, pixel_left_top_row=n_pixels,
                          pixel_left_top_col=n_pixels, img_dest=new_extended_img)

    if border_type == 0:
        border_replicate(new_extended_img, n_pixels, alt, anch)
    elif border_type == 1:
        border_reflect(new_extended_img, n_pixels, alt, anch)
    elif border_type == 2:
        border_reflect_101(new_extended_img, n_pixels, alt, anch)
    elif border_type == 3:
        border_wrap(new_extended_img, n_pixels, alt, anch)
    elif border_type == 4:
        border_constant(new_extended_img, n_pixels, alt, anch, k)

    return new_extended_img


def extend_image_n_pixels(img_src, n_pixels, border_type, k=255):

    if len(img_src.shape) != 3:
        return extend(img_src, n_pixels, border_type, k)
    else:
        b_channel, g_channel, r_channel = cv2.split(img_src)

        b_channel_ext = extend(b_channel, n_pixels, border_type, k)
        g_channel_ext = extend(g_channel, n_pixels, border_type, k)
        r_channel_ext = extend(r_channel, n_pixels, border_type, k)

        return cv2.merge((b_channel_ext, g_channel_ext, r_channel_ext))


def convolution_grey_scale_img(mask, img, n_pixels):
    # Obtenemos el alto y ancho de la imagen
    alt_ext, anch_ext = img.shape[:2]

    cv = convolution
    copy = np.copy

    # Obtenemos una matriz auxiliar
    aux = copy(img)

    for i in range(n_pixels, alt_ext - n_pixels):
        for j in range(n_pixels, anch_ext - n_pixels):
            aux[i, j] = cv(mask, img[i, j - n_pixels:1 + j + n_pixels])

    aux_2 = copy(aux)

    cv2.waitKey(0)

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
    aux_r = zeros((alt_ext, anch_ext), dtype=np.uint8)
    aux_g = zeros((alt_ext, anch_ext), dtype=np.uint8)
    aux_b = zeros((alt_ext, anch_ext), dtype=np.uint8)
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
                      smoothing_mask, sharpering_mask):

    low_frecuencies = my_im_gauss_convolution(img_lowF, smoothing_mask, 0)
    cv2.imshow('low_frecuencies', low_frecuencies)
    cv2.waitKey(0)
    low_HF_aux = my_im_gauss_convolution(img_highF, sharpering_mask, 0)
    cv2.imshow('low_frecuencies', low_HF_aux)
    cv2.waitKey(0)
    sharper = img_highF - low_HF_aux
    cv2.imshow('low_frecuencies', sharper)
    cv2.waitKey(0)
    insert_img_into_other(img_src=sharper, img_dest=low_frecuencies, pixel_left_top_row=0, pixel_left_top_col=0)
    return low_frecuencies

