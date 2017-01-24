import cv2
import functions as fx

if __name__ == "__main__":
    ##############################################
    # EJERCICIO A
    ##############################################
    fish = cv2.imread('data/fish.bmp', flags=cv2.IMREAD_COLOR)
    mask = fx.get_mask_vector(2)
    smoothed_fish = fx.my_im_gauss_convolution(fish, mask)
    canvas = fx.generate_continous_canvas([fish, smoothed_fish])
    fx.show_img(im=canvas, name='Suavizado de una imagen en color con sigma = 2')

    dog = cv2.imread('data/dog.bmp', flags=cv2.IMREAD_GRAYSCALE)
    mask_2 = fx.get_mask_vector(1)
    smoothed_dog = fx.my_im_gauss_convolution(dog, mask_2)
    canvas = fx.generate_continous_canvas([dog, smoothed_dog])
    fx.show_img(im=canvas, name='Suavizado de una imagen en escala de grises con sigma = 1')

    ##############################################
    # EJERCICIO B
    ##############################################
    cat = cv2.imread('data/cat.bmp', flags=cv2.IMREAD_GRAYSCALE)

    cat_dog = fx.make_hybrid_image(cat, dog,
                                smoothing_mask_sigma=8,
                                sharpering_mask_sigma=4,
                                show_images=True)

    ##############################################
    # EJERCICIO C
    ##############################################
    fx.show_img(fx.generate_new_pyramidal_canvas(cat_dog, times_to_show=6),
                'Piramide gaussiana')
