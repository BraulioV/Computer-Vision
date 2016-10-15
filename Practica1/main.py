import cv2
import functions

if __name__ == "__main__":

    # img = cv2.imread('data/motorcycle.bmp', flags=cv2.IMREAD_COLOR)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # mask = functions.get_mask_vector(1)
    # result = functions.my_im_gauss_convolution(img, mask)
    # cv2.imshow('imagen filtrada', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img01 = cv2.imread('data/dog.bmp', flags=cv2.IMREAD_GRAYSCALE)
    img02 = cv2.imread('data/cat.bmp', flags=cv2.IMREAD_GRAYSCALE)

    mask_img01 = functions.get_mask_vector(8)
    mask_img02 = functions.get_mask_vector(3)
    hybrid = functions.make_hybrid_image(
        img01,img02,mask_img01, mask_img02)
    cv2.imshow('hibrida', hybrid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    txt = ["Imagen 1", "Imagen 2", "Imagen 3"]
    # canvas = functions.generate_new_pyramidal_canvas(hybrid, times_to_show= 8)
    cv2.imshow('canvas_largo', functions.generate_continous_canvas([img01,img02,hybrid],txt))
    # cv2.imshow('canvas', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()