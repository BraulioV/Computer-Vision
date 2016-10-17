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

    img01 = cv2.imread('data/cat.bmp', flags=cv2.IMREAD_COLOR)
    img02 = cv2.imread('data/dog.bmp', flags=cv2.IMREAD_COLOR)

    hybrid = functions.make_hybrid_image(
        img01,img02, 8, 3,True)
    cv2.imshow('hibrida', hybrid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # canvas = functions.generate_new_pyramidal_canvas(hybrid, times_to_show= 8)
    # cv2.imshow('canvas_largo', functions.generate_continous_canvas([img01,img02,hybrid]))
    # cv2.imshow('canvas', canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()