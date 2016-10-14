import functions
import cv2

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

    img01 = cv2.imread('data/bird.bmp', flags=cv2.IMREAD_COLOR)
    img02 = cv2.imread('data/plane.bmp', flags=cv2.IMREAD_COLOR)

    mask_img01 = functions.get_mask_vector(3)
    mask_img02 = functions.get_mask_vector(1.5)
    hybrid = functions.make_hybrid_image(
        img01,img02,mask_img01, mask_img02)
    cv2.imshow('hibrida', hybrid)
    cv2.waitKey(0)
    cv2.imshow('hibrida_mini', functions.subsample_image(img_src = hybrid, subsample_factor = 5))
    cv2.waitKey(0)
    cv2.destroyAllWindows()