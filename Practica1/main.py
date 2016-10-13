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

    img01 = cv2.imread('data/motorcycle.bmp', flags=cv2.IMREAD_GRAYSCALE)
    img02 = cv2.imread('data/bicycle.bmp', flags=cv2.IMREAD_GRAYSCALE)

    mask_img01 = functions.get_mask_vector(3)
    mask_img02 = functions.get_mask_vector(1.5)
    print(mask_img01.size)
    print(mask_img02.size)

    cv2.imshow('hibrida', functions.make_hybrid_image(img01,img02,mask_img01, mask_img02))
    cv2.waitKey(0)
    cv2.destroyAllWindows()