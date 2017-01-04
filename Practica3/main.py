import cv2
import camera_calibration as cc

if __name__ == '__main__':
    # Ejercicio 1
   # camera = cc.generate_Pcamera()
   # points = cc.generate_points()
   # hom_points, projected = cc.project_points(points, camera)
   # camera_est, err = cc.DLT_algorithm(real_points=hom_points, projected_points=projected, camera=camera)
   # print(camera)
   # print(camera_est)
   # print(err)
    # Ejercicio 2
    #images = []

    #for i in range(1,26):
    #    images.append(cv2.imread('imagenes/Image'+str(i)+'.tif', flags=cv2.IMREAD_GRAYSCALE))

    #cc.calibrate_camera_from(images,True)
    
    vmort1 = cv2.imread('imagenes/Vmort1.pgm', flags=cv2.IMREAD_GRAYSCALE)
    vmort2 = cv2.imread('imagenes/Vmort2.pgm', flags=cv2.IMREAD_GRAYSCALE)
     
    fundamental_mat, img_points1, img_points2 = cc.estimate_fundamental_matrix_from(vmort1, vmort2)
    
    epip1, epip2 = cc.show_epilines(vmort1, img_points1, vmort2, img_points2, fundamental_mat)
    
    cc.epipolar_line_error(img_points1, img_points2, epip1, epip2)