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
    
    #vmort1 = cv2.imread('imagenes/Vmort1.pgm', flags=cv2.IMREAD_GRAYSCALE)
    #vmort2 = cv2.imread('imagenes/Vmort2.pgm', flags=cv2.IMREAD_GRAYSCALE)
     
    #fundamental_mat, img_points1, img_points2 = cc.estimate_fundamental_matrix_from(vmort1, vmort2)
    
    #epip1, epip2 = cc.show_epilines(vmort1, img_points1, vmort2, img_points2, fundamental_mat)
    
    #cc.epipolar_line_error(img_points1, img_points2, epip1, epip2)
    
    # Archivos de las cámaras
    camera_file_names = ['imagenes/rdimage.000.ppm.camera',
                         'imagenes/rdimage.001.ppm.camera',
                         'imagenes/rdimage.004.ppm.camera',
                        ]
    
    camera_matrix00, radial_distorsion00, rotation_matrix00, translation_matrix00 = \
        cc.read_camera_file(camera_file_names[0])
    camera_matrix01, radial_distorsion01, rotation_matrix01, translation_matrix01 = \
        cc.read_camera_file(camera_file_names[1])
    camera_matrix04, radial_distorsion04, rotation_matrix04, translation_matrix04 = \
        cc.read_camera_file(camera_file_names[2])
        
    # Imágenes    
    rdimages000 = cv2.imread('imagenes/rdimage.000.ppm', flags=cv2.IMREAD_COLOR) 
    rdimages001 = cv2.imread('imagenes/rdimage.001.ppm', flags=cv2.IMREAD_COLOR)
    rdimages004 = cv2.imread('imagenes/rdimage.004.ppm', flags=cv2.IMREAD_COLOR)
    
    kp1, des1, kp2, des2, kp3, des3, matches1to2, matches1to3, matches2to3 = \
        cc.get_matches_of_3(rdimages000, rdimages001, rdimages004)
    
    E_1to2 = cc.calculate_essential_matrix(matches1to2, camera_matrix00, camera_matrix01,
                                          kp1, kp2)
    
    cc.calculate_rotation(E_1to2, camera_matrix01)