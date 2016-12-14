import cv2
import camera_calibration as cc

if __name__ == '__main__':
    camera = cc.generate_Pcamera()
    points = cc.generate_points()
    hom_points, projected = cc.project_points(points, camera)
    camera_est, err = cc.DLT_algorithm(real_points=hom_points, projected_points=projected, camera=camera)
    print(camera)
    print(err)

