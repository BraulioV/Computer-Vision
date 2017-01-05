The cameras file contains the following information:

(3x3) camera matrix K
(3)   radial distortion parameters (if zero: the images are corrected for radial distortion)
(3x3) rotation matrix R
(3)   translation vector t

a 3D point X will be projected into the images in the usual way:
x = K[R^T|-R^T t]X

any problems? mail:
christoph.strecha@esat.kuleuven.ac.be


