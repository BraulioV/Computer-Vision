def findValidImages(images):
    valids = []
    size = (8, 6)
    
    for img in images:
        valids.append(cv2.findChessboardCorners(img, size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE+cv2.CALIB_CB_FAST_CHECK))

    for i in valids:
        # Si la imagen es válida, procedemos a refinar
        # los puntos con cornerSubPix
        if i[1] is None:
            print("no valida")                  
        else:
            cv2.cornerSubPix(image=img.astype(np.float32), corners=,
                             winSize=(5, 5), zeroZone=(-1, -1),
                             criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
