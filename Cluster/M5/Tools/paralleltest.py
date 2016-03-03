# TEST FILE PLEASE IGNORE

def getKeypointsDescriptors(filenames,detector_type,descriptor_type):
    detector=cv2.FeatureDetector_create(detector_type)
    if descriptor_type == 'SIFT':
        descriptor = cv2.DescriptorExtractor_create(descriptor_type)
    K =  [None] * np.length(filenames)
    D =  [None] * np.length(filenames)
    print 'Extracting Local Descriptors'
    init=time.time()
    Parallel(n_jobs=6)(delayed(getKeypointsDescriptors_par)(K, D, filenames, descriptor_type, detector, descriptor, i) for i in np.length(filenames))
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return(K,D)

def getKeypointsDescriptors_par(K, D, filenames, descriptor_type, detector, descriptor, i):
        ima=cv2.imread(filenames[i])
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        if descriptor_type == 'SIFT':
            kpts=detector.detect(gray)
            kpts,des=descriptor.compute(gray,kpts)
            K[i] = kpts
            D[i] = des
        elif descriptor_type == 'HOG':
            des = extractHOGfeatures(gray, detector)
            D[i] = des
        elif descriptor_type == 'LBP':
            des = extractLBPfeatures(gray, detector)
            D[i] = des
