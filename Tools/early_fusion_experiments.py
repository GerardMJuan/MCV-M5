from BOVW_functions import *
from sklearn.decomposition import PCA

##########################
#PARAMETROS DE EJECUCION
##########################
if __name__ == '__main__':

    # MEtodo usado para detectar los puntos de interEs
    detector= ['FAST']
    # MEtodo usado para obtener la descripciOn de los puntos de interEs
    descriptor= 'SIFT'
    # NUmero de puntos de interEs que se utilizan para obtener el vocabulario visual
    num_samples= 50000
    # NUmero de palabras en el vocabulario visual
    k= 5000

    folds = 5
    start=0.1
    end=10
    numparams=30
    doPCA = False # Do PCA before K Means

    # Directorio raiz donde se encuentran todas las imAgenes de aprendizaje
    dataset_folder_train='../MIT_split/train/'
    # Directorio raiz donde se encuentran todas las imAgenes de test
    dataset_folder_test='../MIT_split/test/'


    #Modo de experimentacion. 0 = rehacer experimentos, 1 = cargar datos ya guardados.
    use_precalc_voc = 0

    # Type of color (1 normal, 0 el del profe)
    TypeOfColor = 0

    ##############################################


    # PreparaciOn de los nombres de los ficheros necesarios para guardar el vocabulario y las palabras visuales de las imAgenes de aprendizaje y test
    # Se generan tres vocabularios y, por lo tanto tres conjuntos de palabras visuales utilizando tres configuraciones diferentes de descriptor:
    # 1. SOlo SIFT, 2. Color, 3. ConcatenaciOn de SIFT + color
    for j in range(len(detector)):
        print 'For detector ' + detector[j]
        codebook_filename_SIFT='CB_S_'+detector[j]+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        codebook_filename_COLOR='CB_C_'+detector[j]+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        codebook_filename_EARLY='CB_E_'+detector[j]+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

        visual_words_filename_train_SIFT='VW_train_SIFT_'+detector[j]+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        visual_words_filename_train_COLOR='VW_train_COLOR_'+detector[j]+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        visual_words_filename_train_EARLY='VW_train_EARLY_'+detector[j]+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

        visual_words_filename_test_SIFT='VW_test_SIFT_'+detector[j]+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        visual_words_filename_test_COLOR='VW_test_COLOR_'+detector[j]+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        visual_words_filename_test_EARLY='VW_test_EARLY_'+detector[j]+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

        filenames_train,GT_ids_train,GT_labels_train = prepareFiles(dataset_folder_train)


        if use_precalc_voc == 0:

            # CAlculo de puntos de interEs para todas las imAgenes del conjunto de aprendizaje. La descripciOn se obtiene tanto con SIFT como con el descriptor de color
            KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector[j],descriptor)
            CDSC_train = getLocalColorDescriptors(filenames_train,KPTS_train, TypeOfColor)
            # Early fusion de las dos descripciones para cada punto de interEs: SIFT + color. Simplemente se concatenan los dos descriptores
            FDSC_train=[]
            for i in range(len(DSC_train)):
                FDSC_train.append(np.hstack((DSC_train[i],CDSC_train[i])))

        # ConstrucciOn de los 3 vocabularios visuales: SIFT, Color y Early fusion. Los vocabularios quedan guardados en disco.
        # Comentar estas lIneas si los vocabularios ya estAn creados y guardados en disco de una ejecuciOn anterior
        if use_precalc_voc == 0:
            CB_SIFT=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename_SIFT, doPCA, cfg.pca_ncomponents_sift)
            CB_COLOR=getAndSaveCodebook(CDSC_train, num_samples, k, codebook_filename_COLOR, doPCA, cfg.pca_ncomponents_color)
            CB_EARLY=getAndSaveCodebook(FDSC_train, num_samples, k, codebook_filename_EARLY, doPCA, cfg.pca_ncomponents_earlyf)

        # Carga de los vocabularios visuales previamente creados y guardados en disco en una ejecuciOn anterior.
        # Comentar estas lIneas si se quiere re-calcular los vocabularios o si los vocabularios todavIa no se han creado
        if use_precalc_voc == 1:
            CB_SIFT=cPickle.load(open(codebook_filename_SIFT,'r'))
            CB_COLOR=cPickle.load(open(codebook_filename_COLOR,'r'))
            CB_EARLY=cPickle.load(open(codebook_filename_EARLY,'r'))

        # Obtiene la descripciOn BoW de las imAgenes del conjunto de aprendizaje para las tres descripciones: SIFT, Color y Early fusion
        if use_precalc_voc == 0:
            VW_SIFT_train=getAndSaveBoVWRepresentation(DSC_train,k,CB_SIFT,visual_words_filename_train_SIFT)
            VW_COLOR_train=getAndSaveBoVWRepresentation(CDSC_train,k,CB_COLOR,visual_words_filename_train_COLOR)
            VW_FUSION_train=getAndSaveBoVWRepresentation(FDSC_train,k,CB_EARLY,visual_words_filename_train_EARLY)

        # Carga de las 3 descripciones BoW del conjunto de aprendizaje previamente creadas y guardadas en disco en una ejecuciOn anterior.
        # Comentar estas lIneas si se quiere re-calcular la representaciOn o si la representaciOn todavIa no se ha creado
        if use_precalc_voc == 1:
            VW_SIFT_train=cPickle.load(open(visual_words_filename_train_SIFT,'r'))
            VW_COLOR_train=cPickle.load(open(visual_words_filename_train_COLOR,'r'))
            VW_FUSION_train=cPickle.load(open(visual_words_filename_train_EARLY,'r'))

        # CAlculo de puntos de interEs para todas las imAgenes del conjunto de test. Obtiene las dos descripciones (SIFT y color) y tambiEn las concatena
        # para obtener la descripciOn Early Fusion
        filenames_test,GT_ids_test,GT_labels_test = prepareFiles(dataset_folder_test)
        KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector[j],descriptor)
        CDSC_test = getLocalColorDescriptors(filenames_test,KPTS_test,TypeOfColor)
        FDSC_test=[]
        for i in range(len(DSC_test)):
            FDSC_test.append(np.hstack((DSC_test[i],CDSC_test[i])))

        # Obtiene las 3 descripciones BoW (SIFT, color y early fusion) de las imAgenes del conjunto de test
        VW_SIFT_test=getAndSaveBoVWRepresentation(DSC_test,k,CB_SIFT,visual_words_filename_test_SIFT)
        VW_COLOR_test=getAndSaveBoVWRepresentation(CDSC_test,k,CB_COLOR,visual_words_filename_test_COLOR)
        VW_FUSION_test=getAndSaveBoVWRepresentation(FDSC_test,k,CB_EARLY,visual_words_filename_test_EARLY)

        # Entrena un clasificador SVM con las imAgenes del conjunto de aprendizaje y lo evalUa utilizando las imAgenes del conjunto de test
        # para las 3 descripciones (SIFT, color y early fusion)
        # Devuelve la accuracy como medida del rendimiento del clasificador
        ac_BOVW_SIFT,cm,fpr,tpr,roc_auc = trainAndTestLinearSVM_withfolds(VW_SIFT_train,VW_SIFT_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)
        ac_BOVW_COLOR,cm,fpr,tpr,roc_auc = trainAndTestLinearSVM_withfolds(VW_COLOR_train,VW_COLOR_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)
        ac_BOVW_EF,cm,fpr,tpr,roc_auc = trainAndTestLinearSVM_withfolds(VW_FUSION_train,VW_FUSION_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)

        print 'Accuracy BOVW with LinearSVM SIFT: '+str(ac_BOVW_SIFT)
        print 'Accuracy BOVW with LinearSVM Color: '+str(ac_BOVW_COLOR)
        print 'Accuracy BOVW with LinearSVM Early Fusion: '+str(ac_BOVW_EF)

        names = unique_elements(GT_labels_test)
        # Name of the confusion matrix file
        savename = 'finalexp' + str(detector[j]) + 'CN.png'
        #Name of the ROC Curve file
        save = 'finalexp' + str(detector[j]) +'ROC.png'
        plot_confusion_matrix(cm,names,savename)

        plt.figure()
        plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             linewidth=2)

        for i in range(8):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                           ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig(save)
        plt.close()