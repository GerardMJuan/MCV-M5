from BOVW_functions_color import *
import os
import Queue
import threading

# TODO: NORMALIZAR HISTOGRAMAS

##########################################################
#  Funciones para extraer el histograma del vocabulario  #
##########################################################

# Funcion para normalizar el histograma
def histogramNormalization(current_histogram):
    print 'Normalizing the histogram'
    low, high = np.floor(current_histogram.min()), np.ceil(current_histogram.max())
    bins = np.linspace(low, high, high - low + 1)
    hist, edges = np.histogram(current_histogram, bins=bins, density=True)
    #print hist.sum()
    plt.plot(hist,edges)
    plt.xlabel('Hist')
    plt.ylabel('Edges')
    plt.title('Hist Edges hist.')
    plt.show()
    return np.c_[hist,edges]

# Funcion principal donde se realiza el experimento
def experiment(params, queue):
    
    # Parametros de experimento
    thname              = params[0]
    mode                = params[1]
    current_descriptor  = params[2]
    detector            = params[3]
    num_samples         = params[4]
    k                   = params[5]
    doPCA               = params[6]
    pca_components      = params[7]
        
    print '------ Th. Name: '+thname+', mode: '+mode+', descriptor: '+current_descriptor+', detector: '+detector+', samples: '+str(num_samples)+', k: '+str(k)+', PCA: '+str(doPCA)+', comp.: '+str(pca_components)+'------'
  
    # Computar vocabulario
    filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../MIT_split/train/')
    KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,current_descriptor)
    codebook_filename='CB_'+detector+'_'+current_descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
    CB = getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename, doPCA, pca_components)
    
    if mode.lower() == 'train':
        visual_words_filename_train='VW_train_'+detector+'_'+current_descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        visual_words_SPM_filename_train='VWSPM_train_'+detector+'_'+current_descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        
        queue.put(GT_ids_train)
        VW_train = getAndSaveBoVWRepresentation(DSC_train,k,CB,visual_words_filename_train)
        queue.put(VW_train)
        VWSPM_train = getAndSaveBoVW_SPMRepresentation(DSC_train,KPTS_train,k,CB,visual_words_SPM_filename_train,filenames_train)
        queue.put(VWSPM_train)
        
    else:    
        visual_words_filename_test='VW_test_'+detector+'_'+current_descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        visual_words_SPM_filename_test='VWSPM_test_'+detector+'_'+current_descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
    
        filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../MIT_split/test/')
        queue.put(GT_ids_test)
        KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,current_descriptor)
        VW_test = getAndSaveBoVWRepresentation(DSC_test,k,CB,visual_words_filename_test)
        queue.put(VW_test)
        VWSPM_test = getAndSaveBoVW_SPMRepresentation(DSC_test,KPTS_test,k,CB,visual_words_SPM_filename_test,filenames_test)
        queue.put(VWSPM_test)

##########################################################
#                  Programa principal                    #
##########################################################

# Definimos los parametros del experimento
descriptors         = ['SIFT','color'] # Si len(descriptors) == 1, se usan dos threads. Si len(descriptors) == 2, se usan cuatro threads.
detector            = 'Dense'
num_samples         = 50000
k                   = 5000
doPCA               = False
pca_components      = 60
weighting           = 0.65 # Valor entre 0 y 1. Representa la importancia del primer descriptor respecto el segundo (solo se aplica si len(descriptors) == 2). Por defecto usar 0.5.

# Threads para el primer descriptor
th1_params = ['Thread-1', 'train', descriptors[0], detector, num_samples, k, doPCA, pca_components]
th2_params = ['Thread-2', 'test', descriptors[0], detector, num_samples, k, doPCA, pca_components]

if len(descriptors)>1:
    # Threads para el segundo descriptor
    th3_params = ['Thread-3', 'train', descriptors[1], detector, num_samples, k, doPCA, pca_components]
    th4_params = ['Thread-4', 'test', descriptors[1], detector, num_samples, k, doPCA, pca_components]

# Creamos una cola para cada thread
th1_queue = Queue.Queue()
th2_queue = Queue.Queue()

if len(descriptors)>1:
    th3_queue = Queue.Queue()
    th4_queue = Queue.Queue()

# Creamos cuatro threads
thread1_ = threading.Thread(
                target=experiment,
                name=th1_params[0],
                args=[th1_params, th1_queue],
                )
thread2_ = threading.Thread(
                target=experiment,
                name=th2_params[0],
                args=[th2_params, th2_queue],
                )

if len(descriptors)>1:                
    thread3_ = threading.Thread(
                    target=experiment,
                    name=th3_params[0],
                    args=[th3_params, th3_queue],
                    )
    thread4_ = threading.Thread(
                    target=experiment,
                    name=th4_params[0],
                    args=[th4_params, th4_queue],
                    )

# Iniciamos threads
thread1_.start()
thread2_.start()

if len(descriptors)>1:
    thread3_.start()
    thread4_.start()

# Esperamos hasta que todos los threads se hayan completado
thread1_.join()
thread2_.join()

if len(descriptors)>1:
    thread3_.join()
    thread4_.join()

# Guardamos los datos extraidos de los threads (en colas) a listas
D1 = [] # Contenido descriptor 1: < GT_ids_train, VW_train, VWSPM_train, GT_ids_test, VW_test y VWSPM_test >

if len(descriptors)>1:
    D2 = [] # Contenido descriptor 2: < GT_ids_train, VW_train, VWSPM_train, GT_ids_test, VW_test y VWSPM_test >

while not th1_queue.empty(): # Devuelve: descriptor 1 < GT_ids_train, VW_train y VWSPM_train >
    D1.append(th1_queue.get())

while not th2_queue.empty(): # Devuelve: descriptor 1 < GT_ids_test, VW_test y VWSPM_test >
    D1.append(th2_queue.get())
    
if len(descriptors)>1:    
    while not th3_queue.empty(): # Devuelve: descriptor 2 < GT_ids_train, VW_train y VWSPM_train >
        D2.append(th3_queue.get())
        
    while not th4_queue.empty(): # Devuelve: descriptor 2 < GT_ids_test, VW_test y VWSPM_test >
        D2.append(th4_queue.get())

if len(descriptors)>1:
    # Aplicamos pesos
    D1[1] = D1[1] * weighting
    D2[1] = D2[1] * ( 1 - weighting)
    D1[2] = D1[2] * weighting
    D2[2] = D2[2] * ( 1 - weighting)
    # Unimos los histogramas de los vocabularios
    VW_train    = np.hstack((D1[1],D2[1]))  
    VWSPM_train = np.hstack((D1[2],D2[2]))      
    VW_test     = np.hstack((D1[4],D2[4]))
    VWSPM_test  = np.hstack((D1[5],D2[5]))
else:
    # Cogemos directamente el valor del vocabulario
    VW_train    = D1[1]  
    VWSPM_train = D1[2]      
    VW_test     = D1[4]
    VWSPM_test  = D1[5]

# Extraemos ids
GT_ids_train    = D1[0]
GT_ids_test     = D1[3]

# Usamos el clasificador
ac_BOVW_L       = trainAndTestLinearSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,1)
ac_BOVW_SPM_L   = trainAndTestLinearSVM(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,1)

ac_BOVW_HI      = trainAndTestHISVM(VW_train,VW_test,GT_ids_train,GT_ids_test,1)
ac_BOVW_SPM_HI  = trainAndTestHISVM(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,1)

ac_BOVW_SPM_SPMK= trainAndTestSPMSVM(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,1,k)

# Printamos resultados
print 'Accuracy BOVW with LinearSVM: '+str(ac_BOVW_L)
print 'Accuracy BOVW with HISVM: '+str(ac_BOVW_HI)

print 'Accuracy BOVW with SPM with LinearSVM:'+str(ac_BOVW_SPM_L)
print 'Accuracy BOVW with SPM with HISVM: '+str(ac_BOVW_SPM_HI)
print 'Accuracy BOVW with SPM with SPMKernelSVM: '+str(ac_BOVW_SPM_SPMK)