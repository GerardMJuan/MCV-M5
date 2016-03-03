from BOVW_functions import *


detector='Dense'
descriptor='SIFT'																																															
num_samples=50000
k=32
codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_SPM_filename_train='VWSPM_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
visual_words_SPM_filename_test='VWSPM_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../../Databases/MIT_split/train/')
KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor)
CB=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename)
VW_train=getAndSaveBoVWRepresentation(DSC_train,k,CB,visual_words_filename_train)
VWSPM_train=getAndSaveBoVW_SPMRepresentation(DSC_train,KPTS_train,k,CB,visual_words_SPM_filename_train,filenames_train)


filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../../Databases/MIT_split/test/')
KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor)
VW_test=getAndSaveBoVWRepresentation(DSC_test,k,CB,visual_words_filename_test)
VWSPM_test=getAndSaveBoVW_SPMRepresentation(DSC_test,KPTS_test,k,CB,visual_words_SPM_filename_test,filenames_test)


ac_BOVW_L = trainAndTestLinearSVM(VW_train,VW_test,GT_ids_train,GT_ids_test,1)
ac_BOVW_SPM_L = trainAndTestLinearSVM(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,1)

ac_BOVW_HI = trainAndTestHISVM(VW_train,VW_test,GT_ids_train,GT_ids_test,1)
ac_BOVW_SPM_HI = trainAndTestHISVM(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,1)

ac_BOVW_SPM_SPMK = trainAndTestSPMSVM(VWSPM_train,VWSPM_test,GT_ids_train,GT_ids_test,1,k)


print 'Accuracy BOVW with LinearSVM: '+str(ac_BOVW_L)
print 'Accuracy BOVW with HISVM: '+str(ac_BOVW_HI)

print 'Accuracy BOVW with SPM with LinearSVM:'+str(ac_BOVW_SPM_L)
print 'Accuracy BOVW with SPM with HISVM: '+str(ac_BOVW_SPM_HI)
print 'Accuracy BOVW with SPM with SPMKernelSVM: '+str(ac_BOVW_SPM_SPMK)