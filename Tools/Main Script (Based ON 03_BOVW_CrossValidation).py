#!/usr/bin/python
from BOVW_functions import *
from sklearn.metrics import roc_curve, auc

if __name__ == '__main__':

    detector='SIFT'
    descriptor='SIFT'
    num_samples=50000

    k=5000

    k_knn = 32
    lookfor_k = 0
    num_iterations_lookfor_k = 1
    x_total_k = []
    y_total_acc = []

    folds = 5
    start=0.001
    end=10
    numparams=30

    classifier='LinearSVM' # Choose between KNN and linearSVM
    # Name of the accuracy file
    text_file = open("Output.txt", "a")

    for index_k in range(0, num_iterations_lookfor_k):
        if lookfor_k == 0 or classifier != 'KNN':
            num_iterations_lookfor_k = 0
        else:
            if index_k == 0:
                k_knn = 20
            else:
                k_knn = k_knn + 15

        codebook_filename='CB_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        visual_words_filename_train='VW_train_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'
        visual_words_filename_test='VW_test_'+detector+'_'+descriptor+'_'+str(num_samples)+'samples_'+str(k)+'centroids.dat'

        filenames_train,GT_ids_train,GT_labels_train = prepareFiles('../MIT_split/train/')
        KPTS_train,DSC_train = getKeypointsDescriptors(filenames_train,detector,descriptor)
        CB=getAndSaveCodebook(DSC_train, num_samples, k, codebook_filename)
        #CB=cPickle.load(open(codebook_filename,'r'))

        VW_train=getAndSaveBoVWRepresentation(DSC_train,k,CB,visual_words_filename_train)
        #VW_train=cPickle.load(open(visual_words_filename_train,'r'))

        filenames_test,GT_ids_test,GT_labels_test = prepareFiles('../MIT_split/test/')
        KPTS_test,DSC_test = getKeypointsDescriptors(filenames_test,detector,descriptor)
        VW_test=getAndSaveBoVWRepresentation(DSC_test,k,CB,visual_words_filename_test)

        #ac_BOVW_L = trainAndTestLinearSVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)

        if classifier == 'KNN':
            ac_BOVW_L = trainAndTestKNeighborsClassifier_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,k_knn)
        elif classifier == 'LinearSVM':
            ac_BOVW_L,cm,fpr,tpr,roc_auc = trainAndTestLinearSVM_withfolds(VW_train,VW_test,GT_ids_train,GT_ids_test,folds,start,end,numparams)

        names = unique_elements(GT_labels_test)
        # Name of the confusion matrix file
        savename = str(folds) + 'sift_CN.png'
        #Name of the ROC Curve file
        save = str(folds) +'sift_ROC.png'
        plot_confusion_matrix(cm,names,savename)
        s = 'For lbp_lin The accuracy is ' + str(ac_BOVW_L) + '\n'
        text_file.write(s)


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


    if lookfor_k == 1:
        plt.figure()
        plt.plot(x_total_k, y_total_acc)
        plt.xlabel('K value')
        plt.ylabel('Accuracy')
        plt.title('Relationship between K value and accuracy')
        plt.show()