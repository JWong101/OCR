import cv2
import numpy as np

def deskew(img):
    moments = cv2.moments(img)

    if abs(moments['mu02']) < 1e-2:
        return img.copy()

    skew = moments['mu11'] / moments['mu02']

    size = img.size()
    trans = np.float32([[1, skew, ]-0.5 * size * skew, [0, 1, 0])
    img = cv2.warpAffine(img, trans, (size, size), flags= cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

def singleHog(img):
    winSize = (20, 20) #size of image 20x20
    blockSize = (10, 10) #used for illumination variation. Typically it's 2 * cellSize, however illumination doesn't matter in this case
    blockStride = (5, 5) #overlap between neighbloring blocks. 50% of blockSize
    cellSize = (10, 10) #scale of features, size of descriptor is smaller than in the image a very small cellSize would blow up the size of the feature vector and a large one wouldn't capture any info

    nBins = 9 #number of bins in the histogram gradients. 9: 0 and 180 degrees in 20 degree increments
    derivAperture = 1
    winSigma = -1.0
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = 1
    nlevels = 64
    useSignedGradients = True #signed gradients are between 0 and 360, unsigned 0 and 180 (pointing up or down)
    hog = cv2.HogDescriptor(winSize, blockSize, blockStride, cellSize, nBins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels, useSignedGradients)

    descript = hog.compute(img)

def train():
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)

    #radial bias function
    svm.setType(cv2.ml.SVM_RBF)

    #sets decision boundary
    svm.setC(12.5)

    svm.setGamma(gamma)

    svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)

    svm.save("digits_svm_model.yml")
    testResponse = svm.predict(testdata)[1].ravel()

    #autotune parameters
    #takes 5x longer to train
#    svm.trainAuto(trainData)

