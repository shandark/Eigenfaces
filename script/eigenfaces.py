#!/usr/bin/env python

import matplotlib.pyplot as plt
import scipy.spatial.distance

import imghdr
import numpy
import os
import scipy.misc


dataDirectory    = "faces/trainingSet"
variance         = 0.95

def enumerateImagePaths(dataDirectory1):
  filenames = list()
  for root, _, files in os.walk(dataDirectory1):
    path = root.split('/')
    for f in files:
      filename = os.path.join(root, f)
      if imghdr.what(filename):
        filenames.append(filename)
  return filenames

def printTitle(titleToPrint):
    print("\n\n############################################\n%s\n" % titleToPrint)

def compareVisualyFace(faceList):
   index = 1
   f = plt.figure()
   for name in faceList:
       f.add_subplot(len(faceList), 2, index)
       index += 1
       plt.imshow(scipy.misc.imread(name.split(" ")[0]), cmap=plt.cm.Greys_r)
       f.add_subplot(len(faceList), 2, index)
       index += 1
       plt.imshow(scipy.misc.imread(name.split(" ")[1]), cmap=plt.cm.Greys_r)
   plt.show()


def printResults(title, facesList):
    printTitle(title)
    
    print("%25s %25s %20s" % ("Face","Recognized as","Distance"))
    for nameStr in facesList:
       print("%25s %25s %20s" % (nameStr.split(" ")[0].split("/")[-1], nameStr.split(" ")[1].split("/")[-1], nameStr.split(" ")[2]))
    compareVisualyFace(facesList)



def calcAverageFace(trainingSet):
    avgFace = numpy.zeros(trainingSet[0].shape)
    
    for image in trainingSet:
      avgFace += image/float(len(trainingSet))
    
    return avgFace.astype(int)


filenames          = enumerateImagePaths(dataDirectory)
trainingImageNames = filenames
numTrainingFaces = len(trainingImageNames)

maxWeight = 0

# Calc average face - normalize database

trainingImages = list()

for name in trainingImageNames:
  trainingImages.append( scipy.misc.imread(name) )

averageFace = calcAverageFace(trainingImages)
trainingImages = [ image - averageFace for image in trainingImages ] 

# Calculate eigenvectors

x,y = trainingImages[0].shape
n   = x*y
A   = numpy.matrix( numpy.zeros((n,numTrainingFaces)) )

for i,image in enumerate(trainingImages):
  A[:,i] = numpy.reshape(image,(n,1))

M                         = A.transpose()*A
eigenvalues, eigenvectors = numpy.linalg.eig(M)
indices                   = eigenvalues.argsort()[::-1]
eigenvalues               = eigenvalues[indices]
eigenvectors              = eigenvectors[:,indices]

eigenvalueSum           = sum(eigenvalues)
partialSum              = 0.0
numEffectiveEigenvalues = 0

for index,eigenvalue in enumerate(eigenvalues):
  partialSum += eigenvalue
  if partialSum / eigenvalueSum >= variance:
    numEffectiveEigenvalues = index+1
    break

V = numpy.matrix( numpy.zeros((n,numEffectiveEigenvalues)) )
for i in range(numEffectiveEigenvalues):
  V[:,i] = A*eigenvectors[:,i].real

# Calc weights

personWeights = dict()

for name in filenames:
  image = scipy.misc.imread(name)

  image = image - averageFace 

  weights = list()

  for i in range(numEffectiveEigenvalues):
    weights.append( (V[:,i].transpose() * image.reshape((n,1))).tolist()[0][0] )
  
  if maxWeight < max(weights):
    maxWeight = max(weights)
  
  personWeights[name] = weights

# End Training

# Start recognition

# TODO: Add path and filenames
unknownDirectory = "faces/unknown"
filesToRecognize = enumerateImagePaths(unknownDirectory)
overThreshold = list()
notRecognized = list()
recognizedFaces = list()


#tempT = list() 

for nameToRecognize in filesToRecognize:
  image = scipy.misc.imread(nameToRecognize) 

  # Convert UNKNOWN face to FaceVector
  # Normalize vector by image - averageFace
  image = image - averageFace

  unknownWeights = list()
  d = maxWeight 
  resultName = str()

  # Convert normalized vector to eigenspace
  # Create weights
  for i in range(numEffectiveEigenvalues):
    unknownWeights.append( (V[:,i].transpose() * image.reshape((n,1))).tolist()[0][0] )

  for name,weights in personWeights.iteritems():
    # Calculate d = ||W - Wk||^2
    tempD = scipy.spatial.distance.euclidean(unknownWeights , weights)
    if tempD < d:
       d = tempD
       resultName = name

  #tempT.append(d)

  # TODO: How calc threshold???
  # d < threshold then face K [print filename - recognized person]
  if d < 353495804:
    recognizedFaces.append("%s %s %d" % (nameToRecognize, resultName, d))
  elif d > 353495803 and d < 635087668:
    notRecognized.append("%s %s %d" % (nameToRecognize, resultName, d))
  else:
    overThreshold.append("%s %s %d" % (nameToRecognize, resultName, d))


printResults("Recognized", recognizedFaces)
printResults("NOT Recognized", notRecognized)
printResults("Totaly not recognized", overThreshold)

print("\n\nRecognized: %d/%d" % (len(recognizedFaces), len(filesToRecognize)))
#print(sorted(tempT, key=int))
