#!/usr/bin/env python

import matplotlib.pyplot as plt
import scipy.spatial.distance

import imghdr
import numpy
import os
import random
import scipy.misc
import math


dataDirectory    = "Data/faces/Original"
variance         = 0.95

if variance > 1.0:
  variance = 1.0
elif variance < 0.0:
  variance = 0.0

def enumerateImagePaths(dataDirectory1):
  filenames = list()
  for root, _, files in os.walk(dataDirectory1):
    path = root.split('/')
    for f in files:
      filename = os.path.join(root, f)
      if imghdr.what(filename):
        filenames.append(filename)
  return filenames

filenames          = enumerateImagePaths(dataDirectory)
trainingImageNames = filenames
numTrainingFaces = len(trainingImageNames)

maxWeight = 0

#
# Choose training images
#

trainingImages = list()

for name in trainingImageNames:
  trainingImages.append( scipy.misc.imread(name) )

#
# Calculate & subtract average face
#

meanFace = numpy.zeros(trainingImages[0].shape)

for image in trainingImages:
  meanFace += 1/numTrainingFaces * image

trainingImages = [ image - meanFace for image in trainingImages ] 

#
# Calculate eigenvectors
#

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


#
# Transform remaining images into "face space"
#

personWeights = dict()

for name in filenames:
  image = scipy.misc.imread(name)

  image = image - meanFace 

  weights = list()

  for i in range(numEffectiveEigenvalues):
    weights.append( (V[:,i].transpose() * image.reshape((n,1))).tolist()[0][0] )
  
  if maxWeight < max(weights):
    maxWeight = max(weights)
  
  personWeights[name] = weights

# End Training

# Start recognition

# TODO: Add path and filenames
unknownDirectory = "Data/unknown"
filesToRecognize = enumerateImagePaths(unknownDirectory)
unknownFaceImages = list()
notRecognized = list()
recognizedFaces = list()


#tempT = list() 

for nameToRecognize in filesToRecognize:
  image = scipy.misc.imread(nameToRecognize) 

  # Convert UNKNOWN face to FaceVector
  # Normalize vector by image - meanFace
  image = image - meanFace

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
  if d < 224210797:
    recognizedFaces.append("%25s %25s %20d" % (nameToRecognize.split("/")[-1], resultName.split("/")[-1], d))
  else:
    notRecognized.append("%25s %25s %20d" % (nameToRecognize.split("/")[-1], resultName.split("/")[-1], d))


print("\n\n############################################\n\n")

print("%25s %25s %20s" % ("Face","Recognized as","Distance"))
for nameStr in recognizedFaces:
   print(nameStr)

print("\n\n############################################\n\n")

print("%25s %25s %20s" % ("Face","Nearest face","Distance"))
for nameStr in notRecognized:
   print(nameStr)

print("\n\nRecognized: %d/%d" % (len(recognizedFaces), len(filesToRecognize)))
#print(sorted(tempT, key=int))
