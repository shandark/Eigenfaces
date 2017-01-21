#!/usr/bin/env python

import matplotlib.pyplot as plt
import scipy.spatial.distance

import argparse
import imghdr
import numpy
import os
import random
import scipy.misc
import math

parser = argparse.ArgumentParser(description="Eigenface reconstruction demonstration")
parser.add_argument("data",       metavar="DATA", type=str,   help="Data directory")
parser.add_argument("n",          metavar="N",    type=int,   help="Number of training images", default=50)
parser.add_argument("--variance",                 type=float, help="Desired proportion of variance", default = 0.95)

arguments = parser.parse_args()

dataDirectory    = arguments.data
numTrainingFaces = arguments.n
variance         = arguments.variance

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
    print("Reached", variance * 100, "%", "explained variance with", index+1 , "eigenvalues")
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

print ("End training with success")

# End Training

# Start recognition

# TODO: Add path and filenames
unknownDirectory = "Data/unknown"
filesToRecognize = enumerateImagePaths(unknownDirectory)
unknownFaceImages = list()
recognizedFaces = 0


#tempT = list() 

for nameToRecognize in filesToRecognize:
  image = scipy.misc.imread(nameToRecognize) 

  # Convert UNKNOWN face to FaceVector
  # Normalize vector by image - meanFace
  image = image - meanFace

  unknownWeights = list()
  d = maxWeight 
  resultName = str()

  for i in range(numEffectiveEigenvalues):
    unknownWeights.append( (V[:,i].transpose() * image.reshape((n,1))).tolist()[0][0] )

  for name,weights in personWeights.iteritems():
    tempD = scipy.spatial.distance.euclidean(unknownWeights , weights)
    if tempD < d:
       d = tempD
       resultName = name

  #tempT.append(d)

  # TODO: How calc threshold???
  if d < 224210797:
    print("The %s recognized as %s" % (nameToRecognize, resultName))
    recognizedFaces += 1
  else:
    print("Not %s recognized, nearest face is %s" % (nameToRecognize, resultName)) 

  #print(nameToRecognize)


print("Recognized: %d/%d" % (recognizedFaces, len(filesToRecognize)))
#print(sorted(tempT, key=int))
    

# Convert normalized vector to eigenspace

# Create weights

# Calculate d = ||W - Wk||^2

# d > threshold then not face

# d < threshold then face K [print filename - recognized person]

# other d then unknown person
