#!/usr/bin/env python

import matplotlib.pyplot as plt

import argparse
import imghdr
import numpy
import os
import random
import scipy.misc

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

def enumerateImagePaths(root):
  filenames = list()
  for root, _, files in os.walk(dataDirectory):
    path = root.split('/')
    for f in files:
      filename = os.path.join(root, f)
      if imghdr.what(filename):
        filenames.append(filename)
  return filenames

filenames          = enumerateImagePaths(dataDirectory)
trainingImageNames = filenames
numTrainingFaces = len(trainingImageNames)
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

remainingImages = list()

for name in filenames:
  remainingImages.append( scipy.misc.imread(name) )

remainingImages = [ image - meanFace for image in remainingImages ]

for image in remainingImages:
  weights = list()

  for i in range(numEffectiveEigenvalues):
    weights.append( (V[:,i].transpose() * image.reshape((n,1))).tolist()[0][0] )

print ("End with success")

# Move weight and filenames to dict? Filename could be a person name
# End Training

# Start face recognition

# Convert UNKNOWN face to FaceVector

# Normalize vector by image - meanFace

# Convert normalized vector to eigenspace

# Create weights

# Calculate d = ||W - Wk||^2

# d > threshold then not face

# d < threshold then face K [print filename - recognized person]

# other d then unknown person
