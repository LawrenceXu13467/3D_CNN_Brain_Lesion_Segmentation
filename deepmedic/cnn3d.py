# Copyright (c) 2016, Konstantinos Kamnitsas
# All rights reserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the BSD license. See the accompanying LICENSE file
# or read the terms at https://opensource.org/licenses/BSD-3-Clause.

# Modified for cs 168 class project
from __future__ import absolute_import, print_function, division
from six.moves import xrange
import numpy as np
import random
from math import ceil
import theano
import theano.tensor as T
import pickle

class PathwayTypes(object):
    NORM = 0; SUBS = 1; FC = 2 # static
    
    def pTypes(self): #To iterate over if needed.
        # This enumeration is also the index in various datastructures ala: [ [listForNorm], [listForSubs], [listForFc] ] 
        return [self.NORM, self.SUBS, self.FC]

py = PathwayTypes

def cropRczOf5DimArrayToMatchOther(array5DimToCrop, dimensionsOf5DimArrayToMatchInRcz):
    # dimensionsOf5DimArrayToMatchInRcz : [ batch size, num of fms, r, c, z] 
    output = array5DimToCrop[:,
                            :,
                            :dimensionsOf5DimArrayToMatchInRcz[2],
                            :dimensionsOf5DimArrayToMatchInRcz[3],
                            :dimensionsOf5DimArrayToMatchInRcz[4]]
    return output
    
def repeatRcz5DimArrayByFactor(array5Dim, factor3Dim):
    # array5Dim: [batch size, num of FMs, r, c, z]. Ala input/output of conv layers.
    # Repeat FM in the three last dimensions, to upsample back to the normal resolution space.
    expandedR = array5Dim.repeat(factor3Dim[0], axis=2)
    expandedRC = expandedR.repeat(factor3Dim[1], axis=3)
    expandedRCZ = expandedRC.repeat(factor3Dim[2], axis=4)
    return expandedRCZ
    
def upsampleRcz5DimArrayAndOptionalCrop(array5dimToUpsample,
                                        upsamplingFactor,
                                        upsamplingScheme="repeat",
                                        dimensionsOf5DimArrayToMatchInRcz=None) :
    # array5dimToUpsample : [batch_size, numberOfFms, r, c, z].
    if upsamplingScheme == "repeat" :
        upsampledOutput = repeatRcz5DimArrayByFactor(array5dimToUpsample, upsamplingFactor)
    else :
        print("ERROR: in upsampleRcz5DimArrayAndOptionalCrop(...). Not implemented type of upsampling! Exiting!"); exit(1)
        
    if dimensionsOf5DimArrayToMatchInRcz != None :
        # If the central-voxels are eg 10, the susampled-part will have 4 central voxels. Which above will be repeated to 3*4 = 12.
        # I need to clip the last ones, to have the same dimension as the input from 1st pathway, which will have dimensions equal to the centrally predicted voxels (10)
        output = cropRczOf5DimArrayToMatchOther(upsampledOutput, dimensionsOf5DimArrayToMatchInRcz)
    else :
        output = upsampledOutput
        
    return output
    
def getMiddlePartOfFms(fms, listOfNumberOfCentralVoxelsToGetPerDimension) :
    # fms: a 5D tensor, [batch, fms, r, c, z]
    fmsShape = T.shape(fms) #fms.shape works too, but this is clearer theano grammar.
    # if part is of even width, one voxel to the left is the centre.
    rCentreOfPartIndex = (fmsShape[2] - 1) // 2
    rIndexToStartGettingCentralVoxels = rCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[0] - 1) // 2
    rIndexToStopGettingCentralVoxels = rIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[0]  # Excluding
    cCentreOfPartIndex = (fmsShape[3] - 1) // 2
    cIndexToStartGettingCentralVoxels = cCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[1] - 1) // 2
    cIndexToStopGettingCentralVoxels = cIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[1]  # Excluding
    
    if len(listOfNumberOfCentralVoxelsToGetPerDimension) == 2:  # the input FMs are of 2 dimensions (for future use)
        return fms[ :, :,
                    rIndexToStartGettingCentralVoxels : rIndexToStopGettingCentralVoxels,
                    cIndexToStartGettingCentralVoxels : cIndexToStopGettingCentralVoxels]
    elif len(listOfNumberOfCentralVoxelsToGetPerDimension) == 3 :  # the input FMs are of 3 dimensions
        zCentreOfPartIndex = (fmsShape[4] - 1) // 2
        zIndexToStartGettingCentralVoxels = zCentreOfPartIndex - (listOfNumberOfCentralVoxelsToGetPerDimension[2] - 1) // 2
        zIndexToStopGettingCentralVoxels = zIndexToStartGettingCentralVoxels + listOfNumberOfCentralVoxelsToGetPerDimension[2]  # Excluding
        return fms[ :, :,
                    rIndexToStartGettingCentralVoxels : rIndexToStopGettingCentralVoxels,
                    cIndexToStartGettingCentralVoxels : cIndexToStopGettingCentralVoxels,
                    zIndexToStartGettingCentralVoxels : zIndexToStopGettingCentralVoxels]
    else :  # wrong number of dimensions!
        return -1
        
def makeResidualConnectionBetweenLayersAndReturnOutput( myLogger,
                                                        deeperLayerOutputImagesTrValTest,
                                                        deeperLayerOutputImageShapesTrValTest,
                                                        earlierLayerOutputImagesTrValTest,
                                                        earlierLayerOutputImageShapesTrValTest) :
    # Add the outputs of the two layers and return the output, as well as its dimensions.
    # Result: The result should have exactly the same shape as the output of the Deeper layer. Both #FMs and Dimensions of FMs.
    
    (deeperLayerOutputImageTrain, deeperLayerOutputImageVal, deeperLayerOutputImageTest) = deeperLayerOutputImagesTrValTest
    (deeperLayerOutputImageShapeTrain, deeperLayerOutputImageShapeVal, deeperLayerOutputImageShapeTest) = deeperLayerOutputImageShapesTrValTest
    (earlierLayerOutputImageTrain, earlierLayerOutputImageVal, earlierLayerOutputImageTest) = earlierLayerOutputImagesTrValTest
    (earlierLayerOutputImageShapeTrain, earlierLayerOutputImageShapeVal, earlierLayerOutputImageShapeTest) = earlierLayerOutputImageShapesTrValTest
    # Note: deeperLayerOutputImageShapeTrain has dimensions: [batchSize, FMs, r, c, z]    
    # The deeper FMs can be greater only when there is upsampling. But then, to do residuals, I would need to upsample the earlier FMs. Not implemented.
    if np.any(np.asarray(deeperLayerOutputImageShapeTrain[2:]) > np.asarray(earlierLayerOutputImageShapeTrain[2:])) or \
            np.any(np.asarray(deeperLayerOutputImageShapeVal[2:]) > np.asarray(earlierLayerOutputImageShapeVal[2:])) or \
                np.any(np.asarray(deeperLayerOutputImageShapeTest[2:]) > np.asarray(earlierLayerOutputImageShapeTest[2:])) :
        myLogger.print3("ERROR: In function [makeResidualConnectionBetweenLayersAndReturnOutput] the RCZ-dimensions of a deeper layer FMs were found greater than the earlier layers. Not implemented functionality. Exiting!")
        myLogger.print3("\t (train) Dimensions of Deeper Layer=" + str(deeperLayerOutputImageShapeTrain) + ". Dimensions of Earlier Layer=" + str(earlierLayerOutputImageShapeTrain) )
        myLogger.print3("\t (val) Dimensions of Deeper Layer=" + str(deeperLayerOutputImageShapeVal) + ". Dimensions of Earlier Layer=" + str(earlierLayerOutputImageShapeVal) )
        myLogger.print3("\t (test) Dimensions of Deeper Layer=" + str(deeperLayerOutputImageShapeTest) + ". Dimensions of Earlier Layer=" + str(earlierLayerOutputImageShapeTest) )
        exit(1)
        
    # get the part of the earlier layer that is of the same dimensions as the FMs of the deeper:
    partOfEarlierFmsToAddTrain = getMiddlePartOfFms(earlierLayerOutputImageTrain, deeperLayerOutputImageShapeTrain[2:])
    partOfEarlierFmsToAddVal = getMiddlePartOfFms(earlierLayerOutputImageVal, deeperLayerOutputImageShapeVal[2:])
    partOfEarlierFmsToAddTest = getMiddlePartOfFms(earlierLayerOutputImageTest, deeperLayerOutputImageShapeTest[2:])
    
    # Add the FMs, after taking care of zero padding if the deeper layer has more FMs.
    numFMsDeeper = deeperLayerOutputImageShapeTrain[1]
    numFMsEarlier = earlierLayerOutputImageShapeTrain[1]
    if numFMsDeeper >= numFMsEarlier :
        outputOfResConnTrain = T.inc_subtensor(deeperLayerOutputImageTrain[:, :numFMsEarlier, :,:,:], partOfEarlierFmsToAddTrain, inplace=False)
        outputOfResConnVal = T.inc_subtensor(deeperLayerOutputImageVal[:, :numFMsEarlier, :,:,:], partOfEarlierFmsToAddVal, inplace=False)
        outputOfResConnTest = T.inc_subtensor(deeperLayerOutputImageTest[:, :numFMsEarlier, :,:,:], partOfEarlierFmsToAddTest, inplace=False)
    else : # Deeper FMs are fewer than earlier. This should not happen in most architectures. But oh well...
        outputOfResConnTrain = deeperLayerOutputImageTrain + partOfEarlierFmsToAddTrain[:, :numFMsDeeper, :,:,:]
        outputOfResConnVal = deeperLayerOutputImageVal + partOfEarlierFmsToAddVal[:, :numFMsDeeper, :,:,:]
        outputOfResConnTest = deeperLayerOutputImageTest + partOfEarlierFmsToAddTest[:, :numFMsDeeper, :,:,:]
        
    # Dimensions of output are the same as those of the deeperLayer
    return (outputOfResConnTrain, outputOfResConnVal, outputOfResConnTest)
    
    

class Pathway(object):
    # This is a virtual class.
    
    def __init__(self, pName=None) :
        self._pName = pName
        self._pType = None # Pathway Type.
        
        # === Input to the pathway ===
        self._inputTrain = None
        self._inputVal = None
        self._inputTest = None
        self._inputShapeTrain = None
        self._inputShapeVal = None
        self._inputShapeTest = None
        
        # === Basic architecture parameters === 
        self._layersInPathway = []
        self._subsFactor = [1,1,1]
        self._recField = None # At the end of pathway
        
        # === Output of the block ===
        self._outputTrain = None
        self._outputVal = None
        self._outputTest = None
        self._outputShapeTrain = None
        self._outputShapeVal = None
        self._outputShapeTest = None
        
    def makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(self,
                                                    myLogger,
                                                    
                                                    inputTrain,
                                                    inputVal,
                                                    inputTest,
                                                    inputDimsTrain,
                                                    inputDimsVal,
                                                    inputDimsTest,
                                                    
                                                    numKernsPerLayer,
                                                    kernelDimsPerLayer,
                                                    
                                                    convWInitMethod,
                                                    useBnPerLayer, # As a flag for case that I want to apply BN on input image. I want to apply to input of FC.
                                                    rollingAverageForBatchNormalizationOverThatManyBatches,
                                                    activFuncPerLayer,
                                                    dropoutRatesPerLayer=[],
                                                    
                                                    poolingParamsStructureForThisPathwayType = [],
                                                    
                                                    indicesOfLowerRankLayersForPathway=[],
                                                    ranksOfLowerRankLayersForPathway = [],
                                                    
                                                    indicesOfLayersToConnectResidualsInOutputForPathway=[]
                                                    ) :
        rng = np.random.RandomState(55789)
        myLogger.print3("[Pathway_" + str(self.getStringType()) + "] is being built...")
        
        self._recField = self.calcRecFieldOfPathway(kernelDimsPerLayer)
        
        self._setInputAttributes(inputTrain, inputVal, inputTest, inputDimsTrain, inputDimsVal, inputDimsTest)                
        myLogger.print3("\t[Pathway_"+str(self.getStringType())+"]: Input's Shape: (Train) " + str(self._inputShapeTrain) + \
                ", (Val) " + str(self._inputShapeVal) + ", (Test) " + str(self._inputShapeTest))
        
        inputToNextLayerTrain = self._inputTrain; inputToNextLayerVal = self._inputVal; inputToNextLayerTest = self._inputTest
        inputToNextLayerShapeTrain = self._inputShapeTrain; inputToNextLayerShapeVal = self._inputShapeVal; inputToNextLayerShapeTest = self._inputShapeTest
        numOfLayers = len(numKernsPerLayer)
        for layer_i in xrange(0, numOfLayers) :
            thisLayerFilterShape = [numKernsPerLayer[layer_i],inputToNextLayerShapeTrain[1]] + kernelDimsPerLayer[layer_i]
            
            thisLayerUseBn = useBnPerLayer[layer_i]
            thisLayerActivFunc = activFuncPerLayer[layer_i]
            thisLayerDropoutRate = dropoutRatesPerLayer[layer_i] if dropoutRatesPerLayer else 0
            
            thisLayerPoolingParameters = poolingParamsStructureForThisPathwayType[layer_i]
            
            myLogger.print3("\t[Conv.Layer_" + str(layer_i) + "], Filter Shape: " + str(thisLayerFilterShape))
            myLogger.print3("\t[Conv.Layer_" + str(layer_i) + "], Input's Shape: (Train) " + str(inputToNextLayerShapeTrain) + \
                            ", (Val) " + str(inputToNextLayerShapeVal) + ", (Test) " + str(inputToNextLayerShapeTest))
            
            if layer_i in indicesOfLowerRankLayersForPathway :
                layer = LowRankConvLayer(ranksOfLowerRankLayersForPathway[ indicesOfLowerRankLayersForPathway.index(layer_i) ])
            else : # normal conv layer
                layer = ConvLayer()
            layer.makeLayer(rng,
                            inputToLayerTrain=inputToNextLayerTrain,
                            inputToLayerVal=inputToNextLayerVal,
                            inputToLayerTest=inputToNextLayerTest,
                            inputToLayerShapeTrain=inputToNextLayerShapeTrain,
                            inputToLayerShapeVal=inputToNextLayerShapeVal,
                            inputToLayerShapeTest=inputToNextLayerShapeTest,
                            
                            filterShape=thisLayerFilterShape,
                            poolingParameters=thisLayerPoolingParameters,
                            convWInitMethod=convWInitMethod,
                            useBnFlag = thisLayerUseBn,
                            rollingAverageForBatchNormalizationOverThatManyBatches=rollingAverageForBatchNormalizationOverThatManyBatches,
                            activationFunc=thisLayerActivFunc,
                            dropoutRate=thisLayerDropoutRate
                            ) 
            self._layersInPathway.append(layer)
            
            if layer_i not in indicesOfLayersToConnectResidualsInOutputForPathway : #not a residual connecting here
                inputToNextLayerTrain = layer.outputTrain
                inputToNextLayerVal = layer.outputVal
                inputToNextLayerTest = layer.outputTest
            else : #make residual connection
                myLogger.print3("\t[Pathway_"+str(self.getStringType())+"]: making Residual Connection between output of [Layer_"+str(layer_i)+"] to input of previous layer.")
                deeperLayerOutputImagesTrValTest = (layer.outputTrain, layer.outputVal, layer.outputTest)
                deeperLayerOutputImageShapesTrValTest = (layer.outputShapeTrain, layer.outputShapeVal, layer.outputShapeTest)
                assert layer_i > 0 # The very first layer (index 0), should never be provided for now. Cause I am connecting 2 layers back.
                earlierLayer = self._layersInPathway[layer_i-1]
                earlierLayerOutputImagesTrValTest = (earlierLayer.inputTrain, earlierLayer.inputVal, earlierLayer.inputTest)
                earlierLayerOutputImageShapesTrValTest = (earlierLayer.inputShapeTrain, earlierLayer.inputShapeVal, earlierLayer.inputShapeTest)
                
                (inputToNextLayerTrain,
                inputToNextLayerVal,
                inputToNextLayerTest) = makeResidualConnectionBetweenLayersAndReturnOutput( myLogger,
                                                                                            deeperLayerOutputImagesTrValTest,
                                                                                            deeperLayerOutputImageShapesTrValTest,
                                                                                            earlierLayerOutputImagesTrValTest,
                                                                                            earlierLayerOutputImageShapesTrValTest )
                layer.outputAfterResidualConnIfAnyAtOutpTrain = inputToNextLayerTrain
                layer.outputAfterResidualConnIfAnyAtOutpVal = inputToNextLayerVal
                layer.outputAfterResidualConnIfAnyAtOutpTest = inputToNextLayerTest
            # Residual connections preserve the both the number of FMs and the dimensions of the FMs, the same as in the later, deeper layer.
            inputToNextLayerShapeTrain = layer.outputShapeTrain
            inputToNextLayerShapeVal = layer.outputShapeVal
            inputToNextLayerShapeTest = layer.outputShapeTest
        
        self._setOutputAttributes(inputToNextLayerTrain, inputToNextLayerVal, inputToNextLayerTest,
                                inputToNextLayerShapeTrain, inputToNextLayerShapeVal, inputToNextLayerShapeTest)
        
        myLogger.print3("\t[Pathway_"+str(self.getStringType())+"]: Output's Shape: (Train) " + str(self._outputShapeTrain) + \
                        ", (Val) " + str(self._outputShapeVal) + ", (Test) " + str(self._outputShapeTest))
        
        myLogger.print3("[Pathway_" + str(self.getStringType()) + "] done.")
        
    # Skip connections to end of pathway.
    def makeMultiscaleConnectionsForLayerType(self, convLayersToConnectToFirstFcForMultiscaleFromThisLayerType) :
        
        layersInThisPathway = self.getLayers()
        
        [outputOfPathwayTrain, outputOfPathwayVal, outputOfPathwayTest ] = self.getOutput()
        [outputShapeTrain, outputShapeVal, outputShapeTest] = self.getShapeOfOutput()
        numOfCentralVoxelsToGetTrain = outputShapeTrain[2:]; numOfCentralVoxelsToGetVal = outputShapeVal[2:]; numOfCentralVoxelsToGetTest = outputShapeTest[2:]
        
        for convLayer_i in convLayersToConnectToFirstFcForMultiscaleFromThisLayerType :
            thisLayer = layersInThisPathway[convLayer_i]
                    
            middlePartOfFmsTrain = getMiddlePartOfFms(thisLayer.outputTrain, numOfCentralVoxelsToGetTrain)
            middlePartOfFmsVal = getMiddlePartOfFms(thisLayer.outputVal, numOfCentralVoxelsToGetVal)
            middlePartOfFmsTest = getMiddlePartOfFms(thisLayer.outputTest, numOfCentralVoxelsToGetTest)
            
            outputOfPathwayTrain = T.concatenate([outputOfPathwayTrain, middlePartOfFmsTrain], axis=1)
            outputOfPathwayVal = T.concatenate([outputOfPathwayVal, middlePartOfFmsVal], axis=1)
            outputOfPathwayTest = T.concatenate([outputOfPathwayTest, middlePartOfFmsTest], axis=1)
            outputShapeTrain[1] += thisLayer.getNumberOfFeatureMaps(); outputShapeVal[1] += thisLayer.getNumberOfFeatureMaps(); outputShapeTest[1] += thisLayer.getNumberOfFeatureMaps(); 
            
        self._setOutputAttributes(outputOfPathwayTrain, outputOfPathwayVal, outputOfPathwayTest,
                                outputShapeTrain, outputShapeVal, outputShapeTest)
        
    # The below should be updated, and calculated in here properly with private function and per layer.
    def calcRecFieldOfPathway(self, kernelDimsPerLayer) :
        return calcRecFieldFromKernDimListPerLayerWhenStrides1(kernelDimsPerLayer)
        
    def calcInputRczDimsToProduceOutputFmsOfCompatibleDims(self, thisPathWayKernelDims, dimsOfOutputFromPrimaryPathway):
        recFieldAtEndOfPathway = self.calcRecFieldOfPathway(thisPathWayKernelDims)
        rczDimsOfInputToPathwayShouldBe = [-1,-1,-1]
        rczDimsOfOutputOfPathwayShouldBe = [-1,-1,-1]
        
        rczDimsOfOutputFromPrimaryPathway = dimsOfOutputFromPrimaryPathway[2:]
        for rcz_i in xrange(3) :
            rczDimsOfOutputOfPathwayShouldBe[rcz_i] = int(ceil(rczDimsOfOutputFromPrimaryPathway[rcz_i]/(1.0*self.subsFactor()[rcz_i])))
            rczDimsOfInputToPathwayShouldBe[rcz_i] = recFieldAtEndOfPathway[rcz_i] + rczDimsOfOutputOfPathwayShouldBe[rcz_i] - 1
        return rczDimsOfInputToPathwayShouldBe
        
    # Setters
    def _setInputAttributes(self, inputToLayerTrain, inputToLayerVal, inputToLayerTest, inputToLayerShapeTrain, inputToLayerShapeVal, inputToLayerShapeTest) :
        self._inputTrain = inputToLayerTrain; self._inputVal = inputToLayerVal; self._inputTest = inputToLayerTest
        self._inputShapeTrain = inputToLayerShapeTrain; self._inputShapeVal = inputToLayerShapeVal; self._inputShapeTest = inputToLayerShapeTest
        
    def _setOutputAttributes(self, outputTrain, outputVal, outputTest, outputShapeTrain, outputShapeVal, outputShapeTest) :
        self._outputTrain = outputTrain; self._outputVal = outputVal; self._outputTest = outputTest
        self._outputShapeTrain = outputShapeTrain; self._outputShapeVal = outputShapeVal; self._outputShapeTest = outputShapeTest
        
    # Getters
    def pName(self):
        return self._pName
    def pType(self):
        return self._pType
    def getLayers(self):
        return self._layersInPathway
    def getLayer(self, index):
        return self._layersInPathway[index]
    def subsFactor(self):
        return self._subsFactor
    def getOutput(self):
        return [ self._outputTrain, self._outputVal, self._outputTest ]
    def getShapeOfOutput(self):
        return [ self._outputShapeTrain, self._outputShapeVal, self._outputShapeTest ]
    def getShapeOfInput(self):
        return [ self._inputShapeTrain, self._inputShapeVal, self._inputShapeTest ]
        
    # Other API :
    def getStringType(self) : raise NotImplementedMethod() # Abstract implementation. Children classes should implement this.
    # Will be overriden for lower-resolution pathways.
    def getOutputAtNormalRes(self): return self.getOutput()
    def getShapeOfOutputAtNormalRes(self): return self.getShapeOfOutput()
    
class NormalPathway(Pathway):
    def __init__(self, pName=None) :
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.NORM
    # Override parent's abstract classes.
    def getStringType(self) :
        return "NORMAL"
        
class SubsampledPathway(Pathway):
    def __init__(self, subsamplingFactor, pName=None) :
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.SUBS
        self._subsFactor = subsamplingFactor
        
        self._outputNormResTrain = None
        self._outputNormResVal = None
        self._outputNormResTest = None
        self._outputNormResShapeTrain = None
        self._outputNormResShapeVal = None
        self._outputNormResShapeTest = None
        
    def upsampleOutputToNormalRes(self, upsamplingScheme="repeat",
                            shapeToMatchInRczTrain=None, shapeToMatchInRczVal=None, shapeToMatchInRczTest=None):
        #should be called only once to build. Then just call getters if needed to get upsampled layer again.
        [outputTrain, outputVal, outputTest] = self.getOutput()
        [outputShapeTrain, outputShapeVal, outputShapeTest] = self.getShapeOfOutput()
        
        outputNormResTrain = upsampleRcz5DimArrayAndOptionalCrop(outputTrain,
                                                                self.subsFactor(),
                                                                upsamplingScheme,
                                                                shapeToMatchInRczTrain)
        outputNormResVal = upsampleRcz5DimArrayAndOptionalCrop( outputVal,
                                                                self.subsFactor(),
                                                                upsamplingScheme,
                                                                shapeToMatchInRczVal)
        outputNormResTest = upsampleRcz5DimArrayAndOptionalCrop(outputTest,
                                                                self.subsFactor(),
                                                                upsamplingScheme,
                                                                shapeToMatchInRczTest)
        
        outputNormResShapeTrain = outputShapeTrain[:2] + shapeToMatchInRczTrain[2:]
        outputNormResShapeVal = outputShapeVal[:2] + shapeToMatchInRczVal[2:]
        outputNormResShapeTest = outputShapeTest[:2] + shapeToMatchInRczTest[2:]
        
        self._setOutputAttributesNormRes(outputNormResTrain, outputNormResVal, outputNormResTest,
                                outputNormResShapeTrain, outputNormResShapeVal, outputNormResShapeTest)
        
    def _setOutputAttributesNormRes(self, outputNormResTrain, outputNormResVal, outputNormResTest,
                                    outputNormResShapeTrain, outputNormResShapeVal, outputNormResShapeTest) :
        #Essentially this is after the upsampling "layer"
        self._outputNormResTrain = outputNormResTrain; self._outputNormResVal = outputNormResVal; self._outputNormResTest = outputNormResTest
        self._outputNormResShapeTrain = outputNormResShapeTrain; self._outputNormResShapeVal = outputNormResShapeVal; self._outputNormResShapeTest = outputNormResShapeTest
        
        
    # OVERRIDING parent's classes.
    def getStringType(self) :
        return "SUBSAMPLED" + str(self.subsFactor())
        
    def getOutputAtNormalRes(self):
        # upsampleOutputToNormalRes() must be called first once.
        return [ self._outputNormResTrain, self._outputNormResVal, self._outputNormResTest ]
        
    def getShapeOfOutputAtNormalRes(self):
        # upsampleOutputToNormalRes() must be called first once.
        return [ self._outputNormResShapeTrain, self._outputNormResShapeVal, self._outputNormResShapeTest ]
        
             
class FcPathway(Pathway):
    def __init__(self, pName=None) :
        Pathway.__init__(self, pName)
        self._pType = PathwayTypes.FC
    # Override parent's abstract classes.
    def getStringType(self) :
        return "FC"

try:
    from sys import maxint as MAX_INT
except ImportError:
    # python3 compatibility
    from sys import maxsize as MAX_INT

from deepmedic.neuralnet.ops import applyDropout, makeBiasParamsAndApplyToFms, applyRelu, applyPrelu, applyElu, applySelu, pool3dMirrorPad
from deepmedic.neuralnet.ops import applyBn, createAndInitializeWeightsTensor, convolveWithGivenWeightMatrix, applySoftmaxToFmAndReturnProbYandPredY



def checkDimsOfYpredAndYEqual(y, yPred, stringTrainOrVal) :
    if y.ndim != yPred.ndim:
        raise TypeError( "ERROR! y did not have the same shape as y_pred during " + stringTrainOrVal,
                        ('y', y.type, 'y_pred', yPred.type) )

# Inheritance:
# Block -> ConvLayer -> LowRankConvLayer
#                L-----> ConvLayerWithSoftmax

class Block(object):
    
    def __init__(self) :
        self.inputTrain = None
        self.inputVal = None
        self.inputTest = None
        self.inputShapeTrain = None
        self.inputShapeVal = None
        self.inputShapeTest = None
        
        self._numberOfFeatureMaps = None
        self._poolingParameters = None
        
        self._appliedBnInLayer = None # This flag is a combination of rollingAverageForBn>0 AND useBnFlag, with the latter used for the 1st layers of pathways (on image).
        
        # All trainable parameters
        # NOTE: VIOLATED _HIDDEN ENCAPSULATION BY THE FUNCTION THAT TRANSFERS PRETRAINED WEIGHTS deepmed.neuralnet.transferParameters.transferParametersBetweenLayers.
        # TEMPORARY TILL THE API GETS FIXED (AFTER DA)!
        self.params = [] # W, (gbn), b, (aPrelu)
        self._W = None # Careful. LowRank does not set this. Uses ._WperSubconv
        self._b = None # shape: a vector with one value per FM of the input
        self._gBn = None # ONLY WHEN BN is applied
        self._aPrelu = None # ONLY WHEN PreLu
        
        # ONLY WHEN BN! All of these are for the rolling average! If I fix this, only 2 will remain!
        self._muBnsArrayForRollingAverage = None # Array
        self._varBnsArrayForRollingAverage = None # Arrays
        self._rollingAverageForBatchNormalizationOverThatManyBatches = None
        self._indexWhereRollingAverageIs = 0 #Index in the rolling-average matrices of the layers, of the entry to update in the next batch.
        self._sharedNewMu_B = None # last value shared, to update the rolling average array.
        self._sharedNewVar_B = None
        self._newMu_B = None # last value tensor, to update the corresponding shared.
        self._newVar_B = None
        
        
        # === Output of the block ===
        self.outputTrain = None
        self.outputVal = None
        self.outputTest = None
        self.outputShapeTrain = None
        self.outputShapeVal = None
        self.outputShapeTest = None
        # New and probably temporary, for the residual connections to be "visible".
        self.outputAfterResidualConnIfAnyAtOutpTrain = None
        self.outputAfterResidualConnIfAnyAtOutpVal = None
        self.outputAfterResidualConnIfAnyAtOutpTest = None
        
        # ==== Target Block Connected to that layer (softmax, regression, auxiliary loss etc), if any ======
        self.targetBlock = None
        
    # Setters
    def _setBlocksInputAttributes(self, inputToLayerTrain, inputToLayerVal, inputToLayerTest, inputToLayerShapeTrain, inputToLayerShapeVal, inputToLayerShapeTest) :
        self.inputTrain = inputToLayerTrain
        self.inputVal = inputToLayerVal
        self.inputTest = inputToLayerTest
        self.inputShapeTrain = inputToLayerShapeTrain
        self.inputShapeVal = inputToLayerShapeVal
        self.inputShapeTest = inputToLayerShapeTest
        
    def _setBlocksArchitectureAttributes(self, filterShape, poolingParameters) :
        self._numberOfFeatureMaps = filterShape[0] # Of the output! Used in trainValidationVisualise.py. Not of the input!
        assert self.inputShapeTrain[1] == filterShape[1]
        self._poolingParameters = poolingParameters
        
    def _setBlocksOutputAttributes(self, outputTrain, outputVal, outputTest, outputShapeTrain, outputShapeVal, outputShapeTest) :
        self.outputTrain = outputTrain
        self.outputVal = outputVal
        self.outputTest = outputTest
        self.outputShapeTrain = outputShapeTrain
        self.outputShapeVal = outputShapeVal
        self.outputShapeTest = outputShapeTest
        # New and probably temporary, for the residual connections to be "visible".
        self.outputAfterResidualConnIfAnyAtOutpTrain = self.outputTrain
        self.outputAfterResidualConnIfAnyAtOutpVal = self.outputVal
        self.outputAfterResidualConnIfAnyAtOutpTest = self.outputTest
        
    def setTargetBlock(self, targetBlockInstance):
        # targetBlockInstance : eg softmax layer. Future: Regression layer, or other auxiliary classifiers.
        self.targetBlock = targetBlockInstance
    # Getters
    def getNumberOfFeatureMaps(self):
        return self._numberOfFeatureMaps
    def fmsActivations(self, indices_of_fms_in_layer_to_visualise_from_to_exclusive) :
        return self.outputTest[:, indices_of_fms_in_layer_to_visualise_from_to_exclusive[0] : indices_of_fms_in_layer_to_visualise_from_to_exclusive[1], :, :, :]
    
    # Other API
    def getL1RegCost(self) : #Called for L1 weigths regularisation
        raise NotImplementedMethod() # Abstract implementation. Children classes should implement this.
    def getL2RegCost(self) : #Called for L2 weigths regularisation
        raise NotImplementedMethod()
    def getTrainableParams(self):
        if self.targetBlock == None :
            return self.params
        else :
            return self.params + self.targetBlock.getTrainableParams()
        
    def updateTheMatricesWithTheLastMusAndVarsForTheRollingAverageOfBNInference(self):
        # This function should be erazed when I reimplement the Rolling average.
        if self._appliedBnInLayer :
            muArrayValue = self._muBnsArrayForRollingAverage.get_value()
            muArrayValue[self._indexWhereRollingAverageIs] = self._sharedNewMu_B.get_value()
            self._muBnsArrayForRollingAverage.set_value(muArrayValue, borrow=True)
            
            varArrayValue = self._varBnsArrayForRollingAverage.get_value()
            varArrayValue[self._indexWhereRollingAverageIs] = self._sharedNewVar_B.get_value()
            self._varBnsArrayForRollingAverage.set_value(varArrayValue, borrow=True)
            self._indexWhereRollingAverageIs = (self._indexWhereRollingAverageIs + 1) % self._rollingAverageForBatchNormalizationOverThatManyBatches
            
    def getUpdatesForBnRollingAverage(self) :
        # This function or something similar should stay, even if I clean the BN rolling average.
        if self._appliedBnInLayer :
            #CAREFUL: WARN, PROBLEM, THEANO BUG! If a layer has only 1FM, the .newMu_B ends up being of type (true,) instead of vector!!! Error!!!
            return [(self._sharedNewMu_B, self._newMu_B),
                    (self._sharedNewVar_B, self._newVar_B) ]
        else :
            return []
        
class ConvLayer(Block):
    
    def __init__(self) :
        Block.__init__(self)
        self._activationFunctionType = "" #linear, relu or prelu
        
    def _processInputWithBnNonLinearityDropoutPooling(self,
                rng,
                inputToLayerTrain,
                inputToLayerVal,
                inputToLayerTest,
                inputToLayerShapeTrain,
                inputToLayerShapeVal,
                inputToLayerShapeTest,
                useBnFlag, # Must be true to do BN. Used to not allow doing BN on first layers straight on image, even if rollingAvForBnOverThayManyBatches > 0.
                rollingAverageForBatchNormalizationOverThatManyBatches, #If this is <= 0, we are not using BatchNormalization, even if above is True.
                activationFunc,
                dropoutRate) :
        # ---------------- Order of what is applied -----------------
        #  Input -> [ BatchNorm OR biases applied] -> NonLinearity -> DropOut -> Pooling --> Conv ] # ala He et al "Identity Mappings in Deep Residual Networks" 2016
        # -----------------------------------------------------------
        
        #---------------------------------------------------------
        #------------------ Batch Normalization ------------------
        #---------------------------------------------------------
        if useBnFlag and rollingAverageForBatchNormalizationOverThatManyBatches > 0 :
            self._appliedBnInLayer = True
            self._rollingAverageForBatchNormalizationOverThatManyBatches = rollingAverageForBatchNormalizationOverThatManyBatches
            (inputToNonLinearityTrain,
            inputToNonLinearityVal,
            inputToNonLinearityTest,
            self._gBn,
            self._b,
            # For rolling average :
            self._muBnsArrayForRollingAverage,
            self._varBnsArrayForRollingAverage,
            self._sharedNewMu_B,
            self._sharedNewVar_B,
            self._newMu_B,
            self._newVar_B
            ) = applyBn( rollingAverageForBatchNormalizationOverThatManyBatches, inputToLayerTrain, inputToLayerVal, inputToLayerTest, inputToLayerShapeTrain)
            self.params = self.params + [self._gBn, self._b]
        else : #Not using batch normalization
            self._appliedBnInLayer = False
            #make the bias terms and apply them. Like the old days before BN's own learnt bias terms.
            numberOfInputChannels = inputToLayerShapeTrain[1]
            
            (self._b,
            inputToNonLinearityTrain,
            inputToNonLinearityVal,
            inputToNonLinearityTest) = makeBiasParamsAndApplyToFms( inputToLayerTrain, inputToLayerVal, inputToLayerTest, numberOfInputChannels )
            self.params = self.params + [self._b]
            
        #--------------------------------------------------------
        #------------ Apply Activation/ non-linearity -----------
        #--------------------------------------------------------
        self._activationFunctionType = activationFunc
        if self._activationFunctionType == "linear" : # -1 stands for "no nonlinearity". Used for input layers of the pathway.
            ( inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest ) = (inputToNonLinearityTrain, inputToNonLinearityVal, inputToNonLinearityTest)
        elif self._activationFunctionType == "relu" :
            ( inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest ) = applyRelu(inputToNonLinearityTrain, inputToNonLinearityVal, inputToNonLinearityTest)
        elif self._activationFunctionType == "prelu" :
            numberOfInputChannels = inputToLayerShapeTrain[1]
            ( self._aPrelu, inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest ) = applyPrelu(inputToNonLinearityTrain, inputToNonLinearityVal, inputToNonLinearityTest, numberOfInputChannels)
            self.params = self.params + [self._aPrelu]
        elif self._activationFunctionType == "elu" :
            ( inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest ) = applyElu(inputToNonLinearityTrain, inputToNonLinearityVal, inputToNonLinearityTest)
        elif self._activationFunctionType == "selu" :
            ( inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest ) = applySelu(inputToNonLinearityTrain, inputToNonLinearityVal, inputToNonLinearityTest)
            
        #------------------------------------
        #------------- Dropout --------------
        #------------------------------------
        (inputToPoolTrain, inputToPoolVal, inputToPoolTest) = applyDropout(rng, dropoutRate, inputToLayerShapeTrain, inputToDropoutTrain, inputToDropoutVal, inputToDropoutTest)
        
        #-------------------------------------------------------
        #-----------  Pooling ----------------------------------
        #-------------------------------------------------------
        if self._poolingParameters == [] : #no max pooling before this conv
            inputToConvTrain = inputToPoolTrain
            inputToConvVal = inputToPoolVal
            inputToConvTest = inputToPoolTest
            
            inputToConvShapeTrain = inputToLayerShapeTrain
            inputToConvShapeVal = inputToLayerShapeVal
            inputToConvShapeTest = inputToLayerShapeTest
        else : #Max pooling is actually happening here...
            (inputToConvTrain, inputToConvShapeTrain) = pool3dMirrorPad(inputToPoolTrain, inputToLayerShapeTrain, self._poolingParameters)
            (inputToConvVal, inputToConvShapeVal) = pool3dMirrorPad(inputToPoolVal, inputToLayerShapeVal, self._poolingParameters)
            (inputToConvTest, inputToConvShapeTest) = pool3dMirrorPad(inputToPoolTest, inputToLayerShapeTest, self._poolingParameters)
            
        return (inputToConvTrain, inputToConvVal, inputToConvTest,
                inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest )
        
    def _createWeightsTensorAndConvolve(self, rng, filterShape, convWInitMethod, 
                                        inputToConvTrain, inputToConvVal, inputToConvTest,
                                        inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest) :
        #-----------------------------------------------
        #------------------ Convolution ----------------
        #-----------------------------------------------
        #----- Initialise the weights -----
        # W shape: [#FMs of this layer, #FMs of Input, rKernDim, cKernDim, zKernDim]
        self._W = createAndInitializeWeightsTensor(filterShape, convWInitMethod, rng)
        self.params = [self._W] + self.params
        
        #---------- Convolve --------------
        tupleWithOuputAndShapeTrValTest = convolveWithGivenWeightMatrix(self._W, filterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest)
        
        return tupleWithOuputAndShapeTrValTest
    
    # The main function that builds this.
    def makeLayer(self,
                rng,
                inputToLayerTrain,
                inputToLayerVal,
                inputToLayerTest,
                inputToLayerShapeTrain,
                inputToLayerShapeVal,
                inputToLayerShapeTest,
                filterShape,
                poolingParameters, # Can be []
                convWInitMethod,
                useBnFlag, # Must be true to do BN. Used to not allow doing BN on first layers straight on image, even if rollingAvForBnOverThayManyBatches > 0.
                rollingAverageForBatchNormalizationOverThatManyBatches, #If this is <= 0, we are not using BatchNormalization, even if above is True.
                activationFunc="relu",
                dropoutRate=0.0):
        """
        type rng: numpy.random.RandomState
        param rng: a random number generator used to initialize weights
        
        type inputToLayer:  tensor5 = theano.tensor.TensorType(dtype='float32', broadcastable=(False, False, False, False, False))
        param inputToLayer: symbolic image tensor, of shape inputToLayerShape
        
        type filterShape: tuple or list of length 5
        param filterShape: (number of filters, num input feature maps,
                            filter height, filter width, filter depth)
                            
        type inputToLayerShape: tuple or list of length 5
        param inputToLayerShape: (batch size, num input feature maps,
                            image height, image width, filter depth)
        """
        self._setBlocksInputAttributes(inputToLayerTrain, inputToLayerVal, inputToLayerTest, inputToLayerShapeTrain, inputToLayerShapeVal, inputToLayerShapeTest)
        self._setBlocksArchitectureAttributes(filterShape, poolingParameters)
        
        # Apply all the straightforward operations on the input, such as BN, activation function, dropout, pooling        
        (inputToConvTrain, inputToConvVal, inputToConvTest,
        inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest) = self._processInputWithBnNonLinearityDropoutPooling( rng,
                                                                                        inputToLayerTrain,
                                                                                        inputToLayerVal,
                                                                                        inputToLayerTest,
                                                                                        inputToLayerShapeTrain,
                                                                                        inputToLayerShapeVal,
                                                                                        inputToLayerShapeTest,
                                                                                        useBnFlag,
                                                                                        rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                                        activationFunc,
                                                                                        dropoutRate)
        
        tupleWithOuputAndShapeTrValTest = self._createWeightsTensorAndConvolve( rng, filterShape, convWInitMethod, 
                                                                                inputToConvTrain, inputToConvVal, inputToConvTest,
                                                                                inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest)
        
        self._setBlocksOutputAttributes(*tupleWithOuputAndShapeTrValTest)
        
    # Override parent's abstract classes.
    def getL1RegCost(self) : #Called for L1 weigths regularisation
        return abs(self._W).sum()
    def getL2RegCost(self) : #Called for L2 weigths regularisation
        return (self._W ** 2).sum()
    
    
# Ala Yani Ioannou et al, Training CNNs with Low-Rank Filters For Efficient Image Classification, ICLR 2016. Allowed Ranks: Rank=1 or 2.
class LowRankConvLayer(ConvLayer):
    def __init__(self, rank=2) :
        ConvLayer.__init__(self)
        
        self._WperSubconv = None # List of ._W theano tensors. One per low-rank subconv. Treat carefully. 
        del(self._W) # The ._W of the Block parent is not used.
        self._rank = rank # 1 or 2 dimensions
        
    def _cropSubconvOutputsToSameDimsAndConcatenateFms( self,
                                                        rSubconvOutput, rSubconvOutputShape,
                                                        cSubconvOutput, cSubconvOutputShape,
                                                        zSubconvOutput, zSubconvOutputShape,
                                                        filterShape) :
        assert (rSubconvOutputShape[0] == cSubconvOutputShape[0]) and (cSubconvOutputShape[0] == zSubconvOutputShape[0]) # same batch size.
        
        concatOutputShape = [ rSubconvOutputShape[0],
                                rSubconvOutputShape[1] + cSubconvOutputShape[1] + zSubconvOutputShape[1],
                                rSubconvOutputShape[2],
                                cSubconvOutputShape[3],
                                zSubconvOutputShape[4]
                                ]
        rCropSlice = slice( (filterShape[2]-1)//2, (filterShape[2]-1)//2 + concatOutputShape[2] )
        cCropSlice = slice( (filterShape[3]-1)//2, (filterShape[3]-1)//2 + concatOutputShape[3] )
        zCropSlice = slice( (filterShape[4]-1)//2, (filterShape[4]-1)//2 + concatOutputShape[4] )
        rSubconvOutputCropped = rSubconvOutput[:,:, :, cCropSlice if self._rank == 1 else slice(0, MAX_INT), zCropSlice  ]
        cSubconvOutputCropped = cSubconvOutput[:,:, rCropSlice, :, zCropSlice if self._rank == 1 else slice(0, MAX_INT) ]
        zSubconvOutputCropped = zSubconvOutput[:,:, rCropSlice if self._rank == 1 else slice(0, MAX_INT), cCropSlice, : ]
        concatSubconvOutputs = T.concatenate([rSubconvOutputCropped, cSubconvOutputCropped, zSubconvOutputCropped], axis=1) #concatenate the FMs
        
        return (concatSubconvOutputs, concatOutputShape)
    
    # Overload the ConvLayer's function. Called from makeLayer. The only different behaviour, because BN, ActivationFunc, DropOut and Pooling are done on a per-FM fashion.        
    def _createWeightsTensorAndConvolve(self, rng, filterShape, convWInitMethod, 
                                        inputToConvTrain, inputToConvVal, inputToConvTest,
                                        inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest) :
        # Behaviour: Create W, set self._W, set self.params, convolve, return ouput and outputShape.
        # The created filters are either 1-dimensional (rank=1) or 2-dim (rank=2), depending  on the self._rank
        # If 1-dim: rSubconv is the input convolved with the row-1dimensional filter.
        # If 2-dim: rSubconv is the input convolved with the RC-2D filter, cSubconv with CZ-2D filter, zSubconv with ZR-2D filter. 
        
        #----- Initialise the weights and Convolve for 3 separate, low rank filters, R,C,Z. -----
        # W shape: [#FMs of this layer, #FMs of Input, rKernDim, cKernDim, zKernDim]
        
        rSubconvFilterShape = [ filterShape[0]//3, filterShape[1], filterShape[2], 1 if self._rank == 1 else filterShape[3], 1 ]
        rSubconvW = createAndInitializeWeightsTensor(rSubconvFilterShape, convWInitMethod, rng)
        rSubconvTupleWithOuputAndShapeTrValTest = convolveWithGivenWeightMatrix(rSubconvW, rSubconvFilterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest)
        
        cSubconvFilterShape = [ filterShape[0]//3, filterShape[1], 1, filterShape[3], 1 if self._rank == 1 else filterShape[4] ]
        cSubconvW = createAndInitializeWeightsTensor(cSubconvFilterShape, convWInitMethod, rng)
        cSubconvTupleWithOuputAndShapeTrValTest = convolveWithGivenWeightMatrix(cSubconvW, cSubconvFilterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest)
        
        numberOfFmsForTotalToBeExact = filterShape[0] - 2*(filterShape[0]//3) # Cause of possibly inexact integer division.
        zSubconvFilterShape = [ numberOfFmsForTotalToBeExact, filterShape[1], 1 if self._rank == 1 else filterShape[2], 1, filterShape[4] ]
        zSubconvW = createAndInitializeWeightsTensor(zSubconvFilterShape, convWInitMethod, rng)
        zSubconvTupleWithOuputAndShapeTrValTest = convolveWithGivenWeightMatrix(zSubconvW, zSubconvFilterShape, inputToConvTrain, inputToConvVal, inputToConvTest, inputToConvShapeTrain, inputToConvShapeVal, inputToConvShapeTest)
        
        # Set the W attribute and trainable parameters.
        self._WperSubconv = [rSubconvW, cSubconvW, zSubconvW] # Bear in mind that these sub tensors have different shapes! Treat carefully.
        self.params = self._WperSubconv + self.params
        
        # concatenate together.
        (concatSubconvOutputsTrain, concatOutputShapeTrain) = self._cropSubconvOutputsToSameDimsAndConcatenateFms(rSubconvTupleWithOuputAndShapeTrValTest[0], rSubconvTupleWithOuputAndShapeTrValTest[3],
                                                                                                        cSubconvTupleWithOuputAndShapeTrValTest[0], cSubconvTupleWithOuputAndShapeTrValTest[3],
                                                                                                        zSubconvTupleWithOuputAndShapeTrValTest[0], zSubconvTupleWithOuputAndShapeTrValTest[3],
                                                                                                        filterShape)
        (concatSubconvOutputsVal, concatOutputShapeVal) = self._cropSubconvOutputsToSameDimsAndConcatenateFms(rSubconvTupleWithOuputAndShapeTrValTest[1], rSubconvTupleWithOuputAndShapeTrValTest[4],
                                                                                                        cSubconvTupleWithOuputAndShapeTrValTest[1], cSubconvTupleWithOuputAndShapeTrValTest[4],
                                                                                                        zSubconvTupleWithOuputAndShapeTrValTest[1], zSubconvTupleWithOuputAndShapeTrValTest[4],
                                                                                                        filterShape)
        (concatSubconvOutputsTest, concatOutputShapeTest) = self._cropSubconvOutputsToSameDimsAndConcatenateFms(rSubconvTupleWithOuputAndShapeTrValTest[2], rSubconvTupleWithOuputAndShapeTrValTest[5],
                                                                                                        cSubconvTupleWithOuputAndShapeTrValTest[2], cSubconvTupleWithOuputAndShapeTrValTest[5],
                                                                                                        zSubconvTupleWithOuputAndShapeTrValTest[2], zSubconvTupleWithOuputAndShapeTrValTest[5],
                                                                                                        filterShape)
        
        return (concatSubconvOutputsTrain, concatSubconvOutputsVal, concatSubconvOutputsTest, concatOutputShapeTrain, concatOutputShapeVal, concatOutputShapeTest)
        
        
    # Implement parent's abstract classes.
    def getL1RegCost(self) : #Called for L1 weigths regularisation
        l1Cost = 0
        for wOfSubconv in self._WperSubconv : l1Cost += abs(wOfSubconv).sum()
        return l1Cost
    def getL2RegCost(self) : #Called for L2 weigths regularisation
        l2Cost = 0
        for wOfSubconv in self._WperSubconv : l2Cost += (wOfSubconv ** 2).sum()
        return l2Cost
    def getW(self):
        print("ERROR: For LowRankConvLayer, the ._W is not used! Use ._WperSubconv instead and treat carefully!! Exiting!"); exit(1)
        
        
class SoftmaxLayer(Block):
    """ Softmax for classification. Note, this is simply the softmax function, after adding bias. Not a ConvLayer """
    
    def __init__(self):
        Block.__init__(self)
        self._numberOfOutputClasses = None
        #self._b = None # The only type of trainable parameter that a softmax layer has.
        self._softmaxTemperature = None
        
    def makeLayer(  self,
                    rng,
                    layerConnected, # the basic layer, at the output of which to connect this softmax.
                    softmaxTemperature = 1):
        
        self._numberOfOutputClasses = layerConnected.getNumberOfFeatureMaps()
        self._softmaxTemperature = softmaxTemperature
        
        self._setBlocksInputAttributes(layerConnected.outputTrain, layerConnected.outputVal, layerConnected.outputTest,
                                        layerConnected.outputShapeTrain, layerConnected.outputShapeVal, layerConnected.outputShapeTest)
        
        # At this last classification layer, the conv output needs to have bias added before the softmax.
        # NOTE: So, two biases are associated with this layer. self.b which is added in the ouput of the previous layer's output of conv,
        # and this self._bClassLayer that is added only to this final output before the softmax.
        (self._b,
        biasedInputToSoftmaxTrain,
        biasedInputToSoftmaxVal,
        biasedInputToSoftmaxTest) = makeBiasParamsAndApplyToFms( self.inputTrain, self.inputVal, self.inputTest, self._numberOfOutputClasses )
        self.params = self.params + [self._b]
        
        # ============ Softmax ==============
        #self.p_y_given_x_2d_train = ? Can I implement negativeLogLikelihood without this ?
        ( self.p_y_given_x_train,
        self.y_pred_train ) = applySoftmaxToFmAndReturnProbYandPredY( biasedInputToSoftmaxTrain, self.inputShapeTrain, self._numberOfOutputClasses, softmaxTemperature)
        ( self.p_y_given_x_val,
        self.y_pred_val ) = applySoftmaxToFmAndReturnProbYandPredY( biasedInputToSoftmaxVal, self.inputShapeVal, self._numberOfOutputClasses, softmaxTemperature)
        ( self.p_y_given_x_test,
        self.y_pred_test ) = applySoftmaxToFmAndReturnProbYandPredY( biasedInputToSoftmaxTest, self.inputShapeTest, self._numberOfOutputClasses, softmaxTemperature)
        
        self._setBlocksOutputAttributes(self.p_y_given_x_train, self.p_y_given_x_val, self.p_y_given_x_test, self.inputShapeTrain, self.inputShapeVal, self.inputShapeTest)
        
        layerConnected.setTargetBlock(self)
        
        
    def negativeLogLikelihood(self, y, weightPerClass):
        # Used in training.
        # param y: y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        # weightPerClass is a vector with 1 element per class.
        
        #Weighting the cost of the different classes in the cost-function, in order to counter class imbalance.
        e1 = np.finfo(np.float32).tiny
        addTinyProbMatrix = T.lt(self.p_y_given_x_train, 4*e1) * e1
        
        weightPerClassBroadcasted = weightPerClass.dimshuffle('x', 0, 'x', 'x', 'x')
        log_p_y_given_x_train = T.log(self.p_y_given_x_train + addTinyProbMatrix) #added a tiny so that it does not go to zero and I have problems with nan again...
        weighted_log_p_y_given_x_train = log_p_y_given_x_train * weightPerClassBroadcasted
        # return -T.mean( weighted_log_p_y_given_x_train[T.arange(y.shape[0]), y] )
        
        # Not a very elegant way to do the indexing but oh well...
        indexDim0 = T.arange( weighted_log_p_y_given_x_train.shape[0] ).dimshuffle( 0, 'x','x','x')
        indexDim2 = T.arange( weighted_log_p_y_given_x_train.shape[2] ).dimshuffle('x', 0, 'x','x')
        indexDim3 = T.arange( weighted_log_p_y_given_x_train.shape[3] ).dimshuffle('x','x', 0, 'x')
        indexDim4 = T.arange( weighted_log_p_y_given_x_train.shape[4] ).dimshuffle('x','x','x', 0)
        return -T.mean( weighted_log_p_y_given_x_train[ indexDim0, y, indexDim2, indexDim3, indexDim4] )
    
    
    def meanErrorTraining(self, y):
        # Returns float = number of errors / number of examples of the minibatch ; [0., 1.]
        # param y: y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        
        # check if y has same dimension of y_pred
        checkDimsOfYpredAndYEqual(y, self.y_pred_train, "training")
        
        #Mean error of the training batch.
        tneq = T.neq(self.y_pred_train, y)
        meanError = T.mean(tneq)
        return meanError
    
    def meanErrorValidation(self, y):
        # y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        
        # check if y has same dimension of y_pred
        checkDimsOfYpredAndYEqual(y, self.y_pred_val, "validation")
        
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            tneq = T.neq(self.y_pred_val, y)
            meanError = T.mean(tneq)
            return meanError #The percentage of the predictions that is not the correct class.
        else:
            raise NotImplementedError()
        
    def getRpRnTpTnForTrain0OrVal1(self, y, training0OrValidation1):
        # The returned list has (numberOfClasses)x4 integers: >numberOfRealPositives, numberOfRealNegatives, numberOfTruePredictedPositives, numberOfTruePredictedNegatives< for each class (incl background).
        # Order in the list is the natural order of the classes (ie class-0 RP,RN,TPP,TPN, class-1 RP,RN,TPP,TPN, class-2 RP,RN,TPP,TPN ...)
        # param y: y = T.itensor4('y'). Dimensions [batchSize, r, c, z]
        
        yPredToUse = self.y_pred_train if  training0OrValidation1 == 0 else self.y_pred_val
        checkDimsOfYpredAndYEqual(y, yPredToUse, "training" if training0OrValidation1 == 0 else "validation")
        
        returnedListWithNumberOfRpRnTpTnForEachClass = []
        
        for class_i in xrange(0, self._numberOfOutputClasses) :
            #Number of Real Positive, Real Negatives, True Predicted Positives and True Predicted Negatives are reported PER CLASS (first for WHOLE).
            tensorOneAtRealPos = T.eq(y, class_i)
            tensorOneAtRealNeg = T.neq(y, class_i)

            tensorOneAtPredictedPos = T.eq(yPredToUse, class_i)
            tensorOneAtPredictedNeg = T.neq(yPredToUse, class_i)
            tensorOneAtTruePos = T.and_(tensorOneAtRealPos,tensorOneAtPredictedPos)
            tensorOneAtTrueNeg = T.and_(tensorOneAtRealNeg,tensorOneAtPredictedNeg)
                    
            returnedListWithNumberOfRpRnTpTnForEachClass.append( T.sum(tensorOneAtRealPos) )
            returnedListWithNumberOfRpRnTpTnForEachClass.append( T.sum(tensorOneAtRealNeg) )
            returnedListWithNumberOfRpRnTpTnForEachClass.append( T.sum(tensorOneAtTruePos) )
            returnedListWithNumberOfRpRnTpTnForEachClass.append( T.sum(tensorOneAtTrueNeg) )
            
        return returnedListWithNumberOfRpRnTpTnForEachClass
    
    def predictionProbabilities(self) :
        return self.p_y_given_x_test


try:
    import cPickle
except ImportError:
    # python3 compatibility
    import _pickle as cPickle

    
def load_object_from_file(filenameWithPath) :
    f = file(filenameWithPath, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj

def dump_object_to_file(my_obj, filenameWithPath) :
    """
    my_obj = object to pickle
    filenameWithPath = a string with the full path+name
    
    The function uses the 'highest_protocol' which is supposed to be more storage efficient.
    It uses cPickle, which is coded in c and is supposed to be faster than pickle.
    Remember, this instance is safe to load only from a code which is fully-compatible (same version)
    ...with the code this was saved from, i.e. same classes define.
    If I need forward compatibility, read this: http://deeplearning.net/software/theano/tutorial/loading_and_saving.html
    """
    f = file(filenameWithPath, 'wb')
    cPickle.dump(my_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
def load_object_from_gzip_file(filenameWithPath) :
    f = gzip.open(filenameWithPath, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj

def dump_object_to_gzip_file(my_obj, filenameWithPath) :
    f = gzip.open(filenameWithPath, 'wb')
    cPickle.dump(my_obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()
    
def dump_cnn_to_gzip_file_dotSave(cnnInstance, filenameWithPathToSaveTo, logger=None) :
    filenameWithPathToSaveToDotSave = os.path.abspath(filenameWithPathToSaveTo + ".save")
    cnnInstance.freeGpuTrainingData(); cnnInstance.freeGpuValidationData(); cnnInstance.freeGpuTestingData();
    # Clear out the compiled functions, so that they are not saved with the instance:
    compiledFunctionTrain = cnnInstance.cnnTrainModel; cnnInstance.cnnTrainModel = ""
    compiledFunctionVal = cnnInstance.cnnValidateModel; cnnInstance.cnnValidateModel = ""
    compiledFunctionTest = cnnInstance.cnnTestModel; cnnInstance.cnnTestModel = ""
    compiledFunctionVisualise = cnnInstance.cnnVisualiseFmFunction; cnnInstance.cnnVisualiseFmFunction = ""
    
    if logger != None :
        logger.print3("Saving network to: "+str(filenameWithPathToSaveToDotSave))
    else:
        print("Saving network to: "+str(filenameWithPathToSaveToDotSave))
        
    dump_object_to_gzip_file(cnnInstance, filenameWithPathToSaveToDotSave)
    
    if logger != None :
        logger.print3("Model saved.")
    else:
        print("Model saved.")
        
    # Restore instance's values, which were cleared for the saving of the instance:
    cnnInstance.cnnTrainModel = compiledFunctionTrain
    cnnInstance.cnnValidateModel = compiledFunctionVal
    cnnInstance.cnnTestModel = compiledFunctionTest
    cnnInstance.cnnVisualiseFmFunction = compiledFunctionVisualise
    
    return filenameWithPathToSaveToDotSave


def calculateSubsampledImagePartDimensionsFromImagePartSizePatchSizeAndSubsampleFactor(imagePartDimensions, patchDimensions, subsampleFactor) :
    """
    This function gives you how big your subsampled-image-part should be, so that it corresponds to the correct number of central-voxels in the normal-part. Currently, it's coupled with the patch-size of the normal-scale. I.e. the subsampled-patch HAS TO BE THE SAME SIZE as the normal-scale, and corresponds to subFactor*patchsize in context.
    When the central voxels are not a multiple of the subFactor, you get ceil(), so +1 sub-patch. When the CNN repeats the pattern, it is giving dimension higher than the central-voxels of the normal-part, but then they are sliced-down to the correct number (in the cnn_make_model function, right after the repeat).        
    This function works like this because of getImagePartFromSubsampledImageForTraining(), which gets a subsampled-image-part by going 1 normal-patch back from the top-left voxel of a normal-scale-part, and then 3 ahead. If I change it to start from the top-left-CENTRAL-voxel back and front, I will be able to decouple the normal-patch size and the subsampled-patch-size. 
    """
    #if patch is 17x17, a 17x17 subPart is cool for 3 voxels with a subsampleFactor. +2 to be ok for the 9x9 centrally classified voxels, so 19x19 sub-part.
    subsampledImagePartDimensions = []
    for rcz_i in xrange(len(imagePartDimensions)) :
        centralVoxelsInThisDimension = imagePartDimensions[rcz_i] - patchDimensions[rcz_i] + 1
        centralVoxelsInThisDimensionForSubsampledPart = int(ceil(centralVoxelsInThisDimension*1.0/subsampleFactor[rcz_i]))
        sizeOfSubsampledImagePartInThisDimension = patchDimensions[rcz_i] + centralVoxelsInThisDimensionForSubsampledPart - 1
        subsampledImagePartDimensions.append(sizeOfSubsampledImagePartInThisDimension)
    return subsampledImagePartDimensions

def calcRecFieldFromKernDimListPerLayerWhenStrides1(kernDimPerLayerList) :
    if not kernDimPerLayerList : #list is []
        return 0
    
    numberOfDimensions = len(kernDimPerLayerList[0])
    receptiveField = [1]*numberOfDimensions
    for dimension_i in xrange(numberOfDimensions) :
        for layer_i in xrange(len(kernDimPerLayerList)) :
            receptiveField[dimension_i] += kernDimPerLayerList[layer_i][dimension_i] - 1
    return receptiveField


def checkRecFieldVsSegmSize(receptiveFieldDim, segmentDim) :
    numberOfRFDim = len(receptiveFieldDim)
    numberOfSegmDim = len(segmentDim)
    if numberOfRFDim != numberOfSegmDim :
        print("ERROR: [in function checkRecFieldVsSegmSize()] : Receptive field and image segment have different number of dimensions! (should be 3 for both! Exiting!)")
        exit(1)
    for dim_i in xrange(numberOfRFDim) :
        if receptiveFieldDim[dim_i] > segmentDim[dim_i] :
            print("ERROR: [in function checkRecFieldVsSegmSize()] : The segment-size (input) should be at least as big as the receptive field of the model! This was not found to hold! Dimensions of Receptive Field:", receptiveFieldDim, ". Dimensions of Segment: ", segmentDim)
            return False
    return True

def checkKernDimPerLayerCorrect3dAndNumLayers(kernDimensionsPerLayer, numOfLayers) :
    #kernDimensionsPerLayer : a list with sublists. One sublist per layer. Each sublist should have 3 integers, specifying the dimensions of the kernel at the corresponding layer of the pathway. eg: kernDimensionsPerLayer = [ [3,3,3], [3,3,3], [5,5,5] ] 
    if kernDimensionsPerLayer == None or len(kernDimensionsPerLayer) != numOfLayers :
        return False
    for kernDimInLayer in kernDimensionsPerLayer :
        if len(kernDimInLayer) != 3 :
            return False
    return True

def checkSubsampleFactorEven(subFactor) :
    for dim_i in xrange(len(subFactor)) :
        if subFactor[dim_i]%2 != 1 :
            return False
    return True

#-----helper functions that I use in here---

def padImageWithMirroring(inputImage, voxelsPerDimToPad) :
    # inputImage shape: [batchSize, #channels#, r, c, z]
    # inputImageDimensions : [ batchSize, #channels, dim r, dim c, dim z ] of inputImage
    # voxelsPerDimToPad shape: [ num o voxels in r-dim to add, ...c-dim, ...z-dim ]
    # If voxelsPerDimToPad is odd, 1 more voxel is added to the right side.
    # r-axis
    assert np.all(voxelsPerDimToPad) >= 0
    padLeft = int(voxelsPerDimToPad[0] // 2); padRight = int((voxelsPerDimToPad[0] + 1) // 2);
    paddedImage = T.concatenate([inputImage[:, :, int(voxelsPerDimToPad[0] // 2) - 1::-1 , :, :], inputImage], axis=2) if padLeft > 0 else inputImage
    paddedImage = T.concatenate([paddedImage, paddedImage[ :, :, -1:-1 - int((voxelsPerDimToPad[0] + 1) // 2):-1, :, :]], axis=2) if padRight > 0 else paddedImage
    # c-axis
    padLeft = int(voxelsPerDimToPad[1] // 2); padRight = int((voxelsPerDimToPad[1] + 1) // 2);
    paddedImage = T.concatenate([paddedImage[:, :, :, padLeft - 1::-1 , :], paddedImage], axis=3) if padLeft > 0 else paddedImage
    paddedImage = T.concatenate([paddedImage, paddedImage[:, :, :, -1:-1 - padRight:-1, :]], axis=3) if padRight > 0 else paddedImage
    # z-axis
    padLeft = int(voxelsPerDimToPad[2] // 2); padRight = int((voxelsPerDimToPad[2] + 1) // 2)
    paddedImage = T.concatenate([paddedImage[:, :, :, :, padLeft - 1::-1 ], paddedImage], axis=4) if padLeft > 0 else paddedImage
    paddedImage = T.concatenate([paddedImage, paddedImage[:, :, :, :, -1:-1 - padRight:-1]], axis=4) if padRight > 0 else paddedImage
    
    return paddedImage



class Cnn3d(object):
    def __init__(self):
        
        self.cnnModelName = None
        
        self.pathways = [] # There should be only 1 normal and only one FC pathway. Eg, see self.getFcPathway()
        self.numSubsPaths = 0
        
        self.finalTargetLayer = ""
        
        self.numberOfOutputClasses = None
        
        self.cnnTrainModel = ""
        self.cnnValidateModel = ""
        self.cnnTestModel = ""
        self.cnnVisualiseFmFunction = ""
        
        self.recFieldCnn = ""
        
        self.borrowFlag = ""
        
        self.batchSize = ""
        self.batchSizeValidation = ""
        self.batchSizeTesting = ""
        
        # self.patchesToTrainPerImagePart = ""
        self.dataTypeX = ""
        self.nkerns = ""  # number of feature maps.
        self.nkernsSubsampled = ""
        
        # Fully Connected Layers
        self.kernelDimensionsFirstFcLayer = ""
        
        # Automatically lower CNN's learning rate by looking at validation accuracy:
        self.topMeanValidationAccuracyAchievedInEpoch = [-1, -1]
        self.lastEpochAtTheEndOfWhichLrWasLowered = 0  # refers to CnnTrained epochs, not the epochs in the do_training loop.
        
        # Residual Learning
        self.indicesOfLayersToConnectResidualsInOutput = ""
        
        # Lower rank convolutional layers
        self.indicesOfLowerRankLayersPerPathway = ""
        self.ranksOfLowerRankLayersForEachPathway = ""
        
        
        # ======= Shared Variables with X and Y data for training/validation/testing ======
        self._initializedSharedVarsTrain = False
        self.sharedInpXTrain = ""
        self.sharedInpXPerSubsListTrain = []
        self.sharedLabelsYTrain = ""
        self._initializedSharedVarsVal = False
        self.sharedInpXVal = ""
        self.sharedInpXPerSubsListVal = []
        self.sharedLabelsYVal = ""
        self._initializedSharedVarsTest = False
        self.sharedInpXTest = ""
        self.sharedInpXPerSubsListTest = []
        
        
        self.numberOfEpochsTrained = 0
        
        self._trainingStateAttributesInitialized = False
        
        self.indicesOfLayersPerPathwayTypeToFreeze = None
        self.costFunctionLetter = ""  # "L", "D" or "J"
        self.initialLearningRate = ""  # used by exponential schedule
        self.learning_rate = theano.shared(np.cast["float32"](0.01))  # initial value, changed in make_cnn_model().compileTrainingFunction()
        self.classicMomentum0OrNesterov1 = None
        # SGD + Classic momentum: (to save the momentum)
        self.initialMomentum = ""  # used by exponential schedule
        self.momentum = theano.shared(np.cast["float32"](0.))
        self.momentumTypeNONNormalized0orNormalized1 = None
        self.velocities_forMom = []  # list of shared_variables. Each of the individual Dws is a sharedVar. This whole thing isnt.
        self.sgd0orAdam1orRmsProp2 = None
        # ADAM:
        self.b1_adam = None
        self.b2_adam = None
        self.epsilonForAdam = None
        self.i_adam = theano.shared(np.cast["float32"](0.))  # Current iteration of adam
        self.m_listForAllParamsAdam = []  # list of mean of grads for all parameters, for ADAM optimizer.
        self.v_listForAllParamsAdam = []  # list of variances of grads for all parameters, for ADAM optimizer.
        # RMSProp
        self.rho_rmsProp = None
        self.epsilonForRmsProp = None
        self.accuGradSquare_listForAllParamsRmsProp = []  # the rolling average accumulator of the variance of the grad (grad^2)
        # Regularisation
        self.L1_reg_constant = None
        self.L2_reg_constant = None
        
        
        # Symbolic variables, which stand for the input. Will be loaded by the compiled trainining/val/test function. Can also be pre-set by an existing tensor if required in future extensions.
        self.inputTensorsXToCnnInitialized = False
        self.inputTensorNormTrain = None; self.inputTensorNormVal = None; self.inputTensorNormTest = None;
        self.listInputTensorPerSubsTrain = []; self.listInputTensorPerSubsVal = []; self.listInputTensorPerSubsTest = [];
        
    def getNumSubsPathways(self):
        count = 0
        for pathway in self.pathways :
            if pathway.pType() ==  pt.SUBS :
                count += 1
        return count
    
    def getNumPathwaysThatRequireInput(self):
        count = 0
        for pathway in self.pathways :
            if pathway.pType() != pt.FC :
                count += 1
        return count
    
    def getFcPathway(self):
        for pathway in self.pathways :
            if pathway.pType() == pt.FC :
                return pathway
        return None
    
    def increaseNumberOfEpochsTrained(self):
        self.numberOfEpochsTrained += 1
        
    def change_learning_rate_of_a_cnn(self, newValueForLearningRate, myLogger=None) :
        stringToPrint = "UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) + ") Changing the Cnn's Learning Rate to: " + str(newValueForLearningRate)
        if myLogger != None :
            myLogger.print3(stringToPrint)
        else :
            print(stringToPrint)
        self.learning_rate.set_value(newValueForLearningRate)
        self.lastEpochAtTheEndOfWhichLrWasLowered = self.numberOfEpochsTrained
        
    def divide_learning_rate_of_a_cnn_by(self, divideLrBy, myLogger=None) :
        oldLR = self.learning_rate.get_value()
        newValueForLearningRate = oldLR * 1.0 / divideLrBy
        self.change_learning_rate_of_a_cnn(newValueForLearningRate, myLogger)
        
    def change_momentum_of_a_cnn(self, newValueForMomentum, myLogger=None):
        stringToPrint = "UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) + ") Changing the Cnn's Momentum to: " + str(newValueForMomentum)
        if myLogger != None :
            myLogger.print3(stringToPrint)
        else :
            print(stringToPrint)
        self.momentum.set_value(newValueForMomentum)
        
    def multiply_momentum_of_a_cnn_by(self, multiplyMomentumBy, myLogger=None) :
        oldMom = self.momentum.get_value()
        newValueForMomentum = oldMom * multiplyMomentumBy
        self.change_momentum_of_a_cnn(newValueForMomentum, myLogger)
        
    def changeB1AndB2ParametersOfAdam(self, b1ParamForAdam, b2ParamForAdam, myLogger) :
        myLogger.print3("UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) + ") Changing the Cnn's B1 and B2 parameters for ADAM optimization to: B1 = " + str(b1ParamForAdam) + " || B2 = " + str(b2ParamForAdam))
        self.b1_adam = b1ParamForAdam
        self.b2_adam = b2ParamForAdam
        
    def changeRhoParameterOfRmsProp(self, rhoParamForRmsProp, myLogger) :
        myLogger.print3("UPDATE: (epoch-cnn-trained#" + str(self.numberOfEpochsTrained) + ") Changing the Cnn's Rho parameter for RMSProp optimization to: Rho = " + str(rhoParamForRmsProp))
        self.rho_rmsProp = rhoParamForRmsProp
        
    def checkMeanValidationAccOfLastEpochAndUpdateCnnsTopAccAchievedIfNeeded(self,
                                                                        myLogger,
                                                                        meanValidationAccuracyOfLastEpoch,
                                                                        minIncreaseInValidationAccuracyConsideredForLrSchedule) :
        # Called at the end of an epoch, right before increasing self.numberOfEpochsTrained
        highestAchievedValidationAccuracyOfCnn = self.topMeanValidationAccuracyAchievedInEpoch[0]
        if meanValidationAccuracyOfLastEpoch > highestAchievedValidationAccuracyOfCnn + minIncreaseInValidationAccuracyConsideredForLrSchedule :
            self.topMeanValidationAccuracyAchievedInEpoch[0] = meanValidationAccuracyOfLastEpoch
            self.topMeanValidationAccuracyAchievedInEpoch[1] = self.numberOfEpochsTrained
            myLogger.print3("UPDATE: In this last epoch (cnnTrained) #" + str(self.topMeanValidationAccuracyAchievedInEpoch[1]) + " the CNN achieved a new highest mean validation accuracy of: " + str(self.topMeanValidationAccuracyAchievedInEpoch[0]))
            
    def _initializeSharedVarsForInputsTrain(self) :
        # ======= Initialize sharedVariables ==========
        self._initializedSharedVarsTrain = True
        # Create the needed shared variables. Number of dimensions should be correct (5 for x, 4 for y). But size is placeholder. Changes when shared.set_value during training.
        self.sharedInpXTrain = theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag)
        for subsPath_i in xrange(self.numSubsPaths) :
            self.sharedInpXPerSubsListTrain.append(theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag))
        # When storing data on the GPU it has to be stored as floats (floatX). Later this variable is cast as "int", to be used correctly in computations.
        self.sharedLabelsYTrain = theano.shared(np.zeros([1, 1, 1, 1], dtype="float32") , borrow=self.borrowFlag)
        
    def _initializeSharedVarsForInputsVal(self) :
        self._initializedSharedVarsVal = True
        self.sharedInpXVal = theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag)
        for subsPath_i in xrange(self.numSubsPaths) :
            self.sharedInpXPerSubsListVal.append(theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag))
        self.sharedLabelsYVal = theano.shared(np.zeros([1, 1, 1, 1], dtype="float32") , borrow=self.borrowFlag)
        
    def _initializeSharedVarsForInputsTest(self) :
        self._initializedSharedVarsTest = True
        self.sharedInpXTest = theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag)
        for subsPath_i in xrange(self.numSubsPaths) :
            self.sharedInpXPerSubsListTest.append(theano.shared(np.zeros([1, 1, 1, 1, 1], dtype="float32"), borrow=self.borrowFlag))
            
    def freeGpuTrainingData(self) :
        if self._initializedSharedVarsTrain :  # False if this is called (eg save model) before train/val/test function is compiled.
            self.sharedInpXTrain.set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))  # = []
            for subsPath_i in xrange(self.numSubsPaths) :
                self.sharedInpXPerSubsListTrain[subsPath_i].set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))
            self.sharedLabelsYTrain.set_value(np.zeros([1, 1, 1, 1], dtype="float32"))  # = []
            
    def freeGpuValidationData(self) :
        if self._initializedSharedVarsVal :
            self.sharedInpXVal.set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))  # = []
            for subsPath_i in xrange(self.numSubsPaths) :
                self.sharedInpXPerSubsListVal[subsPath_i].set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))
            self.sharedLabelsYVal.set_value(np.zeros([1, 1, 1, 1], dtype="float32"))  # = []
            
    def freeGpuTestingData(self) :
        if self._initializedSharedVarsTest :
            self.sharedInpXTest.set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))  # = []
            for subsPath_i in xrange(self.numSubsPaths) :
                self.sharedInpXPerSubsListTest[subsPath_i].set_value(np.zeros([1, 1, 1, 1, 1], dtype="float32"))
                
    def checkTrainingStateAttributesInitialized(self):
        return self._trainingStateAttributesInitialized
    
    # for inference with batch-normalization. Every training batch, this is called to update an internal matrix of each layer, with the last mus and vars, so that I can compute the rolling average for inference.
    def updateTheMatricesOfTheLayersWithTheLastMusAndVarsForTheMovingAverageOfTheBatchNormInference(self) :
        self._updateMatricesOfBnRollingAverageForInference()
        
    def _updateMatricesOfBnRollingAverageForInference(self):
        for pathway in self.pathways :
            for layer in pathway.getLayers() :
                layer.updateTheMatricesWithTheLastMusAndVarsForTheRollingAverageOfBNInference()  # Will do nothing if no BN.
   
    
    def _initializeSharedVariablesOfOptimizer(self, myLogger) :
        # ======= Get List Of Trained Parameters to be fit by gradient descent=======
        paramsToOptDuringTraining = self._getTrainableParameters(myLogger)
        if self.sgd0orAdam1orRmsProp2 == 0 :
            self._initializeSharedVariablesOfSgd(paramsToOptDuringTraining)
        elif self.sgd0orAdam1orRmsProp2 == 1 :
            self._initializeSharedVariablesOfAdam(paramsToOptDuringTraining)
        elif self.sgd0orAdam1orRmsProp2 == 2 :
            self._initializeSharedVariablesOfRmsProp(paramsToOptDuringTraining)
        else :
            return False
        return True
        
    def _initializeSharedVariablesOfSgd(self, paramsToOptDuringTraining) :
        self.velocities_forMom = []
        for param in paramsToOptDuringTraining :
            v = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
            self.velocities_forMom.append(v)
            
    def getUpdatesAccordingToSgd(self, cost, paramsToOptDuringTraining) :
        # create a list of gradients for all model parameters
        grads = T.grad(cost, paramsToOptDuringTraining)
        
        updates = []
        # The below will be 1 if nonNormalized momentum, and (1-momentum) if I am using normalized momentum.
        multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum = 1.0 - self.momentum * self.momentumTypeNONNormalized0orNormalized1
        
        for param, grad, v in zip(paramsToOptDuringTraining, grads, self.velocities_forMom) :
            stepToGradientDirection = multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum * self.learning_rate * grad
            newVelocity = self.momentum * v - stepToGradientDirection
            
            if self.classicMomentum0OrNesterov1 == 0 :
                updateToParam = newVelocity
            else :  # Nesterov
                updateToParam = self.momentum * newVelocity - stepToGradientDirection
                
            updates.append((v, newVelocity))  # I can do (1-mom)*learnRate*grad.
            updates.append((param, param + updateToParam))
            
        return updates
    
    def _initializeSharedVariablesOfRmsProp(self, paramsToOptDuringTraining) :
        self.accuGradSquare_listForAllParamsRmsProp = []
        self.velocities_forMom = []
        for param in paramsToOptDuringTraining :
            accu = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)  # accumulates the mean of the grad's square.
            self.accuGradSquare_listForAllParamsRmsProp.append(accu)
            v = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)  # velocity
            self.velocities_forMom.append(v)
            
    def getUpdatesAccordingToRmsProp(self, cost, params) :
        # epsilon=1e-4 in paper. I got NaN in cost function when I ran it with this value. Worked ok with epsilon=1e-6.
        
        # Code taken and updated (it was V2 of paper, updated to V8) from https://gist.github.com/Newmu/acb738767acb4788bac3
        grads = T.grad(cost, params)
        updates = []
        # The below will be 1 if nonNormalized momentum, and (1-momentum) if I am using normalized momentum.
        multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum = 1.0 - self.momentum * self.momentumTypeNONNormalized0orNormalized1
        
        for param, grad, accu, v in zip(params, grads, self.accuGradSquare_listForAllParamsRmsProp, self.velocities_forMom):
            accu_new = self.rho_rmsProp * accu + (1 - self.rho_rmsProp) * T.sqr(grad)
            stepToGradientDirection = multiplierForCurrentGradUpdateForNonNormalizedOrNormalizedMomentum * (self.learning_rate * grad / T.sqrt(accu_new + self.epsilonForRmsProp))
            newVelocity = self.momentum * v - stepToGradientDirection
            
            if self.classicMomentum0OrNesterov1 == 0 :
                updateToParam = newVelocity
            else :  # Nesterov
                updateToParam = self.momentum * newVelocity - stepToGradientDirection
                
            updates.append((accu, accu_new))
            updates.append((v, newVelocity))  # I can do (1-mom)*learnRate*grad.
            updates.append((param, param + updateToParam))
            
        return updates
    
    def _initializeSharedVariablesOfAdam(self, paramsToOptDuringTraining) :
        self.i_adam = theano.shared(np.cast["float32"](0.))  # Current iteration
        self.m_listForAllParamsAdam = []  # list of mean of grads for all parameters, for ADAM optimizer.
        self.v_listForAllParamsAdam = []  # list of variances of grads for all parameters, for ADAM optimizer.
        for param in paramsToOptDuringTraining :
            m = theano.shared(param.get_value() * 0.)
            self.m_listForAllParamsAdam.append(m)
            v = theano.shared(param.get_value() * 0.)
            self.v_listForAllParamsAdam.append(v)
            
    def getUpdatesAccordingToAdam(self, cost, params) :
        # Epsilon on paper was 10**(-8).
        # Code is on par with version V8 of Kingma's paper.
        grads = T.grad(cost, params)
        
        updates = []
        
        i = self.i_adam
        i_t = i + 1.
        fix1 = 1. - (self.b1_adam)**i_t
        fix2 = 1. - (self.b2_adam)**i_t
        lr_t = self.learning_rate * (T.sqrt(fix2) / fix1)
        for param, grad, m, v in zip(params, grads, self.m_listForAllParamsAdam, self.v_listForAllParamsAdam):
            m_t = (self.b1_adam * m) + ((1. - self.b1_adam) * grad)
            v_t = (self.b2_adam * v) + ((1. - self.b2_adam) * T.sqr(grad))  # Double check this with the paper.
            grad_t = m_t / (T.sqrt(v_t) + self.epsilonForAdam)
            param_t = param - (lr_t * grad_t)
            updates.append((m, m_t))
            updates.append((v, v_t))
            updates.append((param, param_t))
        updates.append((i, i_t))
        return updates
    
    def _getUpdatesOfTrainableParameters(self, myLogger, cost) :
        # ======= Get List Of Trained Parameters to be fit by gradient descent=======
        paramsToOptDuringTraining = self._getTrainableParameters(myLogger)
        if self.sgd0orAdam1orRmsProp2 == 0 :
            myLogger.print3("Optimizer used: [SGD]. Momentum used: Classic0 or Nesterov1 : " + str(self.classicMomentum0OrNesterov1))
            updates = self.getUpdatesAccordingToSgd(cost, paramsToOptDuringTraining)
        elif self.sgd0orAdam1orRmsProp2 == 1 :
            myLogger.print3("Optimizer used: [ADAM]. No momentum implemented for Adam.")
            updates = self.getUpdatesAccordingToAdam(cost, paramsToOptDuringTraining)
        elif self.sgd0orAdam1orRmsProp2 == 2 :
            myLogger.print3("Optimizer used: [RMSProp]. Momentum used: Classic0 or Nesterov1 : " + str(self.classicMomentum0OrNesterov1))
            updates = self.getUpdatesAccordingToRmsProp(cost, paramsToOptDuringTraining)
        return updates
    
    def _getTrainableParameters(self, myLogger):
        # A getter. Don't alter anything here!
        paramsToOptDuringTraining = []  # Ws and Bs
        for pathway in self.pathways :
            for layer_i in xrange(0, len(pathway.getLayers())) :
                if layer_i not in self.indicesOfLayersPerPathwayTypeToFreeze[ pathway.pType() ] :
                    paramsToOptDuringTraining = paramsToOptDuringTraining + pathway.getLayer(layer_i).getTrainableParams()
                else : # Layer will be held fixed. Notice that Batch Norm parameters are still learnt.
                    myLogger.print3("WARN: [Pathway_" + str(pathway.getStringType()) + "] The weights of [Layer-"+str(layer_i)+"] will NOT be trained as specified (index, first layer is 0).")
        return paramsToOptDuringTraining
    
    def _getL1RegCost(self) :
        L1 = 0
        for pathway in self.pathways :
            for layer in pathway.getLayers() :    
                L1 += layer.getL1RegCost()
        return L1
    
    def _getL2RegCost(self) :
        L2_sqr = 0
        for pathway in self.pathways :
            for layer in pathway.getLayers() :    
                L2_sqr += layer.getL2RegCost()
        return L2_sqr
    
    # This function should be called at least once prior to compiling train function for the first time. 
    # If I need to "resume" training, this should not be called.
    # However, if I need to use a pretrained model, and train it in a second stage, I should recall this, with the new stage's parameters, and then recompile trainFunction.
    def initializeTrainingState(self,
                                myLogger,
                                indicesOfLayersPerPathwayTypeToFreeze,
                                costFunctionLetter,
                                learning_rate,
                                sgd0orAdam1orRmsProp2,
                                classicMomentum0OrNesterov1,
                                momentum,
                                momentumTypeNONNormalized0orNormalized1,
                                b1ParamForAdam,
                                b2ParamForAdam,
                                epsilonForAdam,
                                rhoParamForRmsProp,
                                epsilonForRmsProp,
                                L1_reg_constant,
                                L2_reg_constant
                                ) :
        myLogger.print3("...Initializing attributes for the optimization...")
        self.numberOfEpochsTrained = 0
        
        # Layers to train (rest are left untouched, eg for pretrained models.
        self.indicesOfLayersPerPathwayTypeToFreeze = indicesOfLayersPerPathwayTypeToFreeze
        
        # Cost function
        if costFunctionLetter != "previous" :
            self.costFunctionLetter = costFunctionLetter
            
        # Regularization
        self.L1_reg_constant = L1_reg_constant
        self.L2_reg_constant = L2_reg_constant
        
        # Learning rate and momentum
        self.initialLearningRate = learning_rate # This is important for the learning rate schedule to work.
        self.change_learning_rate_of_a_cnn(learning_rate, myLogger)
        self.classicMomentum0OrNesterov1 = classicMomentum0OrNesterov1
        self.initialMomentum = momentum
        self.momentumTypeNONNormalized0orNormalized1 = momentumTypeNONNormalized0orNormalized1
        self.change_momentum_of_a_cnn(momentum, myLogger)
        
        # Optimizer
        self.sgd0orAdam1orRmsProp2 = sgd0orAdam1orRmsProp2
        if sgd0orAdam1orRmsProp2 == 1 :
            self.changeB1AndB2ParametersOfAdam(b1ParamForAdam, b2ParamForAdam, myLogger)
            self.epsilonForAdam = epsilonForAdam
        elif sgd0orAdam1orRmsProp2 == 2 :
            self.changeRhoParameterOfRmsProp(rhoParamForRmsProp, myLogger)
            self.epsilonForRmsProp = epsilonForRmsProp
            
        # Important point. Initializing the shareds that hold the velocities etc states of the optimizers.
        self._initializeSharedVariablesOfOptimizer(myLogger)
        
        self._trainingStateAttributesInitialized = True
        
    def _getUpdatesForBnRollingAverage(self) :
        # These are not the variables of the normalization of the FMs' distributions that are optimized during training. These are only the Mu and Stds that are used during inference,
        # ... and here we update the sharedVariable which is used "from the outside during do_training()" to update the rolling-average-matrix for inference. Do for all layers.
        updatesForBnRollingAverage = []
        for pathway in self.pathways :
            for layer in pathway.getLayers() :
                updatesForBnRollingAverage.extend(layer.getUpdatesForBnRollingAverage())  #CAREFUL: WARN, PROBLEM, THEANO BUG! If a layer has only 1FM, the .newMu_B ends up being of type (true,) instead of vector!!! Error!!!
        return updatesForBnRollingAverage
    
    # NOTE: compileTrainFunction() changes the self.initialLearningRate. Which is used for the exponential schedule!
    def compileTrainFunction(self, myLogger) :
        # At the next stage of the refactoring:
        # 1. Take an additional variable that says whether to "initialize" new training, or to "resume" training
        # 2. Build model here. Which internally LOADS the weights, array made by newModel. Dont initialize a model here. If you want to pretrain, have a -pretrainedModel function to create a new model.
        # 3. initializeTrainingState() if the input variable (1) says so (eg to do another training stage). Otherwise, dont call it, to resume training.
        myLogger.print3("...Building the training function...")
        
        if not self.checkTrainingStateAttributesInitialized() :
            myLogger.print3("ERROR: Prior to compiling the training function, training state attributes need to be initialized via a call of [Cnn3d.setTrainingStateAttributes(...)]. Exiting!"); exit(1)
            
        self._initializeSharedVarsForInputsTrain()
        
        # symbolic variables needed:
        index = T.lscalar()
        x = self.inputTensorNormTrain
        listXPerSubs = self.listInputTensorPerSubsTrain
        
        y = T.itensor4('y')  # Input of the theano-compiled-function. Dimensions of y labels: [batchSize, r, c, z]
        # When storing data on the GPU it has to be stored as floats (floatX). Thus the sharedVariable is FloatX/32. Here this variable is cast as "int", to be used correctly in computations.
        intCastSharedLabelsYTrain = T.cast(self.sharedLabelsYTrain, 'int32')
        inputVectorWeightsOfClassesInCostFunction = T.fvector()  # These two were added to counter class imbalance by changing the weights in the cost function
        weightPerClass = T.fvector()  # a vector with 1 element per class.
        
        # The cost Function to use.
        if self.costFunctionLetter == "L" :
            costFromLastLayer = self.finalTargetLayer.negativeLogLikelihood(y, weightPerClass)
        else :
            myLogger.print3("ERROR: Problem in make_cnn_model(). The parameter self.costFunctionLetter did not have an acceptable value( L,D,J ). Exiting."); exit(1)
            
        cost = (costFromLastLayer
                + self.L1_reg_constant * self._getL1RegCost()
                + self.L2_reg_constant * self._getL2RegCost())
        
        updates = self._getUpdatesOfTrainableParameters(myLogger, cost)
        
        updates = updates + self._getUpdatesForBnRollingAverage()
        
        givensSet = { x: self.sharedInpXTrain[index * self.batchSize: (index + 1) * self.batchSize] }
        for subPath_i in xrange(self.numSubsPaths) : # if there are subsampled paths...
            xSub = listXPerSubs[subPath_i]
            sharedInpXSubTrain = self.sharedInpXPerSubsListTrain[subPath_i]
            givensSet.update({ xSub: sharedInpXSubTrain[index * self.batchSize: (index + 1) * self.batchSize] })
        givensSet.update({  y: intCastSharedLabelsYTrain[index * self.batchSize: (index + 1) * self.batchSize],
                            weightPerClass: inputVectorWeightsOfClassesInCostFunction })
        
        myLogger.print3("...Compiling the function for training... (This may take a few minutes...)")
        self.cnnTrainModel = theano.function(
                                [index, inputVectorWeightsOfClassesInCostFunction],
                                [cost] + self.finalTargetLayer.getRpRnTpTnForTrain0OrVal1(y, 0),
                                updates=updates,
                                givens=givensSet
                                )
        myLogger.print3("The training function was compiled.")
        
    def compileValidationFunction(self, myLogger) :
        myLogger.print3("...Building the validation function...")
        
        self._initializeSharedVarsForInputsVal()
        
        # symbolic variables needed:
        index = T.lscalar()
        x = self.inputTensorNormVal
        listXPerSubs = self.listInputTensorPerSubsVal
        y = T.itensor4('y')  # Input of the theano-compiled-function. Dimensions of y labels: [batchSize, r, c, z]
        # When storing data on the GPU it has to be stored as floats (floatX). Thus the sharedVariable is FloatX/32. Here this variable is cast as "int", to be used correctly in computations.
        intCastSharedLabelsYVal = T.cast(self.sharedLabelsYVal, 'int32')
        
        givensSet = { x: self.sharedInpXVal[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation] }
        for subPath_i in xrange(self.numSubsPaths) : # if there are subsampled paths...
            xSub = listXPerSubs[subPath_i]
            sharedInpXSubVal = self.sharedInpXPerSubsListVal[subPath_i]
            givensSet.update({ xSub: sharedInpXSubVal[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation] })
        givensSet.update({ y: intCastSharedLabelsYVal[index * self.batchSizeValidation: (index + 1) * self.batchSizeValidation] })
        
        myLogger.print3("...Compiling the function for validation... (This may take a few minutes...)")
        self.cnnValidateModel = theano.function(
                                    [index],
                                    self.finalTargetLayer.getRpRnTpTnForTrain0OrVal1(y, 1),
                                    givens=givensSet
                                    )
        myLogger.print3("The validation function was compiled.")
        
        
    def compileTestAndVisualisationFunction(self, myLogger) :
        myLogger.print3("...Building the function for testing and visualisation of FMs...")
        
        self._initializeSharedVarsForInputsTest()
        
        # symbolic variables needed:
        index = T.lscalar()
        x = self.inputTensorNormTest
        listXPerSubs = self.listInputTensorPerSubsTest
        
        listToReturnWithAllTheFmActivationsAndPredictionsAppended = []
        for pathway in self.pathways :
            for layer in pathway.getLayers() :  # each layer that this pathway/fc has.
                listToReturnWithAllTheFmActivationsAndPredictionsAppended.append( layer.fmsActivations([0, 9999]) )
                
        listToReturnWithAllTheFmActivationsAndPredictionsAppended.append(self.finalTargetLayer.predictionProbabilities())
        
        givensSet = { x: self.sharedInpXTest[index * self.batchSizeTesting: (index + 1) * self.batchSizeTesting] }
        for subPath_i in xrange(self.numSubsPaths) : # if there are subsampled paths...
            xSub = listXPerSubs[subPath_i]
            sharedInpXSubTest = self.sharedInpXPerSubsListTest[subPath_i]
            givensSet.update({ xSub: sharedInpXSubTest[index * self.batchSizeTesting: (index + 1) * self.batchSizeTesting] })
            
        myLogger.print3("...Compiling the function for testing and visualisation of FMs... (This may take a few minutes...)")
        self.cnnTestAndVisualiseAllFmsFunction = theano.function(
                                                        [index],
                                                        listToReturnWithAllTheFmActivationsAndPredictionsAppended,
                                                        givens=givensSet
                                                        )
        myLogger.print3("The function for testing and visualisation of FMs was compiled.")
        
    def _getInputTensorsXToCnn(self):
        if not self.inputTensorsXToCnnInitialized :
            # Symbolic variables, which stand for the input to the CNN. Will be loaded by the compiled trainining/val/test function. Can also be pre-set by an existing tensor if required in future extensions.
            tensor5 = T.TensorType(dtype='float32', broadcastable=(False, False, False, False, False))
            self.inputTensorNormTrain = tensor5()  # Actually, for these 3, a single tensor5() could be used, as long as I reshape it separately for each afterwards. The actual value is loaded by the compiled functions.
            self.inputTensorNormVal = tensor5()  # myTensor.reshape(inputImageShapeValidation)
            self.inputTensorNormTest = tensor5()
            # For the multiple subsampled pathways.
            for subsPath_i in xrange(self.numSubsPaths) :
                self.listInputTensorPerSubsTrain.append(tensor5())  # Actually, for these 3, a single tensor5() could be used.
                self.listInputTensorPerSubsVal.append(tensor5())  # myTensor.reshape(inputImageSubsampledShapeValidation)
                self.listInputTensorPerSubsTest.append(tensor5())
            self.inputTensorsXToCnnInitialized = True
            
        return (self.inputTensorNormTrain, self.inputTensorNormVal, self.inputTensorNormTest,
                self.listInputTensorPerSubsTrain, self.listInputTensorPerSubsVal, self.listInputTensorPerSubsTest)
        
        
    def _getClassificationLayer(self):
        return SoftmaxLayer()
        
    def make_cnn_model( self,
                        myLogger,
                        cnnModelName,
                        numberOfOutputClasses,
                        numberOfImageChannelsPath1,
                        numberOfImageChannelsPath2,
                        
                        nkerns,
                        kernelDimensions,
                        # THESE NEXT TWO, ALONG WITH THE ONES FOR FC, COULD BE PUT IN ONE STRUCTURE WITH NORMAL, EG LIKE kerns = [ [kernsNorm], [kernsSub], [kernsFc]]
                        nkernsSubsampled, # Used to control if secondary pathways: [] if no secondary pathways. Now its the "factors"
                        kernelDimensionsSubsampled,
                        subsampleFactorsPerSubPath, # Controls how many pathways: [] if no secondary pathways. Else, List of lists. One sublist per secondary pathway. Each sublist has 3 ints, the rcz subsampling factors.
                        fcLayersFMs,
                        kernelDimensionsFirstFcLayer,
                        softmaxTemperature,
                        
                        activationFunc,
                        #---Residual Connections----
                        indicesOfLayersToConnectResidualsInOutput,
                        #--Lower Rank Layer Per Pathway---
                        indicesOfLowerRankLayersPerPathway,
                        ranksOfLowerRankLayersForEachPathway,
                        #---Pooling---
                        maxPoolingParamsStructure,
                        #--- Skip Connections --- #Deprecated, not used/supported
                        convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes,
                        
                        imagePartDimensionsTraining ,
                        imagePartDimensionsValidation,
                        imagePartDimensionsTesting,
                        
                        batch_size,
                        batch_size_validation,
                        batch_size_testing,
                        
                        # Dropout
                        dropoutRatesForAllPathways,  # list of sublists, one for each pathway. Each either empty or full with the dropout rates of all the layers in the path.
                        # Initialization
                        convWInitMethod,
                        # Batch Normalization
                        applyBnToInputOfPathways,  # one Boolean flag per pathway type. Placeholder for the FC pathway.
                        rollingAverageForBatchNormalizationOverThatManyBatches,
                        
                        borrowFlag,
                        dataTypeX='float32',
                        ):

        self.cnnModelName = cnnModelName
        
        self.numberOfOutputClasses = numberOfOutputClasses
        self.numberOfImageChannelsPath1 = numberOfImageChannelsPath1
        self.numberOfImageChannelsPath2 = numberOfImageChannelsPath2
        # === Architecture ===
        self.nkerns = nkerns  # Useless?
        self.nkernsSubsampled = nkernsSubsampled  # Useless?
        self.numSubsPaths = len(subsampleFactorsPerSubPath) # do I want this as attribute? Or function is ok?
        
        # fcLayersFMs???
        self.kernelDimensionsFirstFcLayer = kernelDimensionsFirstFcLayer
        
        # == Other Architectural Params ==
        self.indicesOfLayersToConnectResidualsInOutput = indicesOfLayersToConnectResidualsInOutput
        self.indicesOfLowerRankLayersPerPathway = indicesOfLowerRankLayersPerPathway
        # pooling?

        # == Batch Sizes ==
        self.batchSize = batch_size
        self.batchSizeValidation = batch_size_validation
        self.batchSizeTesting = batch_size_testing
        # == Others ==
        self.dropoutRatesForAllPathways = dropoutRatesForAllPathways
        # == various ==
        self.borrowFlag = borrowFlag
        self.dataTypeX = dataTypeX
        
        # ======== Calculated Attributes =========
        #This recField CNN should in future be calculated with all non-secondary pathways, ie normal+fc. Use another variable for pathway.recField.
        self.recFieldCnn = calcRecFieldFromKernDimListPerLayerWhenStrides1(kernelDimensions)
        

        rng = np.random.RandomState(23455)
    
        myLogger.print3("...Building the CNN model...")
        
        # Symbolic variables, which stand for the input. Will be loaded by the compiled trainining/val/test function. Can also be pre-set by an existing tensor if required in future extensions.
        (inputTensorNormTrain, inputTensorNormVal, inputTensorNormTest,
        listInputTensorPerSubsTrain, listInputTensorPerSubsVal, listInputTensorPerSubsTest) = self._getInputTensorsXToCnn()
        
        thisPathway = NormalPathway()
        self.pathways.append(thisPathway)
        thisPathwayType = thisPathway.pType()
        
        inputToPathwayTrain = inputTensorNormTrain
        inputToPathwayVal = inputTensorNormVal
        inputToPathwayTest = inputTensorNormTest
        inputToPathwayShapeTrain = [self.batchSize, numberOfImageChannelsPath1] + imagePartDimensionsTraining
        inputToPathwayShapeVal = [self.batchSizeValidation, numberOfImageChannelsPath1] + imagePartDimensionsValidation
        inputToPathwayShapeTest = [self.batchSizeTesting, numberOfImageChannelsPath1] + imagePartDimensionsTesting
        
        thisPathWayNKerns = nkerns
        thisPathWayKernelDimensions = kernelDimensions
        
        thisPathwayNumOfLayers = len(thisPathWayNKerns)
        thisPathwayUseBnPerLayer = [rollingAverageForBatchNormalizationOverThatManyBatches > 0] * thisPathwayNumOfLayers
        thisPathwayUseBnPerLayer[0] = applyBnToInputOfPathways[thisPathwayType] if rollingAverageForBatchNormalizationOverThatManyBatches > 0 else False  # For the 1st layer, ask specific flag.
        
        thisPathwayActivFuncPerLayer = [activationFunc] * thisPathwayNumOfLayers
        thisPathwayActivFuncPerLayer[0] = "linear" if thisPathwayType != pt.FC else activationFunc  # To not apply activation on raw input. -1 is linear activation.
        
        thisPathway.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(myLogger,
                                                                         inputToPathwayTrain,
                                                                         inputToPathwayVal,
                                                                         inputToPathwayTest,
                                                                         inputToPathwayShapeTrain,
                                                                         inputToPathwayShapeVal,
                                                                         inputToPathwayShapeTest,
                                                                         
                                                                         thisPathWayNKerns,
                                                                         thisPathWayKernelDimensions,
                                                                         
                                                                         convWInitMethod,
                                                                         thisPathwayUseBnPerLayer,
                                                                         rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                         thisPathwayActivFuncPerLayer,
                                                                         dropoutRatesForAllPathways[thisPathwayType],
                                                                         
                                                                         maxPoolingParamsStructure[thisPathwayType],
                                                                         
                                                                         indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                         ranksOfLowerRankLayersForEachPathway[thisPathwayType],
                                                                         indicesOfLayersToConnectResidualsInOutput[thisPathwayType]
                                                                         )
        # Skip connections to end of pathway.
        thisPathway.makeMultiscaleConnectionsForLayerType(convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes[thisPathwayType])
        
        [dimsOfOutputFrom1stPathwayTrain, dimsOfOutputFrom1stPathwayVal, dimsOfOutputFrom1stPathwayTest] = thisPathway.getShapeOfOutput()
        
        for subPath_i in xrange(self.numSubsPaths) :
            thisPathway = SubsampledPathway(subsampleFactorsPerSubPath[subPath_i])
            self.pathways.append(thisPathway) # There will be at least an entry as a secondary pathway. But it won't have any layers if it was not actually used.
            thisPathwayType = thisPathway.pType()
            
            inputToPathwayTrain = listInputTensorPerSubsTrain[subPath_i]
            inputToPathwayVal = listInputTensorPerSubsVal[subPath_i]
            inputToPathwayTest = listInputTensorPerSubsTest[subPath_i]
            
            thisPathWayNKerns = nkernsSubsampled[subPath_i]
            thisPathWayKernelDimensions = kernelDimensionsSubsampled
            
            thisPathwayNumOfLayers = len(thisPathWayNKerns)
            thisPathwayUseBnPerLayer = [rollingAverageForBatchNormalizationOverThatManyBatches > 0] * thisPathwayNumOfLayers
            thisPathwayUseBnPerLayer[0] = applyBnToInputOfPathways[thisPathwayType] if rollingAverageForBatchNormalizationOverThatManyBatches > 0 else False  # For the 1st layer, ask specific flag.
            
            thisPathwayActivFuncPerLayer = [activationFunc] * thisPathwayNumOfLayers
            thisPathwayActivFuncPerLayer[0] = "linear" if thisPathwayType != pt.FC else activationFunc  # To not apply activation on raw input. -1 is linear activation.
            
            inputToPathwayShapeTrain = [self.batchSize, numberOfImageChannelsPath2] + thisPathway.calcInputRczDimsToProduceOutputFmsOfCompatibleDims(thisPathWayKernelDimensions, dimsOfOutputFrom1stPathwayTrain);
            inputToPathwayShapeVal = [self.batchSizeValidation, numberOfImageChannelsPath2] + thisPathway.calcInputRczDimsToProduceOutputFmsOfCompatibleDims(thisPathWayKernelDimensions, dimsOfOutputFrom1stPathwayVal)
            inputToPathwayShapeTest = [self.batchSizeTesting, numberOfImageChannelsPath2] + thisPathway.calcInputRczDimsToProduceOutputFmsOfCompatibleDims(thisPathWayKernelDimensions, dimsOfOutputFrom1stPathwayTest)
            
            thisPathway.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(myLogger,
                                                                     inputToPathwayTrain,
                                                                     inputToPathwayVal,
                                                                     inputToPathwayTest,
                                                                     inputToPathwayShapeTrain,
                                                                     inputToPathwayShapeVal,
                                                                     inputToPathwayShapeTest,
                                                                     thisPathWayNKerns,
                                                                     thisPathWayKernelDimensions,
                                                                     
                                                                     convWInitMethod,
                                                                     thisPathwayUseBnPerLayer,
                                                                     rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                     thisPathwayActivFuncPerLayer,
                                                                     dropoutRatesForAllPathways[thisPathwayType],
                                                                     
                                                                     maxPoolingParamsStructure[thisPathwayType],
                                                                     
                                                                     indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                     ranksOfLowerRankLayersForEachPathway[thisPathwayType],
                                                                     indicesOfLayersToConnectResidualsInOutput[thisPathwayType]
                                                                     )
            # Skip connections to end of pathway.
            thisPathway.makeMultiscaleConnectionsForLayerType(convLayersToConnectToFirstFcForMultiscaleFromAllLayerTypes[thisPathwayType])
            
            # this creates essentially the "upsampling layer"
            thisPathway.upsampleOutputToNormalRes(upsamplingScheme="repeat",
                                                  shapeToMatchInRczTrain=dimsOfOutputFrom1stPathwayTrain,
                                                  shapeToMatchInRczVal=dimsOfOutputFrom1stPathwayVal,
                                                  shapeToMatchInRczTest=dimsOfOutputFrom1stPathwayTest)
            
            
        inputToFirstFcLayerTrain = None; inputToFirstFcLayerVal = None; inputToFirstFcLayerTest = None; numberOfFmsOfInputToFirstFcLayer = 0
        for path_i in xrange(len(self.pathways)) :
            [outputNormResOfPathTrain, outputNormResOfPathVal, outputNormResOfPathTest] = self.pathways[path_i].getOutputAtNormalRes()
            [dimsOfOutputNormResOfPathTrain, dimsOfOutputNormResOfPathVal, dimsOfOutputNormResOfPathTest] = self.pathways[path_i].getShapeOfOutputAtNormalRes()
            
            inputToFirstFcLayerTrain =  T.concatenate([inputToFirstFcLayerTrain, outputNormResOfPathTrain], axis=1) if path_i != 0 else outputNormResOfPathTrain
            inputToFirstFcLayerVal = T.concatenate([inputToFirstFcLayerVal, outputNormResOfPathVal], axis=1) if path_i != 0 else outputNormResOfPathVal
            inputToFirstFcLayerTest = T.concatenate([inputToFirstFcLayerTest, outputNormResOfPathTest], axis=1) if path_i != 0 else outputNormResOfPathTest
            numberOfFmsOfInputToFirstFcLayer += dimsOfOutputNormResOfPathTrain[1]
            
        thisPathway = FcPathway()
        self.pathways.append(thisPathway)
        thisPathwayType = thisPathway.pType()
        
        # This is the shape of the kernel in the first FC layer.
        # NOTE: If there is no hidden FC layer, this kernel is used in the Classification layer then.
        # Originally it was 1x1x1 only. The pathways themselves where taking care of the receptive field.
        # However I can now define it larger (eg 3x3x3), in case it helps combining the multiresolution features better/smoother.
        # The convolution is seamless, ie same shape output/input, by mirror padding the input.
        firstFcLayerAfterConcatenationKernelShape = self.kernelDimensionsFirstFcLayer
        voxelsToPadPerDim = [ kernelDim - 1 for kernelDim in firstFcLayerAfterConcatenationKernelShape ]
        myLogger.print3("DEBUG: Shape of the kernel of the first FC layer is : " + str(firstFcLayerAfterConcatenationKernelShape))
        myLogger.print3("DEBUG: Input to the FC Pathway will be padded by that many voxels per dimension: " + str(voxelsToPadPerDim))
        inputToPathwayTrain = padImageWithMirroring(inputToFirstFcLayerTrain, voxelsToPadPerDim)
        inputToPathwayVal = padImageWithMirroring(inputToFirstFcLayerVal, voxelsToPadPerDim)
        inputToPathwayTest = padImageWithMirroring(inputToFirstFcLayerTest, voxelsToPadPerDim)
        inputToPathwayShapeTrain = [self.batchSize, numberOfFmsOfInputToFirstFcLayer] + dimsOfOutputFrom1stPathwayTrain[2:5]
        inputToPathwayShapeVal = [self.batchSizeValidation, numberOfFmsOfInputToFirstFcLayer] + dimsOfOutputFrom1stPathwayVal[2:5]
        inputToPathwayShapeTest = [self.batchSizeTesting, numberOfFmsOfInputToFirstFcLayer] + dimsOfOutputFrom1stPathwayTest[2:5]
        for rcz_i in xrange(3) : 
            inputToPathwayShapeTrain[2+rcz_i] += voxelsToPadPerDim[rcz_i]
            inputToPathwayShapeVal[2+rcz_i] += voxelsToPadPerDim[rcz_i]
            inputToPathwayShapeTest[2+rcz_i] += voxelsToPadPerDim[rcz_i]
        
        thisPathWayNKerns = fcLayersFMs + [self.numberOfOutputClasses]
        thisPathWayKernelDimensions = [firstFcLayerAfterConcatenationKernelShape] + [[1, 1, 1]] * (len(thisPathWayNKerns) - 1)
        
        thisPathwayNumOfLayers = len(thisPathWayNKerns)
        thisPathwayUseBnPerLayer = [rollingAverageForBatchNormalizationOverThatManyBatches > 0] * thisPathwayNumOfLayers
        thisPathwayUseBnPerLayer[0] = applyBnToInputOfPathways[thisPathwayType] if rollingAverageForBatchNormalizationOverThatManyBatches > 0 else False  # For the 1st layer, ask specific flag.
        
        thisPathwayActivFuncPerLayer = [activationFunc] * thisPathwayNumOfLayers
        thisPathwayActivFuncPerLayer[0] = "linear" if thisPathwayType != pt.FC else activationFunc  # To not apply activation on raw input. -1 is linear activation.
        
        thisPathway.makeLayersOfThisPathwayAndReturnDimensionsOfOutputFM(myLogger,
                                                                         inputToPathwayTrain,
                                                                         inputToPathwayVal,
                                                                         inputToPathwayTest,
                                                                         inputToPathwayShapeTrain,
                                                                         inputToPathwayShapeVal,
                                                                         inputToPathwayShapeTest,
                                                                         
                                                                         thisPathWayNKerns,
                                                                         thisPathWayKernelDimensions,
                                                                         
                                                                         convWInitMethod,
                                                                         thisPathwayUseBnPerLayer,
                                                                         rollingAverageForBatchNormalizationOverThatManyBatches,
                                                                         thisPathwayActivFuncPerLayer,
                                                                         dropoutRatesForAllPathways[thisPathwayType],
                                                                         
                                                                         maxPoolingParamsStructure[thisPathwayType],
                                                                         
                                                                         indicesOfLowerRankLayersPerPathway[thisPathwayType],
                                                                         ranksOfLowerRankLayersForEachPathway[thisPathwayType],
                                                                         indicesOfLayersToConnectResidualsInOutput[thisPathwayType],
                                                                         )
        
        # =========== Make the final Target Layer (softmax, regression, whatever) ==========
        myLogger.print3("Adding the final Softmax Target layer...")
        
        self.finalTargetLayer = self._getClassificationLayer()
        self.finalTargetLayer.makeLayer(rng, self.getFcPathway().getLayer(-1), softmaxTemperature)
        
        myLogger.print3("Finished building the CNN's model.")
        
        