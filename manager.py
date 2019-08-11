from __future__ import print_function
from properties import Properties
from ensemble import Ensemble
from stream import Stream

from model_kliep import Kliep
import gaussianModel as gm

from model_kmm import KMM
from model_alpha_rulsif import Alpha_RULSIF

from sklearn import svm, grid_search
import time, datetime
import numpy as np



class Manager(object):
	def __init__(self, sourceFile, targetFile):
		self.SDataBufferArr = None #2D array representation of self.SDataBuffer
		self.SDataLabels = None
		self.TDataBufferArr = None #2D array representation of self.TDataBuffer
		self.TDataLabels = None

		self.useKliepCVSigma = Properties.useKliepCVSigma
		self.arulsifAlpha = Properties.arulsifAlpha

		self.useSvmCVParams = Properties.useSvmCVParams

		self.ensemble = Ensemble(Properties.ENSEMBLE_SIZE)

		self.initialWindowSize = int(Properties.INITIAL_DATA_SIZE)
		self.maxWindowSize = int(Properties.MAX_WINDOW_SIZE)

		self.enableForceUpdate = int(Properties.enableForceUpdate)
		self.forceUpdatePeriod = int(Properties.forceUpdatePeriod)

		"""
		- simulate source and target streams from corresponding files.
		"""
		print("Reading the Source Dataset")
		self.source = Stream(sourceFile, Properties.INITIAL_DATA_SIZE)
		print("Reading the Target Dataset")
		self.target = Stream(targetFile, Properties.INITIAL_DATA_SIZE)
		print("Finished Reading the Target Dataset")

		Properties.MAXVAR = self.source.data.shape[0]


	"""
	Write value (accuracy or confidence) to a file with DatasetName as an identifier.
	"""
	def __saveResult(self, acc, datasetName):
		with open(datasetName + '_' + Properties.OUTFILENAME, 'a') as f:
			f.write(str(acc) + "\n")
		f.close()

	def convListOfDictToNDArray(self, listOfDict):
		arrayRep = []
		if not listOfDict:
			return arrayRep
		arrayRep = np.array([[float(v)] for k,v in listOfDict[0].items() if k!=-1])
		for i in range(1, len(listOfDict)):
			arrayRep = np.append(arrayRep, np.array([[float(v)] for k,v in listOfDict[i].items() if k!=-1]), axis=1)
		return arrayRep

	def collectLabels(self, listOfDict):
		labels = []
		for d in listOfDict:
			labels.append(str(d[-1]))
		return labels

	"""
	The main method handling multistream classification using KLIEP.
	"""
	def startClassification(self, datasetName, method='kliep'):

		#save the timestamp
		globalStartTime = time.time()
		Properties.logger.info('Global Start Time: ' + datetime.datetime.fromtimestamp(globalStartTime).strftime('%Y-%m-%d %H:%M:%S'))

		#open files for saving accuracy and confidence
		fAcc = open(datasetName + '_' + Properties.OUTFILENAME, 'w')
		fConf = open(datasetName + '_confidence' + '_' + Properties.OUTFILENAME, 'w')

		#Get data buffer
		self.SDataBufferArr = self.source.data
		self.SDataLabels = self.source.dataLabels

		self.TDataBufferArr = self.target.data

		# now resize the windows appropriately
		self.SDataBufferArr = self.SDataBufferArr[:, -Properties.MAX_WINDOW_SIZE:]
		self.SDataLabels = self.SDataLabels[-Properties.MAX_WINDOW_SIZE:]

		self.TDataBufferArr = self.TDataBufferArr[:, -Properties.MAX_WINDOW_SIZE:]

		weightSrcData = np.zeros(shape=(1, len(self.SDataBufferArr)))

		if 'kliep' in method:
			Properties.logger.info('Using KLIEP method for covariate shift correction.')

			# initialize gaussian models
			gmodel = gm.GaussianModel()

			#first choose a suitable value for sigma
			kliep = Kliep(Properties.kliepParEta, Properties.kliepParLambda, Properties.kliepParB, Properties.kliepParThreshold, Properties.kliepDefSigma)

			if self.useKliepCVSigma==1:
				kliep.kliepDefSigma = kliep.chooseSigma(self.SDataBufferArr, self.TDataBufferArr)

			#calculate alpha values
			#self.kliep.kliepDefSigma = 0.1
			Properties.logger.info('Estimating initial DRM')
			gmodel.alphah, kernelMatSrcData, kernelMatTrgData, gmodel.refPoints = kliep.KLIEP(self.SDataBufferArr, self.TDataBufferArr)
			#initialize the updated gaussian model


			kernelMatSrcData = kernelMatSrcData[-Properties.MAX_WINDOW_SIZE:,:]
			kernelMatTrgData = kernelMatTrgData[-Properties.MAX_WINDOW_SIZE:,:]

			Properties.logger.info('Initializing Ensemble with the first model')
			#target model
			#first calculate weight for source instances

			weightSrcData = kliep.calcInstanceWeights(kernelMatSrcData, gmodel.alphah)
			#since weightSrcData is a column matrix, convert it to a list before sending to generating new model

		elif 'kmm' in method:
			Properties.logger.info('Using KMM method for covariate shift correction.')
			kmm = KMM()
			gammab = kmm.computeKernelWidth(self.SDataBufferArr)
			Xtrain = self.SDataBufferArr.T.tolist()
			Xtest = self.TDataBufferArr.T.tolist()
			beta = kmm.kmm(Xtrain, Xtest, gammab)
			weightSrcData = np.array(beta).reshape(1,len(beta))

		elif 'arulsif' in method:
			Properties.logger.info('Using alpha-relative-uLSIF method for covariate shift correction.')
			arulsif = Alpha_RULSIF()
			beta = arulsif.R_ULSIF(self.SDataBufferArr, self.TDataBufferArr, self.arulsifAlpha)
			weightSrcData = np.array(beta).reshape(1, len(beta))
		else:
			print('Incorrect method. Please try again')
			return

		SDataBufferArrTransposed = self.SDataBufferArr.T
		TDataBufferArrTransposed = self.TDataBufferArr.T

		if self.useSvmCVParams == 1:
			params = {'gamma': [2 ** 2, 2 ** -16], 'C': [2 ** -6, 2 ** 15]}
			svr = svm.SVC()
			opt = grid_search.GridSearchCV(svr, params)
			opt.fit(SDataBufferArrTransposed.tolist(), self.SDataLabels)
			optParams = opt.best_params_

			self.ensemble.generateNewModel(SDataBufferArrTransposed.tolist(), self.SDataLabels,
												TDataBufferArrTransposed, weightSrcData[0].tolist(),
												optParams['C'], optParams['gamma'], Properties.svmKernel)
		else:
			self.ensemble.generateNewModel(SDataBufferArrTransposed.tolist(), self.SDataLabels,
												TDataBufferArrTransposed, weightSrcData[0].tolist(),
												Properties.svmDefC, Properties.svmDefGamma, Properties.svmKernel)

		Properties.logger.info(self.ensemble.getEnsembleSummary())

		tDataIndex = 0
		trueTargetNum = 0
		targetConfSum = 0

		#enoughInstToUpdate is used to see if there are enough instances in the windows to
		#estimate the weights
		
		while self.target.data.shape[1] > tDataIndex:
			
			# Target Stream
			print('#', end="") # '#' indicates new point from target
			newTargetDataArr = self.target.data[:, tDataIndex][np.newaxis].T
			# get Target Accuracy on the new instance
			resTarget = self.ensemble.evaluateEnsemble(np.reshape(newTargetDataArr, (1,-1)))
			if isinstance(resTarget[0], float) and abs(resTarget[0]-self.target.dataLabels[tDataIndex])<0.0001:
				trueTargetNum += 1
			elif resTarget[0] == self.target.dataLabels[tDataIndex]:
				trueTargetNum += 1
			acc = float(trueTargetNum)/(tDataIndex+1)
			if (tDataIndex%100)==0:
				Properties.logger.info('\nTotal test instance: '+ str(tDataIndex+1) + ', correct: ' + str(trueTargetNum) + ', accuracy: ' + str(acc))
			fAcc.write(str(acc)+ "\n")

			conf = resTarget[1]  # confidence
			# save confidence
			targetConfSum += conf
			fConf.write(str(float(targetConfSum)/(tDataIndex+1))+ "\n")
			

			tDataIndex += 1

			
		#save the timestamp
		fConf.close()
		fAcc.close()
		globalEndTime = time.time()
		Properties.logger.info(
			'\nGlobal Start Time: ' + datetime.datetime.fromtimestamp(globalEndTime).strftime('%Y-%m-%d %H:%M:%S'))
		Properties.logger.info('Total Time Spent: ' + str(globalEndTime-globalStartTime) + ' seconds')
		Properties.logger.info('Done !!')
