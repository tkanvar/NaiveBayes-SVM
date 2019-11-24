import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import cvxopt
import cvxopt.solvers
import sys
import datetime
import os
from svmutil import *
import threading
from sklearn.utils import shuffle
import pandas as pd

def readDataFromFile(fileName, workOnDigit1, workOnDigit2, classToConsider=[]):
	file = open(fileName, "r")
	images = file.read().split("\n")
	file.close()

	x = []					# 1 image is stored in 1 row
	y = []					# col vector

	for i in range(len(images)):
		if images[i] == "":
			continue

		xtemp = []
		pixels = images[i].split(",")
		shouldUse = False

		for j in range(len(pixels)):
			if j == (len(pixels) - 1):
				label = int(pixels[j])
				if label == workOnDigit1:
					shouldUse = True
					y.append(+1)
				elif label == workOnDigit2:
					shouldUse = True
					y.append(-1)
				elif workOnDigit1 == -1 and workOnDigit2 == -1 and (label in classToConsider):
					shouldUse = True
					y.append(label)
				elif workOnDigit1 == -1 and workOnDigit2 == -1 and len(classToConsider) == 0:
					shouldUse = True
					y.append(label)
			else:
				xtemp.append(float(pixels[j]))

		if shouldUse == True:
			x.append(xtemp)

	x = np.array(x)
	y = np.array(y)

	return x, y

def normalizeData(x):
	x = x / 255
	return x

def readFromModelFile(modelFileName):
	w = []
	b = 0.0
	supportVectorsIndices = []
	aSpprtVctr = []
	ySpprtVctr = []
	xSpprtVctr = []

	file = open(modelFileName, "r")
	lines = file.readlines()
	file.close()

	mode = 0
	for j in range(len(lines)):
		if lines[j] == " \n":
			mode += 1
			continue

		elif lines[j] == "" or lines[j] == " ":
			continue

		line = lines[j].split(",")
	
		if mode == 0:
			for i in range(len(line)):
				if line[i] == "" or line[i] == " " or line[i] == "\n":
					continue
				w.append(float(line[i]))
		elif mode == 1:
			b = float(line[0])
		elif mode == 2:
			for i in range(len(line)):
				if line[i] == "" or line[i] == " " or line[i] == "\n":
					continue
				supportVectorsIndices.append(int(line[i]))
		elif mode == 3:
			atemp = []
			for i in range(len(line)):
				if line[i] == "" or line[i] == " " or line[i] == "\n":
					continue
				atemp.append(float(line[i]))
			aSpprtVctr.append(atemp)
		elif mode == 4:
			atemp = []
			for i in range(len(line)):
				if line[i] == "" or line[i] == " " or line[i] == "\n":
					continue
				atemp.append(float(line[i]))
			ySpprtVctr.append(atemp)
		elif mode == 5:
			atemp = []
			for i in range(len(line)):
				if line[i] == "" or line[i] == " " or line[i] == "\n":
					continue
				atemp.append(float(line[i]))
			xSpprtVctr.append(atemp)

	w = np.array(w)

	return w, b, supportVectorsIndices, aSpprtVctr, ySpprtVctr, xSpprtVctr

def writeModelToFile(modelFileName, w, b, supportVectorsIndices, aSpprtVctr, ySpprtVctr, xSpprtVctr):
	file = open(modelFileName, "w+")
	for i in range(len(w)):
		file.write(str(w[i]) + ",")
	file.write("\n")
	file.write(" \n")
	file.write(str(b[0]) + ",")
	file.write("\n")
	file.write(" \n")
	for i in range(len(supportVectorsIndices)):
		file.write(str(supportVectorsIndices[i]) + ",")
	file.write("\n")
	file.write(" \n")
	for i in range(len(aSpprtVctr)):
		for j in range(len(aSpprtVctr[i])):
			file.write(str(aSpprtVctr[i][j]) + ",")
		file.write("\n")
	file.write(" \n")
	for i in range(len(ySpprtVctr)):
		for j in range(len(ySpprtVctr[i])):
			file.write(str(ySpprtVctr[i][j]) + ",")
		file.write("\n")
	file.write(" \n")
	for i in range(len(xSpprtVctr)):
		for j in range(len(xSpprtVctr[i])):
			file.write(str(xSpprtVctr[i][j]) + ",")
		file.write("\n")
	file.close()

def kernel(typeOfTraining, x1, x2, Gamma):
	if typeOfTraining == 0:
		return np.dot(x1, x2)
	elif typeOfTraining == 1:
		return np.exp(-Gamma * (np.linalg.norm(x1 - x2) ** 2))

def calculateKernel(typeOfTraining, x1, x2, Gamma, startI, endI, startJ, endJ, k, isSymm):
	for i in range(startI, endI):
		for j in range(startJ, endJ):
			k[i,j] = kernel(typeOfTraining, x1[i], x2[j], Gamma)

			if isSymm:
				k[j,i] = k[i,j]

def weight(aSpprtVctr, ySpprtVctr, xSpprtVctr, typeOfTraining):
	w = None
	if typeOfTraining == 0:
		cntSpprtVctr = len(aSpprtVctr)
		w = np.zeros((len(xSpprtVctr[0])))
		for i in range(cntSpprtVctr):
			w += (aSpprtVctr[i] * ySpprtVctr[i] * xSpprtVctr[i])
		w = np.array(w)
	else:
		w = []

	return w

def testModel(typeOfTraining, x, w, b, aSpprtVctr, ySpprtVctr, xSpprtVctr, Gamma, modelFileName="", printMsg=""):
	y = []

	if typeOfTraining == 0:
		w = np.array(w).reshape((-1, 1))
		x = x.T
		y = np.dot(w.T, x) + b
		y = np.array(y).reshape(-1)

	else:
		# Create k matrix
		noOfRows = len(xSpprtVctr)
		noOfCols = len(x)
		k = np.zeros((noOfRows, noOfCols))

		if modelFileName != "" and os.path.exists(modelFileName):
			file = open(modelFileName, "r")
			lines = file.readlines()
			file.close()

			for i in range(len(lines)):
				if lines[i] == "" or lines[i] == " ":
					continue
				line = lines[i].split(",")
	
				for j in range(len(line)):
					if line[j] == "" or line[j] == " " or line[j] == "\n":
						continue
					k[i][j] = float(line[j])

		else:
			t = []
			noOfThread = 4
			for i in range(noOfThread):
				s1 = math.floor(i*noOfRows/noOfThread)
				e1 = math.floor((i+1)*noOfRows/noOfThread)
				s2 = 0
				e2 = noOfCols

				t.append(threading.Thread(target = calculateKernel, args = (typeOfTraining, 
											xSpprtVctr, 
											x, 
											Gamma, 
											s1,
											e1,
											s2,
											e2,
											k,
											False)))

				t[i].start()

			for i in range(noOfThread):
				t[i].join()

			#if modelFileName != "":
			#	file = open(modelFileName, "w+")
			#	for i in range(len(k)):
			#		for j in range(len(k[i])):
			#			file.write(str(k[i][j]) + ",")
			#		file.write("\n")
			#	file.close()

		# Test
		for j in range(len(x)):
			sum = 0.0
			for i in range(len(aSpprtVctr)):
				sum += (aSpprtVctr[i][0] * ySpprtVctr[i][0] * k[i,j])
			y.append(sum + b)
	
	for i in range(len(y)):
		if y[i] >= 0: 
			y[i] = 1
		elif y[i] < 0: 
			y[i] = -1

	return y

def trainModel(modelFileName, typeOfTraining, C, Gamma, trainFileName, printMsg="", workOnDigit1 = -1, workOnDigit2 = -1):
	w = []
	b = 0.0
	supportVectorsIndices = []
	aSpprtVctr = []
	ySpprtVctr = []
	xSpprtVctr = []

	if os.path.exists(modelFileName):
		w, b, supportVectorsIndices, aSpprtVctr, ySpprtVctr, xSpprtVctr = readFromModelFile(modelFileName)
		
	else:
		# Get Data
		xtrain, ytrain = readDataFromFile(trainFileName, workOnDigit1, workOnDigit2)
		xtrain = normalizeData(xtrain)

		# Tarin
		noOfImages = len(xtrain)
		noOfFeatures = len(xtrain[0])

		k = np.zeros((noOfImages, noOfImages))

		t = []
		noOfThread = 4
		for i in range(noOfThread):
			s1 = math.floor(i*noOfImages/noOfThread)
			e1 = math.floor((i+1)*noOfImages/noOfThread)
			s2 = s1
			e2 = noOfImages

			t.append(threading.Thread(target = calculateKernel, args = (typeOfTraining, 
											xtrain, 
											xtrain, 
											Gamma, 
											s1,
											e1,
											s2,
											e2,
											k,
											True)))

			t[i].start()

		for i in range(noOfThread):
			t[i].join()

		P = cvxopt.matrix(np.outer(ytrain, ytrain) * k, tc='d')
		q = cvxopt.matrix(np.ones(noOfImages) * -1, tc='d')
		A = cvxopt.matrix(ytrain, (1, noOfImages), 'd')
		b = cvxopt.matrix(0.0, tc='d')

		t1 = cvxopt.matrix(np.diag(np.ones(noOfImages) * -1), tc='d')
		t2 = cvxopt.matrix(np.diag(np.ones(noOfImages)), tc='d')
		G = cvxopt.matrix(np.vstack((t1, t2)), tc='d')
		t1 = cvxopt.matrix(np.zeros(noOfImages), tc='d')
		t2 = cvxopt.matrix(np.ones(noOfImages) * C, tc='d')
		h = cvxopt.matrix(np.vstack((t1, t2)), tc='d')

		model = cvxopt.solvers.qp(P, q, G, h, A, b)
		if model['status'] == 'unknown':
			print ("Some Issue ", printMsg)
			exit(2)

		alpha = np.ravel(model['x'])

		supportVectors = alpha > 1e-5
		supportVectorsIndices = np.arange(len(alpha))[supportVectors]
		aSpprtVctr = alpha[supportVectors]
		xSpprtVctr = xtrain[supportVectors]
		ySpprtVctr = ytrain[supportVectors]

		aSpprtVctr = np.array(aSpprtVctr).reshape((-1, 1))
		ySpprtVctr = np.array(ySpprtVctr).reshape((-1, 1))

		# w
		w = weight(aSpprtVctr, ySpprtVctr, xSpprtVctr, typeOfTraining)
		
		# b
		cntSpprtVctr = len(aSpprtVctr)
		summ = 0.0
		sums = 0.0
		for i in range(cntSpprtVctr):
			summ = 0.0
			for j in range(cntSpprtVctr):
				summ += (aSpprtVctr[j][0] * ySpprtVctr[j][0] * np.array(kernel(typeOfTraining, xSpprtVctr[j], xSpprtVctr[i], Gamma)))

			sums += (ySpprtVctr[i] - summ)

		b = sums / cntSpprtVctr		

		# Write Model to File
		#writeModelToFile(modelFileName, w, b, supportVectorsIndices, aSpprtVctr, ySpprtVctr, xSpprtVctr)

	return w, b, supportVectorsIndices, aSpprtVctr, ySpprtVctr, xSpprtVctr

def calculateAccuracy(yActual, yPredict):
	accurate = 0
	total = 0
	for i in range(len(yActual)):
		if yActual[i] == yPredict[i]:
			accurate += 1
		total += 1

	accuracy = accurate * 100 / total
	return accuracy

def trainLibsvmModel(typeOfTraining, C, Gamma, xtrain, ytrain, trainFileName):
	problem = svm_problem(ytrain.tolist(), xtrain.tolist())

	params = 0
	if typeOfTraining == 0:		params = svm_parameter("-s 0 -c " + str(C) + " -t 0")				# Linear - s (type of SVM), c (C in soft margin), t (kernel type), g (gamma in gaussian)
	elif typeOfTraining == 1:	params = svm_parameter("-s 0 -c " + str(C) + " -t 2 -g " + str(Gamma))		# Gaussian

	model = svm_train(problem, params)

	return model

def predictLibsvm(ytest, xtest, model):
	_, p_acc, _ = svm_predict(ytest.tolist(), xtest.tolist(), model)	# 1st param - predicted labels, 3rd param - a list of decision values
	return p_acc

def binaryClassification(trainFileName, testFileName, typeOfPackageToTrain, typeOfTraining, C, Gamma, workOnDigit1, workOnDigit2):
	# Get Data
	xtest, ytest = readDataFromFile(testFileName, workOnDigit1, workOnDigit2)
	xtest = normalizeData(xtest)

	# Train and Predict from Model
	if typeOfPackageToTrain == 0:		# cvxopt
		modelFileName = "binaryClass_0_" + str(typeOfTraining) + "_" + str(workOnDigit1) + str(workOnDigit2) + "_Model.txt"
		w, b, svInd, aSpprtVctr, ySpprtVctr, xSpprtVctr = trainModel(modelFileName, typeOfTraining, C, Gamma, trainFileName, "", workOnDigit1, workOnDigit2)

		ypredict = testModel(typeOfTraining, xtest, w, b, aSpprtVctr, ySpprtVctr, xSpprtVctr, Gamma, "")
		accuracy = calculateAccuracy(ytest, ypredict)
		print ("Accuracy = ", accuracy)

	elif typeOfPackageToTrain == 1:		# libsvm
		xtrain, ytrain = readDataFromFile(trainFileName, workOnDigit1, workOnDigit2)	
		xtrain = normalizeData(xtrain)

		model = trainLibsvmModel(typeOfTraining, C, Gamma, xtrain, ytrain, trainFileName)
		accuracy = predictLibsvm(ytest, xtest, model)
		print ("Accuracy = ", accuracy)

def trainLibsvmMultiClassModel(C, Gamma, xtrain, ytrain, trainFileName):
	problem = svm_problem(ytrain.tolist(), xtrain.tolist())
	params = svm_parameter("-s 0 -c " + str(C) + " -t 2 -g " + str(Gamma))		# Gaussian

	model = svm_train(problem, params)

	return model

def predictLibsvmMultiClass(ytest, xtest, model):
	_, p_acc, _ = svm_predict(ytest.tolist(), xtest.tolist(), model)	# 1st param - predicted labels, 3rd param - a list of decision values
	return p_acc

def oneVsOne(lock, trainFileName, testFileName, typeOfTraining, xtest, C, Gamma, workOnDigit1, workOnDigit2, ytestCnt = [], typeOfKernelCalcInTest=""):
	# Train and Predict from Model
	modelFileName = "multiClass_0_" + str(typeOfTraining) + "_" + str(workOnDigit1) + str(workOnDigit2) + "_Model.txt"
	w, b, svInd, aSpprtVctr, ySpprtVctr, xSpprtVctr = trainModel(modelFileName, typeOfTraining, C, Gamma, trainFileName, str(workOnDigit1)+str(workOnDigit2), workOnDigit1, workOnDigit2)

	modelFileName += "_kernel_"+typeOfKernelCalcInTest+".txt"
	ypredict = testModel(typeOfTraining, xtest, w, b, aSpprtVctr, ySpprtVctr, xSpprtVctr, Gamma, modelFileName, str(workOnDigit1)+str(workOnDigit2))

	### Critical Section ###
	lock.acquire()
	for k in range(len(ypredict)):
		if ypredict[k] == -1:
			ytestCnt[k][workOnDigit2] += 1
		else:
			ytestCnt[k][workOnDigit1] += 1
	lock.release()
	########################

def multiClassificationGaussianCvxopt(trainFileName, testFileName, typeOfTraining, C, Gamma, typeOfKernelCalcInTest, classToClassify, xtest, ytest):
	noOfClassToClassify = classToClassify[len(classToClassify)-1]+2
	ytestCnt = np.zeros((len(xtest), noOfClassToClassify))

	# Train and Predict
	noOfThreads = 5
	if noOfThreads > (len(classToClassify) * (len(classToClassify)-1) / 2):
		noOfThreads = (len(classToClassify) * (len(classToClassify)-1) / 2)

	pvsI = classToClassify[0]
	pvsJ = classToClassify[0]
	endDigit = classToClassify[len(classToClassify)-1]+1

	lock = threading.Lock()
		
	while pvsI < endDigit-1:
		t = []
		i = pvsI
		for i in range(pvsI, endDigit):
			for j in range(pvsJ, endDigit):
				if i >= j: continue

				# i - marked as +1, j - marked as -1
				t.append(threading.Thread(target = oneVsOne, args = (lock, trainFileName, testFileName, typeOfTraining, xtest, C, Gamma, i, j, ytestCnt,typeOfKernelCalcInTest)))
				t[len(t)-1].start()

				if len(t) == noOfThreads or pvsI == endDigit-1:
					pvsI = i
					pvsJ = j+1

					if pvsJ == endDigit:
						pvsI = i+1
						pvsJ = 0

					break

			if len(t) == noOfThreads or pvsI == endDigit-1:
				break
			pvsJ = 0

		for l in range(len(t)):
			t[l].join()

		if i == endDigit-1:
			break

	# Accuracy
	accurate = 0
	ytestCnt = np.array(ytestCnt)
	ynew = []
	for i in range(len(ytestCnt)):
		clas = np.argmax(ytestCnt[i])
		ynew.append(clas)
		if clas == ytest[i]:
			accurate += 1

	accuracy = accurate * 100 / len(ytestCnt)
	print ("Accuracy = ", accuracy)

	return ytest, ynew

def multiClassification(trainFileName, testFileName, typeOfPackageToTrain, typeOfTraining, C, Gamma, runOnTrainTest="traintest"):
	if typeOfPackageToTrain == 0:					# cvxopt
		classToClassify = [0,1,2,3,4,5,6,7,8,9]

		if runOnTrainTest == "traintest":
			########## Prediction on training data
			xtest, ytest = readDataFromFile(trainFileName, -1, -1, classToClassify)
			xtest = normalizeData(xtest)

			ytest, ynew = multiClassificationGaussianCvxopt(trainFileName, testFileName, typeOfTraining, C, Gamma, "trainAll", classToClassify, xtest, ytest)

		########## Prediction on test data
		xtest, ytest = readDataFromFile(testFileName, -1, -1, classToClassify)
		xtest = normalizeData(xtest)

		ytest, ynew = multiClassificationGaussianCvxopt(trainFileName, testFileName, typeOfTraining, C, Gamma, "testAll", classToClassify, xtest, ytest)

		return ytest, ynew

	else:								# libsvm
		classToClassify = [0,1,2,3,4,5,6,7,8,9]

		xtrain, ytrain = readDataFromFile(trainFileName, -1, -1, classToClassify)	
		xtest, ytest = readDataFromFile(testFileName, -1, -1, classToClassify)
		xtrain = normalizeData(xtrain)
		xtest = normalizeData(xtest)

		model = trainLibsvmMultiClassModel(C, Gamma, xtrain, ytrain, trainFileName)

		accuracytest = predictLibsvmMultiClass(ytest, xtest, model)
		accuracytrain = predictLibsvmMultiClass(ytrain, xtrain, model)

		print ("Testing Accuracy = ", accuracytest)
		print ("Training Accuracy = ", accuracytrain)

def confusionMatrix(trainFileName, testFileName, typeOfPackageToTrain, typeOfTraining, C, Gamma, runOnTrainTest):
	actualY, predictedY = multiClassification(trainFileName, testFileName, typeOfPackageToTrain, typeOfTraining, C, Gamma, runOnTrainTest)
	
	y_true = pd.Series(actualY, name='Actual')
	y_pred = pd.Series(predictedY, name='Predicted')
	
	print ("Confusion Matrix")
	print (pd.crosstab(y_true, y_pred, margins=True))

def validationMultiClassLibsvmClassification(trainFileName, testFileName, CList, Gamma):
	classToClassify = [0,1,2,3,4,5,6,7,8,9]

	xtottrain, ytottrain = readDataFromFile(trainFileName, -1, -1, classToClassify)
	xtest, ytest = readDataFromFile(testFileName, -1, -1, classToClassify)
	xtottrain = normalizeData(xtottrain)
	xtest = normalizeData(xtest)

	xtottrain, ytottrain = shuffle(xtottrain, ytottrain, random_state=0)

	totrows = len(xtottrain)
	totvalidrows = math.ceil(totrows * 10 / 100)
	tottrainrows = totrows - totvalidrows

	xtrain = xtottrain[:tottrainrows]
	ytrain = ytottrain[:tottrainrows]
	xvalidate = xtottrain[tottrainrows:tottrainrows+totvalidrows]
	yvalidate = ytottrain[tottrainrows:tottrainrows+totvalidrows]

	for i in range(len(CList)):
		C = CList[i]

		model = trainLibsvmMultiClassModel(C, Gamma, xtrain, ytrain, trainFileName)
		accuracyValid = predictLibsvmMultiClass(yvalidate, xvalidate, model)
		accuracyTest = predictLibsvmMultiClass(ytest, xtest, model)

		print ("Accuracy for validation ", C, " = ", accuracyValid)
		print ("Accuracy for test ", C, " = ", accuracyTest)

def plot(CList):
	vacc = [8.8, 42.05, 97.2, 97.3]
	vline, = plt.plot(CList, vacc, label="Validation Set Accuracy", linestyle='-', color='r', marker='x')

	tacc = [9.58, 45.57, 97.22, 97.26]
	tline, = plt.plot(CList, tacc, label="Testing Set Accuracy", linestyle='-', color='b', marker='o')

	plt.legend(handles=[vline, tline])
	plt.xscale('log')
	plt.xlabel("C")
	plt.ylabel("Accuracy")
	plt.title("Accuracy on validation and testing set as C varies")
	plt.savefig("Q2Part2Partd.png")
	plt.close()

fileX = sys.argv[1] # prints python_script.py
fileY = sys.argv[2] # prints var1
binmulti = int(sys.argv[3])
partnum = sys.argv[4] # prints var2

if binmulti == 0 and partnum == "a":
	######## Binary Class
	######## 1.(a) - Linear Kernel CVXOPT
	binaryClassification(fileX, fileY, 0, 0, 1.0, 0, 7, 8)

elif binmulti == 0 and partnum == "b":
	######## 1.(b) - Gaussian Kernel CVXOPT
	binaryClassification(fileX, fileY, 0, 1, 1.0, 0.05, 7, 8)

elif binmulti == 0 and partnum == "c":
	######## 1.(c) - LIBSVM
	binaryClassification(fileX, fileY, 1, 0, 1.0, 0, 7, 8)		# Linear Kernel
	binaryClassification(fileX, fileY, 1, 1, 1.0, 0.05, 7, 8)		# Gaussian Kernel

elif binmulti == 1 and partnum == "a":
	######### Multi Class
	######## 2.(a) - Gaussian Kernel CVXOPT
	multiClassification(fileX, fileY, 0, 1, 1.0, 0.05)

elif binmulti == 1 and partnum == "b":
	######## 2.(b) - LIBSVM
	multiClassification(fileX, fileY, 1, 1, 1.0, 0.05)

elif binmulti == 1 and partnum == "c":
	######## 2.(c) - Confusion
	confusionMatrix(fileX, fileY, 0, 1, 1.0, 0.05, "test")

elif binmulti == 1 and partnum == "d":
	######## 2.(d) - Validation Using LibSvm
	validationMultiClassLibsvmClassification(fileX, fileY, [0.0001, 0.01, 1, 5, 10], 0.05)
	plot([0.0001, 0.01, 1, 5, 10])
