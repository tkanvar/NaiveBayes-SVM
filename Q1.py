import numpy as np
import sys
from collections import defaultdict
import datetime
import sys
import json
from ass2_data import utils
import nltk
import math
import random
from sklearn.metrics import confusion_matrix
import os.path
from os import path
import pandas as pd
import ast
from sklearn.metrics import f1_score
import re
from nltk.util import ngrams

######### Star1 - 5 in Input Json has been mapped to Star0 - Star4 for ease of coding
# typeOfWordTokenizer - 0 (split at " "), 1 (stem), 2 (2 features), 3 (bigrams)

def getModelFileName(typeOfWordTokenizer, trainFileName):
	if trainFileName == "ass2_data\\train_full.json":
		return "ModelInfo_" + str(typeOfWordTokenizer) + "_fullTrain.txt"
	else:
		return "ModelInfo_" + str(typeOfWordTokenizer) + "_subTrain.txt"

def doesModelFileExist(modelFileName):
	return path.exists(modelFileName)

def readModelFromFile(fileName, wordCountInReviews, starsCount, dictionary, wordInfoMatrix):
	file = open(fileName, "r")

	words = (file.readline()).split(",")
	for i in range(len(words)):
		if words[i] == "\n" or words[i] == " " or words[i] == "":
				continue

		wordCountInReviews[i] = int(words[i])

	words = (file.readline()).split(",")
	for i in range(len(words)):
		if words[i] == "\n" or words[i] == " " or words[i] == "":
			continue

		starsCount[i] = int(words[i])

	line = file.readline()
	cntr = 0
	while line != " \n":
		cntr = cntr + 1
		asci = []
		asciStr = line.split(",")
		for i in range(len(asciStr)):
			if asciStr[i] == "\n" or asciStr[i] == " " or asciStr[i] == "":
				continue
			asci.append(int(asciStr[i]))

		name = ''.join(chr(k) for k in asci)
		value = int(file.readline())

		dictionary[name] = value

		line = file.readline()

	line = file.readline()
	while line != "\n" and line != "" and line != " ":

		words = line.split(",")
		wordInfoLine = []

		for j in range(len(words)):
			if words[j] == " " or words[j] == "\n" or words[j] == "":
				continue

			wordInfoLine.append(int(words[j]))

		wordInfoMatrix.append(wordInfoLine)

		line = file.readline()

	file.close()

def writeModelToFile(fileName, wordCountInReviews, starsCount, dictionary, wordInfoMatrix):
	return

	file = open(fileName, "w+")

	for i in range(len(wordCountInReviews)):
		file.write(str(wordCountInReviews[i]) + ",")

	file.write("\n")

	for i in range(len(starsCount)):
		file.write(str(starsCount[i]) + ",")

	file.write("\n")

	for i,j in dictionary.items():
		asci = [ord(k) for k in i]
		for k in range(len(asci)):
			file.write(str(asci[k]) + ",")

		file.write("\n" + str(j) + "\n")

	file.write(" \n")

	for i in range(len(wordInfoMatrix)):
		for j in range(len(wordInfoMatrix[i])):
			file.write(str(wordInfoMatrix[i][j]) + ",")
		file.write("\n")

	file.close()

def tokenizeLine(line, typeOfWordTokenizer):
	words = []

	if typeOfWordTokenizer == 0:
		words = line.split(" ")

	elif typeOfWordTokenizer == 1:
		words = utils.getStemmedDocuments(line)

	elif typeOfWordTokenizer == 2:
		line = " ".join(utils.getStemmedDocuments(line))
		words = line.split(" ")
		words = words + list(ngrams(line, 2))
		words = words + list(ngrams(line, 3))

	elif typeOfWordTokenizer == 3:
		line = " ".join(utils.getStemmedDocuments(line))
		words = list(nltk.bigrams(line.split()))

	return words

def randomPredict(trainFileName, testFileName):
	accuracyCount = 0
	totalCount = 0

	for line in utils.json_reader(testFileName):
		y = int(line["stars"]) - 1

		totalCount += 1

		y1 = random.randint(1,6)
		if y == y1:
			accuracyCount += 1

	print ("Random Prediction Accuracy = ", (accuracyCount * 100 / totalCount), "%")

def majorityPredict(trainFileName, testFileName):
	starsCount = [0] * 5

	for line in utils.json_reader(testFileName):
		y = int(line["stars"]) - 1
		starsCount[y] += 1

	maxStars = max(starsCount)

	print ("Baseline Prediction Accuracy = ", (maxStars * 100 / sum(starsCount)), "%")

def predict(fileName, starsCount, dictionary, wordInfoMatrix, noOfColsInWordInfoMatrix, typeOfWordTokenizer, printData):
	accurate = 0
	notAccurate = 0

	actualY = []
	predictedY = []

	prob = [0] * 5
	totalReviews = starsCount[0] + starsCount[1] + starsCount[2] + starsCount[3] + starsCount[4]
	prob[0] = starsCount[0] / totalReviews
	prob[1] = starsCount[1] / totalReviews
	prob[2] = starsCount[2] / totalReviews
	prob[3] = starsCount[3] / totalReviews
	prob[4] = starsCount[4] / totalReviews

	cntr = 0

	for line in utils.json_reader(fileName):
		cntr = cntr + 1

		y = int(line["stars"]) - 1
		x = (line["text"]).lower()

		words = tokenizeLine(x, typeOfWordTokenizer)
		probReviewStar = [1] * 5

		for i in range(len(words)):
			word = words[i]					# If same word repeated t times then prob of document being y is multiplied t times
			if dictionary[word] == -1:
				continue
			index = dictionary[word]

			for j in range(5):
				probReviewStar[j] += math.log(wordInfoMatrix[index][int(noOfColsInWordInfoMatrix/2) + j])

		for j in  range(5):
			probReviewStar[j] += math.log(prob[j])

		maxProbIndex = probReviewStar.index(max(probReviewStar))

		actualY.append(y)
		predictedY.append(maxProbIndex)

		if maxProbIndex == y:
			accurate += 1
		else:
			notAccurate +=1

	if printData == True:
		print ("Accuracy = ", ((accurate / (accurate + notAccurate)) * 100), "%")

	return actualY, predictedY

def fillWordInfoMatrix(j, wordInfoMatrix, wordCountInReviews, dictionarySize, noOfColsInWordInfoMatrix, typeOfWordTokenizer, totalNumberOfWordsInAllReviews):
	tdIdfFactor = 0
	if typeOfWordTokenizer == 2 or typeOfWordTokenizer == 3:		# TD-IDF
		totalNumberOfReviewsWithWord = wordInfoMatrix[j][0] + wordInfoMatrix[j][1] + wordInfoMatrix[j][2] + wordInfoMatrix[j][3] + wordInfoMatrix[j][4]
		tdIdfFactor = math.log(totalNumberOfWordsInAllReviews / totalNumberOfReviewsWithWord)

	for i in range(int(noOfColsInWordInfoMatrix/2)):
		index = i + int(noOfColsInWordInfoMatrix/2)

		if typeOfWordTokenizer == 2 or typeOfWordTokenizer == 3:		# TD-IDF
			wordInfoMatrix[j][i] *= tdIdfFactor

		prob = (wordInfoMatrix[j][i] + 1) / (wordCountInReviews[i] + dictionarySize)
		wordInfoMatrix[j][index] = prob

def storeAndPredictStars(typeOfWordTokenizer, predictForTrainData, printData, trainFileName, testFileName):
	noOfColsInWordInfoMatrix = 10		# 1st 5 stores the count of word occurred in star0, star1, star2, star3, star4 review. Next 5 is the probabilty of appearing this word in star0, star1, sta2, star3, and star4 review

	wordCountInReviews = [0] * 5
	starsCount = [0] * 5
	dictionary = {}
	wordInfoMatrix = []
	dictionary = defaultdict(lambda:-1, dictionary)

	modelFileName = getModelFileName(typeOfWordTokenizer, trainFileName)
	if doesModelFileExist(modelFileName) == True:
		readModelFromFile(modelFileName, wordCountInReviews, starsCount, dictionary, wordInfoMatrix)

	else:
		cntr = 0

		for line in utils.json_reader(trainFileName):
			cntr = cntr + 1
			y = int(line["stars"]) - 1
			x = (line["text"]).lower()

			words = tokenizeLine(x, typeOfWordTokenizer)
			starsCount[y] += 1
			wordCountInReviews[y] += len(words)

			for i in range(len(words)):
				word = words[i]

				if dictionary[word] == -1:
					wordInfoMatrix.append([0] * noOfColsInWordInfoMatrix)
					dictionary[word] = len(wordInfoMatrix)-1

				indexAtWhichPerformOp = dictionary[word]
				wordInfoMatrix[indexAtWhichPerformOp][y] += 1

		if typeOfWordTokenizer != 2 and typeOfWordTokenizer != 3:	# because of bigrams cant store in dictionary
			writeModelToFile(modelFileName, wordCountInReviews, starsCount, dictionary, wordInfoMatrix)

	dictionarySize = len(wordInfoMatrix)
	totalNumberOfWordsInAllReviews = wordCountInReviews[0] + wordCountInReviews[1] + wordCountInReviews[2] + wordCountInReviews[3] + wordCountInReviews[4]
	for i in range(len(wordInfoMatrix)):
		fillWordInfoMatrix(i, wordInfoMatrix, wordCountInReviews, dictionarySize, noOfColsInWordInfoMatrix, typeOfWordTokenizer, totalNumberOfWordsInAllReviews)

	if predictForTrainData == True:
		actualY, predictedY = predict(trainFileName, starsCount, dictionary, wordInfoMatrix, noOfColsInWordInfoMatrix, typeOfWordTokenizer, printData)
	actualY, predictedY = predict(testFileName, starsCount, dictionary, wordInfoMatrix, noOfColsInWordInfoMatrix, typeOfWordTokenizer, printData)

	return actualY, predictedY

def confusionMatrix(trainFileName, testFileName):
	actualY, predictedY = storeAndPredictStars(0, False, False, trainFileName, testFileName)
	
	y_true = pd.Series(actualY, name='Actual')
	y_pred = pd.Series(predictedY, name='Predicted')
	
	print ("Confusion Matrix")
	print (pd.crosstab(y_true, y_pred, margins=True))

def calculateF1Score(typeOfWordTokenizer, isDataGiven, actualY, predictedY, trainFileName, testFileName):
	if isDataGiven == False:
		actualY, predictedY = storeAndPredictStars(typeOfWordTokenizer, False, True, trainFileName, testFileName)

	print ("F1 Score for each class = ", f1_score(actualY, predictedY, average=None))
	print ("macro-F1 Score = ", f1_score(actualY, predictedY, average='macro'))
	print ("micro-F1 Score = ", f1_score(actualY, predictedY, average='micro'))

fileX = sys.argv[1] # prints python_script.py
fileY = sys.argv[2] # prints var1
partnum = sys.argv[3] # prints var2

if partnum == "a":
	######## (a)
	storeAndPredictStars(0, True, True, fileX, fileY)

elif partnum == "b":
	######## (b)
	randomPredict(fileX, fileY)
	majorityPredict(fileX, fileY)

elif partnum == "c":
	######## (c)
	confusionMatrix(fileX, fileY)

elif partnum == "d":
	######## (d)
	storeAndPredictStars(1, False, True, fileX, fileY)

elif partnum == "e":
	######## (e)
	storeAndPredictStars(3, False, True, fileX, fileY)

elif partnum == "f":
	######## (f)
	calculateF1Score(3, False, [], [], fileX, fileY)

elif partnum == "g":
	######## (g)
	actualY, predictedY = storeAndPredictStars(3, False, True, fileX, fileY)
	calculateF1Score(3, True, actualY, predictedY, fileX, fileY)
