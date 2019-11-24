#!/bin/bash

if [ "$1" = "1" ]; then
	fileX=$2
	fileY=$3
	partnum=$4

	py Q1.py $fileX $fileY $partnum
fi

if [ "$1" = "2" ]; then
	fileX=$2
	fileY=$3
	binmulti=$4
	partnum=$5

	py Q2.py $fileX $fileY $binmulti $partnum
fi
