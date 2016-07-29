/***
	Copyright 2012 Injung Kim

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
***/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "HGULayer.h"
#include "HGUNeuralNetwork.h"
#include "HGUAutoEncoder.h"

int TestMLP();
int TestAutoEncoder();

int main(int argc, char *argv[])
{
	TestMLP();

	TestAutoEncoder();

	return TRUE;
}

int TestMLP()
{
	int aNetStruct[] = { 2, 4, 1 };
	const int noSample = 4;
	float aSample[4][2] = {
		{ 0.F, 0.F },
		{ 0.F, 1.F },
		{ 1.F, 0.F },
		{ 1.F, 1.F }
	};
	float aDesiredOutput[] = { 0, 1, 1, 0 };
	
	srand((unsigned int)time(NULL));

	HGUNeuralNetwork nn;
	nn.Alloc(2, aNetStruct);
	
	printf("=== Training MLP...\n");
	const int maxEpoch = 1000000;
	for(int epoch = 0; epoch < maxEpoch; epoch++){
		float error = 0.F;

		for(int i = 0; i < noSample; i++){
			nn.ComputeGradient(aSample[i], &aDesiredOutput[i]);
			error += nn.GetError(&aDesiredOutput[i]);
		}
		error /= noSample;

		nn.UpdateWeight(0.3F);
		if((epoch + 1) % 100 == 0)
			printf("epoch = %d, error = %f\n", epoch + 1, error);			

//		if((epoch + 1) % 1000 == 0)
//			getchar();

		if(error < 0.0001F)
			break;
	}

	printf("=== Training MLP... Done.\n");
	system("PAUSE");

	printf("=== Testing MLP...\n");
	for(int j = 0; j < noSample; j++){
		nn.Propagate(aSample[j]);
		float *output = nn.GetOutput();
		float *hidden = nn[0].GetOutput();
		printf("sample %d: (%f %f) --> (%f %f %f %f) --> %f\n", j, aSample[j][0], aSample[j][1],  hidden[0], hidden[1], hidden[2], hidden[3], output[0]);
	}


	system("PAUSE");

	return TRUE;
}

int TestAutoEncoder()
{
	const int noHidden = 5;
	const int noSample = 4;
	float aSample[4][2] = {
		{ 0.F, 0.F },
		{ 0.F, 1.F },
		{ 1.F, 0.F },
		{ 1.F, 1.F }
	};

	srand((unsigned int)time(NULL));

	HGUAutoEncoder ae(2, noHidden);

	const int maxEpoch = 1000000;

	printf("=== Training AutoEncoder...\n");

	for(int epoch = 0; epoch < maxEpoch; epoch++){
		float error = 0.F;

		for(int i = 0; i < noSample; i++){
			ae.ComputeGradient(aSample[i]);
			error += ae.GetReproductionError();
		}
		error /= noSample;


		ae.UpdateWeight(0.3F);
		if((epoch + 1) % 100 == 0){
			printf("epoch = %d, error = %f\n", epoch + 1, error);

			for(int j = 0; j < noSample; j++){
				ae.Reproduce(aSample[j]);
				float *reprod = ae.GetReproduction();
				printf("sample %d: (%f %f) --> (%f %f)\n", j, aSample[j][0], aSample[j][1], reprod[0], reprod[1]);
			}
		}

		if(error < 0.0001F)
			break;
	}

	printf("=== Training AutoEncoder... Done.\n");
	system("PAUSE");
	
	printf("=== Testing AutoEncoder...\n");

	for(int j = 0; j < noSample; j++){
		ae.Reproduce(aSample[j]);
		float *encoded = ae.GetOutput();
		float *reprod = ae.GetReproduction();
		printf("sample %d: (%f %f) --> (%f %f %f %f %f --> (%f %f)\n", j, aSample[j][0], aSample[j][1], encoded[0], encoded[1], encoded[2], encoded[3], encoded[4], reprod[0], reprod[1]);
	}

	system("PAUSE");

	return TRUE;
}