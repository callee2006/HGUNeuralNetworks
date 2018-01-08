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

#ifdef WINDOWS
#include <Windows.h>
#endif // WINDOWS


#include "HGULayer.h"
#include "HGUNeuralNetwork.h"
#include "HGUAutoEncoder.h"
#include "HGURBM.h"

int TestMLP();
int TestAutoEncoder();
int TestRBM();
int TestRBM_MT();

int main(int argc, char *argv[])
{
	srand((unsigned int)time(NULL));

//	TestMLP();

//	TestAutoEncoder();

	TestRBM();

//	TestRBM_MT();

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
	nn.Alloc(2, aNetStruct, NULL);
	
	printf("=== Training MLP...\n");
	const int maxEpoch = 1000000;

	float error = 0.F;
	int n = 0;
	for(int epoch = 0; epoch < maxEpoch; epoch++){
		for(int i = 0; i < noSample; i++){
			nn.ComputeGradient(aSample[i], &aDesiredOutput[i]);
			error += nn.GetError(&aDesiredOutput[i]);
		}
		n += noSample;

		nn.UpdateWeight(0.1F / noSample);
		if(epoch == 0 || (epoch + 1) % 10000 == 0)
			printf("epoch = %d, error = %f\n", epoch + 1, error / n);

		if(error / n < 0.0001F)
			break;
	}

	printf("=== Training MLP... Done.\n");
	system("PAUSE");

	printf("=== Testing MLP...\n");
	for(int j = 0; j < noSample; j++){
		nn.Propagate(aSample[j]);
		float *output = nn.GetOutput();
		float *hidden = nn[0]->GetOutput();
		printf("sample %d: (%f %f) --> (%f %f %f %f) --> %f\n", j, aSample[j][0], aSample[j][1],  hidden[0], hidden[1], hidden[2], hidden[3], output[0]);
	}

	system("PAUSE");

	return TRUE;
}

int TestAutoEncoder()
{
	const int noHidden = 20;
	const int noSample = 10;
	float aSample[10][7] = {
		{ 1, 1, 1, 0, 1, 1, 1 },		// 0
		{ 0, 0, 1, 0, 0, 1, 0 },		// 1
		{ 1, 0, 1, 1, 1, 0, 1 },		// 2
		{ 1, 0, 1, 1, 0, 1, 1 },		// 3
		{ 0, 1, 1, 0, 0, 1, 0 },		// 4
		{ 1, 1, 0, 1, 0, 1, 1 },		// 5
		{ 1, 1, 0, 1, 1, 1, 1 },		// 6
		{ 1, 0, 1, 0, 0, 1, 0 },		// 7
		{ 1, 1, 1, 1, 1, 1, 1 },		// 8
		{ 1, 1, 1, 1, 0, 1, 0 } 		// 9
	};

	srand((unsigned int)time(NULL));

	HGUAutoEncoder ae(7, noHidden, NULL);

	const int maxEpoch = 1000000;

	printf("=== Training AutoEncoder...\n");

	float error = 0.F;
	int n = 0;
	for(int epoch = 0; epoch < maxEpoch; epoch++){
		for(int i = 0; i < noSample; i++){
			ae.ComputeGradient(aSample[i]);
			error += ae.GetReproductionError();
		}
		n += noSample;

		ae.UpdateWeight(0.3F / noSample);
		if((epoch + 1) % 10000 == 0)
			printf("epoch = %d, error = %f\n", epoch + 1, error / n);

		if(error / n < 0.001F)
			break;
	}

	printf("=== Training AutoEncoder... Done.\n");
	system("PAUSE");
	
	printf("=== Testing AutoEncoder...\n");

	for(int j = 0; j < noSample; j++){
		ae.Reproduce(aSample[j]);
		float *reprod = ae.GetReproduction();

		printf("sample %d: ( ", j);

		int k = 0;
		for(k = 0; k < 7; k++)
			printf("%1.0f ", aSample[j][k]);
		printf(") --> (");

		for(k = 0; k < 7; k++)
			printf("%5.2f ", reprod[k]);
		printf(")\n");
	}

	system("PAUSE");

	return TRUE;
}

int TestRBM()
{
	const int noHidden = 20;
	const int noSample = 10;
	float aSample[10][7] = {
		{ 1, 1, 1, 0, 1, 1, 1 },		// 0
		{ 0, 0, 1, 0, 0, 1, 0 },		// 1
		{ 1, 0, 1, 1, 1, 0, 1 },		// 2
		{ 1, 0, 1, 1, 0, 1, 1 },		// 3
		{ 0, 1, 1, 0, 0, 1, 0 },		// 4
		{ 1, 1, 0, 1, 0, 1, 1 },		// 5
		{ 1, 1, 0, 1, 1, 1, 1 },		// 6
		{ 1, 0, 1, 0, 0, 1, 0 },		// 7
		{ 1, 1, 1, 1, 1, 1, 1 },		// 8
		{ 1, 1, 1, 1, 0, 1, 0 } 		// 9
	};

	srand((unsigned int)time(NULL));

	HGURBM rbm(7, noHidden, NULL);

	printf("=== Training RBM...\n");

	const int maxEpoch = 1000000;
	float error = 0.F;
	int n = 0;
	for(int epoch = 0; epoch < maxEpoch; epoch++){
		for(int i = 0; i < noSample; i++){
			rbm.ComputeGradient_CD(aSample[i], 1);
			error += rbm.GetError(NULL);
		}
		n += noSample;

		rbm.UpdateWeight(0.01F / noSample);

		if(epoch == 0 || (epoch + 1) % 10000 == 0)
			printf("epoch = %d, error = %f\n", epoch + 1, error / n);

		if(error / n < 0.001F)
			break;
	}

	printf("=== Training RBM... Done.\n");
	system("PAUSE");
	
	printf("=== Testing RBM...\n");

	for(int j = 0; j < noSample; j++){
		rbm.GibsSampling(aSample[j], 1);
		float *reprod = rbm.GetVisibleProb();
		printf("sample %d: ( ", j);

		int k = 0;
		for(k = 0; k < 7; k++)
			printf("%1.0f ", aSample[j][k]);
		printf(") --> (");

		for(k = 0; k < 7; k++)
			printf("%5.2f ", reprod[k]);
		printf(")\n");
	}

	system("PAUSE");

	return TRUE;
}


struct RBMThreadInfo {
	int m_threadIdx;
	int m_noThread;
	HGURBM *m_pRBM;
	float (*m_pSample)[7];
	int m_noSample;
	float m_error;
};

#ifdef WINDOWS
DWORD WINAPI TrainRBMThreadFn(void *vpTrainInfo);
#else	// WINDOWS
void* TrainRBMThreadFn(void *vpTrainInfo);
#endif // WINDOWS


int TestRBM_MT()
{
	const int noHidden = 20;
	const int noSample = 10;
	const int noThread = 2;

	float aSample[10][7] = {
		{ 1, 1, 1, 0, 1, 1, 1 },		// 0
		{ 0, 0, 1, 0, 0, 1, 0 },		// 1
		{ 1, 0, 1, 1, 1, 0, 1 },		// 2
		{ 1, 0, 1, 1, 0, 1, 1 },		// 3
		{ 0, 1, 1, 0, 0, 1, 0 },		// 4
		{ 1, 1, 0, 1, 0, 1, 1 },		// 5
		{ 1, 1, 0, 1, 1, 1, 1 },		// 6
		{ 1, 0, 1, 0, 0, 1, 0 },		// 7
		{ 1, 1, 1, 1, 1, 1, 1 },		// 8
		{ 1, 1, 1, 1, 0, 1, 0 } 		// 9
	};

	srand((unsigned int)time(NULL));

	HGURBM aRBM[noThread];
	RBMThreadInfo aTrainInfo[noThread];

#ifdef WINDOWS
	HANDLE aHandle[noThread];
#endif // WINDOWS

	for(int t = 0; t < noThread; t++){
		aRBM[t].Alloc(7, noHidden, (t == 0) ? NULL : &aRBM[0]);
		aTrainInfo[t].m_threadIdx = t;
		aTrainInfo[t].m_noThread = noThread;
		aTrainInfo[t].m_pRBM = &aRBM[t];
		aTrainInfo[t].m_pSample = aSample;
		aTrainInfo[t].m_noSample = noSample;
	}

	printf("=== Training RBM...\n");

	const int maxEpoch = 1000000;
	float error = 0.F;
	int n = 0;
	for(int epoch = 0; epoch < maxEpoch; epoch++){
		int t = 0;
		for(t = 1; t < noThread; t++){
#ifdef WINDOWS
			aHandle[t] = CreateThread(NULL, 0, TrainRBMThreadFn, (void*)&aTrainInfo[t], 0, NULL);
#endif // WINDOWS
		}

#ifdef	WINDOWS
		TrainRBMThreadFn(&aTrainInfo[0]);
#endif	//	WINDOWS

#ifdef WINDOWS
		WaitForMultipleObjects(noThread - 1, &aHandle[1], TRUE, INFINITE);
		for(t = 1; t < noThread; t++)
			CloseHandle(aHandle[t]);
#endif // WINDOWS

		for(t = 0; t < noThread; t++){
			if(t > 0)
				aRBM[0].MergeGradient(&aRBM[t]);
			error += aTrainInfo[t].m_error;
		}

		n += noSample;

		aRBM[0].UpdateWeight(0.01F / noSample);

		if(epoch == 0 || (epoch + 1) % 1000 == 0)
			printf("epoch = %d, error = %f\n", epoch + 1, error / n);

		if(error / n < 0.001F)
			break;
	}

	printf("=== Training RBM... Done.\n");
	system("PAUSE");
	
	printf("=== Testing RBM...\n");

	for(int j = 0; j < noSample; j++){
		aRBM[0].GibsSampling(aSample[j], 1);
		float *reprod = aRBM[0].GetVisibleProb();
		printf("sample %d: ( ", j);

		int k = 0;
		for(k = 0; k < 7; k++)
			printf("%1.0f ", aSample[j][k]);
		printf(") --> (");

		for(k = 0; k < 7; k++)
			printf("%5.2f ", reprod[k]);
		printf(")\n");
	}

	system("PAUSE");

	return TRUE;
}

#ifdef WINDOWS
DWORD WINAPI TrainRBMThreadFn(void *vpTrainInfo)
#else	// WINDOWS
void* TrainRBMThreadFn(void *vpTrainInfo)
#endif // WINDOWS
{
	RBMThreadInfo *pParam = (RBMThreadInfo*)vpTrainInfo;

	pParam->m_error = 0.F;

	for(int i = pParam->m_threadIdx; i < pParam->m_noSample; i += pParam->m_noThread){
		pParam->m_pRBM->ComputeGradient_CD(pParam->m_pSample[i], 1);
		pParam->m_error += pParam->m_pRBM->GetError(NULL);
	}

#ifdef WINDOWS
	return 0;
#else	// WINDOWS
	return NULL;	
#endif	// WINDOWS
}