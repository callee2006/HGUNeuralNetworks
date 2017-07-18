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


#ifndef __HGURBM__
#define	__HGURBM__

class HGURBM :public HGULayer {
	float *m_aVisible;					// m_aVisble is allocated in RBM, while m_pInput is a pointer to outside memory block
	float *m_aVisibleProb;
	unsigned char *m_aVisibleClamped;

	// for consistency with other layers, 'hidden layer' is called 'output layer'
//	float *m_aOutput;					// defined in the base class
	float *m_aOutputProb;
	float *m_aOutputProb0;				// for CD training
	unsigned char *m_aOutputClamped;

	// for grouping input nodes such as V = (X, Y)
	int *m_aGroupIndex;
	int m_noGroup;

public:
	HGURBM(): HGULayer() {
		m_aVisible = NULL;
		m_aVisibleProb = NULL;
		m_aVisibleClamped = NULL;

		m_aOutputProb = NULL;
		m_aOutputProb0 = NULL;
		m_aOutputClamped = NULL;

		m_aGroupIndex = NULL;
		m_noGroup = NULL;
	}

	HGURBM(int inputDim, int outputDim, HGULayer *pShareSrc): HGULayer() {
		m_aVisible = NULL;
		m_aVisibleProb = NULL;
		m_aVisibleClamped = NULL;

		m_aOutputProb = NULL;
		m_aOutputProb0 = NULL;
		m_aOutputClamped = NULL;

		m_aGroupIndex = NULL;
		m_noGroup = NULL;

		Alloc(inputDim, outputDim, pShareSrc);
	}

	virtual ~HGURBM(void){
		Delete();
	}

	void Delete();

	virtual int Alloc(int inputDim, int outputDim, HGULayer *pShareSrc);

	float* GetVisible()					{ return m_aVisible; }
	float* GetVisibleProb()				{ return m_aVisibleProb; }
	unsigned char* GetVisibleClamped()	{ return m_aVisibleClamped; }

//	float* GetOutput()					{ return m_aOutput; }		// defined in the base class
	float* GetOutputProb()				{ return m_aOutputProb; }
	unsigned char* GetOutputClamped()	{ return m_aOutputClamped; }

	int ComputeVisibleProb();
	int SampleVisible()					{ return Sample_Binomial(m_aVisibleProb, m_inputDim, m_aVisible, m_aVisibleClamped); }
	int Decode()						{ return ComputeVisibleProb() && SampleVisible(); }

	int ComputeOutputProb(float *pInput);
	int SampleOutput()					{ return Sample_Binomial(m_aOutputProb, m_outputDim, m_aOutput, m_aOutputClamped); }
	int Encode(float *pInput)			{ return ComputeOutputProb(pInput) && SampleOutput(); }

	int GibsSampling(float *pInput, int k, float *pOutputProb0 = NULL);

	int Sample_Binomial(float *pProb, int size, float *pDigit, unsigned char *pClamped);

	int ComputeGradient_CD(float *pInput, int k);
	int UpdateWeight(float learningRate);

	float GetEnergy(float *pInput);
	float GetError(float *pInput);

	void Display();
	void DisplayWeight(float *pWeight, const char *title);
};

#endif	//	__HGURBM__