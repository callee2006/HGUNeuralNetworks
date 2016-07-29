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


#ifndef __HGUAutoEncoder__
#define __HGUAutoEncoder__

class HGUAutoEncoder: public HGULayer {
	HGULayer *m_pDecoder;

public:
	HGUAutoEncoder() : HGULayer() {
		m_pDecoder = NULL;
	}

	HGUAutoEncoder(int inputDim, int outputDim) : HGULayer() {
		m_pDecoder = NULL;
		Alloc(inputDim, outputDim);
	}

	virtual ~HGUAutoEncoder(){
		delete m_pDecoder;
	}

	virtual int Alloc(int inputDim, int outputDim);
	void Delete();

	float* GetReproduction()	{ return m_pDecoder->GetOutput(); }
	float GetReproductionError()	{ return  m_pDecoder->GetError(GetInput()); };

	void Encode(float *pInput)	{ Propagate(pInput); }
	void Decode()				{ m_pDecoder->Propagate(GetOutput()); }
	int Reproduce(float *pInput){ return HGULayer::Propagate(pInput) && m_pDecoder->Propagate(GetOutput());	}

	int ComputeGradient(float *pInput);
	int UpdateWeight(float learningRate);

	void MergeGradientTranspose(float *pDest, int inputDim, int outputDim, float *pSrc);
	void CopyWeightTranspose(float *pDest, int inputDim, int outputDim, float *pSrc);
};

#endif // !__HGUAutoEncoder__
