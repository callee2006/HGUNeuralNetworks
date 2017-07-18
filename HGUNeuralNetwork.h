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


class HGUNeuralNetwork {
	int m_noLayer;
	HGULayer **m_aLayer;		// array of HGULayers
	
public:
	HGUNeuralNetwork(){
		m_noLayer = 0;
		m_aLayer = NULL;
	}
	~HGUNeuralNetwork(){
		Delete();
	}

	int IsAllocated() { return m_aLayer != NULL; }
	int Alloc(int noLayer, int *pNoNode, HGUNeuralNetwork *pShareSrc);
	void Delete();

	float* GetOutput()			{ return m_aLayer[m_noLayer-1]->GetOutput(); }

	int Propagate(float *pInput);
	int GetMaxOutputIndex()		{ return m_aLayer[m_noLayer-1]->GetMaxOutputIndex(); }
	int ComputeGradient(float *pInput, float *pDesiredOutput);
	int UpdateWeight(float learningRate);

	HGULayer* operator[] (int idx) { return m_aLayer[idx]; }

	float GetError(float *pDesiredOutput);

	int MergeGradient(HGUNeuralNetwork *pSrc);
};