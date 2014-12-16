#include "rf_rgrsn_constant_regressor.h"
#include "rf_rgrsn_constant_regressor_builder.h"
#include "mex.h"


// Interface function of matlab
// assume prhs[0]: features, prhs[1]: regression file (.txt)
// plhs[0]: predicted values
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
	const char *error_msg;
	
	if(nrhs == 2)
	{		
		//get training data dimension and point
		unsigned int rows = mxGetM(prhs[0]);
		unsigned int cols = mxGetN(prhs[0]);
		double *pFeature = mxGetPr(prhs[0]);		
		
		char *input_regressor_name = mxArrayToString(prhs[1]);

		if(mxIsDouble(prhs[0]) == 0)
		{
			mexPrintf("Error: testing data type must be double.\n");
			return;
		}

		vector<nl_vector> features;		
		nl_vector feature(cols, 0);
		// column major
		for(int i = 0; i<rows; i++)
		{			
			for(int j = 0; j<cols; j++)
			{
				feature[j] = pFeature[j * rows + i];
			}
		
			features.push_back(feature);		
		}
		
		mexPrintf("testing data size is %lu\n", features.size());

		rf_rgrsn_constant_regressor_builder builder;		
		rf_rgrsn_constant_regressor *pRegressor = builder.new_regressioner();
		bool isRead = pRegressor->read(input_regressor_name);
		if(!isRead)
		{
			mexPrintf("Error can not read file %f\n", input_regressor_name);
			return;
		}
		plhs[0] = mxCreateDoubleMatrix(rows, 1, mxREAL);

		double *predictions = mxGetPr(plhs[0]);
		for(int i = 0; i<features.size(); i++)
		{
			predictions[i] = pRegressor->regression(features[i]);			
		}
		
		mxFree(input_regressor_name);
	} 
    
}