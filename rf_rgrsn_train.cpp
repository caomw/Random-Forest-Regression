#include "rf_rgrsn_constant_regressor.h"
#include "rf_rgrsn_constant_regressor_builder.h"
#include "mex.h"


#if 1
// this file is modified from "svmtrain.c" in libsvm
// Interface function of matlab
// 0: features
// 1: labels
// 2: parameters, size 4
// 3: save_file
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
	const char *error_msg;

	// fix random seed to have same results for each run
	// (for cross validation and probability estimation)
	srand(1);

	if(nlhs > 1)
	{
		mexPrintf("Error: only return a result. It is the save file name.\n");	
		return;	
	}

	if(nrhs != 4)
	{
		mexPrintf("Error: Random Forests needs 4 parameter in order. Features, Labels, training parameters, save file name\n");
		return;
	}
    
	if(nrhs == 4)
	{		
		//get training data dimension and point
		unsigned int rows = mxGetM(prhs[0]);
		unsigned int cols = mxGetN(prhs[0]);
		double *pFeature = mxGetPr(prhs[0]);		
		double *pLabel = mxGetPr(prhs[1]);
		double *pParameter = mxGetPr(prhs[2]);
		unsigned int parameterRows = mxGetM(prhs[2]);
		unsigned int parameterCols = mxGetN(prhs[2]);

		// training data
		mexPrintf("rows, cols is %u %u\n", rows, cols);
		if(mxIsDouble(prhs[0]) == 0 || mxIsDouble(prhs[0]) == 0)
		{
			mexPrintf("Error: training data type must be double.\n");
			return;
		}

		// training parameters
		if(parameterRows != 4)
		{
			mexPrintf("Error: parameter row must be 4.\n");			
			return;
		}
		if(parameterCols != 1)
		{
			mexPrintf("Error: parameter column must be 4.\n");
			return;
		}
		if(mxIsDouble(prhs[2]) == 0)
		{
			mexPrintf("Error: parameter must be double.\n");
			return;
		}

		mexPrintf("Parameter must in an order: min_node_size, max_tree_depth, tree_num, max_sample_numer\n");

		int min_node_size = (int) pParameter[0];
		int tree_depth = (int) pParameter[1];
		int tree_num = (int) pParameter[2];
		int max_sample_num = (int) pParameter[3];
		mexPrintf("min_node_size, tree_depth, tree_num, max_sample_num: %d %d %d %d\n", min_node_size, tree_depth, tree_num, max_sample_num);

		// save file name
		
		char *save_name = mxArrayToString(prhs[3]);		
		mexPrintf("save file is %s\n", save_name);

		vector<nl_vector> features;
		vector<double> labels;	
		nl_vector feature(cols, 0);	
		// mex use column major to store data
		for(int i = 0; i<rows; i++)
		{						 
			for(int j = 0; j<cols; j++)
			{
				feature[j] = pFeature[rows * j + i];
			}		
			features.push_back(feature);		
			labels.push_back(pLabel[i]);			
		}
		assert(features.size() == labels.size());
		mexPrintf("training data size is %lu\n", labels.size());

		rf_rgrsn_constant_regressor_builder builder;

		rf_rgrsn_tree_cost_function_constant costFunction;
		rf_rgrsn_constant_regressor *pRegressor = builder.new_regressioner();

		builder.set_tree_number(tree_num);
		builder.set_tree_depth(tree_depth);
		builder.set_min_node_size(min_node_size);
		builder.set_max_sample_num(max_sample_num);

		builder.build_model(*pRegressor, features, labels, costFunction);

		mexPrintf("training finished\n");

		
		pRegressor->write(save_name);

		// output result
		if(nlhs == 1)
		{
			double *ptr;
			plhs[0] = mxCreateString(save_name);			
		}

		//test by training data
		pRegressor = builder.new_regressioner();
		pRegressor->read(save_name);
		double rms = 0.0;
		for(int i = 0; i<features.size(); i++)
		{
			double dif = pRegressor->regression(features[i]) - labels[i];
			rms += dif * dif;
		}
		rms = sqrt(rms/labels.size());
		mexPrintf("training RMS error is %f\n", rms);

		mxFree(save_name);

	}   
}

#endif