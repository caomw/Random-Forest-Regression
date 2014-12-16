#include "rf_rgrsn_constant_regressor.h"
#include "rf_rgrsn_constant_regressor_builder.h"
#include "mex.h"


#if 1
// this file is modified from "svmtrain.c" in libsvm
// Interface function of matlab
// now assume prhs[0]: features prhs[1]: labels
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
	const char *error_msg;

	// fix random seed to have same results for each run
	// (for cross validation and probability estimation)
	srand(1);

	if(nlhs > 1)
	{
	//	exit_with_help();
	//	fake_answer(nlhs, plhs);
		return;
	}
    
	if(nrhs == 2)
	{		
		//get training data dimension and point
		unsigned int rows = mxGetM(prhs[0]);
		unsigned int cols = mxGetN(prhs[0]);
		double *pFeature = mxGetPr(prhs[0]);		
		double *pLabel = mxGetPr(prhs[1]);

		mexPrintf("rows, cols is %u %u\n", rows, cols); 

		if(mxIsDouble(prhs[0]) == 0 || mxIsDouble(prhs[0]) == 0)
		{
			mexPrintf("Error: training data type must be double.\n");
			return;
		}		

		vector<nl_vector> features;
		vector<double> labels;
	//	mexPrintf("%d \n", 1);
		nl_vector feature(cols, 0);
	//	mexPrintf("%d \n", 2);
		// mex use column major to store data
		for(int i = 0; i<rows; i++)
		{
			//int idx = i * cols;			 
			for(int j = 0; j<cols; j++)
			{
				feature[j] = pFeature[rows * j + i];
			}
		//	mexPrintf("%d \n", 3);
			features.push_back(feature);
		//	mexPrintf("%d \n", 4);
			labels.push_back(pLabel[i]);			
		}
		assert(features.size() == labels.size());
		mexPrintf("training data size is %lu\n", labels.size());

		rf_rgrsn_constant_regressor_builder builder;

		unsigned int tree_num = 10;
		unsigned int max_depth = 6;
		unsigned int min_node_size = 6;
		unsigned int min_sample_num = 5;		

		rf_rgrsn_tree_cost_function_constant costFunction;
		rf_rgrsn_constant_regressor *pRegressor = builder.new_regressioner();

		builder.set_tree_number(tree_num);
		builder.set_tree_depth(max_depth);
		builder.set_min_node_size(min_node_size);
		builder.set_min_sample_num(min_sample_num);

		builder.build_model(*pRegressor, features, labels, costFunction);

		mexPrintf("training finished\n");

		string save_file("temp_regressor_.txt");
		pRegressor->write(save_file.c_str());

		// output result
		if(nlhs == 1)
		{
			double *ptr;
			plhs[0] = mxCreateString(save_file.c_str());			
		}

		//test by training data
		pRegressor = builder.new_regressioner();
		pRegressor->read(save_file.c_str());
		double rms = 0.0;
		for(int i = 0; i<features.size(); i++)
		{
			double dif = pRegressor->regression(features[i]) - labels[i];
			rms += dif * dif;
		}
		rms = sqrt(rms/labels.size());
		mexPrintf("training RMS error is %f\n", rms);

	}   
}

#endif