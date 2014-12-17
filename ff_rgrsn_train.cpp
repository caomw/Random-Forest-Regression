#include "ff_rgrsn_regressor.h"
#include "ff_rgrsn_builder.h"
#include "mex.h"


#if 0
static bool read_sin_noise_3(vector<double> & gds, vector<double> & noisy_gds, const char *file)
{
    assert(file);
    FILE *pf = fopen(file, "r");
    if (!pf) {
        printf("can not open %s\n", file);
        return false;
    }
    while (1) {
        double gd = 0;
        double noisy_gd = 0;
        int ret = fscanf(pf, "%lf %lf ", &gd, &noisy_gd);
        if (ret != 2) {
            break;
        }
        gds.push_back(gd);
        noisy_gds.push_back(noisy_gd);
    }
    fclose(pf);
    assert(gds.size() == noisy_gds.size());
    printf("load %d data\n", (int)gds.size());
    return true;
}

static void generateMultiScaleFeature(const vector<double> & data,
                               const vector<double> & groundTruth,
                               const vector<unsigned int> & windowSize,
                               const int step,
                               vector< vector<nl_vector > > & features,
                               vector<double> & labels)
{
    assert(step >= 1);
    assert(data.size() == groundTruth.size());
    
    const int beingInd = windowSize.back();
    for (int i = beingInd; i<data.size(); i++) {
        vector<nl_vector > multi_scale_feature;
        // loop all windowSize
        for (int j = 0; j<windowSize.size(); j++) {
            const int feat_length = windowSize[j]/step;
            assert(feat_length > 0);
            nl_vector feat(feat_length, 0);
            for (int k = 0; k<feat_length; k += 1) {
                feat[k] = data[i - k * step];
            }
            multi_scale_feature.push_back(feat);
        }
        features.push_back(multi_scale_feature);
        labels.push_back(groundTruth[i]);
    }
    
    assert(features.size() == labels.size());
}

// Interface function of matlab
// 0: features
// 1: labels
// 2: windowsize
// 3: parameters
// 4: save_file
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
	if(nrhs != 5)
	{
		mexPrintf("Error: Filter Forests needs 5 parameter in order. Features, Labels, WindowSize, training parameters, save file name\n");
		return;
	}
    
	if(nrhs == 5)
	{		
		//get training data dimension and point
		unsigned int rows = mxGetM(prhs[0]);  //feature rows
		unsigned int cols = mxGetN(prhs[0]);  //feature columns;
		double *pFeature = mxGetPr(prhs[0]);		
		double *pLabel      = mxGetPr(prhs[1]);
		double *pWindowsize = mxGetPr(prhs[2]);
		double *pParameter  = mxGetPr(prhs[3]);
		unsigned int windowSizeRows = mxGetM(prhs[2]);
		unsigned int windowSizeCols = mxGetN(prhs[2]);
		unsigned int parameterRows = mxGetM(prhs[3]);
		unsigned int parameterCols = mxGetN(prhs[3]);

		// get save file		
		int buflen = 1024;
		char save_name[1024] = {NULL};		
		mxGetString(prhs[4], save_name, mxGetN(prhs[4]) * mxGetM(prhs[4]) + 1);
		mexPrintf("save file is %s\n", save_name);

		if(mxIsDouble(prhs[0]) == 0 || mxIsDouble(prhs[1]) == 0)
		{
			mexPrintf("Error: training data type must be double.\n");
			return;
		}

		// read input feature as a long vector
		vector<nl_vector> long_features;
		vector<double> labels;	
		nl_vector feature(cols, 0);

		// mex use column major to store data
		for(int i = 0; i<rows; i++)
		{						 
			for(int j = 0; j<cols; j++)
			{
				feature[j] = pFeature[rows * j + i];
			}		
			long_features.push_back(feature);		
			labels.push_back(pLabel[i]);			
		}
		assert(long_features.size() == labels.size());
		mexPrintf("training data size is %lu\n", labels.size());

		// LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsUint32(const mxArray *pa);
		if(windowSizeCols != 1)
		{
			mexPrintf("Error: window size must be a vector. Column number must be 1 \n");
			return;
		}

		if(mxIsDouble(prhs[2]) == 0)
		{
			mexPrintf("Error: window size must be double. \n");
			return;
		}
		
		vector<int> segments;
		int total_length = 0; 
		for(unsigned i = 0; i<windowSizeRows; i++)
		{
			int size = (int)pWindowsize[i];
			segments.push_back(size);			
			total_length += segments[i];
			mexPrintf("window size is %d\n", size);
		}	
		
		if(total_length != cols)
		{
			mexPrintf("Error: multiple feature dimension error. Please check feature dimensions.\n");
			return;
		}

		// divide long feature into multiple short features
		vector<vector<nl_vector > > multi_scale_features;
		for(unsigned i = 0; i<long_features.size(); i++)
		{
			int startIndex = 0;
			vector<nl_vector> multi_feat;
			for(unsigned j = 0; j<segments.size(); j++)
			{
				nl_vector feat(segments[j], 0);
				for(unsigned k = 0; k < feat.size(); k++)
				{
					feat[k] = long_features[i][startIndex+k];
				}
				multi_feat.push_back(feat);
				startIndex += segments[j];
			}
			multi_scale_features.push_back(multi_feat);
		}
				
		// set parameters
		if(parameterRows != 4 || parameterCols != 1 || mxIsDouble(prhs[3]) == 0)
		{
			mexPrintf("Error: parameter number must be 4.\n");
			mexPrintf("Error: parameter must be in a vector.\n");
			mexPrintf("Error: parameter must be double.\n");
			return;
		}
		mexPrintf("Parameter must in an order: min_node_size, max_tree_depth, tree_num, max_sample_numer\n");

		int min_node_size = (int) pParameter[0];
		int tree_depth = (int) pParameter[1];
		int tree_num = (int) pParameter[2];
		int max_sample_num = (int) pParameter[3];
		// training
		ff_rgrsn_builder builder;
		ff_rgrsn_regressor *pRegressor = builder.new_regressioner();
    
		ff_tree_cost_function costFunction;
		builder.set_min_node_size(min_node_size);
		builder.set_tree_depth(tree_depth);
		builder.set_tree_number(tree_num);
		builder.set_max_sample_num(max_sample_num);
    
		builder.build_model(*pRegressor, multi_scale_features, labels, costFunction);
    
		// testing
		vector<double> testResult;
		double rms = 0;
		for (int i = 0; i<multi_scale_features.size(); i++) {
			double val = pRegressor->regression(multi_scale_features[i]);
			testResult.push_back(val);
			double dif = val - labels[i];
			rms += dif * dif;
		}
		rms = sqrt(rms/labels.size());
		mexPrintf("traing rms error is %f\n", rms);

		pRegressor->write(save_name);
		mexPrintf("save to file %s\n", save_name);

		if(nlhs == 1)
		{
			double *ptr;
			plhs[0] = mxCreateString(save_name);			
		}
	}   
}

#endif