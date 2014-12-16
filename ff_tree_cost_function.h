//
//  ff_tree_cost_function.h
//  RandomForestRegression
//  filter forest tree cost function
//  Created by jimmy on 12/14/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__ff_tree_cost_function__
#define __RandomForestRegression__ff_tree_cost_function__

#include <vector>
#include "nl_vector.h"

using namespace std;

class ff_rgrsn_tree_node;

// Data Dependent Regularized training, Regularized Least Squares
// "Filter Forests for learning data-dependent convolutional kernels" CVPR2014
// regulation term is determined by data
// The feature must has the same meaning of label
class ff_tree_cost_function
{
public:
    ff_tree_cost_function();
    ~ff_tree_cost_function();   
    
    double cost(const vector<vector<nl_vector > > & multi_scale_features,
                const vector<double> & labels,
                const vector<unsigned int> & feature_index,
                const int scale_index,
                nl_vector & wt) const;
    
    ff_rgrsn_tree_node * new_tree_node() const;
    
    
};


#endif /* defined(__RandomForestRegression__ff_tree_cost_function__) */
