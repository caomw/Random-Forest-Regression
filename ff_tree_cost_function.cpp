//
//  ff_tree_cost_function.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/14/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "ff_tree_cost_function.h"
#include "nl_matrix.h"
#include "lp_least_square.h"
#include "ff_rgrsn_tree_node.h"


/********************** ff_tree_cost_funciton   *************************/

ff_tree_cost_function::ff_tree_cost_function()
{
    
}

ff_tree_cost_function::~ff_tree_cost_function()
{
    
}


double ff_tree_cost_function::cost(const vector<vector<nl_vector> > & multi_scale_features,
                                   const vector<double> & labels,
                                   const vector<unsigned int> & feature_index,
                                   const int scale_index,
                                   nl_vector & wt) const
{
    assert(feature_index.size() > 0);
    assert(scale_index < multi_scale_features[0][scale_index].size());
    
    
    const int N = (int)feature_index.size(); // number of features
    const int D = (int)multi_scale_features[0][scale_index].size(); // number of dimension
    if (D > N) {
        printf("Warning: feature number < dimension %d %d!\n", N, D);
        return INT_MAX;
    }
    
    
    // loop each feature
    nl_matrix X(N, D, 0.0);
    for (int idx = 0; idx < feature_index.size(); idx++) {
        int fid = feature_index[idx];
        for (int j = 0; j<D; j++) {
            X(idx, j) = multi_scale_features[fid][scale_index][j];
        }
    }
    
    
    nl_vector Y(N, 0.0);
    for (int idx = 0; idx<feature_index.size(); idx++) {
        int fid = feature_index[idx];
        Y[idx] = labels[fid];
    }
    
    // regulation term
    // the weight of first term (bias term) is set to zero
    nl_matrix gamma = nl_matrix(D, D, 0);
    for (int i = 0; i<D; i++) {
        double dif_mean = 0;
        // loop each feature
        for (int idx = 0; idx<feature_index.size(); idx++) {
            int fid = feature_index[idx];
            double dif = multi_scale_features[fid][scale_index][i] - labels[fid];
            dif_mean += dif * dif;
        }
        dif_mean /= N;
        gamma(i, i) = dif_mean;
    }
    
    bool isSolved = lp_least_square::RLS(X, Y, gamma, wt);
    if (!isSolved) {
        printf("least square solver errror.\n");
        return INT_MAX;
    }
    
    // calculate residual
    nl_matrix Xwt = X * wt;
    assert(Xwt.cols() == 1);
    double residual = 0.0;
    for (int i = 0; i<Y.size(); i++) {
        residual += (Xwt(i, 0) - Y[i]) * (Xwt(i, 0) - Y[i]);
    }
    return residual;
}

ff_rgrsn_tree_node * ff_tree_cost_function::new_tree_node() const
{
    return new ff_rgrsn_tree_node();
}


