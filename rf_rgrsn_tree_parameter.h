//
//  rf_rgrsn_tree_parameter.h
//  RandomForestRegression
//
//  Created by jimmy on 12/10/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef RandomForestRegression_rf_rgrsn_tree_parameter_h
#define RandomForestRegression_rf_rgrsn_tree_parameter_h

#include <vector>
#include <algorithm>
#include <random>
#include "nl_vector.h"

using namespace std;

struct rf_rgrsn_tree_parameter{
    unsigned int max_tree_depth_;
    unsigned int min_feature_num_;
    unsigned int min_sample_num_;    // minimum sample number in spliting
    
    rf_rgrsn_tree_parameter()
    {
        max_tree_depth_  = 8;
        min_feature_num_ = 8;
        min_sample_num_ = 5;
    }
};

inline bool isEqual(double v1, double v2)
{
    const double sqrteps          = 1.490116119384766e-08;
    return fabs(v1 - v2) < sqrteps;
}

inline void random_sample(vector<double> & data, int dimNum)
{
    assert(dimNum > 0);
    double val_min = *min_element(data.begin(), data.end());
    double val_max = *max_element(data.begin(), data.end());
    
    data.clear();
    if (isEqual(val_min, val_max)) {
        return;
    }
    
    default_random_engine generator;
    uniform_real_distribution<double> distribution(val_min, val_max);
    
    for (int i = 0; i<dimNum; i++) {
        double val = distribution(generator);
        data.push_back(val);
    }
}


inline void bagging(const vector<nl_vector > & inputs,
                    const vector<double> & outputs,
                    vector<nl_vector> & bootstrapped_inputs,
                    vector<double> & bootstrapped_outputs,
                    vector<nl_vector> & out_of_bag_inputs,
                    vector<double> & out_of_bag_outputs )
{
    assert(inputs.size() == outputs.size());
    const int N = (int)outputs.size();
    
    // randomly pick a number in [0 N) N times, as trainning data
    // the data which is not picked as test data
    vector<bool> isPicked(N, false);
    for (int i = 0; i<inputs.size(); i++) {
        int rnd = rand()%N;
        bootstrapped_inputs.push_back(inputs[rnd]);
        bootstrapped_outputs.push_back(outputs[rnd]);
        isPicked[rnd] = true;
    }
    
    for (int i = 0; i<N; i++) {
        if (!isPicked[i]) {
            out_of_bag_inputs.push_back(inputs[i]);
            out_of_bag_outputs.push_back(outputs[i]);
        }
    }
    assert(bootstrapped_inputs.size() == bootstrapped_outputs.size());
    assert(out_of_bag_inputs.size() == out_of_bag_outputs.size());
}


inline void bagging(const vector<vector<nl_vector> > & inputs,
                    const vector<double> & outputs,
                    vector<vector<nl_vector>> & bootstrapped_inputs,
                    vector<double> & bootstrapped_outputs,
                    vector<vector<nl_vector>> & out_of_bag_inputs,
                    vector<double> & out_of_bag_outputs )
{
    assert(inputs.size() == outputs.size());
    const int N = (int)outputs.size();
    
    // randomly pick a number in [0 N) N times, as trainning data
    // the data which is not picked as test data
    vector<bool> isPicked(N, false);
    for (int i = 0; i<inputs.size(); i++) {
        int rnd = rand()%N;
        bootstrapped_inputs.push_back(inputs[rnd]);
        bootstrapped_outputs.push_back(outputs[rnd]);
        isPicked[rnd] = true;
    }
    
    for (int i = 0; i<N; i++) {
        if (!isPicked[i]) {
            out_of_bag_inputs.push_back(inputs[i]);
            out_of_bag_outputs.push_back(outputs[i]);
        }
    }
    assert(bootstrapped_inputs.size() == bootstrapped_outputs.size());
    assert(out_of_bag_inputs.size() == out_of_bag_outputs.size());
}






#endif
