//
//  ff_rgrsn_builder.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/15/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "ff_rgrsn_builder.h"
#include "ff_tree_cost_function.h"
#include "ff_rgrsn_regressor.h"
#include "rf_rgrsn_tree_parameter.h"
#include "ff_rgrsn_tree.h"

ff_rgrsn_builder::ff_rgrsn_builder()
{
    tree_number_ = 10;
    max_depth_ = 5;
    min_node_size_ = 12;
	min_sample_num_ = 5;
    verbose_ = true;
}
ff_rgrsn_builder::~ff_rgrsn_builder()
{
    
}

ff_rgrsn_regressor* ff_rgrsn_builder::new_regressioner() const
{
    return new ff_rgrsn_regressor();
}

bool ff_rgrsn_builder::build_model(ff_rgrsn_regressor& model,
                                   const vector< vector< nl_vector > > & inputs,
                                   const vector< double > & outputs,
                                   const ff_tree_cost_function & costFunction) const
{
    assert(inputs.size() == outputs.size());
    //printf("max depth, %u \n", max_depth_);
    
    vector<ff_rgrsn_tree *> trees;
    
    double training_error = 0;
    double cross_validation_error = 0;
    double cv_ratio = 0;
    for (unsigned int i = 0; i<tree_number_; i++) {
        //randomly select part of the data
        vector<vector< nl_vector > > training_inputs;
        vector<double > training_outputs;
        vector<vector< nl_vector > > testing_inputs;
        vector<double > testing_outputs;
        
        bagging(inputs, outputs, training_inputs, training_outputs, testing_inputs, testing_outputs);
        
        ff_rgrsn_tree * pTree = new ff_rgrsn_tree();
        rf_rgrsn_tree_parameter parameter;
        parameter.max_tree_depth_  = max_depth_;
        parameter.min_feature_num_ = min_node_size_;
        
        pTree->build(training_inputs, training_outputs, parameter, costFunction);
        trees.push_back(pTree);
        
        
        double training_rms = 0;
        double testing_rms = 0;
        for (int j = 0; j<training_inputs.size(); j++) {
            double val = pTree->evaluate(training_inputs[j]);
            double dif = val - training_outputs[j];
            training_rms += dif * dif;
        }
        training_rms = sqrt(training_rms/training_inputs.size());
        
        for (int j = 0; j<testing_inputs.size(); j++) {
            double val = pTree->evaluate(testing_inputs[j]);
            double dif = val - testing_outputs[j];
            testing_rms += dif * dif;
        }
        testing_rms = sqrt(testing_rms/testing_inputs.size());
        
        if(verbose_)
        {
            printf("finished %d in %d trees, training, testing RMS error %f %f\n", i, tree_number_, training_rms, testing_rms);
            
            training_error += training_rms;
            cross_validation_error += testing_rms;
            cv_ratio += 1.0 * testing_inputs.size()/inputs.size();
        }
        
    }
    
    model.trees_ = trees;
    
    if(verbose_)
    {
        training_error /= tree_number_;
        cross_validation_error /= tree_number_;
        cv_ratio /= tree_number_;
        printf("average training, cross validation error is %f %f, cross validation ratio is %f.\n", training_error, cross_validation_error, cv_ratio);
    }
    
    return true;
}
