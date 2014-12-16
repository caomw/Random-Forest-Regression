//
//  rf_rgrsn_constant_regressor_builder.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/10/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "rf_rgrsn_constant_regressor_builder.h"
#include "rf_rgrsn_constant_regressor.h"


rf_rgrsn_constant_regressor* rf_rgrsn_constant_regressor_builder::new_regressioner() const
{
    return new rf_rgrsn_constant_regressor();
}


bool rf_rgrsn_constant_regressor_builder::build_model(rf_rgrsn_constant_regressor& model,
                 const vector< nl_vector > & features,
                 const vector< double > & labels,
                 const rf_rgrsn_tree_cost_function_constant & costFunction) const
{
    assert(features.size() == labels.size());
    
    vector<rf_rgrsn_tree_constant *> trees;
    
    double training_error = 0;
    double cross_validation_error = 0;
    double cv_ratio = 0;
    for (unsigned int i = 0; i<tree_number_; i++) {
        //randomly select part of the data
        vector< nl_vector > training_inputs;
        vector<double > training_outputs;
        vector< nl_vector> cv_inputs;
        vector<double > cv_outputs;
        
        bagging(features, labels, training_inputs, training_outputs, cv_inputs, cv_outputs);
        
        rf_rgrsn_tree_constant *pTree = new rf_rgrsn_tree_constant();
        assert(pTree);
        rf_rgrsn_tree_parameter parameter;
        parameter.max_tree_depth_  = max_depth_;
        parameter.min_feature_num_ = min_node_size_;
        parameter.min_sample_num_ = min_sample_num_;
        
        pTree->build(training_inputs, training_outputs, parameter, costFunction);
        trees.push_back(pTree);
        
        if(verbose_)
        {
            double training_rms = 0;
            double testing_rms = 0;
            for (int j = 0; j<training_inputs.size(); j++) {
                double val = pTree->evaluate(training_inputs[j]);
                double dif = val - training_outputs[j];
                training_rms += dif * dif;
            }
            training_rms = sqrt(training_rms/training_inputs.size());
            
            for (int j = 0; j<cv_inputs.size(); j++) {
                double val = pTree->evaluate(cv_inputs[j]);
                double dif = val - cv_outputs[j];
                testing_rms += dif * dif;
            }
            testing_rms = sqrt(testing_rms/cv_inputs.size());
            
            printf("finished %d in %d trees, training, cross validation RMS error %f %f\n", i, tree_number_, training_rms, testing_rms);
            
            training_error += training_rms;
            cross_validation_error += testing_rms;
            cv_ratio += 1.0 * cv_inputs.size()/labels.size();
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