//
//  rf_rgrsn_constant_regressor_builder.h
//  RandomForestRegression
//  random forest regression constant (leaf node) regressor builder
//  Created by jimmy on 12/10/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__rf_rgrsn_constant_regressor_builder__
#define __RandomForestRegression__rf_rgrsn_constant_regressor_builder__

#include <vector>
#include <string>
#include "nl_vector.h"
#include "rf_rgrsn_tree_cost_function.h"


using namespace std;

class rf_rgrsn_constant_regressor;

class rf_rgrsn_constant_regressor_builder
{
private:
    unsigned int tree_number_;
    unsigned int max_depth_;
    unsigned int min_node_size_;
    unsigned int min_sample_num_;
    
    bool verbose_;
    
public:
    rf_rgrsn_constant_regressor_builder(){verbose_ = true;}
    ~rf_rgrsn_constant_regressor_builder(){;}
    
    void set_tree_number(unsigned int tree_num) {tree_number_ = tree_num;}
    void set_tree_depth(unsigned int max_depth) {max_depth_ = max_depth;}
    void set_min_node_size(unsigned int min_node_size) {min_node_size_ = min_node_size;}
    void set_min_sample_num(unsigned int num){min_sample_num_ = num;}
    
    
    //: Create empty model
    rf_rgrsn_constant_regressor* new_regressioner() const;
    
    //: Build model from data
    bool build_model(rf_rgrsn_constant_regressor& model,
                     const vector< nl_vector > & features,
                     const vector< double > & labels,
                     const rf_rgrsn_tree_cost_function_constant & costFunction) const;
    
    string is_a() const {return string("rf_rgrsn_constant_regressor_builder");}
    
    
protected:
    
    
};


#endif /* defined(__RandomForestRegression__rf_rgrsn_constant_regressor_builder__) */
