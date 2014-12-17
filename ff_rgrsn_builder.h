//
//  ff_rgrsn_builder.h
//  RandomForestRegression
//  filter forest regressor builder
//  Created by jimmy on 12/15/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__ff_rgrsn_builder__
#define __RandomForestRegression__ff_rgrsn_builder__

#include <vector>
#include <string>
#include "nl_vector.h"
#include "ff_tree_cost_function.h"

using std::vector;
using std::string;

class ff_rgrsn_regressor;


class ff_rgrsn_builder
{
private:
    unsigned int tree_number_;
    unsigned int max_depth_;
    unsigned int min_node_size_;
	unsigned int min_sample_num_;
    
    bool verbose_;
    
public:
    ff_rgrsn_builder();    
    ~ff_rgrsn_builder();
    
    void set_tree_number(unsigned int tree_num) {tree_number_ = tree_num;}
    void set_tree_depth(unsigned int max_depth) {max_depth_ = max_depth;}
    void set_min_node_size(unsigned int min_node_size) {min_node_size_ = min_node_size;}
	void set_max_sample_num(unsigned int num) { min_sample_num_ = num;}
    
    //: Create empty model
    ff_rgrsn_regressor* new_regressioner() const;
    
    //: Build model from data
    bool build_model(ff_rgrsn_regressor& model,
                     const vector< vector< nl_vector > > & inputs,
                     const vector< double > & outputs,
                     const ff_tree_cost_function & costFunction) const;
    
    string is_a() const {return string("ff_rgrsn_builder");}
};


#endif /* defined(__RandomForestRegression__ff_rgrsn_builder__) */
