//
//  rf_rgrsn_tree_constant.h
//  RandomForestRegression
//
//  Created by jimmy on 12/10/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__rf_rgrsn_tree_constant__
#define __RandomForestRegression__rf_rgrsn_tree_constant__

#include <vector>
#include "nl_vector.h"
#include "rf_rgrsn_tree_node.h"
#include "rf_rgrsn_tree_parameter.h"
#include "rf_rgrsn_tree_cost_function.h"

using namespace::std;

class rf_rgrsn_tree_constant
{
    rf_rgrsn_tree_node_constant * root_;
    unsigned int feature_dim_;
    
    bool verbose_;
    bool verbose_leaf_;
public:
    rf_rgrsn_tree_constant()
    {
        verbose_ = false;
        verbose_leaf_ = false;
        feature_dim_ = 0;
        root_ = 0;
    }
    ~rf_rgrsn_tree_constant()
    {
        
    }
    
    unsigned int n_dim(){
        return feature_dim_;
    }
    
    rf_rgrsn_tree_node_constant * root_node(){
        return root_;
    }
    
    void set_root_node(rf_rgrsn_tree_node_constant * node){
        assert(node);
        root_ = node;
    }
    
    void set_n_dim(unsigned int dim)
    {
        feature_dim_ = dim;
    }
   
   
   // void set_multi_dims(const vcl_vector<unsigned int> & dims){n_multi_dims_ = dims;}
    
    void build(const vector< nl_vector > & features,
               const vector<double> & labels,
               const rf_rgrsn_tree_parameter & para,
               const rf_rgrsn_tree_cost_function_constant & costFunction);
    
    double evaluate(const nl_vector & feature) const;
    
private:
    
    void configure_node(const vector< nl_vector> & features,
                        const vector< double > & labels,
                        const vector< unsigned int > & indices,
                        const unsigned int depth,
                        rf_rgrsn_tree_node_constant * node,
                        const rf_rgrsn_tree_parameter & para,
                        const rf_rgrsn_tree_cost_function_constant & costFunction) const;
    
    double evaluate(const nl_vector & feature, const rf_rgrsn_tree_node_constant * node) const;
};

#endif /* defined(__RandomForestRegression__rf_rgrsn_tree_constant__) */
