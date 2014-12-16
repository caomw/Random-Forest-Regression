//
//  ff_rgrsn_tree.h
//  RandomForestRegression
//  filter forest regression
//  Created by jimmy on 12/15/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__ff_rgrsn_tree__
#define __RandomForestRegression__ff_rgrsn_tree__

#include "nl_vector.h"
#include "nl_matrix.h"
#include "rf_rgrsn_tree_parameter.h"
#include "ff_tree_cost_function.h"
#include <vector>

using std::vector;
class ff_rgrsn_tree_node;

class ff_rgrsn_tree
{
    ff_rgrsn_tree_node * root_;
    vector< unsigned int> n_multi_dims_;
    
    bool verbose_;
    bool verbose_leaf_;
public:
    ff_rgrsn_tree()
    {
        verbose_ = false;
        verbose_leaf_ = false;
    }
    ~ff_rgrsn_tree()
    {
        
    }
    
    vector<unsigned int> n_multi_dims(){return  n_multi_dims_;}
    
    ff_rgrsn_tree_node * root_node(){return root_;}
    void set_root_node(ff_rgrsn_tree_node *node){root_ = node;}
    void set_multi_dims(const vector<unsigned int> & dims){n_multi_dims_ = dims;}
    
    // multiple scale feature
    // [id][scale_id][feat_dim]
    void build(const vector< vector<nl_vector > > & multi_scale_features,
               const vector<double> & labels,
               const rf_rgrsn_tree_parameter & para,
               const ff_tree_cost_function & costFunction);
    
    double evaluate(const vector< nl_vector > & multi_scale_feature) const;
    
private:
    void configure_node(const vector< vector<nl_vector > > & multi_scale_features,
                        const vector< double >& labels,
                        const vector< unsigned int >& indices,
                        const unsigned int depth,
                        ff_rgrsn_tree_node * node,
                        const rf_rgrsn_tree_parameter & para,
                        const ff_tree_cost_function & costFunction) const;

    
    double evaluate(const vector< nl_vector > & multi_scale_feature, ff_rgrsn_tree_node * node) const;
};



#endif /* defined(__RandomForestRegression__ff_rgrsn_tree__) */
