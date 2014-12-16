//
//  ff_rgrsn_tree_node.h
//  RandomForestRegression
//
//  Created by jimmy on 12/15/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__ff_rgrsn_tree_node__
#define __RandomForestRegression__ff_rgrsn_tree_node__

#include "nl_vector.h"
#include <vector>

using namespace std;

class ff_rgrsn_tree_node
{
public:
    ff_rgrsn_tree_node *left_child_;
    ff_rgrsn_tree_node *right_child_;
    unsigned int depth_;
    bool isLeaf_;
    
    unsigned int split_dim_;  //split dimenstion
    double split_threshold_;  // < threshold goto left child, otherwise right child
    
    // leaf parameter
    unsigned int split_scale_index_; // multip-scale feature
    nl_vector wt_;                   // wt * [x1, x2, ... xn]
    
    ff_rgrsn_tree_node()
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_  = 0;
        isLeaf_ = false;
        
        split_scale_index_ = 0;
        split_dim_ = 0;
        split_threshold_ = 0;
    }
    
    // read/write tree to file,
    static bool write_FF_tree(const char *fileName, ff_rgrsn_tree_node * root);
    static bool read_FF_tree(const char *fileName, ff_rgrsn_tree_node * & root);
    
    
public:
    
    double value(const vector< nl_vector > & multi_scale_feature) const;
};


#endif /* defined(__RandomForestRegression__ff_rgrsn_tree_node__) */
