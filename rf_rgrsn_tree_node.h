//
//  rf_rgrsn_tree_node.h
//  random forest regression tree node.
//  RandomForestRegression
//
//  Created by jimmy on 12/9/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__rf_rgrsn_tree_node__
#define __RandomForestRegression__rf_rgrsn_tree_node__

#include <vector>
#include "nl_vector.h"

using namespace std;

class rf_rgrsn_tree_node
{
public:
    rf_rgrsn_tree_node *left_child_;
    rf_rgrsn_tree_node *right_child_;
    unsigned int depth_;
    bool isLeaf_;
    
    unsigned int split_dim_;  //split dimenstion
    double split_threshold_;  // < threshold goto left child, otherwise right child
    
    rf_rgrsn_tree_node()
    {
        left_child_ = NULL;
        right_child_ = NULL;
        depth_ = 0;
        isLeaf_ = false;
        split_dim_ = 0;
        split_threshold_ = 0;
    }
    virtual ~rf_rgrsn_tree_node(){;}
    
    
    virtual bool set_leaf_parameter(const vector<nl_vector> & features, const vector<double> & labels,
                                    const vector<unsigned> & indices) = 0;
    // regression result
    virtual double value(const nl_vector &x) const = 0;
};


class rf_rgrsn_tree_node_constant: public rf_rgrsn_tree_node
{
    double mean_;
public:
    rf_rgrsn_tree_node_constant():rf_rgrsn_tree_node()
    {
        mean_ = 0;
    }
    virtual bool set_leaf_parameter(const vector<nl_vector> & features, const vector<double> & labels,
                                    const vector<unsigned> & indices);
    virtual double value(const nl_vector &x) const;
    
    // read/write tree to file,
    static bool write_tree(const char *fileName, rf_rgrsn_tree_node_constant * root);
    static rf_rgrsn_tree_node_constant * read_tree(const char *fileName);     
private:
  
    static void write_constant_prediction(FILE *pf, rf_rgrsn_tree_node_constant * node);
    static rf_rgrsn_tree_node_constant * read_constant_prediction(FILE *pf);
    
};





#endif /* defined(__RandomForestRegression__rf_rgrsn_tree_node__) */
