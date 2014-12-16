//
//  rf_rgrsn_tree_cost_function.h
//  random forest regression tree cost function
//  RandomForestRegression
//
//  Created by jimmy on 12/9/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__rf_rgrsn_tree_cost_function__
#define __RandomForestRegression__rf_rgrsn_tree_cost_function__

#include <vector>
#include "nl_vector.h"

class rf_rgrsn_tree_node;

using namespace std;

class rf_rgrsn_tree_cost_function
{
protected:
    
public:
    rf_rgrsn_tree_cost_function();
    virtual ~rf_rgrsn_tree_cost_function();
    
    virtual double cost(const vector<nl_vector > & features, const vector<double> & labels,
                        const vector<unsigned> & indices) const = 0;
    virtual rf_rgrsn_tree_node * new_tree_node() const = 0;
};

// model data as a constant value, squared error
class rf_rgrsn_tree_cost_function_constant: public rf_rgrsn_tree_cost_function
{
public:
    rf_rgrsn_tree_cost_function_constant();
    ~rf_rgrsn_tree_cost_function_constant();
    
    virtual double cost(const vector<nl_vector > & features, const vector<double> & labels,
                        const vector<unsigned> & indices) const;
    
    virtual rf_rgrsn_tree_node * new_tree_node() const;
};


#endif /* defined(__RandomForestRegression__rf_rgrsn_tree_cost_function__) */
