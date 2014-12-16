//
//  rf_rgrsn_tree_cost_function.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/9/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "rf_rgrsn_tree_cost_function.h"
#include "rf_rgrsn_tree_node.h"

rf_rgrsn_tree_cost_function::rf_rgrsn_tree_cost_function()
{
    
}

rf_rgrsn_tree_cost_function::~rf_rgrsn_tree_cost_function()
{
    
}

/*************       rf_rgrsn_tree_cost_function        ******************/

rf_rgrsn_tree_cost_function_constant::rf_rgrsn_tree_cost_function_constant()
{
    
}
rf_rgrsn_tree_cost_function_constant::~rf_rgrsn_tree_cost_function_constant()
{
    
}

double rf_rgrsn_tree_cost_function_constant::cost(const vector<nl_vector > & features, const vector<double> & labels,
                                                  const vector<unsigned> & indices) const
{
    assert(indices.size() > 0);
    
    double mean = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        mean += labels[indices[i]];
    }
    mean /= indices.size();
    
    double cost = 0.0;
    for (int i = 0; i<indices.size(); i++) {
        cost += (labels[indices[i]] - mean) * (labels[indices[i]] - mean);
    }
    return cost;
}

rf_rgrsn_tree_node * rf_rgrsn_tree_cost_function_constant::new_tree_node() const
{
    return new rf_rgrsn_tree_node_constant();
}