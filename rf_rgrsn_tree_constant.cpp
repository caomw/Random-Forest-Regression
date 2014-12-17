//
//  rf_rgrsn_tree_constant.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/10/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "rf_rgrsn_tree_constant.h"
#include <math.h>

void rf_rgrsn_tree_constant::build(const vector< nl_vector > & features,
                                   const vector<double> & labels,
                                   const rf_rgrsn_tree_parameter & para,
                                   const rf_rgrsn_tree_cost_function_constant & costFunction)
{
    assert(features.size() == labels.size());
    assert(features.size() > 0);
    
    feature_dim_ = features.front().size();
    root_ = dynamic_cast<rf_rgrsn_tree_node_constant *> (costFunction.new_tree_node());
    assert(root_);
    root_->depth_ = 0;
    
    vector<unsigned int> indices;
    for (int i = 0; i<labels.size(); i++) {
        indices.push_back(i);
    }
    random_shuffle(indices.begin(), indices.end());
    
    this->configure_node(features, labels, indices, 0, root_, para, costFunction);
}

double rf_rgrsn_tree_constant::evaluate(const nl_vector & feature) const
{
    assert(feature.size() == feature_dim_);
    
    return evaluate(feature, root_);
}

double rf_rgrsn_tree_constant::evaluate(const nl_vector & feature, const rf_rgrsn_tree_node_constant * node) const
{
    assert(node);
    if (node->isLeaf_) {
        return node->value(feature);
    }
    
    double threshold = node->split_threshold_;
    double val = feature[node->split_dim_];
    if (node->left_child_ && val <= threshold) {
        return this->evaluate(feature, dynamic_cast<rf_rgrsn_tree_node_constant *>(node->left_child_));
    }
    else if (node->right_child_ && val > threshold)
    {
        return this->evaluate(feature, dynamic_cast<rf_rgrsn_tree_node_constant *>(node->right_child_));
    }
    else
    {
        printf("Error: can not find corresponding leaf node\n");
        return INT_MAX;
    }
}

static bool bestSplittingInOneDimenstion(const vector< nl_vector >& features,
                                         const vector< double >& labels,
                                         const vector< unsigned int >& indices,
                                         const unsigned int nDim,
                                         
                                         const rf_rgrsn_tree_parameter & para,
                                         const rf_rgrsn_tree_cost_function_constant & costFunction,
                                         
                                         double & loss,
                                         double & threshold,
                                         vector< unsigned int > & leftIndices,
                                         vector< unsigned int > & rightIndices)
{
    assert(indices.size() >= para.min_feature_num_);
    
    //collect data in nDim of inputs
    vector<double > data_x_nDim;
    for (int i = 0; i<indices.size(); i++) {
        data_x_nDim.push_back(features[indices[i]][nDim]);
    }
   
    random_sample(data_x_nDim, std::min((unsigned int)features.front().size(), para.max_sample_num_));
    loss = INT_MAX;
    
    bool isDivided =  false;
    for (int i = 0; i < data_x_nDim.size(); i++) {
        vector<unsigned int> curLeftIndices;
        vector<unsigned int> curRightIndices;
        
        //try each possible threshold
        double curThreshold = data_x_nDim[i];
        
        // loop current data (in one dimension)
        for (int j = 0; j<indices.size(); j++) {
            int idx = indices[j];
            if (features[idx][nDim] < curThreshold) {
                curLeftIndices.push_back(idx);
            }
            else
            {
                curRightIndices.push_back(idx);
            }
        }
        if (curLeftIndices.size() * 2 < para.min_feature_num_ || curRightIndices.size() * 2 < para.min_feature_num_) {
            continue;
        }
        
        double left_cost = 0;
        if (curLeftIndices.size() > 0) {
            left_cost = costFunction.cost(features, labels, curLeftIndices);
        }
        
        double right_cost = 0;
        if (curRightIndices.size() > 0) {
            right_cost = costFunction.cost(features, labels, curRightIndices);
        }
        
        if (left_cost + right_cost < loss) {
            loss = left_cost + right_cost;
            threshold = curThreshold;
            leftIndices  = curLeftIndices;
            rightIndices = curRightIndices;
            isDivided = true;
        }
    }
    return isDivided;
}


static bool bestSplittingDecision(const vector< nl_vector >& features,
                                  const vector< double >& labels,
                                  const vector< unsigned int > & indices,
                                  const vector< unsigned int > & dimensions,
                                  
                                  const rf_rgrsn_tree_parameter & para,
                                  const rf_rgrsn_tree_cost_function_constant & costFunction,
                                  
                                  unsigned int & splitDim,
                                  double & splitThreshold,
                                  vector< unsigned int > & leftIndices,
                                  vector< unsigned int > & rightIndices)
{
    //loop each dimenstion
    double minLoss = INT_MAX;
    bool isSplitOk = false;
    for (int i = 0; i<dimensions.size(); i++) {
        vector<unsigned int> curLeftIndices;
        vector<unsigned int> curRightIndices;
        double curLoss = INT_MAX;
        double curThreshold = 0.0;
       
        bool isSplit = bestSplittingInOneDimenstion(features, labels, indices, dimensions[i], para, costFunction,
                                                    curLoss, curThreshold, curLeftIndices, curRightIndices);
        
        if (isSplit && curLoss < minLoss)
        {
            minLoss = curLoss;
            leftIndices  = curLeftIndices;
            rightIndices = curRightIndices;
            splitDim = dimensions[i];
            splitThreshold = curThreshold;
            isSplitOk = true;
        }
    }
    return isSplitOk;
}


void rf_rgrsn_tree_constant::configure_node(const vector< nl_vector> & features,
                    const vector< double > & labels,
                    const vector< unsigned int > & indices,
                    const unsigned int depth,
                    rf_rgrsn_tree_node_constant * node,
                    const rf_rgrsn_tree_parameter & para,
                    const rf_rgrsn_tree_cost_function_constant & costFunction) const
{
    assert(node);
    
    if (depth > para.max_tree_depth_ || indices.size() < para.min_feature_num_) {
        // leaf node
        node->isLeaf_ = true;
        node->set_leaf_parameter(features, labels, indices);
        return;
    }
    
    // randomly generate dimension subset that used in split
    vector< unsigned int > randomDimensions( feature_dim_ );
    for ( unsigned int i = 0; i < feature_dim_; ++i )
    {
        randomDimensions[i] = i;
    }
    random_shuffle( randomDimensions.begin(), randomDimensions.end());
    
    unsigned int subDims = sqrt((double)feature_dim_);
    assert( subDims > 0 );
    assert( subDims <= feature_dim_ );
    vector< unsigned int > randomSubset( randomDimensions.begin(), randomDimensions.begin() + subDims );
    
    // split currnt node into left and right node
    vector< unsigned int > leftIndices;
    vector< unsigned int > rightIndices;
    unsigned int splitDim = 0;
    double splitThreshold = 0.0;
    bool canSplit = bestSplittingDecision(features, labels, indices, randomSubset, para,
                                          costFunction,
                                          splitDim, splitThreshold, leftIndices, rightIndices);
    
    if (canSplit) {
        assert(leftIndices.size() + rightIndices.size() == indices.size());
        node->split_dim_ = splitDim;
        node->split_threshold_ = splitThreshold;       
        
        if (leftIndices.size() != 0) {
            rf_rgrsn_tree_node_constant *leftNode = dynamic_cast<rf_rgrsn_tree_node_constant *>(costFunction.new_tree_node());
            assert(leftNode);
            leftNode->depth_ = depth + 1;
            configure_node(features, labels, leftIndices, depth + 1, leftNode, para, costFunction);
            node->left_child_ = leftNode;
        }
        
        if (rightIndices.size() != 0) {
            rf_rgrsn_tree_node_constant * rightNode = dynamic_cast<rf_rgrsn_tree_node_constant *>(costFunction.new_tree_node());
            rightNode->depth_ = depth + 1;
            configure_node(features, labels, rightIndices, depth + 1, rightNode, para, costFunction);
            node->right_child_ = rightNode;
        }
        return;
    }
    else
    {
        // leaf node
        node->isLeaf_ = true;
        node->set_leaf_parameter(features, labels, indices);
        return;
    }    
}