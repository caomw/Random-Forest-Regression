//
//  ff_rgrsn_tree.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/15/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "ff_rgrsn_tree.h"
#include "ff_tree_cost_function.h"
#include "ff_rgrsn_tree_node.h"


/**********     ff_rgrsn_tree     ********/


void ff_rgrsn_tree::build(const vector< vector<nl_vector > > & multi_scale_features,
                          const vector<double> & labels,
                          const rf_rgrsn_tree_parameter & para, const ff_tree_cost_function & costFunction)
{
    // save multip feature dimension
    assert(multi_scale_features.size() == labels.size());
    
    n_multi_dims_.clear();
    for (int i = 0; i<multi_scale_features[0].size(); i++) {
        n_multi_dims_.push_back((unsigned)multi_scale_features[0][i].size());
    }
    
    root_ = costFunction.new_tree_node();
    assert(root_);
    root_->depth_ = 0;
    
    vector<unsigned int> indices;
    for (int i = 0; i<multi_scale_features.size(); i++) {
        indices.push_back(i);
    }
    random_shuffle(indices.begin(), indices.end());
    
    this->configure_node(multi_scale_features, labels, indices, 0, root_, para, costFunction);
}

double ff_rgrsn_tree::evaluate(const vector< nl_vector > & multi_scale_feature) const
{
    assert(root_);
    assert(multi_scale_feature.size() == n_multi_dims_.size());
    for (int i = 0; i<n_multi_dims_.size(); i++) {
        assert(n_multi_dims_[i] == multi_scale_feature[i].size());
    }
    
    return this->evaluate(multi_scale_feature, root_);
}

double ff_rgrsn_tree::evaluate(const vector< nl_vector > & multi_scale_feature, ff_rgrsn_tree_node * node) const
{
    assert(node);
    if (node->isLeaf_) {
        return node->value(multi_scale_feature);
    }
    
    unsigned int scaleIndex = node->split_scale_index_;
    unsigned int dim = node->split_dim_;
    double threshold = node->split_threshold_;
    double value = multi_scale_feature[scaleIndex][dim];
    
    if (node->left_child_ && value < threshold) {
        return this->evaluate(multi_scale_feature, node->left_child_);
    }
    else if(node->right_child_ && value >= threshold)
    {
        return this->evaluate(multi_scale_feature, node->right_child_);
    }
    else
    {
        printf("Error: can not find proper splitting node\n");
    }
    return INT_MAX;
}

//random version of bestSplittingInDimenstion
//x = 0 1 2 3, randomly pick up 4 number in (0, 3) as candidate splitting value
static bool bestSplittingInDimenstionWithRandom(const vector< vector< nl_vector > > & inputs,
                                                const vector< double > & outputs,
                                                const vector< unsigned int > & indices,
                                                const unsigned int scaleIndex,
                                                const unsigned int dim,
                                                const unsigned int minSize,
                                                const rf_rgrsn_tree_parameter & para,
                                                const ff_tree_cost_function & costFunction,
                                                
                                                //output
                                                double & loss,
                                                double & threshold,
                                                vector< unsigned int >& leftIndices,
                                                vector< unsigned int >& rightIndices)
{
    assert(indices.size() >= minSize);
    assert(scaleIndex < inputs[0][scaleIndex].size());
    
    //collect data in nDim of inputs
    vector<double > data_x_nDim;
    for (int i = 0; i<indices.size(); i++) {
        int idx = indices[i];
        data_x_nDim.push_back(inputs[idx][scaleIndex][dim]);
    }
    
    random_sample(data_x_nDim, std::max((unsigned int)inputs.front().size(), para.min_sample_num_));
    loss = INT_MAX;
    
    //for each possible threshold
    bool isDivided =  false;
    for (int i = 0; i < data_x_nDim.size(); i++) {
        vector<unsigned int> curLeftIndices;
        vector<unsigned int> curRightIndices;
        
        //try every possible threshold
        double curThreshold = data_x_nDim[i];
        
        // loop current data (in one dimension)
        for (int j = 0; j<indices.size(); j++) {
            int idx = indices[j];
            if (inputs[idx][scaleIndex][dim] < curThreshold) {
                curLeftIndices.push_back(idx);
            }
            else
            {
                curRightIndices.push_back(idx);
            }
        }
        if (curLeftIndices.size() * 2 < minSize || curRightIndices.size() * 2 < minSize) {
            continue;
        }
        
        double left_err = 0;
        nl_vector wtDump;
        if (curLeftIndices.size() > 0) {
            left_err = costFunction.cost(inputs, outputs, curLeftIndices, scaleIndex, wtDump);
            assert(left_err >= 0);
            
        }
        
        double right_err = 0;
        if (curRightIndices.size() > 0) {
            right_err = costFunction.cost(inputs, outputs, curRightIndices, scaleIndex, wtDump);
            assert(right_err >= 0);
        }
        
        if (left_err + right_err < loss) {
            loss = left_err + right_err;
            leftIndices  = curLeftIndices;
            rightIndices = curRightIndices;
            threshold = curThreshold;
            isDivided = true;
        }
    }
    
    return isDivided;
}


static bool bestSplittingDimension(const vector< vector< nl_vector > > & inputs,
                                   const vector< double> & outputs,
                                   const vector< unsigned int > & indices,
                                   const unsigned int scaleIndex,
                                   const unsigned int minSize,
                                   const rf_rgrsn_tree_parameter & para,
                                   const ff_tree_cost_function & costFunction,
                                   // output
                                   unsigned int & splitDim,
                                   double & splitThreshold,
                                   double & loss,
                                   vector< unsigned int > & leftIndices,
                                   vector< unsigned int > & rightIndices)
{
    assert(indices.size() >= minSize);
    
    // randomly generate dimensions that used in split
    unsigned int nDims = (unsigned int)inputs[0][scaleIndex].size();
    vector< unsigned int > randomDimensions( nDims );
    for ( unsigned int i = 0; i < nDims; ++i )
    {
        randomDimensions[i] = i;
    }
    random_shuffle( randomDimensions.begin(), randomDimensions.end());
    
    unsigned int subDims = sqrt(nDims);
    
    assert( subDims > 0 );
    assert( subDims <= nDims );
    //erase some dimensions, to avoid always pick the same dimenstion every time
    vector< unsigned int > randomSubset( randomDimensions.begin(), randomDimensions.begin() + subDims );
    
    //loop each dimenstion
    // double minLoss = INT_MAX;
    bool isSplitOk = false;
    for (int i = 0; i<randomSubset.size(); i++) {
        vector<unsigned int> curLeftIndices;
        vector<unsigned int> curRightIndices;
        double curLoss = INT_MAX;
        double threshold = 0.0;
        
        bool isSplit = bestSplittingInDimenstionWithRandom(inputs, outputs, indices, scaleIndex, randomSubset[i], minSize, para,
                                                           costFunction, curLoss, threshold, curLeftIndices, curRightIndices);
        
        if (isSplit && curLoss < loss) {
            loss = curLoss;
            leftIndices  = curLeftIndices;
            rightIndices = curRightIndices;
            splitDim = randomSubset[i];
            splitThreshold = threshold;
            isSplitOk = true;
        }
    }
    
    return isSplitOk;
    
}


static bool bestSplittingScale(const vector< vector< nl_vector > > & inputs,
                               const vector< double > & outputs,
                               const vector< unsigned int> & indices,
                               const unsigned int minSize,
                               const rf_rgrsn_tree_parameter & para,
                               const ff_tree_cost_function & costFunction,
                               //output
                               unsigned int & splitScaleIndex,
                               unsigned int & splitDim,
                               double & splitThreshold,
                               vector< unsigned int > & leftIndices,
                               vector< unsigned int > & rightIndices)
{
    double minLoss = INT_MAX;
    bool canSplit = false;
    //loop each scale index
    for (unsigned int i = 0; i<inputs[0].size(); i++) {
        unsigned int curScaleIndex = i;
        unsigned int curSplitDim = 0;
        double curSplitThreshold = 0;
        double curLoss = INT_MAX;
        vector<unsigned int> curLeftIndices;
        vector<unsigned int> curRightIndices;
        bool isSplit = bestSplittingDimension(inputs, outputs, indices, curScaleIndex, minSize, para, costFunction,
                                              curSplitDim, curSplitThreshold, curLoss, curLeftIndices, curRightIndices);
        //   printf("isSplit %d, curLoss %f\n", isSplit, curLoss);
        if (isSplit && curLoss < minLoss) {
            minLoss = curLoss;
            splitScaleIndex = curScaleIndex;
            splitDim = curSplitDim;
            splitThreshold = curSplitThreshold;
            leftIndices = curLeftIndices;
            rightIndices = curRightIndices;
            canSplit = true;
        }
    }
    return canSplit;
}

void ff_rgrsn_tree::configure_node(const vector< vector<nl_vector > > & multi_scale_features,
                                   const vector< double >& labels,
                                   const vector< unsigned int >& indices,
                                   const unsigned int depth,
                                   ff_rgrsn_tree_node * node,
                                   const rf_rgrsn_tree_parameter & para,
                                   const ff_tree_cost_function & costFunction) const
{
    assert(node);
    
    if (depth > para.max_tree_depth_ || indices.size() < para.min_feature_num_) {
        double min_cost = INT_MAX;
        int opt_scale_index = -1;
        nl_vector optimal_wt;
        
        // loop each scale
        for (int i = 0; i<multi_scale_features[0].size(); i++) {
            nl_vector wt;
            double cost = costFunction.cost(multi_scale_features, labels, indices, i, wt);
            if (cost < min_cost) {
                min_cost = cost;
                opt_scale_index = i;
                optimal_wt = wt;
            }
        }
        assert(opt_scale_index != -1);
        
        if (verbose_leaf_) {
            printf("depth, scale, leaf node size: %d %d %d\n", (int)depth, opt_scale_index, (int)indices.size());
        }
        
        // set node parameters
        node->isLeaf_ = true;
        node->split_scale_index_ = opt_scale_index;
        node->wt_ = optimal_wt;
        return ;
    }
    
    vector<unsigned int> leftIndices;
    vector<unsigned int> rightIndices;
    unsigned int minNodeSize = para.min_feature_num_;
    unsigned int splitScaleIndex = 0;
    unsigned int splitDim = 0;
    double splitThreshold = 0;
    
    // split in different scale and different dimension
    bool canSplit = bestSplittingScale(multi_scale_features, labels,
                                       indices, minNodeSize, para, costFunction,
                                       splitScaleIndex, splitDim, splitThreshold,
                                       leftIndices, rightIndices);
    
    
    if (canSplit) {
        assert(leftIndices.size() + rightIndices.size() == indices.size());
        
        node->split_scale_index_ = splitScaleIndex;
        node->split_dim_ = splitDim;
        node->split_threshold_ = splitThreshold;
        node->isLeaf_ = false;
        
        assert(splitDim < multi_scale_features[0][splitScaleIndex].size());
        if (verbose_) {
            int scale = (int)multi_scale_features[0][splitScaleIndex].size();
            printf("split scale, dim ,left, right node size: %d %d %d %d\n", scale, splitDim, (int)leftIndices.size(), (int)rightIndices.size());
        }
        
        if (leftIndices.size() != 0) {
            ff_rgrsn_tree_node *leftNode = costFunction.new_tree_node();
            leftNode->depth_ = depth + 1;
            configure_node(multi_scale_features, labels, leftIndices, depth + 1, leftNode, para, costFunction);
            node->left_child_ = leftNode;
        }
        if (rightIndices.size() != 0) {
            ff_rgrsn_tree_node *rightNode = costFunction.new_tree_node();
            rightNode->depth_ = depth + 1;
            configure_node(multi_scale_features, labels, rightIndices, depth + 1, rightNode, para, costFunction);
            node->right_child_ = rightNode;
        }
        
        return;
        
    }
    else
    {
        double min_cost = INT_MAX;
        int opt_scale_index = -1;
        nl_vector optimal_wt;
        
        // loop each scale
        for (int i = 0; i<multi_scale_features[0].size(); i++) {
            nl_vector wt;
            double cost = costFunction.cost(multi_scale_features, labels, indices, i, wt);
            if (cost < min_cost) {
                min_cost = cost;
                opt_scale_index = i;
                optimal_wt = wt;
            }
        }
        assert(opt_scale_index != -1);
        
        if (verbose_leaf_) {
            printf("depth, scale, leaf node size: %d %d %d\n", (int)depth, opt_scale_index, (int)indices.size());
        }
        
        // set node parameters
        node->isLeaf_ = true;
        node->split_scale_index_ = opt_scale_index;
        node->wt_ = optimal_wt;
        return ;
    }   
    
}

