//
//  rf_rgrsn_tree_node.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/9/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "rf_rgrsn_tree_node.h"



bool rf_rgrsn_tree_node_constant::set_leaf_parameter(const vector<nl_vector> & features, const vector<double> & labels,
                                                     const vector<unsigned> & indices)
{
    assert(indices.size() > 0);
    mean_ = 0;
    for (int i =0; i<indices.size(); i++) {
        mean_ += labels[indices[i]];
    }
    mean_ /= indices.size();
    
    return true;
}
double rf_rgrsn_tree_node_constant::value(const nl_vector &x) const
{
    return mean_;
}

void rf_rgrsn_tree_node_constant::write_constant_prediction(FILE *pf, rf_rgrsn_tree_node_constant * node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    
    // write corrent node
    fprintf(pf, "%u\t %d\t %u\t %lf\t %lf\n", node->depth_, (int)node->isLeaf_, node->split_dim_, node->split_threshold_, node->mean_);
    rf_rgrsn_tree_node_constant::write_constant_prediction(pf, dynamic_cast<rf_rgrsn_tree_node_constant *>(node->left_child_));
    rf_rgrsn_tree_node_constant::write_constant_prediction(pf, dynamic_cast<rf_rgrsn_tree_node_constant *>(node->right_child_));
}

bool rf_rgrsn_tree_node_constant::write_tree(const char *fileName, rf_rgrsn_tree_node_constant * root)
{
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not write file: %s\n", fileName);
        return false;
    }
    fprintf(pf, "depth\t isLeaf\t split_dimension\t split_threshold\t mean\n");
    rf_rgrsn_tree_node_constant::write_constant_prediction(pf, root);
    fclose(pf);
    return true;
}

rf_rgrsn_tree_node_constant * rf_rgrsn_tree_node_constant::read_constant_prediction(FILE *pf)
{
    rf_rgrsn_tree_node_constant * node = NULL;
    char lineBuf[1024] = {NULL};
    char *ret = fgets(lineBuf, sizeof(lineBuf), pf);
    if (!ret) {
        return NULL;
    }
    if (lineBuf[0] == '#') {
        // empty node
        return NULL;
    }
    node = new rf_rgrsn_tree_node_constant();
    assert(node);
    int isLeaf = 0;
    int ret_num = sscanf(lineBuf, "%u %d %u %lf %lf",  &node->depth_, &isLeaf, &node->split_dim_, &node->split_threshold_, &node->mean_);
    assert(ret_num == 5);
    node->isLeaf_ = (isLeaf == 1);
    node->left_child_  = rf_rgrsn_tree_node_constant::read_constant_prediction(pf);
    node->right_child_ = rf_rgrsn_tree_node_constant::read_constant_prediction(pf);
    return node;
}

rf_rgrsn_tree_node_constant* rf_rgrsn_tree_node_constant::read_tree(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not read file: %s\n", fileName);
        return NULL;
    }
    //read first line
    char lineBuf[1024] = {NULL};
    fgets(lineBuf, sizeof(lineBuf), pf);
    printf("%s\n", lineBuf);
    rf_rgrsn_tree_node_constant * root = rf_rgrsn_tree_node_constant::read_constant_prediction(pf);
    fclose(pf);
    return root;
}
