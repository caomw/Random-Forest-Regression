//
//  ff_rgrsn_tree_node.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/15/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "ff_rgrsn_tree_node.h"

static void write_FF_linear_prediction(FILE *pf, ff_rgrsn_tree_node * node)
{
    if (!node) {
        fprintf(pf, "#\n");
        return;
    }
    
    // write corrent node
    fprintf(pf, "%u\t %d\t %u\t %lf\t %u\t %lu\n", node->depth_, (int)node->isLeaf_, node->split_dim_, node->split_threshold_, node->split_scale_index_, node->wt_.size());
    for (int i = 0; i<node->wt_.size(); i++) {
        fprintf(pf, "%lf\t", node->wt_[i]);
    }
    if(node->wt_.size() != 0)
    {
        fprintf(pf, "\n");
    }
    
    write_FF_linear_prediction(pf, node->left_child_);
    write_FF_linear_prediction(pf, node->right_child_);
}


bool ff_rgrsn_tree_node::write_FF_tree(const char *fileName, ff_rgrsn_tree_node * root)
{
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not write file: %s\n", fileName);
        return false;
    }
    fprintf(pf, "depth\t isLeaf\t split_dimension\t split_threshold\t splitScaleIndex weightDimension\n");
    write_FF_linear_prediction(pf, root);
    fclose(pf);
    return true;
}

static void read_FF_linear_prediction(FILE *pf, ff_rgrsn_tree_node * & node)
{
    char lineBuf[1024] = {NULL};
    char *ret = fgets(lineBuf, sizeof(lineBuf), pf);
    if (!ret) {
        node = NULL;
        return;
    }
    if (lineBuf[0] == '#') {
        // empty node
        node = NULL;
        return;
    }
    node = new ff_rgrsn_tree_node();
    assert(node);
    int isLeaf = 0;
    unsigned int dim = 0;
    int ret_num = sscanf(lineBuf, "%u %d %u %lf %u %u",  &node->depth_, &isLeaf, &node->split_dim_, &node->split_threshold_, &node->split_scale_index_, &dim);
    assert(ret_num == 6);
    node->isLeaf_ = (isLeaf == 1);
    if(dim != 0)
    {
        node->wt_ = nl_vector(dim, 0);
        for (int i = 0; i<dim; i++) {
            double wt = 0;
            ret_num = fscanf(pf, "%lf ", &wt);
            node->wt_[i] = wt;
            assert(ret_num == 1);
        }
    }
    
    node->left_child_  = NULL;
    node->right_child_ = NULL;
    
    read_FF_linear_prediction(pf, node->left_child_);
    read_FF_linear_prediction(pf, node->right_child_);
}

bool ff_rgrsn_tree_node::read_FF_tree(const char *fileName, ff_rgrsn_tree_node * & root)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not read file: %s\n", fileName);
        return false;
    }
    //read first line
    char lineBuf[1024] = {NULL};
    fgets(lineBuf, sizeof(lineBuf), pf);
    printf("%s\n", lineBuf);
    read_FF_linear_prediction(pf, root);
    pclose(pf);
    return true;
}


double ff_rgrsn_tree_node::value(const vector< nl_vector > & multi_scale_feature) const
{
    assert(isLeaf_);
    assert(split_scale_index_ < multi_scale_feature.size());
    
    nl_vector feature = multi_scale_feature[split_scale_index_];
    if (!(feature.size() == wt_.size())) {
        printf("Error: weight dimension is differdent with feature dimension. (d_wt, d_feature): %d %d\n", (int)wt_.size(), (int)feature.size());
        return INT_MAX;
    }
    
    double value = dot_product(wt_, feature);    
    return value;
}
