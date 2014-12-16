//
//  rf_rgrsn_constant_regressor.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/10/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "rf_rgrsn_constant_regressor.h"
#include <string>

using std::string;

double rf_rgrsn_constant_regressor::regression(const nl_vector & feature) const
{
    double value = 0.0;
    for (int i = 0 ; i<trees_.size(); i++) {
        value += trees_[i]->evaluate(feature);
    }
    return value/trees_.size();
}

bool rf_rgrsn_constant_regressor::write(const char *fileName) const
{
    assert(trees_.size() > 0);    
    //write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not open %s\n", fileName);
        return false;
    }
    fprintf(pf, "%d %d\n", (int)trees_.size(), (int)trees_[0]->n_dim());
    vector<string> tree_files;
    string baseName = string(fileName);
    baseName = baseName.substr(0, baseName.size()-4);
    for (int i = 0; i<trees_.size(); i++) {
        char buf[1024] = {NULL};
        sprintf(buf, "%08d", i);
        string fileName = baseName + string(buf) + string(".txt");
        tree_files.push_back(fileName);
        fprintf(pf, "%s\n", fileName.c_str());
    }
    
    for (int i = 0; i<trees_.size(); i++) {
        rf_rgrsn_tree_node_constant::write_tree(tree_files[i].c_str(), trees_[i]->root_node());
    }
    
    fclose(pf);
    
    return true;
}
bool rf_rgrsn_constant_regressor::read(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not read file %s\n", fileName);
        return false;
    }
    int num = 0;
    int nDim = 0;
    int ret = fscanf(pf, "%d %d", &num, &nDim);
    assert(ret == 2);
    vector<string> treeFiles;
    for (int i = 0; i<num; i++) {
        char buf[1024] = {NULL};
        fscanf(pf, "%s", buf);
        treeFiles.push_back(string(buf));
    }
    fclose(pf);
    
    for (int i = 0; i<trees_.size(); i++) {
        delete trees_[i];
        trees_[i] = 0;
    }
    trees_.clear();
    
    for (int i = 0; i<treeFiles.size(); i++) {
        rf_rgrsn_tree_node_constant * root = rf_rgrsn_tree_node_constant::read_tree(treeFiles[i].c_str());
        
        rf_rgrsn_tree_constant *tree = new rf_rgrsn_tree_constant();
        tree->set_root_node(root);
        tree->set_n_dim(nDim);
        
        trees_.push_back(tree);
    }
    
    return true;
    
}