//
//  ff_rgrsn_regressor.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/15/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "ff_rgrsn_regressor.h"
#include "ff_rgrsn_tree.h"
#include <string>
#include "ff_rgrsn_tree_node.h"

using std::string;

ff_rgrsn_regressor::ff_rgrsn_regressor()
{
    
}

ff_rgrsn_regressor::~ff_rgrsn_regressor()
{
    
}

vector<unsigned int > ff_rgrsn_regressor::n_multi_dimes(void) const
{
    assert(trees_.size() > 0);
    return trees_.front()->n_multi_dims();
}

double ff_rgrsn_regressor::regression(const vector<nl_vector > & input) const
{
    vector<unsigned int> dims = trees_.front()->n_multi_dims();
    assert(dims.size() == input.size());
    for (int i = 0; i<input.size(); i++) {
        assert(dims[i] == input[i].size());
    }
    
    double val = 0.0;
    for (int i = 0; i<trees_.size(); i++) {
        val += trees_[i]->evaluate(input);
    }
    val /= trees_.size();
    return val;
}

bool ff_rgrsn_regressor::write(const char *fileName) const
{
    //write tree number and tree files to file Name
    FILE *pf = fopen(fileName, "w");
    if (!pf) {
        printf("can not write file %s\n", fileName);
        return false;
    }
    fprintf(pf, "%d\t %lu\n", (int)trees_.size(), trees_.front()->n_multi_dims().size());
    for (int i = 0; i<trees_.front()->n_multi_dims().size(); i++) {
        fprintf(pf, "%u\t ", trees_.front()->n_multi_dims()[i]);
    }
    fprintf(pf, "\n");
    
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
        ff_rgrsn_tree_node::write_FF_tree(tree_files[i].c_str(), trees_[i]->root_node());
    }
    
    fclose(pf);
    return true;
}

bool ff_rgrsn_regressor::read(const char *fileName)
{
    FILE *pf = fopen(fileName, "r");
    if (!pf) {
        printf("can not read file %s\n", fileName);
        return false;
    }
    int num = 0;
    int nWindowNum = 0;
    int ret = fscanf(pf, "%d %d", &num, &nWindowNum);
    assert(ret == 2);
    vector<unsigned int> winSizes(nWindowNum, 0);
    for (int i = 0; i<nWindowNum; i++) {
        ret = fscanf(pf, "%u ", &winSizes[i]);
        assert(ret == 1);
    }
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
        ff_rgrsn_tree_node * root = NULL;
        bool isRead = ff_rgrsn_tree_node::read_FF_tree(treeFiles[i].c_str(), root);
        assert(isRead);
        ff_rgrsn_tree *tree = new ff_rgrsn_tree();
        tree->set_root_node(root);
        tree->set_multi_dims(winSizes);
        trees_.push_back(tree);
    }
    return true;
}
