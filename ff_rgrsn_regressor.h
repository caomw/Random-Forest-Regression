//
//  ff_rgrsn_regressor.h
//  RandomForestRegression
//
//  Created by jimmy on 12/15/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__ff_rgrsn_regressor__
#define __RandomForestRegression__ff_rgrsn_regressor__

#include <vector>
#include "nl_vector.h"

using std::vector;
class ff_rgrsn_tree;

class ff_rgrsn_regressor
{
    friend class ff_rgrsn_builder;
    vector<ff_rgrsn_tree*> trees_;
public:
    ff_rgrsn_regressor();
    ~ff_rgrsn_regressor();
    
    vector<unsigned int > n_multi_dimes(void) const;
    double regression(const vector<nl_vector > & input) const;
    
    bool read(const char *fileName);
    bool write(const char *fileName) const;
    
};

#endif /* defined(__RandomForestRegression__ff_rgrsn_regressor__) */
