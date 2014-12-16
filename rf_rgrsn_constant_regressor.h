//
//  rf_rgrsn_constant_regressor.h
//  RandomForestRegression
//  random forest constant (leaf node) regressor
//  Created by jimmy on 12/10/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__rf_rgrsn_constant_regressor__
#define __RandomForestRegression__rf_rgrsn_constant_regressor__

#include <vector>
#include "rf_rgrsn_tree_constant.h"



class rf_rgrsn_constant_regressor
{
    friend class rf_rgrsn_constant_regressor_builder;
    vector<rf_rgrsn_tree_constant*> trees_;
public:
    rf_rgrsn_constant_regressor()
    {
    
    }
    ~rf_rgrsn_constant_regressor()
    {
        for (int i = 0; i<trees_.size(); i++) {
            if (trees_[i]) {
                delete trees_[i];
                trees_[i] = NULL;
            }
        }
        trees_.clear();
    }
    
    double regression(const nl_vector & feature) const;
    
    bool write(const char *fileName) const;
    bool read(const char *fileName);
    
};


#endif /* defined(__RandomForestRegression__rf_rgrsn_constant_regressor__) */
