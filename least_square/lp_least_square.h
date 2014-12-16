//
//  lp_least_square.h
//  RandomForestRegression
//  linpack least square
//  Created by jimmy on 12/14/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__lp_least_square__
#define __RandomForestRegression__lp_least_square__

#include "nl_matrix.h"
#include "nl_vector.h"

class lp_least_square
{
public:
    static bool Axb(const nl_matrix & A, const nl_vector & b, nl_vector & x);
    
    // regulated least square y = f(x) = A * x + sigma * x^2 , x is unknown
    // return squared sum of difference
    static bool RLS(const nl_matrix & A, const nl_vector & b, const nl_matrix & sigma, nl_vector & x);

};

#endif /* defined(__RandomForestRegression__lp_least_square__) */
