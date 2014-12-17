//
//  lp_least_square.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/14/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "lp_least_square.h"
#include "linpack_d.hpp"
#include <iostream>
#include <iomanip>

using namespace std;

bool lp_least_square::Axb(const nl_matrix & A, const nl_vector & b, nl_vector & x)
{
    // check input parameter
    assert(A.rows() == b.size());
    
    // memory copy to continuous address
    // column major order
    double *pA = new double [A.rows() * A.cols()];
    assert(pA);
    for (unsigned c = 0; c<A.cols(); c++) {
        for (unsigned r = 0; r<A.rows(); r++) {
            pA[r + c * A.rows()] = A(r, c);
        }
    }
    double *pb = new double[b.size()];
    assert(pb);
    for (unsigned i = 0; i<b.size(); i++) {
        pb[i] = b[i];
    }
    
    // Factor the matrix.
    int *ipvt = new int[A.cols()];
    double *z = new double[A.cols()];
    double rcond = dgeco ( pA, A.rows(), A.cols(), ipvt, z);
    if ( rcond + 1.0 == 1.0 )
    {
        cout << "  Error!  The matrix is nearly singular!\n";
        delete []pA;
        delete []pb;
        delete []ipvt;
        delete []z;
        return false;
    }
    
    // solver linear
    int job = 0;
    dgesl( pA, A.rows(), A.cols(), ipvt, pb, job );
    
    x = nl_vector(A.cols(), 0.0);
    for (int i = 0; i<A.cols(); i++) {
        x[i] = pb[i];
    //    printf("%lf\n", x[i]);
    }
    
    delete []pA;
    delete []pb;
    delete []ipvt;
    delete []z;
    return true;
}

// w = argmin( (y - wx) ^2 + (sigma w) ^2)
// w = (X^t * X + simga^t * sigma)^(-1) X^t * y
// double RLS(const nl_matrix & A, const nl_vector & b, const nl_matrix & sigma, nl_vector & x);
bool lp_least_square::RLS(const nl_matrix & A, const nl_vector & b, const nl_matrix & sigma, nl_vector & x)
{
    assert(A.rows() == b.size());
    assert(sigma.rows() == A.cols());
    assert(sigma.cols() == A.cols());
    
    nl_matrix A_reg = A.transpose() * A + sigma.transpose() * sigma;
    nl_matrix Xtb = A.transpose() * b;
    assert(Xtb.cols() == 1);
    nl_vector bb_vec(Xtb.rows(), 0);
    for (unsigned i = 0; i<Xtb.rows(); i++) {
        bb_vec[i] = Xtb(i, 0);
    }
    
    return lp_least_square::Axb(A_reg, bb_vec, x);
}
