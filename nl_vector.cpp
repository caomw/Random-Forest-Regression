//
//  nl_vector.cpp
//  RandomForestRegression
//
//  Created by jimmy on 12/9/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#include "nl_vector.h"

/*
nl_vector::nl_vector(unsigned len, double const& v0)
{
    allocate(len);
    for (int i = 0; i<num_elmts; i++) {
        data[i] = v0;
    }
}


nl_vector::~nl_vector()
{
    if (data) {
        clear();
    }
}


nl_vector::nl_vector (const nl_vector & v)
{
    set_size(v.num_elmts);
    for (unsigned i = 0; i < v.num_elmts; i ++)   // For each element in v
    {
        this->data[i] = v.data[i];                  // Copy value
    }
}


nl_vector & nl_vector::operator=(nl_vector const & rhs)
{
    if (this != &rhs) { // make sure *this != m
        if (rhs.data) {
            if (this->num_elmts != rhs.num_elmts){
                this->set_size(rhs.size());
            }
            for (unsigned i = 0; i < this->num_elmts; i++){
                this->data[i] = rhs.data[i];
            }
        }
        else {
            // rhs is default-constructed.
            clear();
        }
    }
    return *this;
}

bool nl_vector::set_size(unsigned n)
{
    if (this->data) {
        // if no change in size, do not reallocate.
        if (this->num_elmts == n){
            return false;
        }
        else
        {
            clear();
            allocate(n);
        }
    }
    else {
        allocate(n);
    }
    return true;
}

void nl_vector::allocate(unsigned n)
{
    num_elmts = n;
    data = new double [n];
    
}
void nl_vector::clear()
{
    if (data && num_elmts != 0) {
        delete  []data;
        num_elmts = 0;
        data = 0;
    }
}
 */


