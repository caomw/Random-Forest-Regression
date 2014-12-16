//
//  nl_vector.h
//  RandomForestRegression
//
//  Created by jimmy on 12/9/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__nl_vector__
#define __RandomForestRegression__nl_vector__

// numerical vector modified from vnl_vector in VXL
#include <assert.h>
#include <vector>
#include <numeric>

#define nl_vector std::vector<double>

inline double dot_product(nl_vector const &v1, nl_vector const &v2)
{
    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
}

/*
class nl_vector
{
public:
    //: Creates an empty vector. O(1).
    nl_vector()
    {
        num_elmts = 0;
        data = NULL;
    }   
    
    //: Creates vector of len elements, all set to v0
    nl_vector(unsigned len, double const& v0);
    
    //: Destructor
    ~nl_vector();
    
    //: Return the length, number of elements, dimension of this vector.
    unsigned size() const { return num_elmts; }
    
    //: Return reference to the element at specified index.
    double       & operator[](unsigned int i)
    {
        assert(i < num_elmts);
        return data[i];
    }
    //: Return reference to the element at specified index.
    double const & operator[](unsigned int i) const
    {
        assert(i < num_elmts);
        return data[i];
    }
    
    //: Copy operator
    nl_vector (const nl_vector  & v);
    nl_vector & operator=(nl_vector const & rhs);
    
    bool set_size(unsigned n);
    
private:
    
    void allocate(unsigned n);
    void clear();
    
protected:
    unsigned num_elmts;                // Number of elements (length)
    double* data;                      // Pointer to the actual data
  
};
*/


#endif /* defined(__RandomForestRegression__nl_vector__) */
