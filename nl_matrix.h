//
//  nl_matrix.h
//  RandomForestRegression
//
//  Created by jimmy on 12/14/14.
//  Copyright (c) 2014 CPSC540. All rights reserved.
//

#ifndef __RandomForestRegression__nl_matrix__
#define __RandomForestRegression__nl_matrix__

#include <vector>
#include <assert.h>
#include "nl_vector.h"

using namespace std;

class nl_matrix
{
public:
    nl_matrix(unsigned r, unsigned c, double v0)
    {
        data_.resize(r);
        for (int i = 0; i<r; i++) {
            data_[i].resize(c);
            for (int j = 0; j<c; j++) {
                data_[i][j] = v0;
            }
        }
    }
    
    ~nl_matrix()
    {
        for (int i = 0; i<data_.size(); i++) {
            data_[i].clear();
        }
        data_.clear();
    }
    
    unsigned rows()    const { return (unsigned)data_.size(); }
    unsigned cols()    const { return (unsigned)data_.front().size(); }
    
    double       & operator()(unsigned r, unsigned c)
    {
        assert(r<data_.size());           // Check the row index is valid
        assert(c<data_.front().size());   // Check the column index is valid
        
        return this->data_[r][c];
    }
    
    double const & operator()(unsigned r, unsigned c) const
    {
        assert(r<data_.size());           // Check the row index is valid
        assert(c<data_.front().size());   // Check the column index is valid
        
        return this->data_[r][c];
    }
    
    nl_matrix transpose() const
    {
        unsigned cols = this->cols();
        unsigned rows = this->rows();
        nl_matrix result(cols, rows, 0.0);
        for (unsigned int i = 0; i < cols; i++){
            for (unsigned int j = 0; j < rows; j++){
                result.data_[i][j] = this->data_[j][i];
            }
        }
        return result;
    }
    
    //: Add rhs to lhs  matrix in situ
    nl_matrix& operator+=(nl_matrix const& other)
    {
        unsigned rows = this->rows();
        unsigned cols = this->cols();
        assert(rows == other.rows() && cols == other.cols());
        
        nl_matrix result(rows, cols, 0.0);
        for (unsigned int i = 0; i<rows; i++) {
            for (unsigned int j = 0; j<cols; j++) {
                this->data_[i][j] += other(i, j);
            }
        }
        return *this;
    }
    
    /*
    //: Subtract rhs from lhs matrix in situ
    vnl_matrix<T>& operator-=(vnl_matrix<T> const&);
    //: Multiply lhs matrix in situ by rhs
    vnl_matrix<T>& operator*=(vnl_matrix<T> const&rhs) { return *this = (*this) * rhs; }
     */
    
    nl_matrix operator+(nl_matrix const& rhs) const
    {
        unsigned rows = this->rows();
        unsigned cols = this->cols();
        assert(rows == rhs.rows() && cols == rhs.cols());
        
        nl_matrix result(rows, cols, 0.0);
        for (unsigned int i = 0; i<rows; i++) {
            for (unsigned int j = 0; j<cols; j++) {
                result(i, j) = this->data_[i][j] + rhs(i, j);
            }
        }
        return result;
    }
    
    nl_matrix operator-(nl_matrix const& rhs) const
    {
        unsigned rows = this->rows();
        unsigned cols = this->cols();
        assert(rows == rhs.rows() && cols == rhs.cols());
        
        nl_matrix result(rows, cols, 0.0);
        for (unsigned int i = 0; i<rows; i++) {
            for (unsigned int j = 0; j<cols; j++) {
                result(i, j) = this->data_[i][j] - rhs(i, j);
            }
        }
        return result;
    }

    
    /*
    //: Matrix subtract rhs from lhs and return result in new matrix
    vnl_matrix<T> operator-(vnl_matrix<T> const& rhs) const { return vnl_matrix<T>(*this, rhs, vnl_tag_sub()); }
    //: Matrix multiply lhs by rhs matrix and return result in new matrix
    vnl_matrix<T> operator*(vnl_matrix<T> const& rhs) const { return vnl_matrix<T>(*this, rhs, vnl_tag_mul()); }
     */
    
    nl_matrix operator*(nl_matrix const& rhs) const
    {
        assert(this->cols() == rhs.rows());
        
        unsigned N = this->rows();
        unsigned D = rhs.cols();
        nl_matrix result(N, D, 0.0);
        for (unsigned i = 0; i<N; i++) {
            for (unsigned j = 0; j<D; j++) {
                double val = 0;
                for (unsigned k = 0; k<this->cols(); k++) {
                    val += this->data_[i][k] * rhs(k, j);
                }
                result(i, j) = val;
            }
        }
        return result;
    }
    
    nl_matrix operator*(nl_vector const &rhs) const
    {
        assert(this->cols() == rhs.size());
        
        unsigned N = this->rows();
        unsigned D = 1;
        nl_matrix result(N, D, 0.0);
        for (unsigned i = 0; i<N; i++) {
            double val = 0;
            for (unsigned k = 0; k<this->cols(); k++) {
                val += this->data_[i][k] * rhs[k];
            }
            result(i, 0) = val;
        }
        return result;
    }
    
    nl_matrix operator*(double const & s) const
    {
        nl_matrix result(this->rows(), this->cols(), 0.0);
        for (unsigned i = 0; i<this->rows(); i++) {
            for (unsigned j = 0; j<this->cols(); j++) {
                result(i, j) = s * data_[i][j];
            }
        }
        return result;
    }
    
    
    
    nl_matrix& operator*=(double scale)
    {
        for (unsigned i = 0; i<this->rows(); i++) {
            for (unsigned j = 0; j<this->cols(); j++) {
                data_[i][j] = scale * data_[i][j];
            }
        }
        return *this;
    }
    
    // must delcare as friend, otherwise has three parameters: this, s, rhs
    friend nl_matrix operator*(const double s, const nl_matrix & rhs)
    {
        nl_matrix result(rhs.rows(), rhs.cols(), 0.0);
        for (unsigned i = 0; i<rhs.rows(); i++) {
            for (unsigned j = 0; j<rhs.cols(); j++) {
                result(i, j) = s * rhs.data_[i][j];
            }
        }
        return result;
    }  
   

    
private:
    vector<vector<double> > data_;
    
};

#endif /* defined(__RandomForestRegression__nl_matrix__) */
