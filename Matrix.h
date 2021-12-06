/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   Matrix.h
 * Author: chelovek
 *
 * Created on 30 марта 2019 г., 20:19
 */

#include <vector>
#include <functional>
#include <cmath>

#ifndef MATRIX_H
#define MATRIX_H

class Matrix {
public:
    Matrix();

    Matrix(unsigned num_rows, unsigned num_cols, bool is_random = false, bool gpu_compute = false, double dropout = 0);
    void init(unsigned num_rows, unsigned num_cols,                      bool gpu_compute = false);

    Matrix(const std::vector<double> & v);
    virtual ~Matrix();

    void random_sort();


    void transpose();// { is_transposed = !is_transposed; };
    inline unsigned get_rows() { return rows; };
    inline unsigned get_cols() { return cols; };
    std::vector<double> & get_matrix_val();// { return matrix_val; };
    inline void operator+=(Matrix & m) { sum(*this, m); };
    void operator+=(double d);
    void operator=(std::vector<double> & v);
    void operator=(Matrix & m);
    void init(Matrix & m);
    void init(std::vector<double> & v);

    void activation(unsigned act);
    //void activation(std::function<void(double&)> fn);

    void softmax_calculate(Matrix & m/*, double regularization_sum*/);
    void last_gradient_calculate(Matrix & m, std::vector<double> & output_vals/*, double regularization_sum = 0.0*/);
    void hidden_gradient_calculate(Matrix & m, unsigned act);

    void ziro_matrix();

    void log();

    void thread_multiply(Matrix & a, Matrix & b);

    double regularization_count(double regularization_rate);
    double regularization_gradient_count(double regularization_rate);

    void multiply(Matrix & a, Matrix & b);
    void multiply(const Matrix & m);
    void multiply(double d);
    void multiply_like_sum(Matrix & a);

private:
    void random_shuffle();
    double get_random_numb();
    void sum(Matrix & a, Matrix & b);
    void multiply_standart(const Matrix & a, const Matrix & b);
    void multiply_left_transposed(const Matrix & a, const Matrix & b);
    void multiply_right_transposed(const Matrix & a, const Matrix & b);

    bool is_memory_used = false;
    bool is_gpu         = false;
    bool is_transposed  = false;
    unsigned rows;
    unsigned cols;
    unsigned length;
    std::vector<double> matrix_val;
    double *gpu_matrix_val;
};

#endif /* MATRIX_H */

