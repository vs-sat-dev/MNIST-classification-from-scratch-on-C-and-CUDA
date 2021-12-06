/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   Matrix.cpp
 * Author: chelovek
 *
 * Created on 30 марта 2019 г., 20:19
 */

#include "Matrix.h"
#include <iostream>
#include <random>
#include <cstdlib>
#include <thread>
#include <fstream>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
//#define BLOCK_SIZE = 16

const unsigned block_size = 16;

void logm(std::string str) {
    std::ofstream fout("log.txt", std::ios_base::app);
    fout << str;
    fout.close();
}

void logm_new() {
    std::ofstream fout("log.txt", std::ios_base::trunc);
    fout.close();
}

using namespace std;

Matrix::Matrix() {
}

void Matrix::log() {
    std::ofstream fout("log.txt", std::ios_base::app);
    for(unsigned r = 0; r < rows; ++r) {
        fout << endl;
        for(unsigned c = 0; c < cols; ++c) {
            fout << matrix_val[r*cols + c] << " ";
        }
    }
    fout.close();
}

Matrix::Matrix(unsigned num_rows, unsigned num_cols, bool is_random, bool gpu_compute, double dropout) {
    rows = num_rows;
    cols = num_cols;
    length = rows*cols;
    matrix_val.reserve(length);

    unsigned msize = unsigned(double(length) * dropout);

    for(unsigned i = 0; i < length; ++i) {
        if(is_random)
                matrix_val.emplace_back(get_random_numb());
            else {
            	if(dropout == 0.0)
            	    matrix_val.emplace_back(0.0);
            	else {
            		if(i >= msize)
            			matrix_val.emplace_back(1.0);
            		else
            			matrix_val.emplace_back(0.0);
            	}
            }
    }

    is_gpu = gpu_compute;
    if(is_gpu) {
    	is_memory_used = true;
    	CUDA_CHECK_RETURN(cudaMalloc((void**)&gpu_matrix_val, sizeof(double) * length));
    	CUDA_CHECK_RETURN(cudaMemcpy(gpu_matrix_val, &matrix_val[0], sizeof(double) * length, cudaMemcpyHostToDevice));
    }
}

void Matrix::init(unsigned num_rows, unsigned num_cols, bool gpu_compute) {
	rows = num_rows;
	cols = num_cols;
	length = rows*cols;
	matrix_val.reserve(length);
	for(unsigned i = 0; i < length; ++i) {
	    matrix_val.emplace_back(0.0);
	}

	is_gpu = gpu_compute;
	if(is_gpu) {
		is_memory_used = true;
	    CUDA_CHECK_RETURN(cudaMalloc((void**)&gpu_matrix_val, sizeof(double) * length));
	    CUDA_CHECK_RETURN(cudaMemcpy(gpu_matrix_val, &matrix_val[0], sizeof(double) * length, cudaMemcpyHostToDevice));
	}
}

Matrix::~Matrix() {
	if(is_gpu) {
		cudaFree(gpu_matrix_val);
	}
}

void Matrix::init(Matrix & m) {
    rows = m.rows;
    cols = m.cols;
    length = rows*cols;

    is_gpu = m.is_gpu;
    if(is_gpu) {
        CUDA_CHECK_RETURN(cudaMalloc((void**)&gpu_matrix_val, sizeof(double) * length));
    	CUDA_CHECK_RETURN(cudaMemcpy(gpu_matrix_val, m.gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToDevice));
    }
    else {
    	for(unsigned r = 0; r < rows; ++r) {
    	    for(unsigned c = 0; c < cols; ++c) {
    	        matrix_val[r*cols + c] = m.matrix_val[r*cols + c];
    	    }
    	}
    }

}

double Matrix::get_random_numb() {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0, 1);

    return dis(gen);
}

//------------------------------------------------------------------------------

/*void multiply_standart_thread(vector<vector<double> >& res, const vector<vector<double> >& a, const vector<vector<double> >& b, unsigned offset_pos, unsigned offset_step, bool & is_exist) {
    for(unsigned r = 0; r < a.size(); ++r) {
        for(unsigned mc = offset_pos; mc < b[0].size(); mc += offset_step) {
            for(unsigned mr = 0; mr < b.size(); ++mr) {
                res[r][mc] += a[r][mr] * b[mr][mc];
            }
        }
    }
    is_exist = false;
    //cout << "\nexist1 " << is_exist;
}

void multiply_left_transposed_thread(vector<vector<double> >& res, const vector<vector<double> >& a, const vector<vector<double> >& b, unsigned offset_pos, unsigned offset_step, bool & is_exist) {
    for(unsigned r = 0; r < a[0].size(); ++r) {
        for(unsigned mc = offset_pos; mc < b[0].size(); mc += offset_step) {
            for(unsigned mr = 0; mr < b.size(); ++mr) {
                res[r][mc] += a[mr][r] * b[mr][mc];
            }
        }
    }
    is_exist = false;
    //cout << "\nexist2 " << is_exist;
}

void multiply_right_transposed_thread(vector<vector<double> >& res, const vector<vector<double> >& a, const vector<vector<double> >& b, unsigned offset_pos, unsigned offset_step, bool & is_exist) {
    for(unsigned r = 0; r < a.size(); ++r) {
        for(unsigned mc = 0; mc < b.size(); ++mc) {
            for(unsigned mr = offset_pos; mr < b[0].size(); mr += offset_step) {
                res[r][mc] += a[r][mr] * b[mc][mr];
            }
        }
    }
    is_exist = false;
    //cout << "\nexist3 " << is_exist;
}

//bool b1, b2, b3, b4;

void Matrix::thread_multiply(Matrix & a, Matrix & b) {
    bool b1, b2, b3, b4;
    b1 = b2 = b3 = b4 = true;

    //cout << "\nThread Start";

    if(a.is_transposed) {
        if(a.rows != b.rows) {
        cout << "\nMatrix a.rows != b.rows";
        cout << "\na.rows:" << a.rows << " a.cols:" << a.cols << " b.rows:" << b.rows << " b.cols:" << b.cols;
        cout << "\nFuck";
        exit(1);
    }
        thread t1(multiply_left_transposed_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 0, 4, ref(b1));
        thread t2(multiply_left_transposed_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 1, 4, ref(b2));
        thread t3(multiply_left_transposed_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 2, 4, ref(b3));
        thread t4(multiply_left_transposed_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 3, 4, ref(b4));

        t1.detach();
        t2.detach();
        t3.detach();
        t4.detach();

        //multiply_left_transposed(a, b);
    }
    else if(b.is_transposed) {
        if(a.cols != b.cols) {
        cout << "\nMatrix a.cols != b.cols";
        cout << "\na.rows:" << a.rows << " a.cols:" << a.cols << " b.rows:" << b.rows << " b.cols:" << b.cols;
        exit(1);
    }

        thread t1(multiply_right_transposed_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 0, 4, ref(b1));
        thread t2(multiply_right_transposed_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 1, 4, ref(b2));
        thread t3(multiply_right_transposed_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 2, 4, ref(b3));
        thread t4(multiply_right_transposed_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 3, 4, ref(b4));

        t1.detach();
        t2.detach();
        t3.detach();
        t4.detach();

        //multiply_right_transposed(a, b);
    }
    else {
        if(a.cols != b.rows) {
            cout << "\nMatrix a.cols != b.rows";
            cout << "\na.rows:" << a.rows << " a.cols:" << a.cols << " b.rows:" << b.rows << " b.cols:" << b.cols;
            exit(1);
        }

        thread t1(multiply_standart_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 0, 4, ref(b1));
        thread t2(multiply_standart_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 1, 4, ref(b2));
        thread t3(multiply_standart_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 2, 4, ref(b3));
        thread t4(multiply_standart_thread, ref(get_matrix_val()), ref(a.get_matrix_val()), ref(b.get_matrix_val()), 3, 4, ref(b4));

        t1.detach();
        t2.detach();
        t3.detach();
        t4.detach();
        //thread t2(a, b, 1, 4, b2).detach();
        //thread t3(a, b, 2, 4, b3).detach();
        //thread t4(a, b, 3, 4, b4).detach();
        //multiply_standart(a, b);
    }

    while(b1 || b2 || b3 || b4) {
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    //cout << "\nb1 " << b1 << " b2 " << b2 << " b3 " << b3 << " b4 " << b4;
}*/

//------------------------------------------------------------------------------

__global__ void multiply_standart_kernel(double *a, double *b, double *c, int a_rows, int b_cols, int ab_rows_cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < a_rows && col < b_cols) {
		double sum = 0.0;
		for(int i = 0; i < ab_rows_cols; ++i) {
			sum += a[row * ab_rows_cols + i] * b[i * b_cols + col];
		}
		c[row * b_cols + col] = sum;
	}
}

__global__ void transpose_kernel(double* mat_in, double* mat_out, unsigned rows, unsigned cols)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < cols && idy < rows)
    {
        unsigned int pos = idy * cols + idx;
        unsigned int trans_pos = idx * rows + idy;
        mat_out[trans_pos] = mat_in[pos];
    }
}

/*__global__ void multiply_left_transposed_kernel(double *a, double *b, double *c, int a_rows, int b_cols, int ab_rows_cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < ab_rows_cols && col < b_cols) {
		double sum = 0.0;
		for(int i = 0; i < a_rows; ++i) {
			sum += a[i * a_rows + row] * b[i * b_cols + col];
		}
		c[row * b_cols + col] = sum;
	}
}

__global__ void multiply_right_transposed_kernel(double *a, double *b, double *c, int a_rows, int b_cols, int ab_rows_cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < a_rows && col < ab_rows_cols) {
		double sum = 0.0;
		for(int i = 0; i < b_cols; ++i) {
			sum += a[row * b_cols + i] * b[col * ab_rows_cols + i];
		}
		c[row * ab_rows_cols + col] = sum;
	}
}*/

__global__ void sum_kernel(double *a, double *b, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		a[row * cols + col] += b[row * cols + col];
	}
}

__global__ void multiply_like_sum_kernel(double *a, double *b, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		a[row * cols + col] *= b[row * cols + col];
	}
}

__global__ void multiply_scalar_kernel(double *a, double b, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		a[row * cols + col] *= b;
	}
}

__global__ void softmax_kernel(double *a, double *b, /*double regularization_sum,*/ unsigned length) {
	double max_val = 0.0;
	for(unsigned i = 0; i < length; ++i) {
		max_val = max(b[i], max_val);
	}

	double sum_softmax = 0.0;
	for(unsigned i = 0; i < length; ++i) {
		a[i] = exp(b[i] - max_val);
		sum_softmax += a[i];
	}

	for(unsigned i = 0; i < length; ++i) {
		a[i] /= sum_softmax;
		//a[i] += regularization_sum;
	}
}

__global__ void relu_kernel(double *a, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		if(a[row * cols + col] < 0.0)
		    a[row * cols + col] = 0.0;
	}
}

__global__ void gradient_relu_prime_kernel(double *a, double *b, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		if(b[row * cols + col] < 0.0)
		    a[row * cols + col] *= 0.0;
		else
			a[row * cols + col] *= 1.0;
	}
}

__global__ void elu_kernel(double *a, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		if(a[row * cols + col] < 0.0)
		    a[row * cols + col] = 0.0005 * (exp(a[row * cols + col]) - 1.0);
		else
			a[row * cols + col] = 0.005 * a[row * cols + col];
	}
}

__global__ void gradient_elu_prime_kernel(double *a, double *b, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		if(b[row * cols + col] < 0.0)
		    a[row * cols + col] *= 0.0005 * exp(b[row * cols + col]);
		else
			a[row * cols + col] *= 1.0;//Wrong derivative but it works better then true
	}
}

__global__ void gradient_last_kernel(double *a, double *b, int target_numb, /*double regularization_sum,*/ int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		if((row*cols + col) != target_numb)
		    a[row * cols + col] = 0.0 - b[row * cols + col];// + regularization_sum;
		else
			a[row * cols + col] = 1.0 - b[row * cols + col];// + regularization_sum;
	}
}

__global__ void regularization_kernel(double *a, double * regularization_sum, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		regularization_sum[0] += a[row * cols + col] * a[row * cols + col];
	}
}

__global__ void regularization_gradient_kernel(double *a, double * regularization_sum, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		regularization_sum[0] += a[row * cols + col];
	}
}

__global__ void regularization_sum_kernel(double *a, double d, int rows, int cols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		a[row*cols + col] += d;
	}
}

void Matrix::operator+=(double d) {
	if(is_gpu) {
		dim3 grid_block((cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);
		dim3 grid_thread(block_size, block_size);
		regularization_sum_kernel<<<grid_block, grid_thread>>>(gpu_matrix_val, d, rows, cols);
	}
	else {
		for(unsigned i = 0; i < length; ++i) {
			matrix_val[i] += d;
		}
	}
}

/*__global__ void random_shuffle_kernel(double *a, int rows, int cols, unsigned length) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row < rows && col < cols) {
		curandState state;
		curand_init(clock64(), row*cols+col, 0, &state);

		unsigned j = curand_uniform(&state) * (length-1);
		unsigned i = row*cols+col;
		double temp = a[j];
		__syncrothreads();
		a[i] = temp;
	}
}*/

void Matrix::random_shuffle() {
	for(unsigned i = length - 1; i > 0; --i) {
		unsigned j = get_random_numb() * (length-1);
		double temp = matrix_val[j];
		matrix_val[j] = matrix_val[i];
		matrix_val[i] = temp;
	}
}

void Matrix::random_sort() {

	if(is_gpu) {
		CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
		random_shuffle();
		CUDA_CHECK_RETURN(cudaMemcpy(gpu_matrix_val, &matrix_val[0], sizeof(double) * length, cudaMemcpyHostToDevice));
	}
	else {
		random_shuffle();
	}

}

double Matrix::regularization_count(double regularization_rate) {
    double regularization_weight = 0.0;

    if(is_gpu) {
    	double rs_cpu = 0.0;
    	double * rs;
    	CUDA_CHECK_RETURN(cudaMalloc((void**)&rs, sizeof(double)));
    	CUDA_CHECK_RETURN(cudaMemcpy(rs, &rs_cpu, sizeof(double), cudaMemcpyHostToDevice));

    	dim3 grid_block((cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);
    	dim3 grid_thread(block_size, block_size);
    	regularization_kernel<<<grid_block, grid_thread>>>(gpu_matrix_val, rs, rows, cols);

    	CUDA_CHECK_RETURN(cudaMemcpy(&rs_cpu, rs, sizeof(double), cudaMemcpyDeviceToHost));
    	regularization_weight = (rs_cpu * regularization_rate);
    	CUDA_CHECK_RETURN(cudaFree(rs));
    }
    else {
    	for(unsigned i = 0; i < length; ++i) {
    	    regularization_weight += /*regularization_rate */ (matrix_val[i] * matrix_val[i]);
    	}
    	regularization_weight *= regularization_rate;
    }

    return (regularization_weight / double(length)) / 2.0;
}

double Matrix::regularization_gradient_count(double regularization_rate) {
    double regularization_weight = 0.0;

    if(is_gpu) {
    	double rs_cpu = 0.0;
    	double * rs;
    	CUDA_CHECK_RETURN(cudaMalloc((void**)&rs, sizeof(double)));
    	CUDA_CHECK_RETURN(cudaMemcpy(rs, &rs_cpu, sizeof(double), cudaMemcpyHostToDevice));

    	dim3 grid_block((cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);
    	dim3 grid_thread(block_size, block_size);
    	regularization_gradient_kernel<<<grid_block, grid_thread>>>(gpu_matrix_val, rs, rows, cols);

    	CUDA_CHECK_RETURN(cudaMemcpy(&rs_cpu, rs, sizeof(double), cudaMemcpyDeviceToHost));
    	regularization_weight = rs_cpu;
    	CUDA_CHECK_RETURN(cudaFree(rs));
    }
    else {
    	for(unsigned i = 0; i < length; ++i) {
    	    regularization_weight += /*regularization_rate */ matrix_val[i];
    	}
    }
    regularization_weight /= double(length);
    regularization_weight *= regularization_rate;

    return regularization_weight;
}

void Matrix::transpose() {
	is_transposed = !is_transposed;

	if(is_gpu) {
		Matrix old_matrix;
		old_matrix.init(*this);

	   	dim3 grid_block((cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);
	   	dim3 grid_thread(block_size, block_size);
	   	transpose_kernel<<<grid_block, grid_thread>>>(old_matrix.gpu_matrix_val, gpu_matrix_val, rows, cols);
	   	rows = old_matrix.cols;
	   	cols = old_matrix.rows;
	   	//CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
	}
}

void Matrix::multiply_standart(const Matrix& a, const Matrix& b) {
    if(a.cols != b.rows) {
        cout << "\nMatrix a.cols != b.rows";
        cout << "\na.rows:" << a.rows << " a.cols:" << a.cols << " b.rows:" << b.rows << " b.cols:" << b.cols;
        exit(1);
    }

    for(unsigned r = 0; r < a.rows; ++r) {
        for(unsigned mc = 0; mc < b.cols; ++mc) {
            for(unsigned mr = 0; mr < b.rows; ++mr) {
                matrix_val[r*cols + mc] += a.matrix_val[r*a.cols + mr] * b.matrix_val[mr*b.cols + mc];
            }
        }
    }
}

void Matrix::multiply_left_transposed(const Matrix& a, const Matrix& b) {
    if(a.rows != b.rows) {
        cout << "\nMatrix a.rows != b.rows";
        cout << "\na.rows:" << a.rows << " a.cols:" << a.cols << " b.rows:" << b.rows << " b.cols:" << b.cols;
        exit(1);
    }

    for(unsigned r = 0; r < a.cols; ++r) {
        for(unsigned mc = 0; mc < b.cols; ++mc) {
            for(unsigned mr = 0; mr < b.rows; ++mr) {
                matrix_val[r*cols + mc] += a.matrix_val[mr*a.cols + r] * b.matrix_val[mr*b.cols + mc];
            }
        }
    }
}

void Matrix::multiply_right_transposed(const Matrix& a, const Matrix& b) {
    if(a.cols != b.cols) {
        cout << "\nMatrix a.cols != b.cols";
        cout << "\na.rows:" << a.rows << " a.cols:" << a.cols << " b.rows:" << b.rows << " b.cols:" << b.cols;
        exit(1);
    }

    for(unsigned r = 0; r < a.rows; ++r) {
        for(unsigned mc = 0; mc < b.rows; ++mc) {
            for(unsigned mr = 0; mr < b.cols; ++mr) {
                matrix_val[r*cols + mc] += a.matrix_val[r*a.cols + mr] * b.matrix_val[mc*b.cols + mr];
            }
        }
    }
}

/*double Matrix::regularization_count(unsigned regularization_size, double regularization_rate) {
    double regularization_weight = 0.0;
    regularization_size *= 2;
    for(unsigned r = 0; r < rows; ++r) {
        for(unsigned c = 0; c < cols; ++c) {
            regularization_weight += regularization_rate * (matrix_val[r*cols + c] * matrix_val[r*cols + c]) / double(regularization_size);
        }
    }
    return regularization_weight;
}*/

void Matrix::multiply(Matrix & a, Matrix & b) {

	if(is_gpu) {
	   	dim3 grid_block((b.cols + block_size - 1) / block_size, (a.rows + block_size - 1) / block_size);
	   	dim3 grid_thread(block_size, block_size);
	   	multiply_standart_kernel<<<grid_block, grid_thread>>>(a.gpu_matrix_val, b.gpu_matrix_val, gpu_matrix_val, a.rows, b.cols, a.cols);
	   	//CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
	}
	else {
		if(a.is_transposed)
			multiply_left_transposed(a, b);
		else if(b.is_transposed)
			multiply_right_transposed(a, b);
		else
			multiply_standart(a, b);
	}
}

void Matrix::multiply(const Matrix & m) {
    Matrix old_matrix;
    old_matrix.init(*this);

    if(is_gpu) {
    	   	dim3 grid_block((m.cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);
    	   	dim3 grid_thread(block_size, block_size);
    	   	multiply_standart_kernel<<<grid_block, grid_thread>>>(old_matrix.gpu_matrix_val, m.gpu_matrix_val, gpu_matrix_val, rows, m.cols, cols);
    	   	//CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
    	}
    	else {
    		if(is_transposed)
    			multiply_left_transposed(old_matrix, m);
    		else if(m.is_transposed)
    			multiply_right_transposed(old_matrix, m);
    		else
    			multiply_standart(old_matrix, m);
    	}
}

void Matrix::multiply(double d) {
	if(is_gpu) {
	    dim3 grid_block((cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);
	    dim3 grid_thread(block_size, block_size);
	    multiply_scalar_kernel<<<grid_block, grid_thread>>>(gpu_matrix_val, d, rows, cols);
	    //CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
	}
	else {
		for(unsigned r = 0; r < rows; ++r) {
		    for(unsigned c = 0; c < cols; ++c) {
		        matrix_val[r*cols + c] *= d;
		    }
		}
	}
}

void Matrix::ziro_matrix() {
	if(!is_gpu) {
		for(unsigned r = 0; r < rows; ++r) {
		    for(unsigned c = 0; c < cols; ++c) {
		        matrix_val[r*cols + c] = 0.0;
		    }
		}
	}
}

void Matrix::softmax_calculate(Matrix & m/*, double regularization_sum*/) {
	if(cols != m.cols) {
	    cout << "\nSoftmax cols != v.size()";
	    exit(1);
	    }

	if(is_gpu) {
	   	softmax_kernel<<<1, 1>>>(gpu_matrix_val, m.gpu_matrix_val, /*regularization_sum,*/ m.length);
	}
	else {
		double sum_values_exit = 0.0;

		double max_val = 0.0;
		for(auto v : m.matrix_val) {
		    max_val = max(max_val, v);
		}

		for(auto v : m.matrix_val)
		    sum_values_exit += exp(v - max_val);

		for(unsigned i = 0; i < m.matrix_val.size(); ++i) {
		    matrix_val[i] = (exp(m.matrix_val[i] - max_val) / sum_values_exit)/* + regularization_sum*/;
		}
	}
}

void Matrix::last_gradient_calculate(Matrix & m, vector<double> & output_vals/*, double regularization_sum*/) {
	if(cols != m.cols) {
	    cout << "\nLastGrad cols != v.size()";
	    exit(1);
	}

	if(is_gpu) {

		unsigned target_numb = 0;
		for(unsigned i = 0; i < output_vals.size(); ++i) {
		    if(output_vals[i] == 1.0) {
		        target_numb = i;
		        break;
		    }
		}

	   	dim3 grid_block((cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);
	    dim3 grid_thread(block_size, block_size);
	    gradient_last_kernel<<<grid_block, grid_thread>>>(gpu_matrix_val, m.gpu_matrix_val, target_numb, /*regularization_sum, */rows, cols);
	    //CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
	}
	else {
		for(unsigned i = 0; i < matrix_val.size(); ++i) {
		    matrix_val[i] = (output_vals[i] - m.matrix_val[i]);// + regularization_sum;
		}
	}
}

void Matrix::hidden_gradient_calculate(Matrix & m, unsigned act) {
	if(cols != m.cols) {
	    cout << "\nHiddenGrad cols != v.size()";
	    exit(1);
	}

	if(is_gpu) {
	    dim3 grid_block((cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);
	    dim3 grid_thread(block_size, block_size);
	    if(act == 3)
	        gradient_elu_prime_kernel<<<grid_block, grid_thread>>>(gpu_matrix_val, m.gpu_matrix_val, rows, cols);
	    else if(act == 2)
	    	gradient_relu_prime_kernel<<<grid_block, grid_thread>>>(gpu_matrix_val, m.gpu_matrix_val, rows, cols);
		//CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
	}
	else {
		function<double(double&)> fn;
		if(act == 3)
			fn = [](double val) { return val > 0 ? 1.0 : 0.1 * exp(val); };
		else if (act == 2)
			fn = [](double val) { return val > 0 ? 1.0 : 0.0; };
		//auto sigmoid_prime = [](double val) { return val * (1.0 - val); };
		//auto relu_prime = [](double val) { return val > 0 ? 1.0 : 0.0; };
		//auto elu_prime = [](double val) { return val > 0 ? 1.0 : 0.1 * exp(val); };

		//1:sigmoid_prime 2:relu_prime 3:elu_prime

		for(unsigned a = 0; a < matrix_val.size(); ++a)
		    matrix_val[a] *= fn(m.matrix_val[a]);
	}
}

void Matrix::activation(unsigned act) {
	if(is_gpu) {
	    dim3 grid_block((cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);
	    dim3 grid_thread(block_size, block_size);
	    if(act == 3)
	        elu_kernel<<<grid_block, grid_thread>>>(gpu_matrix_val, rows, cols);
	    if(act == 2)
	        relu_kernel<<<grid_block, grid_thread>>>(gpu_matrix_val, rows, cols);
	    //CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
	}
	else {
		//auto sigmoid = [](double & val) { val = 1.0 / (1.0 + exp(-val)); };
		//auto relu = [](double & val) { val = max(0.0, val); };
		//auto elu = [](double & val) { return val = (val > 0 ? val : 0.1 * (exp(val) - 1.0)); };

		//1:sigmoid 2:relu 3:elu

		function<void(double&)> fn;
		if(act == 1)
			fn = [](double & val) { val = 1.0 / (1.0 + exp(-val)); };
		else if(act == 2)
			fn = [](double & val) { val = max(0.0, val); };
		else if(act == 3)
			fn = [](double & val) { return val = (val > 0 ? 0.1*val : 0.1 * (exp(val) - 1.0)); };

		for(unsigned r = 0; r < rows; ++r) {
		    for(unsigned c = 0; c < cols; ++c) {
		        fn(matrix_val[r*cols + c]);
		    }
		}
	}
}

void Matrix::sum(Matrix & a, Matrix & b) {
    if(a.cols != b.cols && a.rows != b.rows) {
        cout << "\nMatrix cols != m.cols && rows != m.rows";
        exit(1);
    }

    if(is_gpu) {
        dim3 grid_block((a.cols + block_size - 1) / block_size, (b.rows + block_size - 1) / block_size);
        dim3 grid_thread(block_size, block_size);
        sum_kernel<<<grid_block, grid_thread>>>(a.gpu_matrix_val, b.gpu_matrix_val, a.rows, a.cols);
        //CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
    }
    else {
    	for(unsigned r = 0; r < a.rows; ++r) {
    	    for(unsigned c = 0; c < a.cols; ++c) {
    	        a.matrix_val[r*a.cols + c] += b.matrix_val[r*b.cols + c];
    	    }
    	}
    }
}

void Matrix::multiply_like_sum(Matrix & m) {
    if(cols != m.cols && rows != m.rows) {
        cout << "\nMatrix cols != m.cols && rows != m.rows";
        exit(1);
    }

    if(is_gpu) {
        dim3 grid_block((cols + block_size - 1) / block_size, (m.rows + block_size - 1) / block_size);
        dim3 grid_thread(block_size, block_size);
        multiply_like_sum_kernel<<<grid_block, grid_thread>>>(gpu_matrix_val, m.gpu_matrix_val, rows, cols);
        //CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
    }
    else {
    	for(unsigned r = 0; r < rows; ++r) {
    	    for(unsigned c = 0; c < cols; ++c) {
    	        matrix_val[r*cols + c] *= m.matrix_val[r*cols + c];
    	    }
    	}
    }
}

void Matrix::init(vector<double> & v) {
    if(is_gpu) {
    	CUDA_CHECK_RETURN(cudaMemcpy(gpu_matrix_val, &v[0], sizeof(double) * length, cudaMemcpyHostToDevice));
    }
    else {
    	matrix_val = v;
    }
}

void Matrix::operator=(Matrix & m) {
	rows = m.rows;
	cols = m.cols;
	length = m.length;
	if(is_gpu) {
		//CUDA_CHECK_RETURN(cudaMalloc((void**)&gpu_matrix_val, sizeof(double) * length));
		CUDA_CHECK_RETURN(cudaMemcpy(gpu_matrix_val, m.gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToDevice));
	}
	else {
		matrix_val = m.matrix_val;
	}
}

vector<double> & Matrix::get_matrix_val() {
	if(is_gpu) {
		matrix_val.resize(length);
		CUDA_CHECK_RETURN(cudaMemcpy(&matrix_val[0], gpu_matrix_val, sizeof(double) * length, cudaMemcpyDeviceToHost));
	}

	return matrix_val;
}

static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
