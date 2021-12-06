/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   NeuralNetwork.h
 * Author: chelovek
 *
 * Created on 31 марта 2019 г., 20:43
 */

#include <vector>
#include "Matrix.h"

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<unsigned> & topology_, double learning_rate_ = 0.7, double momentum_ = 0.3, double regularization_rate_ = 0.0,
    		bool gpu_compute = false, double dropout_rate_ = 0.0);
    NeuralNetwork(std::string file_name, bool gpu_compute = false);
    NeuralNetwork(const NeuralNetwork& orig);
    void create_topology(std::vector<unsigned> topology_, bool random_values = true);
    virtual ~NeuralNetwork();

    void set_data(std::vector<std::vector<double> > & data);

    void neurons_ziro();
    void feed_forward(unsigned numb);
    void back_propagation(std::vector<double> & output_vals, unsigned epoch, unsigned numb);
    void get_results(std::vector<double> &result_vals);
    inline double get_error() { return main_error; };

    void save_topology();

    void set_dropout(double neurons_kill, unsigned layer_numb);
    void dropout_sort();
    void dropout_ziro();

private:
    std::vector<Matrix>   neurons;
    std::vector<Matrix>   dropout;
    std::vector<Matrix>   biases;
    std::vector<Matrix>   biases_delta;
    std::vector<Matrix>   gradients;
    std::vector<Matrix>   temporary;
    std::vector<Matrix>   weights;
    std::vector<Matrix>   weights_delta;
    std::vector<Matrix>   memory_neurons;
    std::vector<Matrix>   memory_weights;
    std::vector<Matrix>   memory_weights_delta;
    std::vector<Matrix>   work_data;
    Matrix                softmax;
    double                main_error;
    double                learning_rate;
    double                momentum;
    double                lambda;
    std::vector<double>   regularization_sum;
    double                regularization_rate;
    double                dropout_rate;
    bool                  is_gpu = false;
    bool                  is_dropout = false;

};

#endif /* NEURALNETWORK_H */

