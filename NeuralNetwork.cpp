/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   NeuralNetwork.cpp
 * Author: chelovek
 *
 * Created on 31 марта 2019 г., 20:43
 */

#include <iostream>
#include "NeuralNetwork.h"
#include "CSVWriter.h"

#include <fstream>
#include <sstream>
#include <string>


using namespace std;

void logn(std::string str) {
    std::ofstream fout("log.txt", std::ios_base::app);
    fout << str;
    fout.close();
}

void logn_new() {
    std::ofstream fout("log.txt", std::ios_base::trunc);
    fout.close();
}

void NeuralNetwork::create_topology(std::vector<unsigned> topology_, bool random_values) {
    unsigned tsize = topology_.size();

    neurons.reserve(tsize);
    biases.reserve(tsize - 1);
    biases_delta.reserve(tsize - 1);
    gradients.reserve(tsize - 1);
    temporary.reserve(tsize - 1);
    weights.reserve(tsize - 1);
    weights_delta.reserve(tsize - 1);
    memory_neurons.reserve(tsize - 2);
    dropout.reserve(tsize - 2);

    for(unsigned i = 0; i < tsize; ++i) {
        unsigned rows = topology_[i];
        unsigned cols = 1;

        neurons.emplace_back(1, topology_[i], false, is_gpu);

        if(i + 1 < tsize) {
            cols = topology_[i + 1];
            gradients.emplace_back(1, topology_[i+1], false, is_gpu);
            biases.emplace_back(1, topology_[i+1], random_values, is_gpu);
            biases_delta.emplace_back(1, topology_[i+1], false, is_gpu);
            temporary.emplace_back(rows, cols, false, is_gpu);
            weights.emplace_back(rows, cols, random_values, is_gpu);
            weights_delta.emplace_back(rows, cols, false, is_gpu);
            if(i != 0)
            	dropout.emplace_back(1, topology_[i], false, is_gpu, dropout_rate);
        }
    }

    softmax.init(1, topology_.back(), is_gpu);

    regularization_sum.resize(tsize - 1);
    //softmax_prime.resize(topology_.back());
    //cross_entropy.resize(topology_.back());
    //cross_entropy_prime.resize(topology_.back());
}

void NeuralNetwork::set_data(std::vector<std::vector<double> > & data) {
	work_data.reserve(data.size());
	for(unsigned i = 0; i < data.size(); ++i) {
		work_data.emplace_back(1, data[i].size(), false, is_gpu);
		work_data[i].init(data[i]);
	}
}

NeuralNetwork::NeuralNetwork(string file_name, bool gpu_compute) {

	is_gpu = gpu_compute;

    cout << "\nNetworkWasStarted";
    cout << endl;

    vector<unsigned> topology;
    ifstream file(file_name);
    if(file) {
        bool weight_step = true;
        unsigned layer_weight_bias = 0;
        vector<double> vals;
        unsigned step = 0;
        string line;
        //line.reserve(2000000000);
        string name_for_value;
        while(getline(file, line)) {
            string str;
            if(step == 0) {
                for(unsigned i = 0; i < line.length(); ++i) {
                    bool record = false;
                    if(line[i] != ',') {
                        str += line[i];
                    }
                    else
                    	record = true;
                    if(record || i == line.length() - 1) {
                      if(str == "neurons_size" || str == "learning_rate"
                      || str == "momentum" || str == "regularization_rate") {
                          name_for_value = str;
                          str = "";
                      }
                        else {
                            double value;
                            stringstream vss;
                            vss << str;
                            vss >> value;
                            if(name_for_value == "neurons_size")
                                topology.push_back(unsigned(value));
                            else if(name_for_value == "learning_rate")
                                learning_rate = value;
                            else if(name_for_value == "momentum")
                                momentum = value;
                            else if(name_for_value == "regularization_rate")
                                regularization_rate = value;

                            str = "";
                        }
                    }
                }
                create_topology(topology, false);
                vals.reserve(topology[0]*topology[1]);
                cout << endl;
            }
            else {
                for(unsigned i = 0; i < line.length(); ++i) {
                    bool record = false;
                	if(line[i] != ',') {
                        str += line[i];
                    }
                	else
                		record = true;

                    if(record || i == line.length() - 1) {
                        double value;
                        stringstream vss;
                        vss << str;
                        vss >> value;
                        vals.push_back(value);
                        str = "";
                        if(weight_step) {
                        	if(topology[layer_weight_bias] * topology[layer_weight_bias+1] <= vals.size()) {
                        		weights[layer_weight_bias].init(vals);
                        		vals.clear();
                        		weight_step = false;
                        	}
                        }
                        else {
                        	if(topology[layer_weight_bias+1] <= vals.size()) {
                        		biases[layer_weight_bias].init(vals);
                        		weight_step = true;
                        		vals.clear();
                        		++layer_weight_bias;
                        	}
                        }
                    }
                }
            }
            ++step;
            line.clear();
        }
    }
    else {
        cout << "\nFileNotFound";
        cout << endl;
    }
    file.close();

    cout << "\nNetworkWasEnded";
    cout << endl;

}

NeuralNetwork::NeuralNetwork(const vector<unsigned> & topology_, double learning_rate_, double momentum_, double regularization_rate_, bool gpu_compute, double dropout_rate_) {
    dropout_rate = dropout_rate_;
	learning_rate = learning_rate_;
    momentum = momentum_;
    regularization_rate = regularization_rate_;

    is_gpu = gpu_compute;

    if(dropout_rate > 0.0)
    	is_dropout = true;

    create_topology(topology_);
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& orig) {
}

NeuralNetwork::~NeuralNetwork() {
}

void NeuralNetwork::neurons_ziro() {
    for(auto &m : memory_neurons)
        m.ziro_matrix();
}

void NeuralNetwork::dropout_sort() {
    for(auto &d : dropout) {
    	d.random_sort();
    }
}

void NeuralNetwork::set_dropout(double neurons_kill, unsigned layer_numb) {
	//
}

void NeuralNetwork::dropout_ziro() {
	is_dropout = false;
}

void NeuralNetwork::feed_forward(unsigned numb) {
    //auto sigmoid = [](double & val) { val = 1.0 / (1.0 + exp(-val)); };
    //auto relu = [](double & val) { val = max(0.0, val); };
    //auto elu = [](double & val) { return val = (val > 0 ? val : 0.1 * (exp(val) - 1.0)); };

    //1:sigmoid 2:relu 3:elu
    unsigned act = 3;

    for(unsigned i = 0; i < neurons.size() - 1; ++i) {

    	regularization_sum[i] = 0.0;
    	if(regularization_rate != 0.0)
    	    regularization_sum[i] = weights[i].regularization_count(regularization_rate);

    	neurons[i+1].ziro_matrix();
    	if(i == 0)
            neurons[i+1].multiply(work_data[numb], weights[i]);
    	else
    		neurons[i+1].multiply(neurons[i], weights[i]);
        neurons[i+1] += biases[i];
        neurons[i+1] += regularization_sum[i];

        if(i != 0 && is_dropout) {
        	neurons[i].multiply_like_sum(dropout[i-1]);
        }

        if(i+1 < neurons.size()-1)
            neurons[i+1].activation(act);
    }

    softmax.softmax_calculate(neurons.back()/*, regularization_sum.back()*/);
}

void NeuralNetwork::back_propagation(vector<double> & output_vals, unsigned epoch, unsigned numb) {

	gradients.back().last_gradient_calculate(softmax, output_vals/*, regularization_sum*/);

	//auto sigmoid_prime = [](double val) { return val * (1.0 - val); };
    //auto relu_prime = [](double val) { return val > 0 ? 1.0 : 0.0; };
	//auto elu_prime = [](double val) { return val > 0 ? 1.0 : 0.1 * (exp(val) - 1.0); };

    for(unsigned now_layer = neurons.size() - 1; now_layer > 0; --now_layer) {
        unsigned prev_layer            = now_layer - 1;
        unsigned now_layer_bias_weight = now_layer - 1; //Weights and biases layers size = neurons layers size - 1
        unsigned now_layer_gradient    = now_layer - 1; //Gradients layers size          = neurons layers size - 1
        int      prev_layer_gradient   = now_layer - 2;

        weights[now_layer_bias_weight].transpose();

        if(prev_layer_gradient >= 0) {

        	gradients[prev_layer_gradient].ziro_matrix();
            gradients[prev_layer_gradient].multiply(gradients[now_layer_gradient], weights[now_layer_bias_weight]);

            //1:sigmoid_prime 2:relu_prime 3:elu_prime
            //unsigned act = 3;
            unsigned act = 3;
            if(prev_layer == 0)
                gradients[prev_layer_gradient].hidden_gradient_calculate(work_data[numb], act);
            else
            	gradients[prev_layer_gradient].hidden_gradient_calculate(neurons[prev_layer], act);
            if(prev_layer_gradient < neurons.size() - 1 && prev_layer_gradient != 0 && is_dropout)
            	gradients[prev_layer_gradient].multiply_like_sum(dropout[prev_layer_gradient-1]);
        }

        //Compute new weights
        weights[now_layer_bias_weight].transpose();
        if(prev_layer == 0)
            work_data[numb].transpose();
        else
        	neurons[prev_layer].transpose();

        temporary[now_layer_bias_weight].ziro_matrix();

        if(prev_layer == 0)
            temporary[now_layer_bias_weight].multiply(work_data[numb], gradients[now_layer_gradient]);
        else
        	temporary[now_layer_bias_weight].multiply(neurons[prev_layer], gradients[now_layer_gradient]);

        temporary[now_layer_bias_weight].multiply(learning_rate);
        weights_delta[now_layer_bias_weight].multiply(momentum);
        weights_delta[now_layer_bias_weight] += temporary[now_layer_bias_weight];
        //Regularization
        temporary[now_layer_bias_weight] = weights[now_layer_bias_weight];
        //Not regularization
        weights[now_layer_bias_weight] += weights_delta[now_layer_bias_weight];

        //Regularization
        regularization_sum[now_layer_bias_weight] = 0.0;
        if(regularization_rate != 0.0) {
        	temporary[now_layer_bias_weight].multiply(-learning_rate*regularization_rate);
        	weights[now_layer_bias_weight] += temporary[now_layer_bias_weight];
        }

        if(prev_layer == 0)
            work_data[numb].transpose();
        else
        	neurons[prev_layer].transpose();

        //Compute new biases
        gradients[now_layer_gradient].multiply(learning_rate);
        biases_delta[now_layer_bias_weight].multiply(momentum);
        biases_delta[now_layer_bias_weight] += gradients[now_layer_gradient];
        biases[now_layer_bias_weight] += biases_delta[now_layer_bias_weight];
    }
}

void NeuralNetwork::get_results(vector<double> &result_vals) {
    result_vals.clear();
    result_vals = softmax.get_matrix_val();
}


void NeuralNetwork::save_topology() {
    vector<string> str;
    vector<double> val;

    for(unsigned i = 0; i < neurons.size(); ++i) {
        str.push_back("neurons_size");
        val.push_back(neurons[i].get_cols());
    }

    str.push_back("learning_rate");
    val.push_back(learning_rate);

    str.push_back("momentum");
    val.push_back(momentum);

    str.push_back("regularization_rate");
    val.push_back(regularization_rate);

    CSVWriter csv("topology.csv");
    csv.write(str, val);

    for(unsigned i = 0; i < weights.size(); ++i) {
        csv.write(weights[i].get_matrix_val(), i);
        csv.write(biases[i].get_matrix_val());
    }
}
