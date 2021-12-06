#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include "Matrix.h"
#include "NeuralNetwork.h"
#include "CSVReader.h"

#include <random>
#include <algorithm>

using namespace std;

void write_test(vector<unsigned> &results) {
	std::ofstream fout("submission.txt", std::ios_base::app);
	fout << "ImageId,Label";
	for(unsigned i = 0; i < results.size(); ++i) {
		fout << endl << (i+1) << "," << results[i];
	}
	fout.close();
}

void log(std::string str) {
    std::ofstream fout("log.txt", std::ios_base::app);
    fout << str;
    fout.close();
}

void log_new() {
    std::ofstream fout("log.txt", std::ios_base::trunc);
    fout.close();
}

void random_sort(vector<unsigned> & v) {
	std::random_device rd;
    std::mt19937 g(rd());
	std::shuffle(v.begin(), v.end(), g);
}

int main(int argc, char** argv) {
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    unsigned long hardware_threads = std::thread::hardware_concurrency();
    cout << "\nHardware concurrency:" << hardware_threads << endl;

    CSVReader reader("data/train.csv", true);

    cout << "\nSize " << reader.get_rows_size();
    cout << "\nLength " << reader.get_data_size();
    cout << endl;

    double last_max_percent = 0.0;

    vector<double> result_vals;
    vector<double> output = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    vector<unsigned> topology;
    topology.push_back(reader.get_rows_size());
    topology.push_back(800);
    topology.push_back(10);
    double learning_rate       = 0.1;
    double momentum            = 0.7;
    double regularization_rate = 0.000005;
    bool gpu_compute           = true;
    double dropout             = 0.05;
    NeuralNetwork net(topology, learning_rate, momentum, regularization_rate, gpu_compute, dropout);

    net.set_data(reader.get_all_data());

    unsigned length_epoch = 60;
    unsigned length = reader.get_data_size() - 12000;

    std::vector<unsigned> id_data;
    id_data.reserve(length);
    for(unsigned i = 0; i < length; ++i) {
    	id_data.push_back(i);
    }

    for (unsigned i=0 ; i<length_epoch; ++i) {
        double current_error = 0.0;
        double count_error = 0.0;

        net.dropout_sort();
        random_sort(id_data);

        cout << "\nEpoch " << i+1 << "/" << length_epoch << endl;
        for (unsigned j=0 ; j<length ; ++j) {

        	net.feed_forward(id_data[j]);
            net.get_results(result_vals);

            unsigned numb = reader.get_output_data(id_data[j]);
            output[numb] = 1.0;

            net.back_propagation(output, i+1, id_data[j]);
            current_error += net.get_error();
            count_error += 1.0;

            output[numb] = 0.0;
        }

        current_error /= count_error;
        cout << "\nTrain_Error: " << net.get_error();

        unsigned correct = 0;
        unsigned length_check_start = 30000;
        unsigned length_check_end   = 42000;
        for (unsigned r=length_check_start; r<length_check_end; ++r){
        	net.feed_forward(r);
            net.get_results(result_vals);

            unsigned numb = reader.get_output_data(r);
            output[numb] = 1.0;

            double max_val = -1.0;
            unsigned max_val_numb = 0;
            for (unsigned j=0 ; j<10 ; ++j){
                if(max_val < result_vals[j]) {
                    max_val      = result_vals[j];
                    max_val_numb = j;
                }
            }

            if(max_val_numb == numb) {
                ++correct;
            }

            output[numb] = 0.0;
        }
        double percent = (double(correct) / (double(length_check_end - length_check_start))) * 100.0;
        cout << "\nTest_Error " << percent << "%";

        if(percent > last_max_percent) {
        	last_max_percent = percent;
        	net.save_topology();
        }

    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "\n Time work " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

    return 0;
}

