
#include <iostream>

#include "CSVWriter.h"

using namespace std;

CSVWriter::CSVWriter(std::string file_name) {
    fout.open(file_name);
}

CSVWriter::~CSVWriter() {
    fout.close();
}

void CSVWriter::write(vector<string> & parametr_names, vector<double> & parametr_sizes) {
    if(parametr_names.size() != parametr_sizes.size()) {
        cout << "CSVWriter parametr_names != parametr_sizes";
        cout << endl;
        exit(1);
    }

    cout << "\nWriter was started";

    for(unsigned i = 0; i < parametr_names.size(); ++i) {

    	if(parametr_names[i] == "neurons_size")
    		neuron_layer_size.push_back(parametr_sizes[i]);

        fout << parametr_names[i] << "," << parametr_sizes[i];
        if(parametr_names.size() - 1 != i)
            fout << ",";
    }
}


void CSVWriter::write(const vector<double> & parametrs, int numb) {
    fout << endl;
    unsigned endline = parametrs.size();
    if(numb >= 0) {
    	endline = neuron_layer_size[numb+1];
    }

    double length = double(parametrs.size());
    double math_middle        = 0.0;
    double math_middle_square = 0.0;
    double max_val            = 0.0;
    double min_val            = 0.0;
    double disperse           = 0.0;

    unsigned now_numb = 0;
    unsigned now_numb_not_clear = 0;
    for(unsigned i = 0; i < parametrs.size(); ++i) {

    	if(parametrs[i] > max_val)
    		max_val = parametrs[i];
    	if(parametrs[i] < min_val)
    	    min_val = parametrs[i];

    	math_middle        += parametrs[i] / length;
    	math_middle_square += (parametrs[i]*parametrs[i]) / length;

        fout << parametrs[i];
        ++now_numb;
        ++now_numb_not_clear;
        if(now_numb >= endline) {
        	if(now_numb_not_clear < parametrs.size())
        	    fout << endl;
        	now_numb = 0;
        }
        else {
        	fout << ",";
        }
    }

    disperse = math_middle_square - (math_middle*math_middle);
    cout << "\n\nmax_val " << max_val << " min_val " << min_val << " math_middle " << math_middle << " disperse " << disperse << " length " << parametrs.size();

}
