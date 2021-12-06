/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   CSVReader.cpp
 * Author: chelovek
 *
 * Created on 14 июня 2019 г., 20:43
 */

#include "CSVReader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace std;

CSVReader::CSVReader(string file_name, bool is_output) {
    unsigned rows_size = 1;
    unsigned cols_size = 0;
    ifstream fout(file_name.c_str());
    if(fout) {
        string line;
        while(getline(fout, line)) {
            if(cols_size == 0) {
                for(auto &l : line) {
                    if(l == ',')
                        ++rows_size;
                }
                if(is_output)
                    --rows_size;
            }
            ++cols_size;
        }
        cout << "\nFile " << file_name << " rows_size " << rows_size << " cols_size " << cols_size;
    }
    else {
        cout << "\nFile not exist";
        exit(1);
    }
    fout.clear();
    fout.seekg(0);

    data.resize(cols_size - 1);
    output_data.reserve(cols_size - 1);
    string line;
    unsigned i = 0;
    while(getline(fout, line)) {
        string str;
        bool first_comma = true;

        if(!is_output) {
            first_comma = false;
        }

        if(i != 0) {
            data[i-1].reserve(rows_size);
            for(auto &l : line) {
                if(!first_comma) {
                    if(l != ',')
                        str += l;
                    else {
                        double d;
                        stringstream dss;
                        dss << str;
                        dss >> d;
                        data[i-1].emplace_back(d / 255.0);
                        str = "";
                    }
                }
                else {
                    if(l != ',')
                        str += l;
                    else {
                        first_comma = false;

                        double d;
                        stringstream dss;
                        dss << str;
                        dss >> d;
                        output_data.emplace_back(d);
                        //cout << "\n\nOutput_Data " << d;
                        str = "";
                    }
                }
            }
            double d;
            stringstream dss;
            dss << str;
            dss >> d;
            data[i-1].emplace_back(d / 255.0);
            //cout << "\ndata[].size " << data[i-1].size();
        }
        ++i;
    }
    cout << "\ndata.size " << data.size() << " output.size " << output_data.size();
}

CSVReader::CSVReader(const CSVReader& orig) {
}

CSVReader::~CSVReader() {
}

