/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   CSVReader.h
 * Author: chelovek
 *
 * Created on 14 июня 2019 г., 20:43
 */

#ifndef CSVREADER_H
#define CSVREADER_H

#include <vector>
#include <string>

class CSVReader {
public:
    CSVReader(std::string file_name, bool is_output = false);
    CSVReader(const CSVReader& orig);
    virtual ~CSVReader();
    inline std::vector<double> & get_data(unsigned i) { return data[i]; };
    inline std::vector<std::vector<double> > & get_all_data() { return data; };
    inline double get_output_data(unsigned i) { return output_data[i]; };
    inline unsigned get_data_size() { return data.size(); };
    inline unsigned get_rows_size() { return data.front().size(); };
private:
    std::vector<std::vector<double> > data;
    std::vector<unsigned>             output_data;
};

#endif /* CSVREADER_H */

