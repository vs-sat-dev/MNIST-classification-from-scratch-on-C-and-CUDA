#ifndef CSVWRITER_H
#define CSVWRITER_H

#include <vector>
#include <string>
#include <fstream>

class CSVWriter {
public:
    CSVWriter(std::string file_name);
    virtual ~CSVWriter();

    void write(std::vector<std::string> & parametr_names, std::vector<double> & parametr_sizes);
    void write(const std::vector<double> & parametrs, int numb = -1);

private:
    std::ofstream fout;
    std::vector<unsigned> neuron_layer_size;
};

#endif /* CSVWRITER_H */
