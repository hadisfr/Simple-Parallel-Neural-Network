#ifndef __NEURON_H__
#define __NEURON_H__

#include <fstream>
#include <stdexcept>
#include <vector>

class Neuron {
private:
  double bias;
  std::vector<double> weights;
  double (*activation)(double);

public:
  Neuron(double bias, std::vector<double> weights,
         double (*activation)(double));
  double calc(std::vector<double> inputs) const throw(std::exception);
  size_t get_input_size();
};

class Input_Node {
private:
  std::ifstream fd;

public:
  Input_Node(std::string addr) throw(std::exception);
  ~Input_Node();
  std::vector<double> get_next_data();
  bool eof();
};

class EofException : public std::logic_error {
public:
  EofException() : std::logic_error("EOF") {}
};

#endif
