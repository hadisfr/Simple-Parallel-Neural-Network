#include "neuron.h"
#include "utils.h"
#include <stdexcept>

using namespace std;

Neuron::Neuron(double _bias, vector<double> _weights,
               double (*_activation)(double))
    : bias(_bias), weights(_weights), activation(_activation) {}

double Neuron::calc(std::vector<double> inputs) const throw(exception) {
  double res = bias;
  if (inputs.size() != weights.size())
    throw std::invalid_argument(
        "Neuron 'inputs' size not matched with 'weights' size.");
  std::vector<double> middle_stage(weights.size());
#pragma omp parallel for
  for (size_t i = 0; i < weights.size(); i++)
    middle_stage[i] = weights[i] * inputs[i];
  for (size_t i = 0; i < weights.size(); i++)
    res += middle_stage[i];
  return activation(res);
}

size_t Neuron::get_input_size() { return weights.size(); }

Input_Node::Input_Node(string addr) throw(std::exception) {
  fd.open(addr.c_str());
  if (!fd)
    throw runtime_error("could not open file " + addr);
}

Input_Node::~Input_Node() { fd.close(); }

vector<double> Input_Node::get_next_data() {
  string line;
  if (fd.eof())
    throw EofException();
  getline(fd, line);
  vector<string> splitted = utils::split(line, " ");
  vector<double> res;
  for (vector<string>::iterator i = splitted.begin(); i != splitted.end(); i++)
    res.push_back(utils::stod(*i));
  return res;
}
bool Input_Node::eof() { return fd.eof(); }
