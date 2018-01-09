#include "neuron.h"
#include "utils.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <semaphore.h>
#include <stdexcept>
#include <vector>

#define BIAS_CONFIG_SYMBOL "b"
#define WEIGHT_CONFIG_SYMBOL "IW"
#define PARALLELIZE true

using namespace std;

struct NeuronSpecs {
  double bias;
  vector<double> weights;
};

typedef vector<Neuron> Layer;

bool is_running;
size_t data_size;
size_t function_input_number;
size_t hidden_layer_neuron_number;

pthread_t output_layer_thread, input_layer_thread;
vector<pthread_t> hidden_layer_threads;

vector<double> input2hidden, hidden2output;
Layer hidden_layer, output_layer;

size_t input2hidden_counter;
sem_t input2hidden_write, input2hidden_counter_sem;
vector<sem_t> input2hidden_read;
vector<sem_t> hidden2output_write, hidden2output_read;

vector<vector<NeuronSpecs> > get_specs(string addr) {
  vector<vector<NeuronSpecs> > res;
  ifstream f(addr.c_str(), ios::in);
  if (!f)
    throw runtime_error("could not open " + addr);
  string line;
  size_t inner_index = -1;
  bool is_reading_bias;
  while (getline(f, line)) {
    if (utils::starts_with(line, BIAS_CONFIG_SYMBOL)) {
      res.push_back(vector<NeuronSpecs>());
      is_reading_bias = true;
    } else if (utils::starts_with(line, WEIGHT_CONFIG_SYMBOL)) {
      is_reading_bias = false;
      inner_index = 0;
    } else {
      if (is_reading_bias) {
        res.back().push_back(NeuronSpecs());
        res.back().back().bias = utils::stod(line);
      } else {
        vector<string> splitted = utils::split(line, " ");
        for (vector<string>::iterator i = splitted.begin(); i != splitted.end();
             i++)
          res.back()[inner_index].weights.push_back(utils::stod(*i));
        inner_index++;
      }
    }
  }
  f.close();
  return res;
}

double self_func(double f) { return f; }

int get_data_size(int argc, char const *argv[]) {
  if (argc != 4) {
    cerr << "usage:\t" << string(argv[0])
         << " <data_input_number> <config_file_addr> <input_file_addr>" << endl;
    throw invalid_argument("data_input_number not found");
  }
  return utils::stoi(argv[1]);
}

void fill_layers(const vector<vector<NeuronSpecs> > &neuron_specs,
                 Layer &hidden_layer, Layer &output_layer) {
  for (size_t i = 0; i < hidden_layer_neuron_number; i++)
    hidden_layer.push_back(
        Neuron(neuron_specs[0][i].bias, neuron_specs[0][i].weights, tanh));
  output_layer.push_back(
      Neuron(neuron_specs[1][0].bias, neuron_specs[1][0].weights, self_func));
}

void *input_layer_func(void *arg) {
  Input_Node *neuron = (Input_Node *)arg;
  while (is_running) {
    vector<double> input2hidden_local;
    try {
      input2hidden_local = neuron->get_next_data();
    } catch (EofException) {
      break;
    }
    sem_wait(&input2hidden_write);
    input2hidden = input2hidden_local;
    if (!input2hidden.size())
      cerr << "*" << endl;
    input2hidden_counter = hidden_layer_neuron_number;
    for (size_t i = 0; i < hidden_layer_neuron_number; i++)
      sem_post(&input2hidden_read[i]);
  }
  return NULL;
}

void *output_layer_func(void *arg) {
  Neuron neuron = output_layer[0];
  for (size_t j = 0; j < data_size; j++) {
    vector<double> hidden2output_local(hidden_layer_neuron_number);
#pragma omp parallel for
    for (size_t i = 0; i < hidden_layer_neuron_number; i++) {
      sem_wait(&hidden2output_read[i]);
      hidden2output_local[i] = hidden2output[i];
      sem_post(&hidden2output_write[i]);
    }
    printf("%4.10f\n", neuron.calc(hidden2output_local));
  }
  is_running = false;
  return NULL;
}

void *hidden_layer_func(void *arg) {
  size_t index = *(size_t *)arg;
  Neuron neuron = hidden_layer[index];
  delete (size_t *)arg;
  while (is_running) {
    sem_wait(&input2hidden_read[index]);
    vector<double> input2hidden_local = input2hidden;
    sem_wait(&input2hidden_counter_sem);
    input2hidden_counter--;
    if (!input2hidden_counter)
      sem_post(&input2hidden_write);
    sem_post(&input2hidden_counter_sem);

    double hidden2output_local = neuron.calc(input2hidden_local);

    sem_wait(&hidden2output_write[index]);
    hidden2output[index] = hidden2output_local;
    sem_post(&hidden2output_read[index]);
  }
  return NULL;
}

void run_serial(string input_addr,
                const vector<vector<NeuronSpecs> > &neuron_specs) {
  Input_Node input_neuron(input_addr);
  for (size_t i = 0; i < data_size; i++) {
    input2hidden = input_neuron.get_next_data();
    for (size_t j = 0; j < hidden_layer_neuron_number; j++)
      hidden2output[j] = hidden_layer[j].calc(input2hidden);
    printf("%4.10f\n", output_layer[0].calc(hidden2output));
  }
}

void run_parallel(string input_addr,
                  const vector<vector<NeuronSpecs> > &neuron_specs) {
  is_running = true;
  input2hidden_counter = 0;

  sem_init(&input2hidden_write, 0, 1);
  sem_init(&input2hidden_counter_sem, 0, 1);
  input2hidden_read = vector<sem_t>(hidden_layer_neuron_number, sem_t());
  for (size_t i = 0; i < hidden_layer_neuron_number; i++)
    sem_init(&(input2hidden_read[i]), 0, 0);
  hidden2output_write = vector<sem_t>(hidden_layer_neuron_number, sem_t());
  for (size_t i = 0; i < hidden_layer_neuron_number; i++)
    sem_init(&(hidden2output_write[i]), 0, 1);
  hidden2output_read = vector<sem_t>(hidden_layer_neuron_number, sem_t());
  for (size_t i = 0; i < hidden_layer_neuron_number; i++)
    sem_init(&(hidden2output_read[i]), 0, 0);

  Input_Node input_neuron(input_addr);
  pthread_create(&input_layer_thread, NULL, input_layer_func, &input_neuron);
  pthread_create(&output_layer_thread, NULL, output_layer_func, NULL);
  for (size_t i = 0; i < hidden_layer_neuron_number; i++) {
    pthread_t hidden_layer_thread;
    hidden_layer_threads.push_back(hidden_layer_thread);
    size_t *index = new size_t;
    *index = i;
    pthread_create(&hidden_layer_thread, NULL, hidden_layer_func, index);
  }
  pthread_join(output_layer_thread, NULL);
}

int main(int argc, char const *argv[]) {
  vector<vector<NeuronSpecs> > neuron_specs;
  try {
    data_size = get_data_size(argc, argv);
    neuron_specs = get_specs(argv[2]);
    if (neuron_specs.size() != 2)
      throw runtime_error("invalid specs");
  } catch (runtime_error e) {
    cerr << "ERR:\t" << e.what() << endl;
    return 2;
  } catch (logic_error e) {
    cerr << "ERR:\t" << e.what() << endl;
    return 2;
  } catch (exception e) {
    cerr << "ERR:\t" << e.what() << endl;
    return 2;
  }
  function_input_number = neuron_specs[0].back().weights.size();
  hidden_layer_neuron_number = neuron_specs[0].size();

  input2hidden = vector<double>(function_input_number);
  hidden2output = vector<double>(hidden_layer_neuron_number);
  fill_layers(neuron_specs, hidden_layer, output_layer);

  try {
    if (PARALLELIZE)
      run_parallel(argv[3], neuron_specs);
    else
      run_serial(argv[3], neuron_specs);
  } catch (runtime_error e) {
    cerr << "ERR:\t" << e.what() << endl;
    return 2;
  } catch (exception e) {
    cerr << "ERR:\t" << e.what() << endl;
    return 2;
  }

  return 0;
}
