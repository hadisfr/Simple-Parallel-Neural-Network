#ifndef __UTILS_H__
#define __UTILS_H__

#include <sstream>
#include <string>
#include <vector>

namespace utils {
template <typename From, typename To> To cast(From);
template <typename T> std::string to_string(T);
double stod(std::string);
int stoi(std::string);
long stol(std::string);
std::string trim(std::string);
std::vector<std::string> split(std::string str, std::string delimiter);
bool starts_with(std::string str, std::string start);
} // namespace utils

template <typename From, typename To> To utils::cast(From from) {
  std::stringstream ss;
  ss << from;
  To to;
  ss >> to;
  return to;
}

template <typename T> std::string utils::to_string(T var) {
  return cast<T, std::string>(var);
}

#endif
