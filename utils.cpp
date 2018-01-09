#include "utils.h"

using namespace std;

double utils::stod(string str) { return cast<string, double>(str); }

int utils::stoi(string str) { return cast<string, int>(str); }

long utils::stol(string str) { return cast<string, long>(str); }

vector<string> utils::split(string text, string delimiter) {
  text = trim(text);
  vector<string> res;
  size_t last_start = 0;
  for (size_t i = 0; i < (text.size() - delimiter.size() + 1); i++) {
    if (text.substr(i, delimiter.size()) == delimiter) {
      res.push_back(trim(text.substr(last_start, i - last_start)));
      last_start = i + delimiter.size();
      i = last_start - 1;
    }
  }
  if (text.size() - last_start)
    res.push_back(trim(text.substr(last_start, text.size() - last_start)));
  return res;
}

bool is_ws(char ch) {
  return ch == ' ' || ch == '\n' || ch == '\t' || ch == '\r' || ch == '\f';
}

string utils::trim(string str) {
  size_t start = -1, end = str.size();
  while (is_ws(str[++start]))
    ;
  while (is_ws(str[--end]))
    ;
  return str.substr(start, end - start + 1);
}

bool utils::starts_with(string str, string start) {
  return (str.substr(0, start.size()) == start);
}
