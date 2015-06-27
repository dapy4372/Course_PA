#include <string>
#include <ctype.h>
#include "util.h"
using namespace std;

bool myStr2Int(const string &str, int &num)
{
  num = 0;
  size_t i = 0;
  int sign = 1;
  if (str[0] == '-') { sign = -1; i = 1; }
  bool valid = false;
  for (; i < str.size(); ++i) {
    if (isdigit(str[i])) {
      num *= 10;
      num += int(str[i] - '0');
      valid = true;
    }
    else return false;
  }
  num *= sign;
  return valid;
}
