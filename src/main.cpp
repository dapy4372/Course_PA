#include <iostream>
#include <fstream>
#include "dtw_v1.h"
#include <string>
using namespace std;

int main(int argc, char *argv[])
{
  DTW dtw;
  dtw.readFeat(argv[1], false);
  dtw.readFeat(argv[2], true);
  dtw.buildMap();
  //dtw.clear(true);
  //dtw.clear(false);

  cout << dtw.run() << endl;
/*  
  ifstream fin(argv[2]);

  string buf;

  while(getline(fin, buf)){
    dtw.readFeat(buf.c_str(), true);
    dtw.buildMap();
//    dtw.run();
    dtw.clear(true);
  }
  dtw.clear(false);
  */
 // fin.close();
  return 0;
}
