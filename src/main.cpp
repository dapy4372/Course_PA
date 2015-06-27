#include <iostream>
#include <fstream>
#include "dtw_v1.h"
#include <string>
#include "../lib/tm_usage.h"
#include "util.h"
using namespace std;
int T_0;

enum type{Template, Test};

int main(int argc, char *argv[])
{/*
  CommonNs::TmUsage tmusg;
  CommonNs::TmStat stat;
  tmusg.periodStart(); 
  */
  //myStr2Int(argv[1], T_0);
  
  DTW dtw;
  
  dtw.readFeat(argv[1], Template);
  dtw.readFeat(argv[2], Test);
  
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
  /*
  tmusg.getPeriodUsage(stat);
  cout <<"# run time = " << (stat.uTime + stat.sTime) / 1000000.0 << "sec" << endl;
  cout <<"# memory =" << stat.vmPeak / 1000.0 << "MB" << endl;
*/
  return 0;
}
