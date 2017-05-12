// **************************************************************************
//  File       [WorkerAnt.h]
//  Author     [Yu-Hao Ho]
//  Synopsis   [The header file of worker ant]
//  Modify     [2015/03/20 Yu-Hao Ho]
// **************************************************************************
#ifndef _WORKERANT_H
#define _WORKERANT_H
#include <fstream>
#include <iostream>
#include <vector>
#include <string.h>
using namespace std;

struct POINT{ int p; int x; int y; int c; }; //p is order of the food; c is weight of food; 
struct SubSol{ int record; int total_dis; }; 

class WorkerAnt
{

public:
  WorkerAnt();
  ~WorkerAnt(){ delete [] _food; };
  void load_data( const char * );
  void write_data( const char *, const char * );
  const int dis( const POINT &, const POINT &);
  bool is_over_cap( const int & );
  void Greedy();
  void Dynamic();
  const int total_until( const int &, const int & );
private:
  int _max_cap;
  int _food_num;
  int _now_cap;
  int _total_dis;
  POINT *_food;
  POINT _ori;
  POINT _now;
  SubSol* _subsol;
  vector<int> _record_list; 
};

#endif
