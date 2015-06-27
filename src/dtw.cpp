#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include "dtw.h"

using namespace std;
# define INTMAX 9999.0
# define MAX_SHIFT 50
# define W 1.05
# define E 0.3
void DTW::readFeat(const char *filename, Data &data)
{
  ifstream fin(filename);
  stringstream ss("");
  string buf;
  Frame frame;
  
  getline(fin, buf);
  ss << buf;
   
  ss >> data.spkId;
  
  ss.str("");
  ss.clear();
  
  while(getline(fin, buf)){
    ss << buf;
    for(int i=0; i < DIM; ++i)
      ss >> frame.val[i];

    data.queFrame.push_back(frame);
  
    ss.str("");
    ss.clear();
  }
   
  data.len = data.queFrame.size();
  
  fin.close();
}

double DTW::distance(const Data &a, const Data &b, const int &x, const int &y)
{
    double innerProduct = 0;
    double x_tmp = 0;
    double y_tmp = 0;
    for(int i = 0; i < DIM; i++){
      x_tmp += pow(a.queFrame[x].val[i], 2);
      y_tmp += pow(b.queFrame[y].val[i], 2);
      innerProduct += a.queFrame[x].val[i] * b.queFrame[y].val[i];
    }
    //cosine similarity
    return (innerProduct/(sqrt(x_tmp)*sqrt(y_tmp)));
}

void DTW::buildCostTable(const Data &a, const Data &b)
{
  clearCostTable();

  _xLen = a.len;
  _yLen = b.len;
  int m = _xLen / _yLen; // constraints slope
  _costTable = new double* [_xLen];
  for(int x=0; x < _xLen; ++x){
    _costTable[x] = new double [_yLen];
    for(int y=0; y < _yLen; ++y){
      if(m*x - y > MAX_SHIFT || m*x - y < -1*MAX_SHIFT) //add constraints
        _costTable[x][y] = -1 * INTMAX;
      else
        _costTable[x][y] = distance(a, b, x, y);
    }
  } 
}

void DTW::clearCostTable()
{
  if(_costTable != NULL){
    for(int i=0; i < _xLen; ++i)
      delete [] _costTable[i];
    delete [] _costTable;
    _costTable = NULL;
  }
}

//content added by Frank
double DTW::run(const Data &a, const Data &b){
  
  buildCostTable(a, b);
  double   _dynamic[_xLen][_yLen];
  _dynamic[0][0] = _costTable[0][0];
  //initialization of the _dynamic array
  for(int i = 1; i < _yLen; i++){
    _dynamic[0][i] = ( _dynamic[0][i-1] + _costTable[0][i] );
  }
  for(int i = 1; i < _xLen; i++){
    _dynamic[i][0] = ( _dynamic[i-1][0] + _costTable[i][0] );
  }
  //running the dynamic programming
  for(int i = 1; i < _xLen; i++){
    for(int j = 1; j < _yLen; j++){
      double L_choice = ( _dynamic[i-1][j] + _costTable[i][j] );//the left path
      double T_choice = ( _dynamic[i-1][j-1] + 2 * _costTable[i][j] );//the tilt path
      double D_choice = ( _dynamic[i][j-1] + _costTable[i][j]);
      double indicator = 'L';
      if( L_choice > T_choice ){//the left choice is better, further compare it with the down choice
        if( L_choice > D_choice ){//the left choice is the best
          _dynamic[i][j] = L_choice;
        }
        else{//the down choice is the best
          _dynamic[i][j] = D_choice;
          indicator = 'D';
        }
      }
      else{//the tilt choice is better, further compare it with the down choice
        if( T_choice > D_choice ){//the tilt choice is the best
          _dynamic[i][j] = T_choice;
          indicator = 'T';
        }
        else{//the down choice is the best
          _dynamic[i][j] = D_choice;
          indicator = 'D';
        }
      }
    }
  }
  return _dynamic[_xLen-1][_yLen-1] / (_xLen + _yLen);
}

void DTW::cutData(const Data &ori, Data &cut, const int &a, const int &b)
{
  cut.spkId = ori.spkId + "(cut)";
  for(int i=a; i<b; ++i)
    cut.queFrame.push_back(ori.queFrame[i]);
  cut.len = cut.queFrame.size(); 
}

void DTW::search(const Data &key, const Data &obj)
{
  Data cut;
  int subLen = key.len * W;
  int check = key.len * E; 
  cutData(obj, cut, 0, subLen);
  int num = obj.len - cut.len;
  double max = 0;
  double tmp;
  int pos;
  for(int i=0; i < num; ++i){
    //clearCostTable();
    //buildCostTable(key, cut);
    tmp = run(key, cut);
    if(tmp > max){
      pos = i;
      max = tmp;
    }
    cut.queFrame.pop_front();
    cut.queFrame.push_back(obj.queFrame[subLen+i]);
  }
  
  Data exact = cut;
  for(int i = 0; i < check; ++i){
    cut.queFrame.pop_back();
    --cut.len;
    //clearCostTable();
    //buildCostTable(key, cut);
    tmp = run(key, cut);
    if(tmp > max){
      max = tmp;
      exact = cut;
      --subLen;
    } 
  }

  for(int i = 0; i < check; ++i){
    exact.queFrame.pop_front();
    --exact.len;
    //clearCostTable();
    //buildCostTable(key, exact);
    tmp = run(key, exact);
    if(tmp > max){
      max = tmp;
      ++pos;
      --subLen;
    }
  }
  cout << "s = " << pos << " e = " << subLen+pos-1 <<endl; 
  cout << "max = " << max << endl;
}
