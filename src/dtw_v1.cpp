#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include "dtw_v1.h"

using namespace std;
# define INT_MAX 9999.0
extern int T_0;
extern bool type; // type 1 or not

void DTW::readFeat(const char *filename, const bool &type)
{
  ifstream fin(filename);
  stringstream ss("");
  string buf;
  Frame frame;
  
  getline(fin, buf);
  ss << buf;
  ss >> (type ? _spkTest : _spkTemp); //false for temp
  ss.str("");
  ss.clear();
  
  while(getline(fin, buf)){
    ss << buf;
    for(size_t i=0; i < DIM; ++i)
      ss >> frame.val[i];

    type ? _dataX.push_back(frame) : _dataY.push_back(frame);
  
    ss.str("");
    ss.clear();
  }
  
  type ? _xSize = _dataX.size() : _ySize = _dataY.size();
  
  fin.close();
}

double DTW::distance(const int &x, const int &y)
{
	//if( x-y > 50 || x-y < -50)
	  //return INT_MAX;
	//else{
    double tmp = 0;
    for(size_t i=0; i < DIM; ++i){
      tmp += pow((_dataX[x].val[i] - _dataY[y].val[i]), 2);
    }
    return sqrt(tmp);
	//}
}

void DTW::buildMap()
{
  _costTable = new double* [_xSize];
  for(size_t x=0; x < _xSize; ++x){
    _costTable[x] = new double [_ySize];
    for(size_t y=0; y < _ySize; ++y){
      if(constraints(x, y))
        _costTable[x][y] = INT_MAX;
      else
        _costTable[x][y] = distance(x, y);
    }
  }
  
}

bool DTW::constraints(const int &x, const int &y)
{
  //add Maximum allowable absolute time deviation
  int sub = x-y;
  if(sub > T_0 || sub < (-1 * T_0))
    return true;
  //else if(tpye && sub > ){}
  return false;
}

void DTW::clear(const bool &type)
{
  if(type){
    for(size_t i=0; i < _ySize; ++i)
      delete [] _costTable[i];
    delete [] _costTable;
    //clear test
    _spkTest.clear();
    _dataY.clear();
  }
  //clear Temp type=0
  else if(!type){
    _spkTemp.clear();
    _dataX.clear();
  }
}

//content added by Frank
double DTW::run(){
  double   _dynamic[_xSize][_ySize];
  //char     _direction[_xSize][_ySize];
  //_direction: 'D' = down 'L' = left 'T' = tilt
  //do we need backtracking?
  for(size_t i = 0; i < _xSize; i++){
    _dynamic[i][0] = _costTable[i][0];
  }//initialization of X axis
  for(size_t i = 0; i < _ySize; i++){
    _dynamic[0][i] = _costTable[0][i];
  }//initialization of Y axis
  for(size_t i = 1; i < _xSize; i++){
    for(size_t j = 1; j < _ySize; j++){
      double cost = _costTable[i][j];
      char dummy;//dummy indicates the minimal cost
      
			double minCost = _costTable[i-1][j];
      if( minCost < _costTable[i-1][j-1]){
			    
			}

      if( _costTable[i-1][j] < _costTable[i-1][j-1] )
        { dummy = 'L'; }
      else{ dummy = 'T'; }

      if(dummy == 'L'){
        if( _costTable[i-1][j] < _costTable[i][j-1] ){
          dummy = 'L'; cost += _costTable[i-1][j];
        }
        else{
          dummy = 'D'; cost += _costTable[i][j-1];
        }
      }
      else{//dummy == 'T'
        if( _costTable[i-1][j-1] < _costTable[i][j-1] ){
          dummy = 'T'; cost += 2 * _costTable[i-1][j-1];
        }
        else{
          dummy = 'D'; cost += _costTable[i][j-1];
        }
      }
      _dynamic[i][j] = cost;
    }
  }
  return _dynamic[_xSize-1][_ySize-1];
}
