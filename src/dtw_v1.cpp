#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include "dtw_v1.h"

using namespace std;
# define INT_MAX 9999.0
# define MAX_SHIFT 20
extern int T_0;

void DTW::readFeat(const char *filename, const int &type)
{
  ifstream fin(filename);
  stringstream ss("");
  string buf;
  Frame frame;
  
  getline(fin, buf);
  ss << buf;
  
  if(type == 0)    //0 for template
    ss >> _spkTemp;
  if(type == 1)
    ss >> _spkTest;
  
  ss.str("");
  ss.clear();
  
  while(getline(fin, buf)){
    ss << buf;
    for(size_t i=0; i < DIM; ++i)
      ss >> frame.val[i];

    if(type == 0)    //0 for template
      _dataX.push_back(frame);
    if(type == 1)
      _dataY.push_back(frame);
  
    ss.str("");
    ss.clear();
  }
   
  if(type == 0)    //0 for template
    _xSize = _dataX.size();
  if(type == 1)
    _ySize = _dataY.size();
  
  fin.close();
}

double DTW::distance(const int &x, const int &y)
{
	//if( x-y > 50 || x-y < -50)
	  //return INT_MAX;
	//else{
    /*double tmp = 0;
    for(size_t i=0; i < DIM; ++i){
      tmp += pow((_dataX[x].val[i] - _dataY[y].val[i]), 2);
    }
    return sqrt(tmp);
	//}*/
    double temp = 0;
    double x_temp = 0;
    double y_temp = 0;
    for(int i = 0; i < DIM; i++){
        x_temp += pow(_dataX[x].val[i], 2);
        y_temp += pow(_dataY[y].val[i], 2);
        temp   += _dataX[x].val[i] * _dataY[y].val[i];
    }
    return (temp/(sqrt(x_temp)*sqrt(y_temp)));
}

void DTW::buildMap()
{
  _costTable = new double* [_xSize];
  for(int x=0; x < _xSize; ++x){
    _costTable[x] = new double [_ySize];
    for(int y=0; y < _ySize; ++y){
        _costTable[x][y] = distance(x, y);
    }
  }
  
}

void DTW::addConstraint(){
  int dummyX = _xSize;
  int dummyY = _ySize;
  for(int i = 0; i < dummyX; i++)
    for(int j = 0; j < dummyY; j++)
      if( ((j - i*dummyY/dummyX) < -1*MAX_SHIFT)||((j - i*dummyY/dummyX) > MAX_SHIFT) )
        _costTable[i][j] = -1 * INT_MAX;
}

void DTW::clear(const bool &type)
{
  if(type){
    for(int i=0; i < _xSize; ++i)
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
  _dynamic[0][0] = _costTable[0][0];
  //initialization of the _dynamic array
  for(int i = 1; i < _ySize; i++){
    _dynamic[0][i] = ( _dynamic[0][i-1] + _costTable[0][i] );
  }
  for(int i = 1; i < _xSize; i++){
    _dynamic[i][0] = ( _dynamic[i-1][0] + _costTable[i][0] );
  }
  //running the dynamic programming
  for(int i = 1; i < _xSize; i++){
    for(int j = 1; j < _ySize; j++){
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
  return _dynamic[_xSize-1][_ySize-1] / (_xSize + _ySize);
}
