// **************************************************************************
//  File       [main.cpp]
//  Author     [Yu-Hao Ho]
//  Synopsis   [The implementation of worker ant function]
//  Modify     [2015/03/20 Yu-Hao Ho]
// **************************************************************************
#include <fstream>
#include "WorkerAnt.h"
#include <cmath>
WorkerAnt::WorkerAnt()
{
  _total_dis = 0;
  _now_cap = 0;
  _ori.x = 0;
  _ori.y = 0;
  _ori.c = 0;
  _ori.p = 0;
  _now = _ori;
}

void WorkerAnt::load_data( const char *filename )
{
  ifstream fin( filename );
  fin >> _max_cap;
  fin >> _food_num;
  _food = new POINT[_food_num];

  for( int i=0; !fin.eof(); ++i ){
    _food[i].p = i;
    fin >> _food[i].x >> _food[i].y >> _food[i].c;
  }
  fin.close();
}

void WorkerAnt::write_data( const char *filename, const char *command )
{ //+1 to correct the index.
  fstream fout;
  fout.open( filename, ios::out );
  if( strcmp( command, "-DP" ) == 0 ){
    for( int i = _record_list.size()-1; i>=0; --i)
      fout << _record_list[i]+1 << endl; 
  }
  else if ( strcmp( command, "-GD" ) == 0 ){
    for( int i=0; i<_record_list.size(); ++i)
      fout << _record_list[i]+1 << endl;
  }
  fout<< _total_dis; 
  fout.close();
}

const int WorkerAnt::dis( const POINT &a, const POINT &b )
{
  return 
    ( ( ( a.x > b.x ) ? ( a.x - b.x ):( b.x - a.x ) ) 
      + ( ( a.y > b.y ) ? ( a.y - b.y ):( b.y - a.y ) ) ); 
}

bool WorkerAnt::is_over_cap( const int &c )
{
  if( _now_cap + c > _max_cap) 
    return true;
  else 
    return false;
}

void WorkerAnt::Greedy()
{
  for( int i=0; i < _food_num; ++i){
    if( is_over_cap( _food[i].c ) ){
      _record_list.push_back( i-1 );
      _total_dis  += dis( _food[i-1], _ori );
      _now = _ori;
      _now_cap = 0;
      --i;
    }
    else{
      _total_dis += dis ( _now, _food[i] );
      if( i == _food_num - 1 ){
        _record_list.push_back(i);
        _total_dis += dis( _ori, _food[i] );
      }
      else{
        _now = _food[i];
        _now_cap += _food[i].c;
      }
    }
  }
}

void WorkerAnt::Dynamic()
{
  _subsol = new SubSol [ _food_num ];
  _subsol[0].total_dis = dis( _ori, _food[0] ) * 2;
  _subsol[0].record = 0;
  _now_cap = _food[0].c;
  int once = 1;
  while( once < _food_num ){
    if( !is_over_cap( _food[once].c ) ){
      _now_cap += _food[once].c;
      _subsol[once].total_dis = total_until( 0, once );
      _subsol[once].record = once;
      ++once;
    }
    else
      break;
  }

for( int j = once; j<_food_num; ++j ){

  _subsol[j].total_dis = _subsol[j-1].total_dis + ( dis( _ori, _food[j])*2 ); //first at j-1
  _subsol[j].record = j-1; //cut at j-1

  int q = _subsol[j].total_dis; //tmp min distance
  int tmp;
  _now_cap = _food[j].c; //take j food

  for( int i=1; i<j; ++i ){ //second cut at j-i-1
    if( is_over_cap( _food[j-i].c  ) ) // want to take j-i food
      break;
    else{
      tmp = total_until( j-i, j ); //correct indx
      _now_cap += _food[j-i].c;
    }

    if( q > _subsol[j-i-1].total_dis + tmp ){  //find min distance
      q = _subsol[j-i-1].total_dis + tmp;
      _subsol[j].record = j-i-1; //cut at j-i-1
    }   
  }
  _subsol[j].total_dis = q;
}
_total_dis = _subsol[_food_num-1].total_dis;

//store to the record list.
int record_print = _food_num-1;
while(1){
  int print = _subsol[record_print].record;
  _record_list.push_back( record_print );
  if( record_print == _subsol[record_print].record )
    break;
  record_print = _subsol[record_print].record;
}
delete [] _subsol;
}
const int WorkerAnt::total_until( const int &start, const int &end )
{ 
  //total distance from taking j, j-1, j-2 .... to ori at the same time
  int tmp = dis( _ori, _food[start] );
  for( int i=start; i<end; ++i )
    tmp += dis( _food[i], _food[i+1] );
  tmp += dis( _food[end], _ori );
  return tmp;
}
