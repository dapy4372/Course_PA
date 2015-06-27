# ifndef DTW_V1_H
# define DTW_V1_H

#include <vector>
#include <string>
using namespace std;

# define DIM 40
typedef struct
{
  double val[DIM];
}Frame;


class DTW
{
  public:
  
    void readFeat(const char *, const bool &);
    //void readTest(const char *);
    void buildMap();
    double run();
    void clear(const bool &);

  private:
    
    double distance(const unsigned &, const unsigned &);
    
    vector<Frame> _dataX;
    vector<Frame> _dataY;
    string _spkTemp;
    string _spkTest;
    size_t _xSize;//modified
    size_t _ySize;//modified
    double **_costTable; //init in readFeat

};

# endif
