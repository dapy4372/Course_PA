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
  
    void readFeat(const char *, const int &);
    //void readTest(const char *);
    void buildMap();
    double run();
    void clear(const bool &);
    void addConstraint();

  private:
    
    double distance(const int &, const int &);
    bool constraints(const int &, const int &);

    vector<Frame> _dataX;
    vector<Frame> _dataY;
    string _spkTemp;
    string _spkTest;
    int _xSize;//modified
    int _ySize;//modified
    double **_costTable; //init in readFeat

};

# endif
