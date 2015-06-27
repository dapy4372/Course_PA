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

typedef struct
{
  string spkId;
  vector<Frame> vecFrame;
  int len;
}Data;

class DTW
{
  public:
  
    void readFeat(const char *, Data &data);
    //void readTest(const char *);
    void buildCostTable(const Data &, const Data &);
    double run();
    void clearCostTable();
    void addConstraint();

  private:
    
    double distance(const Data &, const Data &, const int &, const int &);

    int _xLen;//modified
    int _yLen;//modified
    double **_costTable; //init in readFeat

};

# endif
