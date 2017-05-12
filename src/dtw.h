# ifndef DTW_H
# define DTW_H

#include <deque>
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
  deque<Frame> queFrame;
  int len;
}Data;

class DTW
{
  public:
    DTW(){_costTable = NULL;}; 
    void readFeat(const char *, Data &data);
    double run(const Data &, const Data &);
    void cutData(const Data &, Data &, const int &, const int &);
    void search(const Data &, const Data &);

  private:
    
    double distance(const Data &, const Data &, const int &, const int &);
    void buildCostTable(const Data &, const Data &);
    void clearCostTable();

    int _xLen;//modified
    int _yLen;//modified
    double **_costTable; //init in readFeat

};

# endif
