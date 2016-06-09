# include <iostream>
# include <vector>
# include <stdlib.h>
# include <math.h>
# include <stdio.h>
# include <sys/times.h>

# include "kdtree.h"
# include "Node.h"
# include "utils.h"

using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 2) {
        cout << "Usage: \n"
             << "      " << argv[0] << " <input file>\n\n"
             << "Example: \n"
             << "      " << argv[0] << " ./data/input1.txt\n\n";
        exit(0);
    }

    // read file
    vector< Element<double> > el_vec = readFile<double>( argv[1] );

    struct tms tmsstart, tmsend;
    clock_t start, end;
    if( (start = times(&tmsstart)) == -1 )
        err_sys("time error");

    // build kdtree
    KdTree<double> myKdtree;
    for( int i = el_vec.size() - 1; i >= 0; --i )
        myKdtree.insert(el_vec[i]);

    if( (end = times(&tmsend)) == -1 )
        err_sys("time error");

    // print the time for build kd tree
    cout << "The time for building Kd tree:\n";
    printTimes(end - start, &tmsstart, &tmsend);

    // (a) Find the nearest neighbor of (0.5, 0.5)
    Element<double> query = {{0.5, 0.5}};
    double tmp = myKdtree.NNSearch(query);
    cout << endl << sqrt(tmp) << endl << endl;

    // (b) How many points there are in the rectangle (0.3, 0.3),(0.3, 0.41),(0.6, 0.3),(0.6, 0.41)?
    double range[2][2] = { {0.3, 0.6}, {0.3, 0.41} };
    myKdtree.rangeSearch(range);
    //myKdtree.deleteNode(tt);
    //myKdtree.rangeSearch(range);
    //Node<double> *tmp = kdtree.search(tt);
    //Node<double> *tmp = kdtree.smallest(kdtree._root, 0, 0);
    //tmp->print();
    return 0;
}
