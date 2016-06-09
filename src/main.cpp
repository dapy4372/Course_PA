# include <iostream>
# include <stdlib.h>
# include <math.h>
# include <stdio.h>
# include <sys/times.h>

# include "kdtree.h"
# include "kdtree.cpp"
# include "Node.h"
# include "Node.cpp"
# include "utils.h"

using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 4){
        fprintf(stderr, "input error!");
        exit(1);
    }

    // read file
    vector< Element<double> > el_vec = readFile<double>( argv[1] );

    // build kdtree
    KdTree<double> myKdtree;
    for( int i = el_vec.size() - 1; i >= 0; --i )
        myKdtree.insert(el_vec[i]);

    Element<double> tt;
    tt.keys[0] = strtold(argv[2], NULL);
    tt.keys[1] = strtold(argv[3], NULL);
    double range[2][2] = { {0, 20}, {0, 20} };
    //myKdtree.rangeSearch(range);
    double tmp = myKdtree.NNSearch(tt);
    cout << endl << sqrt(tmp) << endl << endl;
    //myKdtree.deleteNode(tt);
    //myKdtree.rangeSearch(range);
    //Node<double> *tmp = kdtree.search(tt);
    //Node<double> *tmp = kdtree.smallest(kdtree._root, 0, 0);
    //tmp->print();
    return 0;
}
