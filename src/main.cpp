# include <iostream>
# include <stdlib.h>
# include "kdtree.h"
# include "kdtree.cpp"
# include "Node.h"
# include "Node.cpp"
# include "utils.h"

using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 2){
        fprintf(stderr, "input error!");
        exit(1);
    }
    queue< Element<float> > el_que = readFile<float>( argv[1] );
    KdTree<float> myKdtree;
    while(!el_que.empty()){
        myKdtree.insert(el_que.front());
        el_que.pop();
    }
    Element<float> tt;
    tt.keys[0] = 1;
    tt.keys[1] = 10;
    float range[2][2] = { {0, 10}, {0, 20} };
    myKdtree.rangeSearch(range);
    myKdtree.deleteNode(tt);
    myKdtree.rangeSearch(range);
    //Node<float> *tmp = kdtree.search(tt);
    //Node<float> *tmp = kdtree.smallest(kdtree._root, 0, 0);
    //tmp->print();
    return 0;
}
