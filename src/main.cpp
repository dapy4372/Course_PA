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
    KdTree<float> kdtree;
    while(!el_que.empty()){
        kdtree.insert(el_que.front());
        el_que.pop();
    }
    float range[2][2] = { {1, 1}, {10, 10} };
    Element<float> tt;
    tt.keys[0] = 1;
    tt.keys[1] = 10;
    //kdtree.rangeSearch(range);
    Node<float> *tmp = kdtree.search(tt);
    tmp->print();
    return 0;
}
