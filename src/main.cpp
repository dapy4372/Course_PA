# include <iostream>
# include "kdtree.h"
# include "kdtree.cpp"
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
    float range[2][2] = { {0, 100}, {0, 100} };
    kdtree.rangeSearch(range);
    return 0;
}
