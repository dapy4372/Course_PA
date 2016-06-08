# include "Node.h"
using namespace std;

template < class T >
void Node<T>::print()
{
    for(int i = 0; i < DIM - 1; ++i)
        cout << el.keys[i] << " ";
    cout << el.keys[DIM - 1] << endl;
}
