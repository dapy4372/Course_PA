# include <iostream>
using namespace std;

template < class T >
void Node<T>::print() const
{
    for(int i = 0; i < DIM - 1; ++i)
        cout << _el.keys[i] << " ";
    cout << _el.keys[DIM - 1] << endl;
}

template class Node<double>;
