// **************************************************************************
//  File       [sort_tool.cpp]
//  Author     [Yu-Hao Ho]
//  Synopsis   [The implementation of the SortTool Class]
//  Modify     [2015/03/02 Yu-Hao Ho]
// **************************************************************************

#include "sort_tool.h"
#include<iostream>
#include<limits.h>
#include<math.h>

// Constructor
SortTool::SortTool() {}

// Insertsion sort method
void SortTool::InsertionSort(vector<int>& data) {
    // Function : Insertion sort
    // TODO : Please complete insertion sort code here
 
	for(int j=1;j<data.size();j++){
	    int key=data[j];
		int i=j-1;
		
		while(i>=0 && data[i]>key){
		    data[i+1]=data[i];
			i=i-1;
		}
		data[i+1]=key;
	}

}

// Merge sort method
void SortTool::MergeSort(vector<int>& data){
    SortSubVector(data, 0, data.size() - 1);
}

// Sort subvector
void SortTool::SortSubVector(vector<int>& data, int low, int high) {
    // Function : Sort subvector
    // TODO : Please complete SortSubVector code here
    // Hint : recursively call itself
    //        Merge function is needed
    if(low < high){
	    int middle1 = floor((high + low)/2);
        int middle2 = middle1+1;
 		SortSubVector(data, low, middle1);
        SortSubVector(data, middle2, high);
        Merge(data, low, middle1, middle2, high);
	}
}

// Merge
void SortTool::Merge(vector<int>& data, int low, int middle1, int middle2, int high) {
    // Function : Merge two sorted subvector
    // TODO : Please complete the function
    int n1 = middle1-low+1;
    int n2 = high-middle2+1;
    int* L = new int [n1+1];
    int* R = new int [n2+1];
    
    for(int i=0; i<n1; ++i)
        L[i] = data[low+i];
    for(int j=0; j<n2; ++j)
        R[j] = data[middle2+j];
    L[n1] = R[n2] = INT_MAX;

    int ptr_L = 0, ptr_R = 0;
    
    for(int k=low; k<=high; k++){
        if(L[ptr_L] <= R[ptr_R]){
            data[k] = L[ptr_L];
            ++ptr_L;
        }
        else{
            data[k] = R[ptr_R];
            ++ptr_R;
        }
    }
    delete [] L;
    delete [] R;
}

// Heap sort method
void SortTool::HeapSort(vector<int>& data) {
    // Build Max-Heap
    Build_Max_Heap(data);
    // 1. Swap data[0] which is max value and data[i] so that the max value will be in correct location
    // 2. Do max-heapify for data[0]
    for (int i = data.size() - 1; i >= 1; --i) {
        swap(data[0],data[i]);
        --heapSize;
        Max_Heapify(data,0);
    }
}

//Max heapify
void SortTool::Max_Heapify(vector<int>& data, int root) {
    // Function : Make tree with given root be a max-heap if both right and left sub-tree are max-heap
    // TODO : Please complete max-heapify code here
    int left = root*2+1;
    int right = root*2+2;
    int largest;

    if(left<heapSize && data[left]>data[root])
        largest = left;
    else 
        largest = root;
    if(right<heapSize && data[right]>data[largest])
        largest = right;
    if(largest != root){
        swap(data[root], data[largest]);
        Max_Heapify(data, largest);
    }
}

//Build max heap
void SortTool::Build_Max_Heap(vector<int>& data){
    heapSize=data.size();
    for(int i=floor(heapSize/2)-1; i>=0; --i)
            Max_Heapify(data,i);
}
