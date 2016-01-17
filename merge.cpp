#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <algorithm>

using namespace std;

typedef struct
{
    int start, end;
}Range;

void *mysort(void *);
void sort_seg(int, int);

int *sequence;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t c = PTHREAD_COND_INITIALIZER;

int main(int argc, char *argv[])
{
    int seg_size = atoi(argv[1]);
    int total_num;
    
    scanf("%d", &total_num);
    sequence = new int[total_num];

    int i;
    for(i = 0; i < total_num; ++i)
        scanf("%d", &sequence[i]);
    
    for(i=0; i < total_num; ++i)
        fprintf(stderr, "%d ", sequence[i]);
    fprintf(stderr,"\n");

    sort_seg(total_num, seg_size);

    for(i=0; i < total_num; ++i)
        fprintf(stderr, "%d ", sequence[i]);
    fprintf(stderr,"\n");
    
    //merge_seq()
    return 0;
}

void *mysort(void *p)
{   
    pthread_mutex_lock(&m);
    Range* r = (Range *)(p);
    int s = r->start;
    int e = r->end;
    int i;
    fprintf(stderr, "Handling elements:\n");
    for(i = s; i < e; ++i)
        fprintf(stderr, "%d ", sequence[i]);
    fprintf(stderr, "\n");
    fprintf(stderr, "Sorted %d elements.\n", e-s);
    sort( (sequence + s), (sequence + e) );
    pthread_mutex_unlock(&m);
    pthread_exit(NULL);
}

void sort_seg(int total_num, int seg_size)
{
    int num_seg = total_num / seg_size + 1;
    pthread_t thread[num_seg];
    Range range[num_seg];
    int i;
    for(i = 0; i < total_num / seg_size; ++i){
        range[i].start = seg_size * i;
        range[i].end = seg_size * (i+1);
        if(pthread_create(&thread[i], NULL, mysort, &range[i]) < 0)
            perror("pthread_create error\n");
    }
    if(total_num % seg_size != 0){
        range[i].start = seg_size * (total_num / seg_size);
        range[i].end = total_num;
        if(pthread_create(&thread[i], NULL, mysort, &range[i]) < 0)
            perror("pthread_create error\n");
    }
    for(i = 0; i < num_seg; ++i)
        pthread_join(thread[i], NULL);
}

