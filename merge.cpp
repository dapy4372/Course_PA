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

typedef struct
{
    Range a, b;
}RangePair;

void *mysort(void *);
void sort_seg(int, int);
void merge_seg(int, int);

int *sequence;
int num_seg;
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
    
    merge_seg(total_num, seg_size);
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
    if(total_num % seg_size != 0)
        num_seg = total_num / seg_size + 1;
    else
        num_seg = total_num / seg_size;
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

void merge_seg(int total_num, int seg_size)
{
    pthread_t thread[num_seg/2];
    RangePair range_pair[num_seg];
    int seg_width[num_seg] = {3,3,3,1};
    int i;
    while(num_seg > 1){
        int current = 0;
        for(i = 0; i < num_seg / 2; ++i){
            range_pair[i].a.start = current;
            range_pair[i].a.end = range_pair[i].a.start + seg_width[2*i];
            range_pair[i].b.start = range_pair[i].a.end;
            range_pair[i].b.end = range_pair[i].b.start + seg_width[2*i+1];
            current = range_pair[i].b.end;
            seg_width[i] = seg_width[2*i] + seg_width[2*i+1];
        }
        if(num_seg % 2 == 1)
            seg_width[i] = seg_width[num_seg - 1];

        for(i = 0; i < num_seg/2; ++i)
            fprintf(stderr, "a=(%d, %d); b=(%d, %d)\n", range_pair[i].a.start, range_pair[i].a.end, range_pair[i].b.start, range_pair[i].b.end);
        num_seg -= num_seg / 2;
    }
}
