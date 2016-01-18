#include <iostream>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include <algorithm>
#include <sys/times.h>

using namespace std;

typedef struct
{
    int start, end;
}Range;

typedef struct
{
    Range a, b;
}RangePair;

void print_seq(int, int);
void *mysort(void *);
void sort_seg(int, int);
void *mymerge(void *);
void merge_seg(int, int);
static void pr_times(clock_t, struct tms *, struct tms *);

int *sequence;
int num_seg;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;

int main(int argc, char *argv[])
{
    int seg_size = atoi(argv[1]);
    int total_num;

    scanf("%d", &total_num);
    sequence = new int[total_num];

    if(total_num % seg_size != 0)
        num_seg = total_num / seg_size + 1;
    else
        num_seg = total_num / seg_size;

    int i;
    for(i = 0; i < total_num; ++i)
        scanf("%d", &sequence[i]);

    clock_t start, end;
    struct tms tmsstart, tmsend;
    if ((start = times(&tmsstart)) == -1)    /* starting values */
        cout << "times error" << endl;

    sort_seg(total_num, seg_size);
    merge_seg(total_num, seg_size);

    if ((end = times(&tmsend)) == -1)
        cout << "times error" << endl;                                          
    pr_times(end-start, &tmsstart, &tmsend);

    #ifdef PRINT 
    print_seq(0, total_num);
    #endif

    return 0;
}

void print_seq(int s, int e)
{
    fprintf(stdout, "%d", sequence[s]);
    int i;
    for(i=1; i < e-s; ++i)
        fprintf(stdout, " %d", sequence[s+i]);
    fprintf(stdout,"\n");
}

void *mysort(void *p)
{   
    pthread_mutex_lock(&m);
    Range *r = (Range *)(p);
    int s = r->start;
    int e = r->end;
    int i;
    #ifdef PRINT 
    fprintf(stdout, "Handling elements:\n");
    print_seq(s, e);
    fprintf(stdout, "Sorted %d elements.\n", e-s);
    #endif
    sort( (sequence + s), (sequence + e) );
    pthread_mutex_unlock(&m);
    pthread_exit(NULL);
}

void sort_seg(int total_num, int seg_size)
{
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

void *mymerge(void *p)
{
    pthread_mutex_lock(&m);
    RangePair *rp = (RangePair *)(p);
    int idx_seg1 = rp->a.start, idx_seg2 = rp->b.start;
    int width_seg1 = rp->a.end - rp->a.start, width_seg2 = rp->b.end - rp->b.start, width_merge = width_seg2 + width_seg1;
    int *buf = new int[width_merge];
    int duplicates = 0, idx = 0;

    #ifdef PRINT 
    fprintf(stdout, "Handling elements:\n");
    print_seq(rp->a.start, rp->b.end);
    #endif

    while(idx_seg1 < rp->a.end && idx_seg2 < rp->b.end){
        if(sequence[idx_seg1] < sequence[idx_seg2] || sequence[idx_seg1] == sequence[idx_seg2]){
            if(sequence[idx_seg1] == sequence[idx_seg2])
                ++duplicates;
            buf[idx] = sequence[idx_seg1];
            ++idx_seg1;
        }
        else{
            buf[idx] = sequence[idx_seg2];
            ++idx_seg2;
        }
        ++idx;
    }
    // the reset number
    if(idx_seg1 == rp->a.end){
        while(idx_seg2 < rp->b.end){
            buf[idx] = sequence[idx_seg2];
            ++idx_seg2;
            ++idx;
        }
    }
    else if(idx_seg2 == rp->b.end){
        while(idx_seg1 < rp->a.end){
            buf[idx] = sequence[idx_seg1];
            ++idx_seg1;
            ++idx;
        }
    }
    // update the sequence
    int i;
    for(i = 0; i < width_merge; ++i)
        sequence[rp->a.start + i] = buf[i];

    #ifdef PRINT 
    fprintf(stdout, "Merged %d and %d elements with %d duplicates.\n", width_seg1, width_seg2, duplicates);
    #endif

    delete [] buf;
    pthread_mutex_unlock(&m);
    pthread_exit(NULL);
}

void merge_seg(int total_num, int seg_size)
{
    pthread_t thread[num_seg/2];
    RangePair range_pair[num_seg];
    int seg_width[num_seg];
    int i;
    for(i = 0; i < total_num / seg_size; ++i)
        seg_width[i] = seg_size;
    if(total_num % seg_size != 0)
        seg_width[i] = total_num % seg_size;

    while(num_seg > 1){
        int current = 0;
        for(i = 0; i < num_seg / 2; ++i){
            range_pair[i].a.start = current;
            range_pair[i].a.end = range_pair[i].a.start + seg_width[2*i];
            range_pair[i].b.start = range_pair[i].a.end;
            range_pair[i].b.end = range_pair[i].b.start + seg_width[2*i+1];
            current = range_pair[i].b.end;
            seg_width[i] = seg_width[2*i] + seg_width[2*i+1];
            pthread_create(&thread[i], NULL, mymerge, &range_pair[i]);
        }
        if(num_seg % 2 == 1)
            seg_width[i] = seg_width[num_seg - 1];
        for(i = 0; i < num_seg / 2; ++i)
            pthread_join(thread[i], NULL);
        num_seg -= num_seg / 2;
    }
}

static void pr_times(clock_t real, struct tms *tmsstart, struct tms *tmsend)
{
    static long     clktck = 0;

    if (clktck == 0)    /* fetch clock ticks per second first time */
        if ((clktck = sysconf(_SC_CLK_TCK)) < 0)
            cout << "sysconf error" << endl;
    printf(" real:  %7.2f\n", real / (double) clktck);
    printf(" user:  %7.2f\n",
            (tmsend->tms_utime - tmsstart->tms_utime) / (double) clktck);
    printf(" sys:   %7.2f\n",
            (tmsend->tms_stime - tmsstart->tms_stime) / (double)
            clktck);
    printf(" child user:   %7.2f\n",
            (tmsend->tms_cutime - tmsstart->tms_cutime) /
            (double) clktck);
    printf(" child sys:    %7.2f\n",
            (tmsend->tms_cstime - tmsstart->tms_cstime) /
            (double) clktck);
}
