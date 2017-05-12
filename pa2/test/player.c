#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/select.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>

#define BUF_SIZE 128
#define PLAYER_NUM 4

void name(char *, char *, char *);
void err_sys(const char *);

int main(int argc, char **argv)
{
    
    if(argc != 4)
        err_sys("ERROR! Usage: $./player [host_id] [player_index] [random_key]\n");
    char *host_id = argv[1];
    char *player_idx = argv[2];
    char *key = argv[3];

    /* give FIFO name */
    char FIFO_name[2][15];
    name(FIFO_name[0], host_id, player_idx);
    name(FIFO_name[1], host_id, NULL);
      
    int fdr = open(FIFO_name[0], O_RDWR);
    FILE *fpr = fdopen(fdr, "r+");
    setbuf(fpr, NULL);
    
    int fdw = open(FIFO_name[1], O_RDWR);
    FILE *fpw = fdopen(fdw, "w");
    setbuf(fpw, NULL);

    /* correct response times */
    int done = 0;
    int prev_money[PLAYER_NUM] = {0, 0, 0, 0};

    while(1){
        /* get a message from host */
        char buf[BUF_SIZE];
        int money[PLAYER_NUM];
        fgets(buf, sizeof(buf), fpr);
        //fgets(buf, sizeof(buf), stdin);
        sscanf(buf, "%d %d %d %d\n", &money[0], &money[1], &money[2], &money[3]);


        
        fseek(fpw, 0, SEEK_SET);

        /* decide the bidding price */
        int price = 0;
        if( (done % PLAYER_NUM) == (player_idx[0] - 'A'))
            price = money[player_idx[0] - 'A'];
        
        fprintf(fpw, "%s %s %d\n", player_idx, key, price);
        fflush(fpw);
        fsync(fdw);
        //fprintf(stdout, "%s %s %d\n", player_idx, key, price);
        //fflush(stdout);
        

        //++done;
        //TODO
        /* check whether the previous responses is valid */
        int i;
        for(i = 0; i < PLAYER_NUM; ++i){
            if(prev_money[i] != money[i]){
                ++done;
                break;
            }
        }

        /* record the previous money */
        memcpy(prev_money, money, PLAYER_NUM * sizeof(int));
        /* correctly gives out 10 responses */
        if(done == 10)
            break;
    }

    /* close all fd and fp */
    fclose(fpr);
    close(fdr);
    fclose(fpw);
    close(fdw);

    exit(0);
}
    /*
    lseek(fdr, 0, SEEK_SET);

    int tmp[4];
    fscanf(fpr, "%d %d %d %d", &tmp[0], &tmp[1], &tmp[2], &tmp[3]);
    //read(fdr, buf, BUF_SIZE);
    //fprintf(stderr, "IN G: %s\n", buf);
    //FILE *tmp = fdopen(stdout, "w");*/
    /* sent msg to child*/
    //fprintf(fpw, "%d %d %d %d\n", tmp[0], tmp[1], tmp[2], (tmp[3]+1));
    //fflush(fpw);
    //scanf(buf2, "%s\n", buf);
    //write(fdw, buf2, sizeof(buf2));
    //fsync(fdw);
    //printf("IN G: %s", buf);
    //fprintf(fpw, "%s\n", buf);
    //fflush(fpw);

void err_sys(const char * x)
{
    perror(x);
    exit(1);
}

void name(char *target, char *id, char *index)
{
    char str1[] = "host";
    char str2[] = ".FIFO";
    strcpy(target, str1);
    strcat(target, id);
    if (index != NULL)
    {
        char temp[3] = "_";
        strcat(temp, index);
        strcat (target, temp);
    }
    strcat(target, str2);
}

