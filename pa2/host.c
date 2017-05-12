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

#define BUF_SIZE 20
#define MAX_ROUND 10
#define PLAYER_NUM 4

typedef struct
{
    int money;
    int score;
}Player;

enum{A = 0, B = 1, C = 2, D = 3};

void stoi(const char * const, int *);
void err_sys(const char *);
void init_player(Player *);
void update_player_info(const int * const, Player *);
void add_money(Player *);
void name(char *, const char *, const char *);
int check_key(const char *, const int *);
int find_winner(const int *);
int isvalue_in_array(const int *, const int *, int);

int main(int argc, char **argv)
{
    if(argc != 2)
        err_sys("ERROR! Usage: $./host [host_id]\n");

    srand( (unsigned)time(NULL));
    char *host_id = argv[1];
    
    /* give FIFO name */
    char FIFO_name[5][15];
    name(FIFO_name[0], host_id, "NULL");
    name(FIFO_name[1], host_id, "_A");
    name(FIFO_name[2], host_id, "_B");
    name(FIFO_name[3], host_id, "_C");
    name(FIFO_name[4], host_id, "_D");

    /* momey message */
    char money_msg[20];

    fd_set h_set, readset, tmp_readset;
    while(1)
    {   
        /* make FIFOs */
        int idx = 0;
        for(idx = 0; idx < 5; ++idx) mkfifo(FIFO_name[idx], 0666);

        int playerid[PLAYER_NUM] = {0, 0, 0, 0};
        pid_t pid;
        pid_t player_pid[4];
        FD_ZERO(&h_set);
        FD_ZERO(&readset);
        FD_SET(STDIN_FILENO, &h_set);
        
        /* get player id from bidding system */
        char distributed_msg[BUF_SIZE];
        fgets(distributed_msg, BUF_SIZE, stdin);

        /* recept a message from bidding system */
        if(strncmp(distributed_msg, "-1 -1 -1 -1", 11) == 0){        /* all competition are done */
            for(idx=0; idx < 5; ++idx)
                unlink(FIFO_name[idx]);
            return 0;
        }
        else{        /* competition is not yet finished */
            stoi(distributed_msg, playerid);

            /* read respond from 4 player */
            int fdr = open(FIFO_name[0], O_RDONLY);
            FILE *fpr = fdopen(fdr, "r");
            FD_SET(fdr, &readset);

            /* file desc and FILE object */
            int fdw[PLAYER_NUM];
            FILE *fpw[PLAYER_NUM];

            /* initial each player's money and score */
            Player players[PLAYER_NUM];
            init_player(players);

            /* fork 4 players */
            int random_int_key[PLAYER_NUM] = {0, 0, 0, 0};
            for(idx = 0; idx < PLAYER_NUM; ++idx){
                
                /* generate random number for avoiding cheating */
                random_int_key[idx] = ( ( (unsigned short)rand() ) % 65536 );

                if( ( pid = fork() ) < 0 ){
                    err_sys("fork error");
                }
                else if( pid == 0){        /* child*/

                    char player_idx[2] = "A";
                    char random_ch_key[6] = "";
                    player_idx[0] = ('A' + idx);
                    snprintf(random_ch_key, sizeof(random_ch_key), "%d", random_int_key[idx]);
                    if( execl("./player", "player", host_id, player_idx, random_ch_key, NULL) == -1)
                        err_sys("execl error");

                }
                else{        /* parent*/
                   
                    /* record child pid */
                    player_pid[idx] = pid;

                    /* sent first message to player */
                    snprintf(money_msg, sizeof(money_msg), "%d %d %d %d", players[0].money, players[1].money, players[2].money, players[3].money); 
                    fdw[idx] = open(FIFO_name[idx+1], O_WRONLY);
                    fpw[idx] = fdopen(fdw[idx], "w");
                    fprintf(fpw[idx], "%s\n", money_msg);
                    fflush(fpw[idx]);
                }
            } /* end of for loop: fork child */

            struct timeval timeout;

            int cur_round;
            for(cur_round = 0; cur_round < MAX_ROUND; ++cur_round){
                
                /* infomation of each round */
                int price_arr[PLAYER_NUM];
                bool already_reply[PLAYER_NUM] = {false, false, false, false}; 

                /* receive player's reply in current round */
                while( !(already_reply[0] && already_reply[1] && already_reply[2] && already_reply[3]) ){
                    tmp_readset = readset;

                    /* select for player reply */
                    timeout.tv_sec = 5;
                    timeout.tv_usec = 0;
                    int sl;
                    sl = select(fdr + 1, &tmp_readset, NULL, NULL, &timeout);
                    if(sl == -1){
                        err_sys("error select");
                    }
                    else if(sl == 0){
                        printf("select timeout\n");
                        continue;
                    }
                    
                    /* read reply from player */
                    char player_reply_msg[BUF_SIZE];
                    if(FD_ISSET(fdr, &tmp_readset)){
                        fseek(fpr, 0, SEEK_SET);
                        fgets(player_reply_msg, BUF_SIZE, fpr);
                    }
                   
                    /* check key and host */
                    int msg, price;
                    msg, price = check_key(player_reply_msg, random_int_key);

                    /* get bid price for each player if the message is valid */
                    switch(msg){
                        case A:
                            already_reply[A] = true;
                            price_arr[A] = price;
                            break;
                        case B:
                            already_reply[B] = true;
                            price_arr[B] = price;
                            break;
                        case C:
                            already_reply[C] = true;
                            price_arr[C] = price;
                            break;
                        case D:
                            already_reply[D] = true;
                            price_arr[D] = price;
                            break;
                        default:
                            break;
                    } // end of switch
                } // end of while loop: receive player's reply in current round

                /* find winner of this round */
                int winner_idx;
                winner_idx = find_winner(price_arr);

                /* update player's score and money */
                update_player_info(&price_arr[winner_idx], &players[winner_idx]);
                add_money(players);

                for(idx=0; idx < PLAYER_NUM; ++idx){
                    /* sent message to players */
                    snprintf(money_msg, sizeof(money_msg), "%d %d %d %d", players[0].money, players[1].money, players[2].money, players[3].money); 
                    fdw[idx] = open(FIFO_name[idx+1], O_WRONLY);
                    fpw[idx] = fdopen(fdw[idx], "w");
                    fprintf(fpw[idx], "%s\n", money_msg);
                    fflush(fpw[idx]);
                }

            } // end of for loop: total round  


        } // end of read the message from bidding system
    }
    int i;
    for(i = 0; i < PLAYER_NUM + 1; ++i){
        if( unlink(FIFO_name[i]) != 0 )
            err_sys("remove error\n");
    }
    return 0;
}

void stoi(const char * const str, int *id)
{
    int a, b, c, d;
    sscanf(str, "%d %d %d %d", &a, &b, &c, &d);
    id[0] = a;
    id[1] = b;
    id[2] = c;
    id[3] = d;
}

void err_sys(const char *x) 
{ 
    perror(x); 
    exit(EXIT_FAILURE); 
}

/*
int compare(const void *a, const void *b)
{
    return *(int *)a - *(int *)b;
}
*/

void name(char *name, const char *id, const char *idx)
{
    char str1[] = "host";
    char str2[] = ".FIFO";
    strcpy(name, str1);
    strcat(name, id);
    if(idx != "NULL") strcat(name, idx);
    strcat(name, str2);
}

void init_player(Player *p)
{
    int i;
    for(i = 0; i < PLAYER_NUM; ++i){
        p[i].money = 1000;
        p[i].score = 0;
    }
}

int check_key(const char *reply_msg, const int *key_arr)
{
    char idx;
    int price, key;
    sscanf(reply_msg, "%c %d %d", &idx, &price, &key);
    if (price >= 0)
        return isvalue_in_array((&key), key_arr, PLAYER_NUM), price;
    else
        return -1, -1;
}

/* check wheter value in array or not and return idx if it exists */
int isvalue_in_array(const int *val, const int *key_arr, int size)
{   
    int i;
    for(i = 0; i < size; ++i){
        if( (*val) == key_arr[i] )
            return i;
    }
    return -1;
}

int find_winner(const int *price_arr)
{
    int frt_id = 0;
    int snd_id = -1;
    int same = -1;
    int i;
    for(i = 1; i < PLAYER_NUM; ++i){
         if(price_arr[i] == price_arr[frt_id])
             same = 1;
         if(price_arr[i] > price_arr[frt_id]){
             snd_id = frt_id;
             frt_id = i;
         }
         if(same)
           return snd_id;
         else
           return frt_id;
    }
}

void update_player_info(const int * const price, Player *player)
{
    ++(*player).score;
    (*player).money -= (*price); 
}

void add_money(Player *p)
{
    int i;
    for(i = 0; i < PLAYER_NUM; ++i)
        p[i].money += 1000;
}
