#include <stdlib.h>
#include <sys/wait.h>
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
#define MAX_ROUND 10
#define PLAYER_NUM 4

enum{A, B, C, D};
typedef struct
{
    char idx[2];
    int id, money, score, rank, price;
    unsigned short key;
}Player;

void err_sys(const char *);
void err_quit(const char *);
void TELL_PARENT(pid_t , int);
void WAIT_PARENT(int);
void TELL_CHILD(int);
void WAIT_CHILD(int);

void name(char [15], const char *, const char *);
void init_players(Player *);
bool check_price(Player *);
int cmp_price(const void *, const void *);
int find_winner(Player *, int);
bool check_key(Player, int);
void update_player(Player *);
void add_money(Player *);
int cmp_score(const void *, const void *);
void rank(Player *, int);

int main(int argc, char **argv)
{
    
    int i;
    if(argc != 2)
        err_sys("ERROR! Usage: $./host [host_id]\n");
    char *host_id = argv[1];

    /* give FIFO name */
    char FIFO_name[5][15];
    name(FIFO_name[0], host_id, "_A");
    name(FIFO_name[1], host_id, "_B");
    name(FIFO_name[2], host_id, "_C");
    name(FIFO_name[3], host_id, "_D");
    name(FIFO_name[4], host_id, NULL);

    int fdw[PLAYER_NUM];
    FILE *fpw[PLAYER_NUM];  

    while(1){

        /* make FIFOs */
        for(i = 0; i < PLAYER_NUM + 1; ++i)
            mkfifo(FIFO_name[i], 0777);

        int fdr = open(FIFO_name[4], O_RDWR);
        FILE *fpr = fdopen(fdr, "r+");
        setbuf(fpr, NULL);

        /* get player information from bidding system */
        char buf[BUF_SIZE];
        fgets(buf, sizeof(buf), stdin);

        /* all competition are done, host will be kill */
        if(strncmp(buf, "-1 -1 -1 -1", 11) == 0){
            for(i = 0; i < PLAYER_NUM + 1; ++i)
                if( unlink(FIFO_name[i]) != 0 )
                    err_sys("remove error\n");
            return 0;
        }

        Player players[PLAYER_NUM];
        init_players(players);
        sscanf(buf, "%d %d %d %d\n", &players[A].id, &players[B].id, &players[C].id, &players[D].id);

        for(i = 0; i < PLAYER_NUM; ++i){
            fdw[i] = open(FIFO_name[i], O_RDWR);
            fpw[i] = fdopen(fdw[i], "r+");
            setbuf(fpw[i], NULL);
        }

        /* generate random key */
        //for(i = 0; i < PLAYER_NUM; ++i){
            //players[i].key = ( ( (unsigned short)rand()) % 65536 );
            //fprintf(stderr, "i = %d, key = %d, idx = %s, id = %d, rank = %d\n", i, players[i].key, players[i].idx, players[i].id, players[i].rank);
        //}
        pid_t player_pid[PLAYER_NUM];
        for(i = 0; i < PLAYER_NUM; ++i){
            /* pipe for avoiding race condition */
            int fd[2];
            if(pipe(fd) < 0)
                err_sys("pipe error");
           // fprintf(stderr, "fd[0] = %d, fd[1] = %d\n", fd[0], fd[1]);
           
            pid_t pid;
            if( (pid = fork()) < 0 )
                err_sys("fork error");
            else if(pid == 0){
                if( ( pid = fork() ) < 0)
                    err_sys("fork error");
                else if(pid == 0){
                    /* first child go first */
                    WAIT_PARENT(fd[0]);
                    close(fd[0]);
                    close(fd[1]);
                    char ch_key[6] = "";
                    snprintf(ch_key, sizeof(ch_key), "%d", players[i].key);
                    //fprintf(stderr, "%s\n", ch_key);
                    if( execl("./player", "player", host_id, players[i].idx, ch_key, (char*)0) == -1 ){
                        fprintf(stderr, "execl error");
                        err_sys("execl error");
                    }
                }
                else{
                    TELL_CHILD(fd[1]);
                    exit(0);
                }
            }
            else{
                if( waitpid(pid, NULL, 0) != pid)
                    err_sys("waitpid error");
                player_pid[i] = pid;
                close(fd[0]);
                close(fd[1]);
                /* sent first message to player */
                //char msg_money[BUF_SIZE];
                //sprintf(msg_money, "%d %d %d %d\n", players[A].money, players[B].money, players[C].money, players[D].money);
                //fprintf(stderr, "=== %s === \n", msg_money);
                //write(fdw[i], msg_money, sizeof(msg_money));
                //fsync(fdw[i]);
                //fpw[i] = fdopen(fdw[i], "w");
                //fprintf(fpw[i], "%d %d %d %d\n", players[A].money, players[B].money, players[C].money, players[D].money);
                //fflush(fpw[i]);
                //fsync(fdw[i]);
                //fprintf(stderr, "fdw[%d] = %d\n", i, fdw[i]);
            }
        }
       
        //FILE *fpr = fdopen(fdr, "r");

        fd_set reply_set, ready_read_set;
        FD_ZERO(&reply_set);
        FD_ZERO(&ready_read_set);
        FD_SET(fdr, &reply_set);
        //fprintf(stderr, "fdr = %d\n", fdr);
         
        int cur_round;
        for(cur_round = 0; cur_round < MAX_ROUND; ++cur_round){

            /* sent msg to players*/
            for(i = 0; i < PLAYER_NUM; ++i){
                char msg_money[BUF_SIZE];
                sprintf(msg_money, "%d %d %d %d\n", players[A].money, players[B].money, players[C].money, players[D].money);

                fseek(fpw[i], 0, SEEK_SET);
                //fdw[i] = open(FIFO_name[i], O_RDWR);
                //write(fdw[i], msg_money, sizeof(msg_money));
                //fsync(fdw[i]);
                fprintf(fpw[i], "%d %d %d %d\n", players[A].money, players[B].money, players[C].money, players[D].money);
                fflush(fpw[i]);        
                fsync(fdw[i]);
            }

            /* information of each round */
            bool done[PLAYER_NUM] = {false, false, false, false};

            /* get reply msg from players */
            while( !(done[0] && done[1] && done[2] && done[3]) ){

                ready_read_set = reply_set;
                /* select for players reply */
                int sl;
                sl = select(fdr + 1, &ready_read_set, NULL, NULL, NULL);
                if(sl == -1)
                    err_sys("select error\n");
                
                /* read reply from players */
                if( FD_ISSET(fdr, &ready_read_set) ){

                    char idx[2];
                    int key, price, int_idx; 
                    char msg[BUF_SIZE];
                    fgets(msg, sizeof(msg), fpr);
                    //read(fdr, msg, sizeof(msg));
                    sscanf(msg, "%s %d %d\n", idx, &key, &price);
                    //fprintf(stderr, "%s\n", msg);                
                    //fseek(fpr, 0, SEEK_SET);
                    int_idx = idx[0] - 'A';
                    if( check_key(players[int_idx], key) ){
                        players[int_idx].price = price;
                        done[int_idx] = true;
                    }
                }
            }

            /* check price is smaller than player's money */
            if(check_price(players)){
                --cur_round;
                continue;
            }

            /* find winner in this round */
            int winner_idx;
            if( (winner_idx = find_winner(players, PLAYER_NUM)) < 0 ){
                --cur_round;
                continue;
            }

            update_player(&players[winner_idx]);

            /* add money for next round */
            add_money(players);
        
        }    /* done for all rounds */
     
        rank(players, PLAYER_NUM);

        for(i = 0; i < PLAYER_NUM; ++i){
            fprintf(stdout, "%d %d\n", players[i].id, players[i].rank);
            fflush(stdout);
        }

        int j;
        for(j = 0; j < PLAYER_NUM; ++j){
          //  kill(player_pid[j], SIGKILL);
          //  int status = -1;
          //  wait(&status);
            fclose(fpw[j]);
            close(fdw[j]);
        }
        fclose(fpr);
        close(fdr);
    }
    return 0;
}
    /* read from parent */
  /*  char msg1[BUF_SIZE];
    char msg2[BUF_SIZE];
    char buf[BUF_SIZE];
    //fscanf(stdin, "%s %s", msg1,msg2);
    //fprintf(stdout, "%s %s \n", msg1, msg2);
    int ii[4];
    
    fscanf(stdin, "%d %d %d %d", &ii[0], &ii[1], &ii[2], &ii[3]);
//    fprintf(stdout, "%d %d %d %d\n", ii[0], ii[1], ii[2], (ii[3]+1));
//    fflush(stdout);

    int fdr = open(FIFO_name1, O_RDWR);
    FILE *fpr = fdopen(fdr,"r"); 

    fd_set reply_set, ready_read_set;
    FD_ZERO(&reply_set);
    FD_ZERO(&ready_read_set);
    int max_fd = 0;
    FD_SET(fdr, &reply_set);
    if(fdr > max_fd)
        max_fd = fdr;

    struct timeval timeout;
*/
    /* sent a message to gradchild A */
  /*  int fdw;
    FILE *fpw;
    fdw = open(FIFO_name2, O_RDWR);
    fpw = fdopen(fdw, "w");
*/
    /* sent to grandchild*/
    /*fprintf(fpw, "%d %d %d %d\n", ii[0], ii[1], ii[2], ii[3]);
    fflush(fpw);

    while(1){
        ready_read_set = reply_set;
        timeout.tv_sec = 10;
        timeout.tv_usec = 0;
        int sl;
        sl = select(max_fd+1, &ready_read_set, NULL, NULL, &timeout);
        if( sl == -1 )
            perror( "select error\n");

        if(FD_ISSET(fdr, &ready_read_set)){
            int i[4];
            fscanf(fpr, "%d %d %d %d", &i[0], &i[1], &i[2], &i[3]);
            //read(fdr, buf, BUF_SIZE);
            fprintf(stdout,"%d %d %d %d\n", i[0], i[1], i[2], (i[3]+1));
            fflush(stdout);
            //fprintf(stderr,"%d %d %d %d", i[0], i[1], i[2], i[3]);
        }
    }
    exit(0);
}*/

void err_sys(const char * x)
{
    perror(x);
}

void err_quit(const char * x)
{
    perror(x);
    exit(1);
}

void TELL_PARENT(pid_t pid, int fd)
{
    if(write(fd, "c", 1) != 1)
        err_sys("write error");
}

void WAIT_PARENT(int fd)
{
    char c;
    if(read(fd, &c, 1) != 1)
        err_sys("read error");
    if(c != 'p')
        err_quit("WAIT_PARENT: incorrect data");
}

void TELL_CHILD(int fd)
{
    if(write(fd, "p", 1) != 1)
      err_sys("write error");
}

void WAIT_CHILD(int fd)
{
    char c;
    if(read(fd, &c, 1) != 1)
        err_sys("read error");
    if(c != 'c')
        err_quit("WAIT_CHILD: incorrect data");
}

void name(char name[15], const char *id, const char *idx)
{
    char str1[] = "host";
    char str2[] = ".FIFO";
    strcpy(name, str1);
    strcat(name, id);
    if(idx != NULL) strcat(name, idx);
    strcat(name, str2);
}

void init_players(Player *p)
{
    int i;
    unsigned short tmpkey = ( ( (unsigned short)rand() ) % 65532 );
    for(i = 0; i < PLAYER_NUM; ++i){
        p[i].id = i;
        p[i].money = 1000;
        p[i].score = 0;
        p[i].key = tmpkey+i;
        p[i].idx[0] = ('A' + i);
        p[i].idx[1] = '\0';
        p[i].rank = 0;
        p[i].price = 0;
    }
}

bool check_price(Player *players)
{   
    int i;
    for(i = 0; i < PLAYER_NUM; ++i){
        if(players[i].price > players[i].money)
            return true;
    }
    return false;
}

int cmp_price(const void *a, const void *b)
{
    return (*(Player*)b).price - (*(Player*)a).price;
}

int find_winner(Player *players, int player_num)
{
    Player tmp_players[player_num];
    memcpy(tmp_players, players, player_num * sizeof(Player));
    qsort(tmp_players, player_num, sizeof(Player), cmp_price);

    if(tmp_players[0].price > tmp_players[1].price)
        return (tmp_players[0].idx[0] - 'A');
    else if(tmp_players[0].price > tmp_players[2].price){
        if(tmp_players[2].price > tmp_players[3].price)
            return (tmp_players[2].idx[0] - 'A');
        else
            return -1;
    }
    else if(tmp_players[0].price > tmp_players[3].price)
        return (tmp_players[3].idx[0] - 'A');
    return -1;
}

bool check_key(Player p, int key)
{
    if(p.key == key)
        return true;
    else
        return false;
}

void update_player(Player *player)
{
    ++(*player).score;
    (*player).money -= (*player).price; 
}

void add_money(Player *p)
{
    int i;
    for(i = 0; i < PLAYER_NUM; ++i)
        p[i].money += 1000;
}

int cmp_score(const void *a, const void *b)
{
    return (*(Player*)b).score - (*(Player*)a).score;
}

//TODO check rank 
void rank(Player *players, int player_num)
{
    Player tmp_players[player_num];
    memcpy(tmp_players, players, player_num * sizeof(Player));
    qsort(tmp_players, player_num, sizeof(Player), cmp_score);

    int i, prev_score, cur_rank = 1, same_score_num = 0;
    for(i = 0; i < player_num; ++i){
        Player *tmp = &players[(tmp_players[i].id)];
        if(i == 0){
            (*tmp).rank = 1;    
        }
        else if(i != 0 && (*tmp).score == prev_score ){
            (*tmp).rank = cur_rank;
            ++same_score_num;
        }
        else{
            (*tmp).rank = cur_rank = (cur_rank + same_score_num + 1);
            same_score_num = 0;
        }
        prev_score = (*tmp).score;
    }
}
