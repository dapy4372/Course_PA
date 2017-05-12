#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/select.h>
#include <string.h>
#include <stdbool.h>

#define READ_END 0
#define WRITE_END 1
#define BUF_SIZE 128

typedef struct
{
    int id, score, rank;
}Player;

typedef struct
{
    int a, b, c, d;
}Comp;

typedef struct
{
    int player_rank[4];
    int player_id[4];
    int ready;
    bool idle, off;
}Host;

enum{ A, B, C, D };

void err_sys(const char *);
void TELL_PARENT(pid_t , int);
void WAIT_PARENT(int);
void TELL_CHILD(int);
void WAIT_CHILD(int);

void init_comps(Comp *, int);
void init_host(Host *);
void init_players(Player *, int);
void init_comps_status(int *, int max_comps);
int get_undone_comp(int *, int);
int get_comps_idx(int *, int, int);
void update_score(Host *, Player *, int);
int cmp(const void *, const void *);
void rank(Player *, int);

int main(int argc, char **argv)
{
    if(argc != 3)
        err_sys("ERROR! Usage: $./bidding_system [host_num] [player_num]\n");

    int host_num = atoi(argv[1]);
    int player_num = atoi(argv[2]);

    if(host_num > 12 || host_num < 0 || player_num > 20 || player_num < 0)
        err_sys("ERROR! 0 < host_num < 12, 0 < player_num < 20\n");

    int bid_to_host[host_num][2], host_to_bid[host_num][2];
    FILE *fpr[host_num];
    FILE *fpw[host_num];
    
    int max_comps = (player_num)*(player_num-1)*(player_num-2)*(player_num-3)/24;
    Comp comps[max_comps];
    init_comps(comps, player_num); 
    

    Host hosts[host_num];
    int i;
    for(i = 0; i < host_num; ++i)
        init_host(&hosts[i]);
    
    Player players[player_num];
    init_players(players, player_num);

    int host_pid[host_num];
    //int cur_comp_idx = 0;

    /* -1 means the competition doesn't be done. */
    /* -2 means the competition is done. */
    /* >0 means the competition is processed by host now */
    int comps_status[max_comps]; 
    init_comps_status(comps_status, max_comps);

    for(i = 0; i < host_num; ++i){
        if( pipe(bid_to_host[i]) < 0 || pipe(host_to_bid[i]) < 0 )
            err_sys("pipe error\n");
        int tmp_pipe[2];
        pipe(tmp_pipe);

        pid_t pid;
        if( (pid = fork()) < 0)
            err_sys( "fork error\n");
        else if (pid == 0) {        /* beginning of first child */
            if( (pid = fork()) < 0 )
                err_sys("fork error");
            else if(pid == 0){        /* beginning of second child */

                /* first child go first */
                /* child to parent pipe cuz it doesn't dup2 */
                WAIT_PARENT(tmp_pipe[0]);
                
                /* redirect stdin and stdout */
                dup2(bid_to_host[i][0], STDIN_FILENO);
                dup2(host_to_bid[i][1], STDOUT_FILENO);

                /* close unnecessary fd */
                close(bid_to_host[i][0]);
                close(bid_to_host[i][1]);
                close(host_to_bid[i][0]);
                close(host_to_bid[i][1]);
                
                char host_id[3];
                snprintf(host_id, sizeof(host_id), "%d", i+1);
                /* exec */
                if( execl("./host", "host", host_id, (char*)0) == -1)
                    err_sys("execl error");

            }     /* end of second child */ 
            else{        /* beginning of first child */        
                TELL_CHILD(tmp_pipe[1]);
                exit(0); 
            }    /* end of first child */
        }    /* end of fork child to avoid Zombie */
        else{        /* parent */
            /* we don't need to use waitpid cuz we use vfork */
            if(waitpid(pid, NULL, 0) != pid)
                err_sys("waitpid error");

            /* close unnecessary fd */
            close(bid_to_host[i][0]);
            close(host_to_bid[i][1]);

            fpw[i] = fdopen(bid_to_host[i][1], "w");
            fpr[i] = fdopen(host_to_bid[i][0], "r");
            /*
            if(cur_comp_idx > max_comp){
                if(!hosts[i].off)
                    fprintf(fpw[i], "-1 -1 -1 -1\n");
                    fflush(fpw[i]);
                    hosts[i].off = true;
            }
            else{*/
            //TODO using array to record
            int undone_idx = get_undone_comp(comps_status, max_comps);
            if(undone_idx != -1){
                fprintf(fpw[i], "%d %d %d %d\n", comps[undone_idx].a, comps[undone_idx].b, comps[undone_idx].c, comps[undone_idx].d);
                fflush(fpw[i]);
                hosts[i].idle = false;
                //++cur_comp_idx;
                comps_status[undone_idx] = i;
            }
            host_pid[i] = pid;
        }
    }

    fd_set reply_set, ready_read_set;
    FD_ZERO(&reply_set);
    FD_ZERO(&ready_read_set);
    
    int max_fd = 0;
    for(i = 0; i < host_num; ++i){
        FD_SET(host_to_bid[i][0], &reply_set);
        if(host_to_bid[i][0] > max_fd)
            max_fd = host_to_bid[i][0];
    }

    struct timeval timeout;

    int finish_comp = 0;
    bool done_looping_select = false;

    while(!done_looping_select){
        
        ready_read_set = reply_set;
        
        timeout.tv_sec = 5;
        timeout.tv_usec = 0;
        int sl;
        sl = select(max_fd+1, &ready_read_set, NULL, NULL, NULL);
        if( sl == -1 )
            err_sys( "select error\n");
        
        /* find current replyed host idx */
        int cur_reply_host_idx = 0;
        while(cur_reply_host_idx < host_num){
            if(FD_ISSET(host_to_bid[cur_reply_host_idx][0], &ready_read_set))
                break;
            ++cur_reply_host_idx;
        }
       /* 
        int id[4], rank[4];
        fscanf(fpr[cur_reply_host_idx], "%d %d\n%d %d\n%d %d\n%d %d", &id[0], &rank[0], &id[1], &rank[1], &id[2], &rank[2], &id[3], &rank[3]);
        for(i = 0; i < 4; ++i)
            players[id[i]-1].score += (4 - rank[i]);
            */
/*
        for(i = 0; i < 3; ++i){
            fscanf(fpr[cur_reply_host_idx], "%d %d\n", &id[i], &rank[i]);
            players[id[i]-1].score += (4 - rank[i]);
        }
        fscanf(fpr[cur_reply_host_idx], "%d %d", &id[3], &rank[3]);
        players[id[3]-1].score += (4 - rank[3]);
  */    
        char buf[BUF_SIZE];
        int id, rank;
        bool error_return = false;
        for(i = 0; i < 4; ++i){
            fgets(buf, sizeof(buf), fpr[cur_reply_host_idx]);
            sscanf(buf, "%d %d\n", &id, &rank);
            if(id < 0 || rank < 0 || id > player_num || rank > player_num){
                error_return = true;
                fprintf(stderr, "id rank error");
                break;
            }
            players[id-1].score += (4 - rank);
        }

        int comps_idx = get_comps_idx(comps_status, max_comps, cur_reply_host_idx);
        if(!error_return){
            ++finish_comp;
            comps_status[comps_idx] = -2;
        }
        else{
            comps_status[comps_idx] = -1;
        }

        if(finish_comp > max_comps){
            fprintf(fpw[cur_reply_host_idx], "-1 -1 -1 -1\n");
            fflush(fpw[cur_reply_host_idx]);
            hosts[cur_reply_host_idx].off = true;
        }
        else{
            int undone_idx = get_undone_comp(comps_status, max_comps); 
            if(undone_idx != -1){
                fprintf(fpw[cur_reply_host_idx], "%d %d %d %d\n", comps[undone_idx].a, comps[undone_idx].b, comps[undone_idx].c, comps[undone_idx].d);
                fflush(fpw[cur_reply_host_idx]);
                hosts[cur_reply_host_idx].idle = false;
                comps_status[undone_idx] = cur_reply_host_idx;
            }
        }
        
        //int id, rank;
        //int buf[BUF_SIZE];
        //int id[4], rank[4];
        //fscanf(fpr[cur_reply_host_idx], "%d %d", &id, &rank);
        
        /* recept the msg from current reply host */
        /*
        if(!hosts[cur_reply_host_idx].idle && hosts[cur_reply_host_idx].ready != 4){
            if(hosts[cur_reply_host_idx].player_rank[id] == -1){
                hosts[cur_reply_host_idx].player_rank[id] = rank;
                ++hosts[cur_reply_host_idx].ready;
            }
        }
        */
        /* if the hosts have already finished, we update the player's score in this competition */
      /*  if(hosts[cur_reply_host_idx].ready == 4){
            update_score(&hosts[cur_reply_host_idx], players, player_num);
            init_host(&hosts[cur_reply_host_idx]);
            ++finish_comp;
        */
            /* sent a new competition */
        /*
            if(cur_comp_idx > max_comp){
                fprintf(fpw[cur_reply_host_idx], "-1 -1 -1 -1\n");
                fflush(fpw[cur_reply_host_idx]);
                hosts[cur_reply_host_idx].off = true;
            }
            else{
                fprintf(fpw[cur_reply_host_idx], "%d %d %d %d\n", comps[cur_comp_idx].a, comps[cur_comp_idx].b, comps[cur_comp_idx].c, comps[cur_comp_idx].d);
                fflush(fpw[cur_reply_host_idx]);
                ++cur_comp_idx;
                hosts[cur_reply_host_idx].idle = false;
            }
        }*/
        //fprintf(stderr,"%d %d %d %d", tmp[0], tmp[1], tmp[2], tmp[3]);

        if(finish_comp == max_comps)
            done_looping_select = true;
    }

    /* check for turning off all host */
    for(i = 0; i < host_num; ++i){
        if(!hosts[i].off){
            fprintf(fpw[i], "-1 -1 -1 -1\n");
            fflush(fpw[i]);
            hosts[i].off = true;
        }
    }
    
    rank(players, player_num);

    /* print rank */
    for(i = 0; i < player_num; ++i){
        fprintf(stdout, "%d %d\n", (players[i].id+1), players[i].rank);
        fflush(stdout);
    }
    exit(0);
}

void err_sys(const char * x)
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
        err_sys("WAIT_PARENT: incorrect data");
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
        err_sys("WAIT_CHILD: incorrect data");
}

void init_comps(Comp *comps, int player_num)
{
    int i, j, k, l, idx = 0;
    for(i = 0; i < player_num; ++i)
        for(j = 1; j < player_num; ++j)
            for(k = 2; k < player_num; ++k)
                for(l = 3; l < player_num; ++l)
                    if(l > k && k > j && j > i){
                        Comp tmp = {i+1, j+1, k+1, l+1};
                        comps[idx] = tmp; 
                        ++idx;
                    }
}

void init_host(Host *host)
{
    int i;
    for(i = 0; i < 4; ++i){
        (*host).player_rank[i] = -1;
        (*host).player_id[i] = -1;
    }
    (*host).ready = 0;
    (*host).idle = true;
    (*host).off = false;
}

void init_players(Player *players, int player_num)
{
    int i;
    for(i = 0; i < player_num; ++i){
        players[i].id = i;
        players[i].score = 0;
        players[i].rank = -1;
    }
}

void init_comps_status(int *comps_status, int max_comps)
{
    int i;
    for(i = 0; i < max_comps; ++i)
        comps_status[i] = -1;
}

int get_undone_comp(int *comps_status, int max_comps)
{
    int i;
    for(i = 0; i < max_comps; ++i){
        if(comps_status[i] == -1)
            return i;
    }
    /* retrun -1 means all comps are processed or done */
    return -1;
}

int get_comps_idx(int *comps_status, int max_comps, int cur_reply_host_idx)
{
    int i;
    for(i = 0; i < max_comps; ++i)
        if(comps_status[i] == cur_reply_host_idx)
            return i;
}

void update_score(Host *host, Player *players, int player_num)
{
    int i;
    for(i = 0; i < 4; ++i){
        if( (*host).player_id[i] < player_num )
            players[(*host).player_id[i]].score = 4 - (*host).player_rank[i];
        else
            err_sys("reply player's id error");
    }
}

int cmp(const void *a, const void *b)
{
    return (*(Player*)b).score - (*(Player*)a).score;
}

void rank(Player *players, int player_num)
{
    Player tmp_players[player_num];
    memcpy(tmp_players, players, player_num * sizeof(Player));
    qsort(tmp_players, player_num, sizeof(Player), cmp);

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
