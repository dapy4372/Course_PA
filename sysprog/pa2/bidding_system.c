#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/select.h>
#include <string.h>

#define READ_END 0
#define WRITE_END 1
#define BUF_SIZE 128

typedef struct
{
    int id;
    int score;
}Player_info;

/* competition */
typedef struct
{
    int a, b, c, d;
}Comp;

void err_sys(const char*);
int main(int argc, char **argv)
{
    if(argc != 3){
        fprintf(stderr, "ERROR! Usage: $./bidding_system [host_num] [player_num]\n");
        exit(EXIT_FAILURE);
    }
    
    int h_num = atoi(argv[1]);
    int p_num = atoi(argv[2]);
    int b_to_h[h_num][2], h_to_b[h_num][2];
    int host_pid[h_num];
    FILE *fpr[h_num];
    FILE *fpw[h_num];
    pid_t pid;
    fd_set reply_set, ready_read_set;
    FD_ZERO(&reply_set);
    Player_info player_info[p_num];

    /* initial players info */
    int idx;
    for(idx = 0; idx < p_num; ++idx){
        player_info[idx].id = idx+1;
        player_info[idx].score = 0;
    }


    /* C(p_num, 4) */
    int max_comp = (p_num)*(p_num-1)*(p_num-2)*(p_num-3)/24;
    Comp comps[max_comp];

    /* initial all combination of players */
    int i, j, k, l, comp_idx=0;
    for(i = 0; i < p_num; ++i)
        for(j = 1; j < p_num; ++j)
            for(k = 2; k < p_num; ++k)
                for(l = 3; l < p_num; ++l)
                    if(l > k && k > j && j > i){
                        Comp tmp = {i+1, j+1, k+1, l+1};
                        comps[comp_idx] = tmp; 
                        ++comp_idx;
                    }

    int max_fd = 0;
    //bool closed_host[h_num];
    
    /* recording the number of sent competition */
    int cur_comp = 0;

    /* create pipe and fork child process */
    for(idx = 0; idx < h_num; ++idx){
        //closed_host[idx] = false;
        
        /* create pipe */
        if( pipe(b_to_h[idx]) < 0 || pipe(h_to_b[idx]) < 0 )
            err_sys("pipe error");
        
        /* add to reply set */
        FD_SET(h_to_b[idx][READ_END], &reply_set);
        if(h_to_b[idx][READ_END] > max_fd)
            max_fd = h_to_b[idx][READ_END];
        
        if( ( pid = fork() ) < 0 ){
            err_sys("fork error");
        }
        else if (pid == 0){                /* child */

            /* redirect fd of child */
            dup2(b_to_h[idx][READ_END], STDIN_FILENO);
            dup2(h_to_b[idx][WRITE_END], STDOUT_FILENO);

            /* close unnecessary fd */
            close(b_to_h[idx][READ_END]);
            close(h_to_b[idx][READ_END]);

            /* close unnecessary fd */
            close(b_to_h[idx][WRITE_END]);
            close(h_to_b[idx][WRITE_END]);
            
            /* execl */
            char h_id[2]; 
            snprintf(h_id, sizeof(h_id),"%d", idx+1);
            if( execl("./host", "host", h_id, (char*)0) )
                err_sys("execl error for");

        }
        else{                /* parent */

            /* close unnecessary fd */
            close(b_to_h[idx][READ_END]);
            close(h_to_b[idx][WRITE_END]);
            
            /* standard I/O */
            fpw[idx] = fdopen(b_to_h[idx][WRITE_END], "w");
            fpr[idx] = fdopen(h_to_b[idx][READ_END], "r");

            /* close unnecessary host */
            if(idx > max_comp){
                fprintf(fpw[idx], "0 0 0 0\n");
                fflush(fpw[idx]);
                //colsed_host[idx] = 1;
            }else{        /* sent fisrt distribute */
                fprintf(fpw[idx], "%d %d %d %d\n", comps[idx].a, comps[idx].b, comps[idx].c, comps[idx].d);
                fflush(fpw[idx]);
                ++cur_comp;
            }
            /* record host pid */
            host_pid[idx] = pid;
        }
    
    }

    /* I/O multiplexing */

    /* set timeout */
    struct timeval timeout;
    timeout.tv_sec = 30;
    timeout.tv_usec = 0;

    int host_id, find = 0;
    char buf[BUF_SIZE];
    
    int comp_done = 0;

    while(comp_done < max_comp){
        
        /* select for ready host */
        ready_read_set = reply_set;
        int sl;
        int find = 0;
        sl = select(max_comp+1, &ready_read_set, NULL, NULL, &timeout);
        if(sl==-1)
            err_sys("select error");
        else if (sl==0){
            printf("select timeout\n");
            continue;
        }
        
        /* find currently ready host */
        int cur_ready_host;
        for(cur_ready_host = 0; cur_ready_host < h_num && !find; ++cur_ready_host)
            if(FD_ISSET(h_to_b[cur_ready_host][READ_END], &ready_read_set))
                find = 1;

//TODO if find = 0...

        /* read the message from currently ready host */
        while(!feof(fpr[cur_ready_host])){
            fgets(buf, BUF_SIZE, fpr[cur_ready_host]);
            printf("%s\n", buf);
        }

        /* sent new competition to currently read host */
        if(cur_comp < max_comp){
            fprintf(fpw[cur_ready_host], "%d %d %d %d\n", comps[cur_ready_host].a, comps[cur_ready_host].b, comps[cur_ready_host].c, comps[cur_ready_host].d);
            fflush(fpw[cur_ready_host]);
            ++cur_comp;
        }

        /* record the finished competition */
        ++comp_done;
    }        /* finish all competition */

        /*
        for(idx = 0; idx < h_num; ++idx){
            if(closed_host[idx]) continue;
            while(!feof(fpr[idx][READ_END])){
                fgets(buf, BUF_SIZE, fpr[idx]);
                printf("%s" % buf);
            }
        }*/

        /*
        for(; comp_idx < max_comp; ++comp_idx){
        for(idx = 0; idx < host_num; ++i){
            snprintf(buf, sizeof(buf), "%d %d %d %d\n", players[idx].a, player[idx].b, player[idx].c, player[idx].d);
            if( write(b_to_h[idx][WRITE_END], buf, sizeof(buf)) != sizeof(buf) )
                err_sys("write error");     
        }*/
}

void err_sys(const char* x) 
{ 
    perror(x); 
    exit(1); 
}
