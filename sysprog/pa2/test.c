#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/select.h>
#include <string.h>
#include <fcntl.h>

typedef struct
{
    char idx;
    int num;
}Player;

void check_array(int *, const int * const, const char * const);
void init(Player *, int );
enum{a=1, b=0};
enum{true=1, false=0};

int main()
{
    char aa[2] = "A";
    aa[0] = aa[0] + 1;
    printf("%s\n", aa);
    int c = aa[0]-'A';
    printf("%d\n", c);

    printf("\n === \n%d %d \n === \n", a, b);
    printf("\n === \n%d %d \n === \n", true, false);
    Player player1;
    int fd = open("./abc", O_RDONLY);
    FILE *fp;
    fp = fdopen(fd, "r");
    char buf[20], out[20];
    int i;
    fgets(buf, sizeof(buf), fp);
    sscanf(buf, "%s %d", out, &i);
    //printf("%s", buf);
    //player1.idx = fgetc(fp);
//    printf("%s = %d", out, i); 
    //fgets((char)player1.num, sizeof(player1.num), fp);
    //printf("%c,%d", player1.idx, player1.num); 
    //printf("%c %d", player1.idx, player1.num);
    
    Player players[5];
    init(players, 5);

    for(i=0; i< 5; ++i){
        printf("%d %d =", players[i].idx, players[i].num);
    }

    int n = 10;
    int a[10];   
    check_array(a, &n, "abcdf");

    int j;
    for(j = 0; j < 10; ++j)
        printf("%d\n", a[j]);
    
    int tmp[20] = {1};
    for(j = 0; j < 20; ++j){
       printf("%d", tmp[j]);
    }



    return 0;
}

void init(Player *p, int n)
{
    int i;
    for(i=0; i<n; ++i){
        p[i].idx = i;
        p[i].num = i;
    }
}

void check_array(int *a, const int * const n, const char * const str)
{
    printf("\n === \n %s \n === \n", str);
    int i;
    for(i = 0; i < (*n); ++i)
        a[i] = i;
}
