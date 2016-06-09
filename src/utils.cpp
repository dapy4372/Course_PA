static void err_sys(const char *x)
{
    perror(x);
    exit(1);
}

template < class T >
static vector< Element<T> > readFile( char *filename)
{
    FILE *fp = fopen( filename, "r");
    if( fp == NULL )
        err_sys("open file failure!");

    char *line = NULL;
    size_t len = 0;
    Element<T> el;
    vector< Element<T> > el_vec;
    while( getline(&line, &len, fp) != -1){
        sscanf(line, "%lf %lf", &el.keys[0], &el.keys[1] );
        el_vec.push_back(el);
    }
    fclose(fp);
    return el_vec;
}

template < class T >
static T square(const T &a)
{
    return a*a;
}

static void printTimes(clock_t real, struct tms *tmsstart, struct tms *tmsend)
{
    static long clktck = 0;
    if( clktck == 0 )
        if( (clktck = sysconf(_SC_CLK_TCK)) < 0 )
            err_sys("sysconf error");
    printf("   real: %7.2f (s)\n", real / (double) clktck);
    printf("   user: %7.2f (s)\n", (tmsend->tms_utime - tmsstart->tms_utime) / (double) clktck);
    printf("   sys:  %7.2f (s)\n", (tmsend->tms_stime - tmsstart->tms_stime) / (double) clktck);
}
