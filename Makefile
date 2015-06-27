# CC and CFLAGS are varilables
CC = g++
CFLAGS = -c
AR = ar
ARFLAGS = rcv
# -c option ask g++ to compile the source files, but do not link.
# -g option is for debugging version
# -O2 option is for optimized version
DBGFLAGS = -g -D_DEBUG_ON_
OPTFLAGS = -o2

all	: bin/dtw
	@echo -n ""
	rm -rf *.o lib/*.a lib/*.o

# optimized version
bin/dtw	: dtw_v1_opt.o main_opt.o lib
			$(CC) $(OPTFLAGS) dtw_v1_opt.o main_opt.o -ltm_usage -Llib -o bin/dtw
main_opt.o 	   	: src/main.cpp src/util.h lib/tm_usage.h
			$(CC) $(CFLAGS) $< -Ilib -o $@
dtw_v1_opt.o	: src/dtw_v1.cpp src/dtw_v1.h
			$(CC) $(CFLAGS) $(OPTFLAGS) $< -o $@

# DEBUG Version
dbg : bin/dtw_dbg
	@echo -n ""

bin/dtw_dbg	: dtw_v1_opt.o main_dbg.o lib
			$(CC) $(DBGFLAGS) pattern.o pattern_dbg.o -ltm_usage -Llib -o bin/lllw_dbg
main_dbg.o: src/main.cpp src/util.h lib/tm_usage.h
			$(CC) $(CFLAGS) $< -Ilib -o $@
dtw_v1_opt.o	: src/dtw_v1.cpp src/dtw_v1.h
			$(CC) $(CFLAGS) $(DBGFLAGS) $< -o $@

lib: lib/libtm_usage.a

lib/libtm_usage.a: tm_usage.o
	$(AR) $(ARFLAGS) $@ $<
tm_usage.o: lib/tm_usage.cpp lib/tm_usage.h
	$(CC) $(CFLAGS) $<

# clean all the .o and executable files
clean:
		rm -rf *.o lib/*.a lib/*.o bin/*

