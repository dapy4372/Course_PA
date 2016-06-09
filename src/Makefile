TARGET1=../bin/kdtree
OBJS=main.o
CXX      := g++
#CPPFLAGS := -g -Wall -D DIM=2
CPPFLAGS := -O2 -Wall -D DIM=2

all : $(TARGET1)
	@echo -n ""

$(TARGET1) : $(OBJS)
	$(CXX) $(CPPFLAGS) $^ -o $@

main.o : main.cpp kdtree.h kdtree.cpp utils.h utils.cpp Node.h Node.cpp
	$(CXX) $(CPPFLAGS) -c -o main.o main.cpp

.PHONY: clean

clean :
	-rm -f $(OBJS) $(TARGET)
