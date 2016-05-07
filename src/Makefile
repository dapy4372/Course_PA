CXX      := g++
CPPFLAGS := -g -Wall
#CPPFLAGS := -O2 -I../lib

all : ../bin/kdtree
	@echo -n ""

clean :
	rm *.o ../bin/kdtree

../bin/kdtree : kdtree.o main.o
	$(CXX) $(CPPFLAGS) $^ -o $@

kdtree.o : kdtree.cpp kdtree.h
	$(CXX) $(CPPFLAGS) -c $<

main.o : main.cpp kdtree.h utils.h
	$(CXX) $(CPPFLAGS) -c $<
