AR       := ar
ARFLAGS  := cr
CXX      := g++
CPPFLAGS := -g

all : ../bin/dtw
	@echo -n ""
	rm *.o

clean :
	rm *.o ../bin/dtw

../bin/dtw : main.o dtw.o
	$(CXX) $^ -o $@

main.o : main.cpp dtw.h 
	$(CXX) $(CPPFLAGS) -c $< -o $@

dtw.o : dtw.cpp dtw.h
	$(CXX) $(CPPFLAGS) -c $< -o $@

#word.o : word.cpp word.h
#	$(CXX) $(CPPFLAGS) -c $< -o $@

