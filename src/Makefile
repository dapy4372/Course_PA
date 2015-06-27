AR       := ar
ARFLAGS  := cr
CXX      := g++
CPPFLAGS := -g


all : ../bin/dtw
	@echo -n ""

clean :
	rm *.o ../bin/dtw_v1

../bin/dtw : main.o dtw_v1.o
	$(CXX) $^ -o $@

main.o : main.cpp dtw_v1.h
	$(CXX) $(CPPFLAGS) -c $< -o $@

pattern.o : dtw_v1.cpp dtw_v1.h
	$(CXX) $(CPPFLAGS) -c $< -o $@
