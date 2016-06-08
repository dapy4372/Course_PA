TARGET=../bin/kdtree
OBJS= main.o Node.o kdtree.o
CXX      := g++
CPPFLAGS := -g -Wall
#CPPFLAGS := -O2 -I../lib

all : $(TARGET)
	@echo -n ""

$(TARGET) : $(OBJS)
	$(CXX) $(CPPFLAGS) $^ -o $@

%.o : %.cpp
	$(CXX) $(CPPFLAGS) -c $<

.PHONY: clean

clean :
	-rm -f $(OBJS) $(TARGET)

#kdtree.o : kdtree.cpp
#	$(CXX) $(CPPFLAGS) -c $<

#Node.o : Node.cpp
#	$(CXX) $(CPPFLAGS) -c $<

#main.o : main.cpp
#	$(CXX) $(CPPFLAGS) -c $<
