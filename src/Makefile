#The following four macros should be defined.
TARGET1=ds_prog1
OBJECT1=ds_prog1.o
CC=gcc
CXX=g++
LD_FLAGS=
C_FLAGS=
# end of user configuration
all: $(TARGET1) $(TARGET2)
$(TARGET1) : $(OBJECT1)
	$(CXX) -w $^ -o $@ $(LD_FLAGS) 
%.o : %.c
	$(CC) -w $(C_FLAGS) -c $<
.PHONY: clean
clean :
	-rm -f $(OBJECT1) $(TARGET1)
