CC=g++
TARGET=merger
LIBS=-pthread
N=10000000

all: $(TARGET).cpp
	$(CC) -Wall $(TARGET).cpp -o $(TARGET) $(LIBS)

exp: $(TARGET).cpp
	$(CC) -Wall $(TARGET).cpp -D EXP -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET) 
