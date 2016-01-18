CC=g++
TARGET=merger
all: $(TARGET).cpp
	$(CC) -o $(TARGET) (TARGET).cpp -lpthread 
run:
	./merger 100000 < test_10000000
clean:
	rm -f merger 
