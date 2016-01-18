all: merger.cpp 
	g++ merger.cpp -o merger -lpthread 
run:
	./merger 6 < sample_input.txt 
clean:
	rm -f merger 
