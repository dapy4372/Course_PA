all: merge.cpp 
	g++ -g merge.cpp -o merge -lpthread 
run:
	./merge 3 < input.txt 
clean:
	rm -f merge 
