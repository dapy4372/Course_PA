all: bidding_system.c #customer.c 
	gcc -g bidding_system.c -o bidding_system
#gcc -g customer.c -o customer 
clean:
	rm -f bidding_system customer 
