CC = gcc

all: test

test: test.c tensor.o
	$(CC) -o main test.c tensor.o -lm

tensor.o: tensor.c tensor.h
	$(CC) -c tensor.c 

clean:
	rm -f main
	rm -f *.o
