CC = gcc

all: lenet

lenet: lenet.c tensor.o 
	mkdir -p bin
	$(CC) -o bin/lenet lenet.c tensor.o -lm

test: test.c tensor.o
	mkdir -p bin
	$(CC) -o bin/main test.c tensor.o -lm

tensor.o: tensor.c tensor.h
	$(CC) -c tensor.c 

clean:
	rm -f *.o
	rm -rf bin
