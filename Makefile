CC = gcc

all: lenet

lenet: lenet.c tensor.o 
	mkdir -p bin
	$(CC) -o bin/lenet lenet.c tensor.o -lm

net: lenet.c tensor.o
	$(CC) -shared -o bin/lenet.so lenet.c tensor.o -lm

test: net tensor
	pytest .

# test: test.c tensor.o
# 	mkdir -p bin
# 	$(CC) -o bin/main test.c tensor.o -lm

tensor.o: tensor.c tensor.h
	$(CC) -c tensor.c 

tensor: tensor.c tensor.h
	$(CC) -shared -o bin/tensor.so tensor.c

clean:
	rm -f *.o
	rm -rf bin
