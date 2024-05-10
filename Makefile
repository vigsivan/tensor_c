CC = gcc

all: lenet

lenet: src/lenet.c tensor.o 
	mkdir -p bin
	$(CC) -o bin/lenet src/lenet.c bin/tensor.o -lm

net: src/lenet.c tensor.o
	$(CC) -shared -o bin/lenet.so src/lenet.c bin/tensor.o -lm

test: net tensor
	pytest test/

# test: test.c tensor.o
# 	mkdir -p bin
# 	$(CC) -o bin/main test.c tensor.o -lm

tensor.o: src/tensor.c src/tensor.h
	mkdir -p bin
	$(CC) -o bin/tensor.o -c src/tensor.c  

tensor: src/tensor.c src/tensor.h
	mkdir -p bin
	$(CC) -shared -o bin/tensor.so src/tensor.c

clean:
	rm -rf bin
