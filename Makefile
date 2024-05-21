CC = gcc

all: tensor

test: tensor
	pytest test/

tensor: src/tensor.c src/tensor.h
	mkdir -p bin
	$(CC) -shared -o bin/tensor.so src/tensor.c -lm -fPIC
	$(CC) -o bin/tensor.o -c src/tensor.c  

clean:
	rm -rf bin
