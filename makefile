CC=g++
CF=-ggdb -fopenmp -Wall
LF=-lpthread -ggdb -fopenmp -Wall

all: main.o neuron.o utils.o
	${CC} *.o -o main.out ${LF}

main.o: main.cpp neuron.h utils.h
	${CC} main.cpp -c ${CF}

neuron.o: neuron.cpp neuron.h utils.h
	${CC} neuron.cpp -c ${CF}

utils.o: utils.cpp utils.h
	${CC} utils.cpp -c ${CF}

clean:
	rm *.out *.o