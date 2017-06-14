all: test DNN

test: testMNIST.cpp DNN.hpp DNN.o
	g++ -O2 -w -o test testMNIST.cpp DNN.o

DNN: DNN.cpp DNN.hpp
	g++ -O2 -w -c DNN.cpp

clean:
	rm *.o test
