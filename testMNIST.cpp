#include <iostream>
#include <Eigen/Eigen>
#include <cstdio>
#include <iomanip>
#include "DNN.hpp"
#include "loadMNIST.hpp"

using namespace std;
using namespace Eigen;

float ErrorRate(MatrixXf y_out, MatrixXf y_label);
VectorXi Decode(MatrixXf y_out);

int main(){
	//DNN test("SAVE/test.model", "SAVE/test.param");
	
	DNN test;
	test.AddLayer(784,500,LAYER_RELU);
	test.AddLayer(500,300,LAYER_RELU);
	test.AddLayer(300,10,LAYER_SOFTMAX);

	test.SetLearnRate(0.05);
	test.SetDecayRate(0.002);
	test.SetLossType(LOSS_CROSS_ENTROPY);
	test.SetOptimizeType(OPTIMIZE_MOMENTUM);
	
	test.ShowModel();
	test.SaveModel("SAVE/test.model");

	MatrixXf train_image = read_image_binary("MNIST/train-images-idx3-ubyte");
	MatrixXf train_label = read_label_binary("MNIST/train-labels-idx1-ubyte");
	
	MatrixXf test_image = read_image_binary("MNIST/t10k-images-idx3-ubyte");
	MatrixXf test_label = read_label_binary("MNIST/t10k-labels-idx1-ubyte");

	FILE *fp;
	fp = fopen("SAVE/Log.txt", "w+");

	for(int i=1; i<=6000; ++i){	
		float train_loss = test.Train(train_image, train_label, 200);
		float test_loss = test.Test(test_image, test_label, 200);

		cout << "[" << i << " epoch]";
		cout << " train : " << setw(8) << train_loss;
		cout << " test : " << setw(8) << test_loss;
		cout << endl;
		
		fprintf(fp, "%f\t%f\n", train_loss, test_loss);

		if(i%1000 == 0){
			cout << "Total Test Loss: " << test.Test(test_image, test_label) << endl;
			cout << "Total Error Rate: " << ErrorRate(test.GetOutputTensor(), test_label) << endl;
			cout << "Save parameter ..." << endl;
			test.SaveParam("SAVE/test.param");
		}
	}
	
	cout << "==============================" << endl;
	cout << "Total Test Loss: " << test.Test(test_image, test_label) << endl;
	cout << "Total Error Rate: " << ErrorRate(test.GetOutputTensor(), test_label) << endl;

	cout << endl << "Save model ..." << endl;
	test.SaveModel("SAVE/test.model");
	test.SaveParam("SAVE/test.param");

	return 0;
}

float ErrorRate(MatrixXf y_out, MatrixXf y_label){
	int count = 0;
	VectorXi y_decode = Decode(y_out);
	for(int i=0; i<y_label.cols(); ++i)
		if(y_label(y_decode(i), i) != 1)
			++count;

	return (float)count / y_label.cols(); 
}

VectorXi Decode(MatrixXf y_out){
	float temp;
	VectorXi y_decode(y_out.cols());

	for(int j=0; j<y_out.cols(); ++j){
		float temp = -9999;
		for(int i=0; i<y_out.rows(); ++i){
			if(y_out(i,j) > temp){
				temp = y_out(i,j);
				y_decode(j) = i;
			}
		}
	}

	return y_decode;
}
