#ifndef LOAD_MNIST_HPP
#define LOAD_MNIST_HPP

#include <Eigen/Eigen>
#include <cstdio>

using namespace std;
using namespace Eigen;

int reverse_binary(unsigned char *buffer);
MatrixXf read_image_binary(const char *filename);
MatrixXf read_label_binary(const char *filename);

int reverse_binary(unsigned char *buffer){
	return (int)buffer[3] | (int)buffer[2]<<8 | (int)buffer[1]<<16 | (int)buffer[0]<<24;
}

MatrixXf read_image_binary(const char *filename){
	int mnum;
	int img_total;
	int img_row;
	int img_col;
	
	unsigned char cbuf[4];
	FILE *fp;
	fp = fopen(filename, "rb");
	
	fread(&cbuf, 1, 4, fp);
	mnum = reverse_binary(cbuf);
	fread(&cbuf, 1, 4, fp);
	img_total = reverse_binary(cbuf);
	fread(&cbuf, 1, 4, fp);
	img_row = reverse_binary(cbuf);
	fread(&cbuf, 1, 4, fp);
	img_col = reverse_binary(cbuf);

	cout << mnum << " " << img_total << " " << img_row << " " << img_col << endl;
	//img_total = 200;
	MatrixXf img_buf(img_row*img_col, img_total);
	unsigned char rbuf;
	for(int j=0; j<img_total; ++j){
		for(int i=0; i<img_col*img_row; ++i){
				fread(&rbuf, 1, 1, fp);
				img_buf(i,j) = ((double)rbuf)/255;
		}
	}
	
	return img_buf;
}

MatrixXf read_label_binary(const char *filename){
	int mnum;
	int label_total;
	
	unsigned char cbuf[4];
	FILE *fp;
	fp = fopen(filename, "rb");
	
	fread(&cbuf, 1, 4, fp);
	mnum = reverse_binary(cbuf);
	fread(&cbuf, 1, 4, fp);
	label_total = reverse_binary(cbuf);

	cout << mnum << " " << label_total << endl;
	//label_total = 200;
	MatrixXf label_buf = MatrixXf::Zero(10, label_total);
	unsigned char rbuf;
	for(int i=0; i<label_total; ++i){
		fread(&rbuf, 1, 1, fp);
		label_buf((int)rbuf,i) = 1;
	}
	
	return label_buf;
}

#endif // LOAD_MNIST_HPP