#ifndef DNN_HPP
#define DNN_HPP

#include <Eigen/Eigen>
#include <vector>
#include <iostream>
#include <string>

//Layer type
#define LAYER_RELU      0   //relu activation layer
#define LAYER_SIGMOID   1   //sigmoid activation layer
#define LAYER_TANH      2   //tanh activation layer
#define LAYER_LINEAR    3   //linear activation layer
#define LAYER_SOFTMAX   4   //softmax layer

//Loss type
#define LOSS_MEAN_SQUARE    0   //mean square loss
#define LOSS_CROSS_ENTROPY  1   //cross entropy loss

//Optimize type
#define OPTIMIZE_SGD        0   //SGD optimize
#define OPTIMIZE_MOMENTUM   1   //mementum optimize

using namespace std;
using namespace Eigen;

struct LayerInfo{
    int input;
    int output;
    int type;

    MatrixXf weight;
    VectorXf bias;
};

class DNN{
    public:
        //Constructor & Destructor
        DNN();
        DNN(string model_path);
        DNN(string model_path, string param_path);
        void Init();

        //User operation
        int AddLayer(unsigned int input, unsigned int output, unsigned int type);
        int AddLayer(LayerInfo new_Layer);

        int InitModel();
        int ShowModel();
        int SaveModel(string path);
        int LoadModel(string path);
        int SaveParam(string path);
        int LoadParam(string path);

        float Train(MatrixXf sample, MatrixXf label);
        float Train(MatrixXf sample, MatrixXf label, int batch_size);
        float Test(MatrixXf sample, MatrixXf label);
        float Test(MatrixXf sample, MatrixXf label, int batch_size);

        //Get Set operation
        LayerInfo GetLayer(int layer_id);
        MatrixXf GetNet(int layer_id);
        MatrixXf GetDelta(int layer_id);
        MatrixXf GetOutputTensor();
        float GetTotalLoss();

        int SetLayer(int layer_id, LayerInfo layer);
        int SetLearnRate(float rate);
        int SetDecayRate(float rate);
        int SetLossType(int type);
        int SetOptimizeType(int type);

        MatrixXf GetInputTensor();
        int SetInputTensor(MatrixXf input_ts);
        MatrixXf GetLabelTensor();
        int SetLabelTensor(MatrixXf label_ts);

        //Inner operation
        int FrontProp(unsigned int layer_id, MatrixXf input_ts);
        int FrontPropTotal(MatrixXf input_ts);
        int FrontPropTotal();
        
        int BackProp(unsigned int layer_id, MatrixXf delta_ts);
        int BackPropTotal(MatrixXf label_ts);
        int BackPropTotal();

        int Optimize(int type);
        MatrixXf Loss(MatrixXf &y_out, MatrixXf &y_label);
        
        float Activation(float x, int type);
        MatrixXf TensorActivation(MatrixXf x, int type);
        float ActivationDiff(float x, int type);
        MatrixXf TensorActivationDiff(MatrixXf x, int type);

        MatrixXf InitWeight(int rows, int cols);
        float Gaussrand(float exp, float var);
        int *ListShuffle(int total);
        int Swap(int *a, int *b);

    private:
        int total_layer;
        float learn_rate;
        float decay_rate;
        int loss_type;
        int optimize_type;

        vector <LayerInfo> layer;
        vector <MatrixXf> net;
        vector <MatrixXf> delta;

        vector <MatrixXf> weight_delta;
        vector <VectorXf> bias_delta;
        vector <MatrixXf> weight_rec;
        vector <VectorXf> bias_rec;
        
        MatrixXf input_tensor;
        MatrixXf label_tensor;
};

#endif // DNN_HPP
