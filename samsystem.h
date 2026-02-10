#ifndef SAMSYSTEM_H
#define SAMSYSTEM_H

#include "pch.h"
#include "dataframe.h"
#include "samview.h"
#include "sammodel.h"

class SamTraining;
class ForwardPass;
class BackPropagation;
class SamTest;

enum class Activation {
    LINEAR,
    RELU,
    SOFTMAX,
    SIGMOID,
    TANH
};

class SamSystem : public QObject {
    Q_OBJECT
    DataFrame* data;
    SamModel* model;

    int curr_epochs;

    cl_context context;
    bool first_activation;

    SamView* main_window;
    bool is_standartized;
    bool is_inited;
    cl_device_id curr_device;
    bool ocl_inited;
    bool training_now;

    static void ReLU_func(QVector<float>& vector);
    static void SoftMax_func(QVector<float>::Iterator begin, QVector<float>::Iterator end);
    static void Sigmoid_func(QVector<float>& vector);
    static void Tanh_func(QVector<float>& vector);

    static float MSE_loss(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals);
    static float MAE_loss(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals);
    static float CrossEntropy_loss(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals);

    QVector<QPair<cl_device_id, QString>> devices;

    SamTraining* training_view;

    int t;
    float beta1;
    float beta2;
    // Инициализация m и v для каждого веса и bias
    QVector<QVector<float>> m_w;
    QVector<QVector<float>> m_b;
    QVector<QVector<float>> v_w;
    QVector<QVector<float>> v_b;

    QVector<QVector<float>> best_m_w;
    QVector<QVector<float>> best_m_b;
    QVector<QVector<float>> best_v_w;
    QVector<QVector<float>> best_v_b;
    QVector<float*> best_weights;
    QVector<float*> best_bias;
    float weight_decay;
    int best_epoch;
    int best_t;

    friend ForwardPass;
    friend BackPropagation;
    friend SamTest;

    cl_kernel kernel_matrix_mult;

    // Ядро backprop_linear
    cl_kernel kernel_backprop_linear;

    // Ядро vectors_mult
    cl_kernel kernel_vectors_mult;

    // Ядро weights_first_step
    cl_kernel kernel_weights_first_step;

    // Ядро bias_first_step
    cl_kernel kernel_bias_first_step;

    // Ядро weights_last_step
    cl_kernel kernel_weights_last_step;

    // Ядро bias_last_step
    cl_kernel kernel_bias_last_step;

    // ReLU
    cl_kernel kernel_relu;

    // Sigmoid
    cl_kernel kernel_sigmoid;

    // Tanh
    cl_kernel kernel_tanh;

    // Производная ReLU
    cl_kernel kernel_relu_deriv;

    // Производная Sigmoid
    cl_kernel kernel_sigmoid_deriv;

    // Производная Tanh
    cl_kernel kernel_tanh_deriv;

    // Производная MSE
    cl_kernel kernel_mse_deriv;

    // Производная MAE
    cl_kernel kernel_mae_deriv;

    // Производная Softmax
    cl_kernel kernel_softmax_deriv;
public:
    SamSystem(SamView* main_window);
    ~SamSystem();
    bool load_data();
    bool z_score(int num_x);
    void reset_data();
    void reset_standartization();
    void modelize();
    bool get_is_inited() const;
    bool get_ocl_inited() const;
    void set_neuros(int num, int index);
    bool data_inited() const;
    void init_model();
    void reset_model();
    bool process_data();
    bool test_data();
    void set_training_view(SamTraining* training);
    void set_device(cl_device_id index);
    QVector<QPair<cl_device_id, QString>> get_devices() const;
    int get_epochs() const;
    bool get_is_training() const;
    void set_is_training(bool val);
    void set_curr_epochs(int epoch);

    bool add_layer(Layer* layer);
    bool add_layer(Layer* layer, int index);
    bool add_func(ActivationFunction* func);

    float best_loss;
    void steal_weights_bias(QVector<float*> best_weights, QVector<float*> best_bias);
    void set_best_model();
    int get_best_epoch() const;

    void remove_layer(int index);
    void remove_func(int num_layer);

    bool backpropagation();

    QVector<Layer*> get_layers() const;
    QVector<ActivationFunction*> get_funcs() const;
    QPair<int, int> get_shape_data() const;
};


#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

void reportError(cl_int err, const QString &filename, int line);

#endif // SAMSYSTEM_H
