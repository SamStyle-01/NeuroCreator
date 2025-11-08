#ifndef SAMMODEL_H
#define SAMMODEL_H

#include "pch.h"
#include <random>

class SamView;
class SamSystem;

struct Layer {
    int num_neuros;
};
struct ActivationFunction {
    ActivationFunction(QString func, int num_layer);
    int num_layer;
    QString func;
};

struct LinearLayer : public Layer {
    LinearLayer();
    ~LinearLayer();
};

class SamModel {
    SamView* main_window;
    SamSystem* system;
    QVector<Layer*> layers;
    QVector<ActivationFunction*> funcs;

    std::random_device rd;
    std::mt19937 gen;

    QVector<float*> weights;
    QVector<float*> weights_T;
    QVector<float*> bias;
public:
    SamModel(SamView* main_window, SamSystem* system);
    ~SamModel();

    float* get_weight_T(int index) const;
    float* get_bias(int index) const;

    bool add_layer(Layer* layer);
    bool add_layer(Layer* layer, int index);
    bool add_func(ActivationFunction* func);

    void init_model();

    void set_neuros(int num, int index);

    void remove_layer(int index);
    void remove_func(int num_layer);
    void remove_func_bias(int num_layer);
    void reset_model();

    QVector<Layer*> get_layers() const;
    QVector<ActivationFunction*> get_funcs() const;
};

#endif // SAMMODEL_H
