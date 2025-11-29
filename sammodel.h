#ifndef SAMMODEL_H
#define SAMMODEL_H

#include "pch.h"
#include <random>

class SamView;
class SamSystem;
class BackPropagation;
class ForwardPass;
class SamTest;

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
    QVector<float*> bias;

    friend BackPropagation;
    friend ForwardPass;
    friend SamTest;
public:
    SamModel(SamView* main_window, SamSystem* system);
    ~SamModel();

    float* get_weight(int index) const;
    float* get_bias(int index) const;

    int get_weights_size() const;
    int get_bias_size() const;

    bool add_layer(Layer* layer);
    bool add_layer(Layer* layer, int index);
    bool add_func(ActivationFunction* func);

    void init_model();

    void set_neuros(int num, int index);

    void remove_layer(int index);
    void remove_func(int num_layer);
    void remove_func_bias(int num_layer);
    void reset_model();
    void set_model(QVector<float*> best_weights, QVector<float*> best_bias);

    QVector<Layer*> get_layers() const;
    QVector<ActivationFunction*> get_funcs() const;
};

#endif // SAMMODEL_H
