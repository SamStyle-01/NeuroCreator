#include "sammodel.h"
#include "samsystem.h"

SamModel::SamModel(SamView* main_window, SamSystem* system) {
    this->main_window = main_window;
    this->system = system;

    std::random_device rd;
    std::mt19937 gen(rd());
}

void SamModel::remove_layer(int index) {
    remove_func_bias(index);
    layers.erase(layers.begin() + index);
}

void SamModel::remove_func_bias(int num_layer) {
    for (int i = 0; i < funcs.size(); i++) {
        if (num_layer == funcs[i]->num_layer) {
            funcs.erase(funcs.begin() + i);
            break;
        }
    }
    for (int j = 0; j < funcs.size(); j++) {
        if (funcs[j]->num_layer >= num_layer)
            funcs[j]->num_layer--;
    }
}

void SamModel::remove_func(int num_layer) {
    for (int i = 0; i < funcs.size(); i++) {
        if (num_layer == funcs[i]->num_layer) {
            funcs.erase(funcs.begin() + i);
            return;
        }
    }
}

float* SamModel::get_weight_T(int index) const {
    return weights_T[index];
}

float* SamModel::get_bias(int index) const {
    return bias[index];
}

void SamModel::set_neuros(int num, int index) {
    this->layers[index]->num_neuros = num;
}

void SamModel::init_model() {
    // Веса
    for (int i = 0; i < layers.size() - 1; i++) {
        int in_neurons = layers[i]->num_neuros;
        int out_neurons = layers[i + 1]->num_neuros;

        float bound = sqrt(6.0f / (in_neurons + out_neurons));
        std::uniform_real_distribution<float> dist(-bound, bound);

        weights.push_back(new float[out_neurons * in_neurons]);
        for (int j = 0; j < out_neurons * in_neurons; j++) {
            weights[i][j] = dist(gen);
        }

        weights_T.push_back(new float[out_neurons * in_neurons]);
        for (int row = 0; row < in_neurons; row++) {
            for (int col = 0; col < out_neurons; col++) {
                weights_T[i][col * in_neurons + row] = weights[i][row * out_neurons + col];
            }
        }
    }

    // Смещения
    std::normal_distribution<float> dist(0.0f, 0.01f);

    for (int i = 0; i < layers.size() - 1; i++) {
        int out_neurons = layers[i + 1]->num_neuros;

        bias.push_back(new float[out_neurons]);
        for (int j = 0; j < out_neurons; j++) {
            bias[i][j] = dist(gen);
        }
    }

};

LinearLayer::LinearLayer() {
    num_neuros = 1;
}

ReLU::ReLU(int num_layer) {
    this->num_layer = num_layer;
}

SamModel::~SamModel() {
    this->reset_model();
    for (int i = 0; i < layers.size(); i++) delete layers[i];
    for (int i = 0; i < funcs.size(); i++) delete funcs[i];
}

void SamModel::reset_model() {
    for (int i = 0; i < weights.size(); i++) {
        delete[] weights[i];
    }
    weights.clear();

    for (int i = 0; i < weights_T.size(); i++) {
        delete[] weights_T[i];
    }
    weights_T.clear();

    for (int i = 0; i < bias.size(); i++) {
        delete[] bias[i];
    }
    bias.clear();
}

bool SamModel::add_layer(Layer* layer) {
    layers.push_back(layer);
    return true;
}

bool SamModel::add_layer(Layer* layer, int index) {
    layers.insert(index, layer);
    for (int i = 0; i < funcs.size(); i++) {
        if (funcs[i]->num_layer >= index)
            funcs[i]->num_layer++;
    }
    return true;
}

bool SamModel::add_func(ActivationFunction* func) {
    funcs.push_back(func);
    return true;
}

QVector<Layer*> SamModel::get_layers() const {
    return layers;
}

QVector<ActivationFunction*> SamModel::get_funcs() const {
    return funcs;
}

LinearLayer::~LinearLayer() {}

ReLU::~ReLU() {}

ActivationFunction::~ActivationFunction() {}
