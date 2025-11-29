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

float* SamModel::get_weight(int index) const {
    return weights[index];
}

float* SamModel::get_bias(int index) const {
    return bias[index];
}

void SamModel::set_neuros(int num, int index) {
    this->layers[index]->num_neuros = num;
}

int SamModel::get_weights_size() const {
    return this->weights.size();
}

int SamModel::get_bias_size() const {
    return this->bias.size();
}

void SamModel::set_model(QVector<float*> best_weights, QVector<float*> best_bias) {
    for (int l = 1; l < this->layers.size(); l++) {
        int N_l = this->layers[l]->num_neuros;
        int N_prev = this->layers[l - 1]->num_neuros;
        for (int w = 0; w < N_l; w++) {
            for (int w2 = 0; w2 < N_prev; w2++) {
                int index = w * N_prev + w2;
                this->weights[l - 1][index] = best_weights[l - 1][index];
            }
        }

        for (int b = 0; b < N_l; b++) {
            this->bias[l - 1][b] = best_bias[l - 1][b];
        }
    }
}

void SamModel::init_model() {
    std::mt19937 gen(std::random_device{}());
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);

    // === ВЕСА ===
    for (int i = 0; i < layers.size() - 1; i++) {
        int in_neurons  = layers[i]->num_neuros;
        int out_neurons = layers[i + 1]->num_neuros;

        QString activation = "LINEAR";
        for (const auto* func : funcs) {
            if (func->num_layer == i + 1) {
                activation = func->func;
                break;
            }
        }

        float std_dev = 0.01f;

        if (activation == "ReLU") {
            std_dev = std::sqrt(2.0f / static_cast<float>(in_neurons));
        }
        else if (activation == "Tanh") {
            std_dev = std::sqrt(1.0f / static_cast<float>(in_neurons));
        }
        else if (activation == "Sigmoid") {
            std_dev = std::sqrt(1.0f / static_cast<float>(in_neurons));
        }
        else {
            std_dev = 0.01f;
        }

        weights.push_back(new float[out_neurons * in_neurons]);

        for (int row = 0; row < in_neurons; ++row) {
            for (int col = 0; col < out_neurons; ++col) {
                float val = normal_dist(gen) * std_dev;

                weights[i][row * out_neurons + col] = val;
            }
        }
    }

    // === СМЕЩЕНИЯ (bias) ===
    for (int i = 0; i < layers.size() - 1; i++) {
        int out_neurons = layers[i + 1]->num_neuros;
        bias.push_back(new float[out_neurons]{});

        std::fill_n(bias[i], out_neurons, 0.0f);

        for (int j = 0; j < out_neurons; ++j)
            bias[i][j] = normal_dist(gen) * 0.01f;
    }
}

LinearLayer::LinearLayer() {
    num_neuros = 1;
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

ActivationFunction::ActivationFunction(QString func, int num_layer){
    this->func = func;
    this->num_layer = num_layer;
}
