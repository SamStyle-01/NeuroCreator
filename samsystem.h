#ifndef SAMSYSTEM_H
#define SAMSYSTEM_H

#include "pch.h"
#include "dataframe.h"
#include "samview.h"
#include "sammodel.h"

class SamSystem {
    DataFrame* data;
    SamModel* model;
    DataFrame* processing_data;

    SamView* main_window;
    bool is_standartized;
    bool is_inited;
    cl_device_id curr_device;
    bool ocl_inited;

    QVector<QPair<cl_device_id, QString>> devices;
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
    bool process_data() const;
    void set_device(cl_device_id index);
    QVector<QPair<cl_device_id, QString>> get_devices() const;

    bool add_layer(Layer* layer);
    bool add_layer(Layer* layer, int index);
    bool add_func(ActivationFunction* func);

    void remove_layer(int index);
    void remove_func(int num_layer);

    QVector<Layer*> get_layers() const;
    QVector<ActivationFunction*> get_funcs() const;
    QPair<int, int> get_shape_data() const;
};


#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

void reportError(cl_int err, const QString &filename, int line);

#endif // SAMSYSTEM_H
