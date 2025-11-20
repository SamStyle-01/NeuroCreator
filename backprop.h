#ifndef BACKPROP_H
#define BACKPROP_H

#include <QObject>
#include "pch.h"

class SamSystem;
class DataFrame;

class BackPropagation : public QObject {
    Q_OBJECT
    SamSystem *system;

    static void ReLU_func_deriv(QVector<float>& vector);
    static QVector<float> SoftMax_func_deriv(const QVector<float>& predicted, const QVector<QVector<float>>& target,
                                             int outputs, int col_first, int curr_el);
    static void Sigmoid_func_deriv(QVector<float>& vector);
    static void Tanh_func_deriv(QVector<float>& vector);

    static QVector<float> MSE_deriv(const QVector<float>& predicted, const QVector<QVector<float>>& true_vals, int outputs,
                                    int col_first, int curr_el);
    static QVector<float> MAE_deriv(const QVector<float>& predicted, const QVector<QVector<float>>& true_vals, int outputs,
                                    int col_first, int curr_el);
    static QVector<float> CrossEntropy_deriv(const QVector<float>& predicted, const QVector<QVector<float>>& true_vals, int outputs,
                                             int col_first, int curr_el);

    static void clip_gradients(QVector<float>& grad, float clip_value = 1.0f);

public:
    explicit BackPropagation(SamSystem *system, QObject *parent = nullptr);

signals:
    void finished(const bool &success, QString log);
    void epoch_done(float train_loss, float valid_loss);

public slots:
    void doWork(cl_context& context);
};

#endif // BACKPROP_H
