#ifndef BACKWARD_H
#define BACKWARD_H

#include <QObject>
#include "pch.h"

class SamSystem;
class DataFrame;

class BackWard : public QObject {
    Q_OBJECT
    SamSystem *system;

    static void ReLU_func_deriv(QVector<float>& vector);
    static QVector<float> SoftMax_func_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& target);
    static void Sigmoid_func_deriv(QVector<float>& vector);
    static void Tanh_func_deriv(QVector<float>& vector);

    static QVector<float> MSE_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals);
    static QVector<float> MAE_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals);
    static QVector<float> CrossEntropy_deriv(const QVector<QVector<float>>& predicted, const QVector<QVector<float>>& true_vals);

    static void clip_gradients(QVector<float>& grad, float clip_value = 1.0f);

public:
    explicit BackWard(SamSystem *system, QObject *parent = nullptr);

signals:
    void finished(const bool &success, QString log);
    void epoch_done(float train_loss, float valid_loss);

public slots:
    void doWork(cl_context& context);
};

#endif // BACKWARD_H
