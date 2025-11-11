#ifndef BACKWARD_H
#define BACKWARD_H

#include <QObject>

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

public:
    explicit BackWard(SamSystem *system, QObject *parent = nullptr);

signals:
    void finished(const bool &success, QString log);
    void epoch_done(const bool &success, QString log);

public slots:
    void doWork();
};

#endif // BACKWARD_H
