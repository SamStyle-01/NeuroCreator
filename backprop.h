#ifndef BACKPROP_H
#define BACKPROP_H

#include <QObject>
#include "pch.h"

class SamSystem;
class DataFrame;

class BackPropagation : public QObject {
    Q_OBJECT
    SamSystem *system;
public:
    explicit BackPropagation(SamSystem *system, QObject *parent = nullptr);

signals:
    void finished(const bool &success, QString log);
    void epoch_done(float train_loss, float valid_loss);

public slots:
    void doWork(cl_context& context);
};

#endif // BACKPROP_H
