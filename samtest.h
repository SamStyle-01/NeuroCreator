#ifndef SAMTEST_H
#define SAMTEST_H

#include <QObject>
#include "pch.h"

class SamSystem;
class DataFrame;
class SamTraining;

class SamTest : public QObject {
    Q_OBJECT
    SamSystem *system;
public:
    explicit SamTest(SamSystem *system, QObject *parent = nullptr);

signals:
    void finished(const bool &success, QString log, float result);

public slots:
    void doWork(DataFrame* processing_data, bool delete_data, cl_context& context);
    QPair<QString, float> doWork(DataFrame* processing_data, cl_context& context);
};

#endif // SAMTEST_H
