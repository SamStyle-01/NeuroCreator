#ifndef SAMTEST_H
#define SAMTEST_H

#include <QObject>

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
    void doWork(DataFrame* processing_data);
};

#endif // SAMTEST_H
