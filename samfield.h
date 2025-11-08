#ifndef SAMFIELD_H
#define SAMFIELD_H

#include "pch.h"

struct Layer;
struct ActivationFunction;
class SamSystem;

enum class TypeLayer {
    LAYER,
    FUNC
};

class SamField : public QFrame {
    Q_OBJECT
    QGridLayout* layout;
    SamSystem* system;
    int x_coord;
    QSvgRenderer ReLU;
    QSvgRenderer SoftMax;
    QSvgRenderer Sigmoid;
    QSvgRenderer Tanh;
    bool writing;
    int curr_num_ch;

    QVector<Layer*> layers;
    QVector<ActivationFunction*> funcs;

    QPair<TypeLayer, int> curr_layer;

    bool ShiftOn;

protected:
    void wheelEvent(QWheelEvent *event) override;
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;
public:
    explicit SamField(QWidget *parent, SamSystem *system);
    void set_layers(QVector<Layer*> layers);
    void set_funcs(QVector<ActivationFunction*> funcs);
    QVector<Layer*> get_layers() const;
    QPair<TypeLayer, int> get_curr_layer() const;
};

#endif // SAMFIELD_H
