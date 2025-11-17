#include "samfield.h"
#include "samsystem.h"

extern double scale;

SamField::SamField(QWidget *parent, SamSystem *system) : QFrame{parent} {
    this->setStyleSheet("background-color: #F8F8FF; border-radius: 40px;");
    this->setMinimumSize(750 * scale, 550 * scale);
    this->layout = new QGridLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    this->system = system;
    x_coord = 0;

    ReLU.load(QString("../../ReLU.svg"));
    SoftMax.load(QString("../../SoftMax.svg"));
    Sigmoid.load(QString("../../Sigmoid.svg"));
    Tanh.load(QString("../../Tanh.svg"));

    this->curr_layer = qMakePair(TypeLayer::LAYER, -1);
    this->setFocusPolicy(Qt::StrongFocus);
    this->setFocus();

    this->writing = false;
    this->curr_num_ch = -1;
    ShiftOn = false;
}

void SamField::paintEvent(QPaintEvent* event) {
    QPainter painter;
    painter.begin(this);
    painter.setRenderHint(QPainter::Antialiasing, true);

    QBrush brush(QColor(255, 202, 202));
    QPen pen(Qt::black);
    QFont font("Inter", 24);
    painter.setFont(font);
    pen.setWidth(2);
    painter.setBrush(brush);
    painter.setPen(pen);

    int height_element = 615 * (scale + (1 - scale) / 2);
    int width_layer = 115;
    QFontMetrics fm(QFont("Inter", 24));

    // Слои
    for (int i = 0; i < layers.size(); i++) {
        painter.drawRect(-x_coord + 85 + i * 300, height() / 2 - height_element / 2, width_layer, height_element);

        QRect boundingRect = fm.boundingRect(-x_coord + 83 + i * 300 + width_layer / 2, height() / 2 - height_element / 2 + 60,
                                             0, 0, Qt::AlignLeft, QString::number(layers[i]->num_neuros));

        painter.drawText(QPoint(-x_coord + 83 + i * 300 + width_layer / 2 - boundingRect.width() / 2, height() / 2 - height_element / 2 + 60),
                         !system->data_inited() && i == layers.size() - 1 ? "-" : QString::number(layers[i]->num_neuros));
    }

    pen.setWidth(1);
    painter.setPen(pen);
    QBrush brush2(QColor(113, 219, 75));
    painter.setBrush(brush2);
    // Кружки для полносвязных слоёв
    for (int i = 0; i < layers.size(); i++) {
        if (layers[i]->num_neuros == 1 || layers[i]->num_neuros == 0) {
            painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2), 18, 18);
        }
        else if (layers[i]->num_neuros == 2) {
            for (int j = -1; j < 1; j++)
                painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 - j * 60 * (scale + (1 - scale) / 2) - 30), 18, 18);
        }
        else if (layers[i]->num_neuros == 3) {
            for (int j = -1; j < 2; j++)
                painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 - j * 60 * (scale + (1 - scale) / 2)), 18, 18);
        }
        else if (layers[i]->num_neuros == 4) {
            for (int j = -2; j < 2; j++)
                painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 - j * 60 * (scale + (1 - scale) / 2) - 30), 18, 18);
        }
        else if (layers[i]->num_neuros == 5) {
            for (int j = -2; j < 3; j++)
                painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 - j * 60 * (scale + (1 - scale) / 2)), 18, 18);
        }
        else {
            for (int j = -2; j < 3; j++) {
                if (j == -1) j++;
                painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 - j * 60 * (scale + (1 - scale) / 2)), 18, 18);
                brush2.setColor(QColor(89, 89, 89));
                painter.setBrush(brush2);
                for (int k = -1; k < 2; k++) {
                    painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 + (60 + k * 15) * (scale + (1 - scale) / 2)), 4, 4);
                }
                brush2.setColor(QColor(113, 219, 75));
                painter.setBrush(brush2);
            }
        }
    }
    pen.setWidth(2);
    painter.setPen(pen);

    // Функции активации
    QBrush brush3(QColor(196, 227, 233));
    painter.setBrush(brush3);
    for (int i = 0; i < funcs.size(); i++) {
        const int x = 200 + funcs[i]->num_layer * 300;
        const int y = height() / 2 - height_element / 2;
        const int func_width = 100;
        const int func_height = 615 * (scale + (1 - scale) / 2);
        const int radius = 31;

        QPainterPath path;

        path.moveTo(x - x_coord, y + func_height);
        path.lineTo(x - x_coord, y);

        path.lineTo(x - x_coord + func_width - radius, y);
        path.arcTo(x - x_coord + func_width - 2 * radius, y, 2 * radius, 2 * radius, 90.0, -90.0);

        path.lineTo(x - x_coord + func_width, y + func_height - radius);

        path.arcTo(x - x_coord + func_width - 2 * radius, y + func_height - 2 * radius, 2 * radius, 2 * radius, 0.0, -90.0);

        path.closeSubpath();

        painter.drawPath(path);

        // Картинка функции активации
        if (funcs[i]->func == "ReLU") {
            painter.drawText(QPoint(-x_coord + 200 + funcs[i]->num_layer * 300 + func_width / 2 - 40, height() / 2 - height_element / 2 + 60), "ReLU");

            QRect relu_rect(-x_coord + 200 + funcs[i]->num_layer * 300 + func_width / 2 - 40, height() / 2 * (scale + (1 - scale) / 2) - height_element / 2 + 211, 80, 80);
            ReLU.render(&painter, relu_rect);
        }
        else if (funcs[i]->func == "SoftMax") {
            painter.drawText(QPoint(-x_coord + 200 + funcs[i]->num_layer * 300 + func_width / 2 - 40, height() / 2 - height_element / 2 + 60), "SoftM");

            QRect softmax_rect(-x_coord + 200 + funcs[i]->num_layer * 300 + func_width / 2 - 40, height() / 2 * (scale + (1 - scale) / 2) - height_element / 2 + 211, 80, 80);
            SoftMax.render(&painter, softmax_rect);
        }
        else if (funcs[i]->func == "Sigmoid") {
            painter.drawText(QPoint(-x_coord + 200 + funcs[i]->num_layer * 300 + func_width / 2 - 40, height() / 2 - height_element / 2 + 60), "Sigm");

            QRect sigmoid_rect(-x_coord + 200 + funcs[i]->num_layer * 300 + func_width / 2 - 40, height() / 2 * (scale + (1 - scale) / 2) - height_element / 2 + 211, 80, 80);
            Sigmoid.render(&painter, sigmoid_rect);
        }
        else if (funcs[i]->func == "Tanh") {
            painter.drawText(QPoint(-x_coord + 200 + funcs[i]->num_layer * 300 + func_width / 2 - 40, height() / 2 - height_element / 2 + 60), "Tanh");

            QRect tanh_rect(-x_coord + 200 + funcs[i]->num_layer * 300 + func_width / 2 - 40, height() / 2 * (scale + (1 - scale) / 2) - height_element / 2 + 211, 80, 80);
            Tanh.render(&painter, tanh_rect);
        }
    }

    QPen pen2(QColor(89, 89, 89));
    pen2.setWidth(8);
    painter.setPen(pen2);

    // Стрелки
    for (int i = 0; i < layers.size() - 1; i++) {
        painter.drawLine(QPoint(-x_coord + 90 + i * 300 + width_layer, height() / 2), QPoint(-x_coord + 78 + (i + 1) * 300, height() / 2));

        painter.drawLine(QPoint(-x_coord + 80 + (i + 1) * 300, height() / 2), QPoint(-x_coord + 66 + (i + 1) * 300, height() / 2 + 15));
        painter.drawLine(QPoint(-x_coord + 80 + (i + 1) * 300, height() / 2), QPoint(-x_coord + 66 + (i + 1) * 300, height() / 2 - 15));
    }

    if (curr_layer.second != -1) {
        if (curr_layer.first == TypeLayer::LAYER) {
            QBrush brush(QColor(210, 210, 70));
            painter.setFont(font);
            pen.setWidth(2);
            painter.setBrush(brush);
            painter.setPen(pen);
            painter.drawRect(-x_coord + 85 + curr_layer.second * 300, height() / 2 - height_element / 2, width_layer, height_element);

            pen.setColor(Qt::black);
            pen.setWidth(2);
            painter.setPen(pen);

            QRect boundingRect = fm.boundingRect(-x_coord + 83 + curr_layer.second * 300 + width_layer / 2, height() / 2 - height_element / 2 + 60,
                                                 0, 0, Qt::AlignLeft, QString::number(layers[curr_layer.second]->num_neuros));

            painter.drawText(QPoint(-x_coord + 83 + curr_layer.second * 300 + width_layer / 2 - boundingRect.width() / 2, height() / 2 - height_element / 2 + 60),
                             !system->data_inited() && curr_layer.second == layers.size() - 1 ? "-" : QString::number(layers[curr_layer.second]->num_neuros));

            pen.setWidth(1);
            painter.setPen(pen);
            QBrush brush2(QColor(113, 219, 75));
            painter.setBrush(brush2);
            for (int i = 0; i < layers.size(); i++) {
                if (layers[i]->num_neuros == 1 || layers[i]->num_neuros == 0) {
                    painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2), 18, 18);
                }
                else if (layers[i]->num_neuros == 2) {
                    for (int j = -1; j < 1; j++)
                        painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 - j * 60 * (scale + (1 - scale) / 2) - 30), 18, 18);
                }
                else if (layers[i]->num_neuros == 3) {
                    for (int j = -1; j < 2; j++)
                        painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 - j * 60 * (scale + (1 - scale) / 2)), 18, 18);
                }
                else if (layers[i]->num_neuros == 4) {
                    for (int j = -2; j < 2; j++)
                        painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 - j * 60 * (scale + (1 - scale) / 2) - 30), 18, 18);
                }
                else if (layers[i]->num_neuros == 5) {
                    for (int j = -2; j < 3; j++)
                        painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 - j * 60 * (scale + (1 - scale) / 2)), 18, 18);
                }
                else {
                    for (int j = -2; j < 3; j++) {
                        if (j == -1) j++;
                        painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 - j * 60 * (scale + (1 - scale) / 2)), 18, 18);
                        brush2.setColor(QColor(89, 89, 89));
                        painter.setBrush(brush2);
                        for (int k = -1; k < 2; k++) {
                            painter.drawEllipse(QPoint(-x_coord + 85 + i * 300 + width_layer / 2, height() / 2 + (60 + k * 15) * (scale + (1 - scale) / 2)), 4, 4);
                        }
                        brush2.setColor(QColor(113, 219, 75));
                        painter.setBrush(brush2);
                    }
                }
            }
            pen.setWidth(2);
            painter.setPen(pen);
        }
    }

    // Набираемый текст. Выделение
    if (curr_num_ch != -1) {
        QPen pen_n(QColor(200, 0, 0));
        pen_n.setWidth(2);

        painter.setPen(pen_n);

        QRect boundingRect = fm.boundingRect(-x_coord + 83 + curr_num_ch * 300 + width_layer / 2, height() / 2 - height_element / 2 + 60,
                                             0, 0, Qt::AlignLeft, QString::number(layers[curr_num_ch]->num_neuros));

        painter.drawText(QPoint(-x_coord + 83 + curr_num_ch * 300 + width_layer / 2 - boundingRect.width() / 2, height() / 2 - height_element / 2 + 60),
                         QString::number(layers[curr_num_ch]->num_neuros));
    }

    // Граница виджета
    painter.setPen(pen);
    painter.setBrush(Qt::NoBrush);
    QRect rect = this->rect();
    rect.setTopLeft(rect.topLeft() + QPoint(1, 1));
    rect.setBottomRight(rect.bottomRight() + QPoint(-1, -1));
    painter.drawRoundedRect(rect, 40, 40);
    painter.end();
    QWidget::paintEvent(event);
}

void SamField::wheelEvent(QWheelEvent *event) {
    if (event->angleDelta().y() < 0) {
        if (x_coord < 85 + layers.size() * 300 - 375) {
            x_coord += 40;
        }
    }
    else {
        if (x_coord > 0) {
            x_coord -= 40;
        }
    }
    repaint();
    QWidget::wheelEvent(event);
}

void SamField::set_layers(QVector<Layer*> layers) {
    this->layers = layers;
    repaint();
}
void SamField::set_funcs(QVector<ActivationFunction*> funcs) {
    this->funcs = funcs;
    repaint();
}

void SamField::mousePressEvent(QMouseEvent *event) {
    int height_element = 615 * (scale + (1 - scale) / 2);
    int width_layer = 115;

    if (curr_num_ch != -1) {
        writing = false;
        this->system->set_neuros(layers[curr_num_ch]->num_neuros, curr_num_ch);
        if (curr_num_ch == 0 && system->data_inited()) {
            this->system->set_neuros(layers[curr_num_ch]->num_neuros < system->get_shape_data().first
                                         ? layers[curr_num_ch]->num_neuros : system->get_shape_data().first - 1, curr_num_ch);
            this->system->set_neuros(system->get_shape_data().first
                                         - layers[curr_num_ch]->num_neuros, layers.size() - 1);
        }
        curr_num_ch = -1;
        this->set_layers(system->get_layers());
    }

    for (int i = 0; i < layers.size(); i++) {
        if (event->pos().x() >= -x_coord + 85 + i * 300 && event->pos().y() >= height() / 2 - height_element / 2
                && event->pos().x() <= -x_coord + 85 + i * 300 + width_layer && event->pos().y() <= height() / 2 + height_element / 2) {
            this->curr_layer = qMakePair(TypeLayer::LAYER, i);
            repaint();
            return;
        }
    }
    for (int i = 0; i < funcs.size(); i++) {
        if (event->pos().x() >= -x_coord + 200 + i * 300 && event->pos().y() >= height() / 2 - height_element / 2
            && event->pos().x() <= -x_coord + 300 + i * 300 && event->pos().y() <= height() / 2 + height_element / 2) {
            this->curr_layer = qMakePair(TypeLayer::FUNC, i);
            repaint();
            return;
        }
    }
    this->curr_layer = qMakePair(TypeLayer::LAYER, -1);
    QWidget::mousePressEvent(event);
    repaint();
}

QPair<TypeLayer, int> SamField::get_curr_layer() const {
    return curr_layer;
}

void SamField::keyPressEvent(QKeyEvent *event) {
    if (writing) {
        if (event->key() == Qt::Key_Backspace) {
            layers[curr_num_ch]->num_neuros /= 10;
        }
        else if (event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) {
            writing = false;
            this->system->set_neuros(layers[curr_num_ch]->num_neuros, curr_num_ch);
            if (curr_num_ch == 0 && system->data_inited()) {
                this->system->set_neuros(layers[curr_num_ch]->num_neuros < system->get_shape_data().first
                                             ? layers[curr_num_ch]->num_neuros : system->get_shape_data().first - 1, curr_num_ch);
                this->system->set_neuros(system->get_shape_data().first
                                             - layers[curr_num_ch]->num_neuros, layers.size() - 1);
            }
            curr_num_ch = -1;
            this->set_layers(system->get_layers());
        }
        else {
            if (event->text().toInt() || event->text() == "0") {
                layers[curr_num_ch]->num_neuros *= 10;
                layers[curr_num_ch]->num_neuros += event->text().toInt();
                if (layers[curr_num_ch]->num_neuros > 999999)
                    layers[curr_num_ch]->num_neuros = 999999;
            }
        }
        repaint();
    }
    else {
        if ((event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) && curr_layer.second != layers.size() - 1
            && !system->get_is_inited()) {
            writing = true;
            curr_num_ch = curr_layer.second;
            layers[curr_num_ch]->num_neuros = 0;
            repaint();
        }
        if (event->key() == Qt::Key_Shift) {
            ShiftOn = true;
        }
        if (event->key() == Qt::Key_Backspace && curr_layer.second != -1) {
            if (!this->system->get_is_inited()) {
                if (ShiftOn) {
                    this->system->remove_func(curr_layer.second);
                    this->funcs = system->get_funcs();
                }
                else {
                    if (curr_layer.second == this->system->get_layers().size() - 1 && this->system->get_layers().size() > 1) {
                        auto temp_layers = this->system->get_layers();
                        this->system->set_neuros(temp_layers[curr_layer.second]->num_neuros, curr_layer.second - 1);
                    }

                    this->system->remove_layer(curr_layer.second);
                    this->layers = system->get_layers();
                    this->funcs = system->get_funcs();
                    curr_layer.second--;
                }
                repaint();
            }
            else {
                QMessageBox::warning(this->parentWidget(), "Ошибка", "Модель была инициализирована");
            }
        }
        if (event->key() == Qt::Key_Right && curr_layer.second != -1 && curr_layer.second < layers.size() - 1) {
            curr_layer.second++;
            repaint();
        }
        if (event->key() == Qt::Key_Left && curr_layer.second != -1 && curr_layer.second > 0) {
            curr_layer.second--;
            repaint();
        }
    }

    QWidget::keyPressEvent(event);
}

QVector<Layer*> SamField::get_layers() const {
    return layers;
}

void SamField::keyReleaseEvent(QKeyEvent *event) {
    if (event->key() == Qt::Key_Shift) {
        ShiftOn = false;
    }

    QWidget::keyPressEvent(event);
}

void SamField::mouseDoubleClickEvent(QMouseEvent *event) {
    QFontMetrics fm(QFont("Inter", 24));

    int height_element = 615 * (scale + (1 - scale) / 2);
    int width_layer = 115;

    for (int i = 0; i < layers.size() - 1; i++) {
        QRect boundingRect = fm.boundingRect(-x_coord + 83 + i * 300 + width_layer / 2, height() / 2 - height_element / 2 + 60,
                                             0, 0, Qt::AlignLeft, QString::number(layers[i]->num_neuros));

        int x_offset = boundingRect.width() / 2;

        QRect hitBox(
            boundingRect.x() - x_offset,
            boundingRect.y(),
            boundingRect.width(),
            -boundingRect.height()
            );

        auto pos = event->pos();
        pos.setX(pos.x());
        if (!this->system->get_is_inited()) {
            if (hitBox.contains(pos)) {
                writing = true;
                curr_num_ch = i;
                layers[curr_num_ch]->num_neuros = 0;
                repaint();
                return;
            }
        }
    }

    QWidget::mouseDoubleClickEvent(event);
}
