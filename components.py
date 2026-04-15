# 只使用 PyQt6
from PyQt6.QtWidgets import (QTableWidget, QApplication, QTableWidgetItem, 
                             QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QSlider, QDoubleSpinBox, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal

class SpreadsheetTable(QTableWidget):
    def keyPressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_C: self.copy_to_clipboard()
            elif event.key() == Qt.Key.Key_V: self.paste_from_clipboard()
        else: super().keyPressEvent(event)
    def copy_to_clipboard(self):
        sel = self.selectedRanges()
        if not sel: return
        rows = []
        for i in range(sel[0].topRow(), sel[0].bottomRow()+1):
            rows.append("\t".join([self.item(i, j).text() if self.item(i, j) else "" for j in range(sel[0].leftColumn(), sel[0].rightColumn()+1)]))
        QApplication.clipboard().setText("\n".join(rows))
    def paste_from_clipboard(self):
        text = QApplication.clipboard().text()
        if not text: return
        r, c = max(0, self.currentRow()), max(0, self.currentColumn())
        for i, line in enumerate(text.splitlines()):
            if r+i >= self.rowCount(): self.insertRow(r+i)
            for j, val in enumerate(line.split('\t')):
                if c+j < self.columnCount(): self.setItem(r+i, c+j, QTableWidgetItem(val.strip()))

# === 新增: 单个变量的控制组件 (滑块 + 输入框) ===
class VariableController(QWidget):
    value_changed = pyqtSignal(float)

    def __init__(self, name, min_val, max_val, default_val):
        super().__init__()
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 标签
        lbl = QLabel(f"{name}:")
        lbl.setFixedWidth(80)
        layout.addWidget(lbl)
        
        # 滑块 (QSlider 是整数，需要转换)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000) # 1000级精度
        self.slider.valueChanged.connect(self._on_slider_change)
        layout.addWidget(self.slider)
        
        # 数字框
        self.spin = QDoubleSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setValue(default_val)
        self.spin.setSingleStep((max_val - min_val) / 100.0)
        self.spin.setDecimals(3)
        self.spin.valueChanged.connect(self._on_spin_change)
        self.spin.setFixedWidth(80)
        layout.addWidget(self.spin)
        
        # 初始化滑块位置
        self._update_slider_from_val(default_val)

    def _on_slider_change(self, val):
        # 0-1000 -> min-max
        ratio = val / 1000.0
        real_val = self.min_val + ratio * (self.max_val - self.min_val)
        self.spin.blockSignals(True)
        self.spin.setValue(real_val)
        self.spin.blockSignals(False)
        self.value_changed.emit(real_val)

    def _on_spin_change(self, val):
        self._update_slider_from_val(val)
        self.value_changed.emit(val)

    def _update_slider_from_val(self, val):
        if self.max_val == self.min_val: return
        ratio = (val - self.min_val) / (self.max_val - self.min_val)
        slider_val = int(ratio * 1000)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_val)
        self.slider.blockSignals(False)

    def get_value(self):
        return self.spin.value()

# === 新增: 实时预测面板 ===
class RealTimePredictor(QWidget):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.controllers = {}
        
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(10)
        
        # 输入区
        self.input_group = QGroupBox("输入变量调节 (What-if Simulator)")
        self.input_layout = QVBoxLayout(self.input_group)
        self.layout.addWidget(self.input_group)
        
        # 结果区
        self.res_group = QGroupBox("实时预测结果")
        self.res_layout = QHBoxLayout(self.res_group)
        
        self.lbl_pred_val = QLabel("0.0000")
        self.lbl_pred_val.setStyleSheet("font-size: 24px; font-weight: bold; color: #0984e3;")
        self.lbl_pred_std = QLabel("± 0.0000")
        self.lbl_pred_std.setStyleSheet("font-size: 16px; color: #636e72;")
        
        self.res_layout.addStretch()
        self.res_layout.addWidget(QLabel("Predicted Y: "))
        self.res_layout.addWidget(self.lbl_pred_val)
        self.res_layout.addSpacing(20)
        self.res_layout.addWidget(self.lbl_pred_std)
        self.res_layout.addStretch()
        
        self.layout.addWidget(self.res_group)
        self.layout.addStretch() # 底部顶起

    def setup_inputs(self):
        # 清空旧控件
        for i in reversed(range(self.input_layout.count())): 
            self.input_layout.itemAt(i).widget().setParent(None)
        self.controllers = {}
        
        if not self.engine.X_cols:
            self.input_layout.addWidget(QLabel("请先训练模型以启用模拟器"))
            return

        for col in self.engine.X_cols:
            if col in self.engine.bounds_info:
                mn, mx, mean = self.engine.bounds_info[col]
            else:
                mn, mx, mean = 0, 1, 0.5
            
            ctl = VariableController(col, mn, mx, mean)
            ctl.value_changed.connect(self.update_prediction)
            self.input_layout.addWidget(ctl)
            self.controllers[col] = ctl
            
        self.update_prediction()

    def update_prediction(self):
        if not self.engine.model_predictive: return
        
        inputs = {name: ctl.get_value() for name, ctl in self.controllers.items()}
        y, std = self.engine.predict_value(inputs)
        
        self.lbl_pred_val.setText(f"{y:.4f}")
        self.lbl_pred_std.setText(f"± {std:.4f}")