import sys
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 导入主应用类
from app import DOEApp

# 只使用 PyQt6
from PyQt6.QtWidgets import QApplication
QT_MODULE = "PyQt6"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = DOEApp()
    w.show()
    sys.exit(app.exec())