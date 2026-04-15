import math
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib import cm, rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# PyQt6 / PySide6 兼容导入
try:
    from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QDialog, 
                                 QSizePolicy, QMessageBox, QApplication)
    from PyQt6.QtGui import QColor
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except ImportError:
    from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QDialog, 
                                 QSizePolicy, QMessageBox, QApplication)
    from PySide6.QtGui import QColor
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        
        # 初始化策略：水平方向扩展，垂直方向保持最小要求
        # 关键点：Vertical 使用 Minimum 策略，配合 setMinimumHeight 防止被压缩
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.updateGeometry()

    def wheelEvent(self, event): event.ignore()

    # === 核心修复：强制调整画布高度并锁定 ===
    def adjust_canvas_height(self, height_inches):
        """
        根据图表内容的 inch 高度，自动计算所需的像素高度，
        并设置给 Qt Widget，强制触发 ScrollArea 的滚动条。
        """
        # 1. 设置 Matplotlib 内部图表高度
        self.fig.set_figheight(height_inches)
        
        # 2. 计算像素高度 (Inches * DPI)
        # 额外加 50px 缓冲，防止边缘遮挡
        required_height_px = int(height_inches * self.fig.dpi) + 50
        
        # 3. 强制锁定最小高度！这会阻止 ScrollArea 压缩组件
        self.setMinimumHeight(required_height_px)
        
        # 4. 通知布局系统尺寸已改变
        self.updateGeometry()
        
        # 5. 重绘
        self.draw()

    def plot_pareto(self, effects, t_crit, title=None):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        
        # 动态高度计算: 基础 5.0 inch，每多一个变量增加 0.4 inch
        n_vars = len(effects)
        h = max(5.0, n_vars * 0.4 + 1.5)
        
        if 'Importance' in effects.columns:
            bars = ax.barh(effects['Variable'], effects['Importance'], color='#66c2a5', edgecolor='k', height=0.6)
            if title:
                ax.set_title(title, fontsize=12)
            else:
                ax.set_title('Feature Importance (Random Forest)', fontsize=12)
            ax.set_xlabel('Importance Value')
        else:
            bars = ax.barh(effects['Variable'], effects['Abs_t_value'], color='skyblue', edgecolor='navy', height=0.6)
            for coef, bar in zip(effects['Coefficient'], bars):
                bar.set_color('#99ff99' if coef>=0 else '#ff9999')
            
            if t_crit > 0:
                ax.axvline(x=t_crit, color='red', linestyle='--', label='Critical t (95%)')
                ax.legend()
            
            if title:
                ax.set_title(title, fontsize=12)
            else:
                ax.set_title('Pareto Chart (Standardized Effect)', fontsize=12)
                
        ax.bar_label(bars, fmt=' %.3f', padding=3)
        self.fig.tight_layout()
        
        # 应用强制高度
        self.adjust_canvas_height(h)

    def plot_shap_beeswarm(self, engine):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        
        default_h = 6.5
        
        shap_vals, X_scaled = engine.get_shap_data()
        
        if isinstance(shap_vals, str) and shap_vals == "MISSING_LIB":
            ax.text(0.5, 0.5, "未检测到 shap 库\n请安装: pip install shap", ha='center', fontsize=14)
            self.adjust_canvas_height(4.0)
            return
        if shap_vals is None:
            ax.text(0.5, 0.5, "无法计算 SHAP 值 (非随机森林模型?)", ha='center')
            self.adjust_canvas_height(4.0)
            return

        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        sorted_idx = np.argsort(mean_abs_shap) 
        
        features = engine.X_cols
        n_features = len(features)
        
        if n_features > 10:
            default_h = max(6.5, n_features * 0.5)

        for i, idx in enumerate(sorted_idx):
            s_vals = shap_vals[:, idx]
            f_vals = X_scaled[:, idx]
            try:
                kde = gaussian_kde(s_vals)
                density = kde(s_vals)
                density = density / density.max() 
            except:
                density = np.ones_like(s_vals)
            
            jitter = (np.random.rand(len(s_vals)) - 0.5) * 0.5 * density
            sc = ax.scatter(s_vals, i + jitter, c=f_vals, cmap='coolwarm', s=15, alpha=0.8, edgecolors='none')

        ax.set_yticks(range(n_features))
        ax.set_yticklabels([features[i] for i in sorted_idx])
        ax.set_xlabel("SHAP Value (Impact on Model Output)")
        ax.set_title("SHAP Summary (Beeswarm Plot)")
        ax.axvline(x=0, color="#999999", linestyle="-", linewidth=0.5, zorder=-1)
        
        cbar = self.fig.colorbar(sc, ax=ax, aspect=30)
        cbar.set_label("Feature Value (Low --> High)")
        cbar.set_ticks([]) 
        
        self.fig.tight_layout()
        self.adjust_canvas_height(default_h)

    def plot_main_effects(self, engine):
        self.fig.clf()
        cols = engine.X_cols
        rows_num = math.ceil(len(cols)/3)
        # 动态高度: 行数越多，高度越大
        h = max(5.0, rows_num * 3.5)
        
        base = np.array([engine.bounds_info[c][2] for c in cols])
        all_pred = []
        plot_data = []
        for i, name in enumerate(cols):
            vmin, vmax, _ = engine.bounds_info[name]
            x = np.linspace(vmin, vmax, 20)
            mat = np.tile(base, (20, 1)); mat[:, i] = x
            y, _ = engine.predict_value_batch(mat)
            all_pred.extend(y); plot_data.append((x, y))
            
        ymin, ymax = min(all_pred), max(all_pred)
        margin = (ymax-ymin)*0.1 if ymax!=ymin else 1.0
        
        for i, name in enumerate(cols):
            ax = self.fig.add_subplot(rows_num, 3, i+1)
            ax.plot(plot_data[i][0], plot_data[i][1], 'b-', lw=2)
            ax.set_ylim(ymin-margin, ymax+margin)
            ax.set_title(name); ax.grid(True, ls=':')
            
        self.fig.suptitle(f"Main Effects ({engine.model_type})", fontsize=12)
        self.fig.tight_layout(rect=[0, 0, 1, 0.97])
        
        self.adjust_canvas_height(h)

    def plot_interaction_matrix(self, engine, top_vars):
        self.fig.clf()
        n = len(top_vars)
        if n < 2: return
        
        # 交互矩阵通常比较大，给予足够空间
        # 3x3 -> 8.5 inch, 4x4 -> 11.0 inch
        h = max(6.0, n * 2.8)
        
        axes = self.fig.subplots(n, n)
        base = np.array([engine.bounds_info[c][2] for c in engine.X_cols])
        
        for i, row_var in enumerate(top_vars):
            for j, col_var in enumerate(top_vars):
                ax = axes[i, j]
                if i == j:
                    ax.text(0.5, 0.5, row_var, ha='center', fontsize=10, weight='bold')
                    ax.axis('off'); continue
                
                c_min, c_max, _ = engine.bounds_info[col_var]
                r_min, r_max, _ = engine.bounds_info[row_var]
                x_vals = np.linspace(c_min, c_max, 20)
                r_idx = engine.X_cols.index(row_var); c_idx = engine.X_cols.index(col_var)
                
                mat_low = np.tile(base, (20, 1)); mat_low[:, c_idx] = x_vals; mat_low[:, r_idx] = r_min
                mat_high = np.tile(base, (20, 1)); mat_high[:, c_idx] = x_vals; mat_high[:, r_idx] = r_max
                
                y_low, _ = engine.predict_value_batch(mat_low) 
                y_high, _ = engine.predict_value_batch(mat_high)
                
                ax.plot(x_vals, y_low, 'r--', label='Low')
                ax.plot(x_vals, y_high, 'b-', label='High')
                ax.grid(True, ls=':')
                if j == 0: ax.set_ylabel(row_var, fontsize=8)
                if i == n-1: ax.set_xlabel(col_var, fontsize=8)

        self.fig.suptitle("Interaction Matrix", fontsize=12)
        self.fig.tight_layout(rect=[0, 0, 1, 0.97])
        
        self.adjust_canvas_height(h)

    def plot_surface_3d(self, engine, x_name, y_name):
        self.fig.clf()
        ax = self.fig.add_subplot(111, projection='3d')
        
        h = 6.5
        
        x_min, x_max, _ = engine.bounds_info[x_name]
        y_min, y_max, _ = engine.bounds_info[y_name]
        X, Y = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30))
        flat_X, flat_Y = X.ravel(), Y.ravel()
        base = np.array([engine.bounds_info[c][2] for c in engine.X_cols])
        mat = np.tile(base, (len(flat_X), 1))
        mat[:, engine.X_cols.index(x_name)] = flat_X; mat[:, engine.X_cols.index(y_name)] = flat_Y
        Z, _ = engine.predict_value_batch(mat)
        Z = Z.reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8)
        ax.contour(X, Y, Z, zdir='z', offset=Z.min(), cmap=cm.viridis, alpha=0.5)
        ax.set_xlabel(x_name); ax.set_ylabel(y_name); ax.set_zlabel('Y')
        ax.set_title(f"Response Surface ({engine.model_type})")
        self.fig.colorbar(surf, shrink=0.5, aspect=10)
        
        self.adjust_canvas_height(h)

    def plot_diagnostics(self, y_true, y_pred, metrics):
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.scatter(y_true, y_pred, c='blue', alpha=0.6, edgecolors='k', label='Data Points')
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        margin = (max_val - min_val) * 0.1
        lims = [min_val - margin, max_val + margin]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect Fit')
        ax.set_aspect('equal'); ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Actual Value'); ax.set_ylabel('Predicted Value')
        ax.set_title('Model Diagnostics')
        ax.grid(True, ls=':')
        r2_text = f"CV R² = {metrics.get('R2', 0):.4f}\nRMSE = {metrics.get('RMSE', 0):.4f}"
        ax.text(0.05, 0.95, r2_text, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        self.draw() 

class DiagnosticDialog(QDialog):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.setWindowTitle("模型诊断 (Diagnostics)")
        self.resize(600, 500)
        l = QVBoxLayout(self)
        cv = MplCanvas(self)
        l.addWidget(cv)
        y_true, y_pred = engine.get_diagnostic_data()
        if len(y_true) > 0:
            cv.plot_diagnostics(y_true, y_pred, engine.metrics)
        else:
            l.addWidget(QLabel("无数据或模型未训练"))

class LHSPlotDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LHS 采样分布")
        self.resize(700, 500)
        l = QVBoxLayout(self)
        cv = MplCanvas(self)
        l.addWidget(cv)
        
        cv.fig.clf()
        cols = df.columns[:-1]; n = min(len(cols), 5)
        cols = cols[:n]
        axes = cv.fig.subplots(n, n)
        if n == 1: axes = np.array([[axes]])
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if i == j:
                    ax.hist(df[cols[i]], bins=10, color='skyblue', edgecolor='black')
                    ax.set_title(cols[i], fontsize=8)
                else:
                    ax.scatter(df[cols[j]], df[cols[i]], alpha=0.7, s=15, c='blue')
                if i != n-1: ax.set_xticklabels([])
                if j != 0: ax.set_yticklabels([])
        cv.fig.tight_layout()
        cv.draw()