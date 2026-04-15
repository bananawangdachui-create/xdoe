import sys
import numpy as np
import pandas as pd
import time  
from datetime import datetime
import os

# 纯 PyQt6 导入
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QLabel, QPushButton, QSpinBox, QSplitter, QGroupBox, QFormLayout,
                                QMessageBox, QHeaderView, QTabWidget, QFileDialog, 
                                QTextEdit, QScrollArea, QComboBox, QDoubleSpinBox, 
                                QProgressDialog, QSizePolicy, QRadioButton, QCheckBox,
                                QButtonGroup, QListWidget, QListWidgetItem, QStackedWidget, QFrame,
                                QTableWidgetItem)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QColor, QFont
QT_MODULE = "PyQt6"

# 导入自定义模块
from analysis_engine import AnalysisEngine
from workers import TrainingWorker, OptimizationWorker, AdaptiveSamplingWorker
from plotting import MplCanvas, DiagnosticDialog, LHSPlotDialog
# 导入新增的组件
from components import SpreadsheetTable, RealTimePredictor

class DOEApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = AnalysisEngine()
        self.setWindowTitle("DOE Master Pro V10.2: 智能实验设计与优化系统")
        self.resize(1350, 880)
        self.setup_ui()
        self.init_default_vars()
        self.statusBar().showMessage("准备就绪")
        self.train_start_time = 0.0 

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 左侧导航栏 (Sidebar)
        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(200)
        self.sidebar.setStyleSheet("""
            QListWidget { background-color: #2D3436; color: white; outline: 0; border: none; }
            QListWidget::item { padding: 15px; font-size: 14px; font-weight: bold; border-bottom: 1px solid #444; }
            QListWidget::item:selected { background-color: #0984e3; color: white; }
            QListWidget::item:hover { background-color: #636e72; }
        """)
        items = [
            ("1. 实验设计", "📝"), 
            ("2. 模型训练", "⚙️"), 
            ("3. 结果分析", "📊"), 
            ("4. 预测与优化", "🚀")
        ]
        for name, icon in items:
            it = QListWidgetItem(f"{icon}  {name}")
            self.sidebar.addItem(it)
        self.sidebar.currentRowChanged.connect(self.change_page)
        main_layout.addWidget(self.sidebar)

        # 2. 右侧内容区 (Stacked Widget)
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # 添加页面
        self.page_design = self.create_design_page()
        self.page_train = self.create_train_page()
        self.page_analysis = self.create_analysis_page()
        self.page_opt = self.create_opt_page()
        
        self.stack.addWidget(self.page_design)
        self.stack.addWidget(self.page_train)
        self.stack.addWidget(self.page_analysis)
        self.stack.addWidget(self.page_opt)
        
        # 默认选中第一页
        self.sidebar.setCurrentRow(0)

    # --- 页面 1: 实验设计 ---
    def create_design_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 顶部：变量定义
        gb_var = QGroupBox("1.1 变量定义")
        v_layout = QVBoxLayout(gb_var)
        self.var_table = SpreadsheetTable()
        self.var_table.setColumnCount(3)
        self.var_table.setHorizontalHeaderLabels(["Name", "Min", "Max"])
        self.var_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        v_layout.addWidget(self.var_table)
        
        h_btn = QHBoxLayout()
        b_add = QPushButton("➕ 添加变量"); b_add.clicked.connect(self.add_var)
        b_del = QPushButton("➖ 删除变量"); b_del.clicked.connect(self.del_var)
        h_btn.addWidget(b_add); h_btn.addWidget(b_del); h_btn.addStretch()
        v_layout.addLayout(h_btn)
        layout.addWidget(gb_var)

        # 中部：抽样设置
        gb_sample = QGroupBox("1.2 抽样与数据生成")
        s_layout = QHBoxLayout(gb_sample)
        s_layout.addWidget(QLabel("样本数:"))
        self.sp_samp = QSpinBox(); self.sp_samp.setRange(5, 5000); self.sp_samp.setValue(30)
        s_layout.addWidget(self.sp_samp)
        
        s_layout.addWidget(QLabel("小数位:"))
        self.sp_dec = QSpinBox(); self.sp_dec.setRange(0, 6); self.sp_dec.setValue(2)
        s_layout.addWidget(self.sp_dec)
        
        b_gen = QPushButton("🎲 生成实验设计 (LHS)"); b_gen.clicked.connect(self.do_sampling)
        b_vis = QPushButton("👁️ 查看分布"); b_vis.clicked.connect(self.show_lhs_dist)
        s_layout.addWidget(b_gen); s_layout.addWidget(b_vis); s_layout.addStretch()
        layout.addWidget(gb_sample)

        # 底部：数据预览 + 导入导出功能
        h_data_head = QHBoxLayout()
        h_data_head.addWidget(QLabel("<b>1.3 实验数据 (请在此填入实验结果 Response_Y)</b>"))
        h_data_head.addStretch()
        
        b_import = QPushButton("📂 导入数据 (Excel/CSV)"); b_import.clicked.connect(self.import_data)
        b_export = QPushButton("💾 导出数据"); b_export.clicked.connect(self.export_data)
        b_import.setStyleSheet("background-color: #ffeaa7; color: #d35400; font-weight: bold;")
        h_data_head.addWidget(b_import)
        h_data_head.addWidget(b_export)
        
        layout.addLayout(h_data_head)
        
        self.data_tbl = SpreadsheetTable()
        layout.addWidget(self.data_tbl)
        
        return page

    # --- 页面 2: 模型训练 ---
    def create_train_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # 2.1 顶部配置栏
        gb_config = QGroupBox("模型训练配置")
        gb_config.setFixedHeight(90)
        l_config = QHBoxLayout(gb_config)
        l_config.setContentsMargins(20, 15, 20, 15)
        l_config.setSpacing(15)
        
        l_config.addWidget(QLabel("<b>选择算法:</b>"))
        self.combo_model = QComboBox()
        self.combo_model.addItems(["Kriging", "RandomForest", "Polynomial"])
        self.combo_model.setMinimumWidth(140)
        self.combo_model.currentTextChanged.connect(self.update_recommendation_ui)
        l_config.addWidget(self.combo_model)
        
        self.b_train = QPushButton("🚀 开始训练")
        self.b_train.clicked.connect(self.do_train)
        self.b_train.setFixedWidth(120)
        self.b_train.setStyleSheet("background-color: #0984e3; color: white; font-weight: bold; padding: 6px; border-radius: 4px;")
        l_config.addWidget(self.b_train)

        b_save_model = QPushButton("💾 保存模型"); b_save_model.clicked.connect(self.save_model_file)
        b_load_model = QPushButton("📂 加载模型"); b_load_model.clicked.connect(self.load_model_file)
        l_config.addWidget(b_save_model)
        l_config.addWidget(b_load_model)
        
        self.lbl_recommend = QLabel("💡 推荐: 请先加载数据")
        self.lbl_recommend.setStyleSheet("color: #636e72; font-style: italic; margin-left: 10px;")
        l_config.addWidget(self.lbl_recommend)
        l_config.addStretch()
        
        layout.addWidget(gb_config)
        
        # 2.2 主体内容区
        content_layout = QHBoxLayout()
        content_layout.setSpacing(15)
        
        gb_log = QGroupBox("训练日志 / 评估指标")
        l_log = QVBoxLayout(gb_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("font-family: Consolas; font-size: 13px; background-color: #f5f6fa; color: #2d3436; border: 1px solid #b2bec3;")
        l_log.addWidget(self.txt_log)
        content_layout.addWidget(gb_log, 3)
        
        gb_diag = QGroupBox("模型诊断 (Predicted vs Actual)")
        gb_diag.setStyleSheet("QGroupBox { font-weight: bold; color: #2d3436; }")
        l_diag = QVBoxLayout(gb_diag)
        l_diag.setContentsMargins(5, 10, 5, 5)
        
        self.cv_diagnostics = MplCanvas(self)
        l_diag.addWidget(self.cv_diagnostics)
        
        content_layout.addWidget(gb_diag, 7)
        
        layout.addLayout(content_layout)
        
        return page

    # --- 页面 3: 结果分析 ---
    def create_analysis_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # 顶部工具栏
        h_tools = QHBoxLayout()
        h_tools.addWidget(QLabel("<b>图表类型:</b>"))
        self.combo_plot_type = QComboBox()
        self.combo_plot_type.addItems(["关键因子(Pareto/Sobol)", "主效应图", "交互效应图", "响应面(Surface)"])
        self.combo_plot_type.currentTextChanged.connect(self.update_analysis_plot)
        h_tools.addWidget(self.combo_plot_type)
        
        self.lbl_plot_hint = QLabel("(请先训练模型)")
        self.lbl_plot_hint.setStyleSheet("color: gray")
        h_tools.addWidget(self.lbl_plot_hint)
        
        h_tools.addStretch()
        b_save = QPushButton("💾 保存图片"); b_save.clicked.connect(lambda: self.save_img(self.cv_analysis, "AnalysisPlot"))
        h_tools.addWidget(b_save)
        layout.addLayout(h_tools)
        
        # 参数控制区
        self.ctl_surf = QWidget(); hl_surf = QHBoxLayout(self.ctl_surf)
        hl_surf.setContentsMargins(0,0,0,0)
        hl_surf.addWidget(QLabel("X轴:")); self.cb_x = QComboBox(); self.cb_x.currentTextChanged.connect(self.update_analysis_plot)
        hl_surf.addWidget(self.cb_x)
        hl_surf.addWidget(QLabel("Y轴:")); self.cb_y = QComboBox(); self.cb_y.currentTextChanged.connect(self.update_analysis_plot)
        hl_surf.addWidget(self.cb_y); hl_surf.addStretch()
        layout.addWidget(self.ctl_surf)
        
        self.ctl_inter = QWidget(); hl_inter = QHBoxLayout(self.ctl_inter)
        hl_inter.setContentsMargins(0,0,0,0)
        hl_inter.addWidget(QLabel("Top N 因子:")); self.sp_top = QSpinBox(); self.sp_top.setRange(2, 6); self.sp_top.setValue(3)
        self.sp_top.valueChanged.connect(self.update_analysis_plot)
        hl_inter.addWidget(self.sp_top); hl_inter.addStretch()
        layout.addWidget(self.ctl_inter)
        
        # 滚动区域配置
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.stack_analysis = QStackedWidget()
        
        # 层1: 绘图画布
        self.cv_analysis = MplCanvas(self)
        self.stack_analysis.addWidget(self.cv_analysis)
        
        # 层2: Sobol 提示与操作区
        self.pg_sobol = QWidget()
        l_sobol = QVBoxLayout(self.pg_sobol)
        l_sobol.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        lbl_info = QLabel("⚠️ Kriging模型训练后无自动生成Pareto图\n\n(Kriging是黑盒模型，需要通过方差分析计算敏感度)")
        lbl_info.setStyleSheet("font-size: 16px; color: #636e72; font-weight: bold;")
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        l_sobol.addWidget(lbl_info)
        
        l_sobol.addSpacing(15)
        
        h_sobol_cfg = QHBoxLayout()
        h_sobol_cfg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        h_sobol_cfg.addWidget(QLabel("Sobol样本数:"))
        self.sp_sobol_n = QSpinBox()
        self.sp_sobol_n.setRange(100, 100000)
        self.sp_sobol_n.setValue(2000)
        self.sp_sobol_n.setSingleStep(500)
        self.sp_sobol_n.setFixedWidth(100)
        h_sobol_cfg.addWidget(self.sp_sobol_n)
        
        h_sobol_cfg.addSpacing(20)
        
        self.chk_export_sobol = QCheckBox("导出样本(CSV)")
        h_sobol_cfg.addWidget(self.chk_export_sobol)
        
        l_sobol.addLayout(h_sobol_cfg)
        l_sobol.addSpacing(15)
        
        b_sobol = QPushButton("▶ 继续 Sobol 排序分析")
        b_sobol.setFixedSize(220, 50)
        b_sobol.setStyleSheet("font-size: 15px; font-weight: bold; background-color: #0984e3; color: white; border-radius: 8px;")
        b_sobol.clicked.connect(self.run_sobol_sort)
        l_sobol.addWidget(b_sobol, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.stack_analysis.addWidget(self.pg_sobol)
        
        scroll_area.setWidget(self.stack_analysis)
        layout.addWidget(scroll_area)
        
        return page

    # --- 页面 4: 预测与优化 ---
    def create_opt_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        # 4.1 全局寻优
        gb_opt = QGroupBox("4.1 自动参数寻优 (Global Optimization)")
        gb_opt.setMaximumHeight(150)
        l_opt = QVBoxLayout(gb_opt)
        h_opt = QHBoxLayout()
        self.rb_max = QRadioButton("目标: 最大化"); self.rb_max.setChecked(True)
        self.rb_min = QRadioButton("目标: 最小化")
        b_run = QPushButton("⚡ 寻找最优解"); b_run.clicked.connect(self.run_optimizer)
        b_run.setStyleSheet("color: green; font-weight: bold;")
        h_opt.addWidget(self.rb_max); h_opt.addWidget(self.rb_min); h_opt.addWidget(b_run); h_opt.addStretch()
        l_opt.addLayout(h_opt)
        
        self.opt_tbl = SpreadsheetTable(); self.opt_tbl.setFixedHeight(60)
        l_opt.addWidget(self.opt_tbl)
        layout.addWidget(gb_opt)

        # 4.2 智能采样推荐
        gb_al = QGroupBox("4.2 自适应采样推荐 (Adaptive Sampling / Active Learning)")
        gb_al.setMaximumHeight(200) # 增加高度以容纳多行
        gb_al.setStyleSheet("QGroupBox { border: 1px solid #e17055; margin-top: 5px; } QGroupBox::title { color: #e17055; font-weight: bold; }")
        l_al = QVBoxLayout(gb_al)
        
        h_al_desc = QHBoxLayout()
        h_al_desc.addWidget(QLabel("基于 <b>EI (Expected Improvement)</b> 算法。"))
        
        # 新增: 数量选择
        h_al_desc.addWidget(QLabel("推荐点数量:"))
        self.sp_al_batch = QSpinBox()
        self.sp_al_batch.setRange(1, 10)
        self.sp_al_batch.setValue(1)
        h_al_desc.addWidget(self.sp_al_batch)
        
        h_al_desc.addStretch()
        b_al_run = QPushButton("🎯 推荐下一次实验")
        b_al_run.setStyleSheet("background-color: #e17055; color: white; font-weight: bold;")
        b_al_run.clicked.connect(self.run_adaptive_sampling)
        h_al_desc.addWidget(b_al_run)
        l_al.addLayout(h_al_desc)
        
        # 增加表格高度
        self.al_tbl = SpreadsheetTable(); self.al_tbl.setFixedHeight(120) 
        l_al.addWidget(self.al_tbl)
        layout.addWidget(gb_al)

        # === 界面分割：左侧实时预测，右侧多方案推演 ===
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # 左侧: 4.3 实时模拟器 (NEW)
        self.realtime_predictor = RealTimePredictor(self.engine)
        splitter.addWidget(self.realtime_predictor)

        # 右侧: 4.4 多方案表格
        gb_pred = QGroupBox("4.4 多方案批量推演")
        l_pred = QVBoxLayout(gb_pred)
        h_pred = QHBoxLayout()
        b1 = QPushButton("➕"); b1.clicked.connect(self.add_scenario)
        b2 = QPushButton("➖"); b2.clicked.connect(self.del_scenario)
        b3 = QPushButton("▶ 计算表格"); b3.clicked.connect(self.calc_scenarios)
        h_pred.addWidget(b1); h_pred.addWidget(b2); h_pred.addWidget(b3); h_pred.addStretch()
        l_pred.addLayout(h_pred)
        
        self.pred_tbl = SpreadsheetTable()
        l_pred.addWidget(self.pred_tbl)
        
        w_right = QWidget(); w_right.setLayout(l_pred)
        splitter.addWidget(w_right)
        
        # 设置初始比例 4:6
        splitter.setSizes([400, 600])

        return page

    # --- 逻辑控制 ---
    def change_page(self, row):
        self.stack.setCurrentIndex(row)
        if row == 2:
            self.update_plot_options()
            self.update_analysis_plot()
        elif row == 3:
            # 切换到预测页时，刷新模拟器的滑块
            self.realtime_predictor.setup_inputs()

    def init_default_vars(self):
        d = [("Pressure", 10, 50), ("Temp", 100, 200), ("Speed", 1, 10)]
        self.var_table.setRowCount(len(d))
        for i, (n, l, u) in enumerate(d):
            self.var_table.setItem(i, 0, QTableWidgetItem(n))
            self.var_table.setItem(i, 1, QTableWidgetItem(str(l)))
            self.var_table.setItem(i, 2, QTableWidgetItem(str(u)))

    def add_var(self):
        r = self.var_table.rowCount(); self.var_table.insertRow(r)
        self.var_table.setItem(r, 0, QTableWidgetItem(f"X{r+1}"))
        self.var_table.setItem(r, 1, QTableWidgetItem("0"))
        self.var_table.setItem(r, 2, QTableWidgetItem("1"))

    def del_var(self):
        if self.var_table.currentRow()>=0: self.var_table.removeRow(self.var_table.currentRow())

    def get_vars_config(self):
        res = []
        for i in range(self.var_table.rowCount()):
            try:
                res.append({
                    'name': self.var_table.item(i, 0).text(),
                    'low': float(self.var_table.item(i, 1).text()),
                    'up': float(self.var_table.item(i, 2).text())
                })
            except: pass
        return res

    def get_table_data(self, table):
        r, c = table.rowCount(), table.columnCount()
        if r==0: return None
        d = []
        h = [table.horizontalHeaderItem(i).text() for i in range(c)]
        try:
            for i in range(r):
                row = []
                for j in range(c):
                    it = table.item(i, j)
                    row.append(float(it.text()) if it and it.text().strip() else 0.0)
                d.append(row)
            return pd.DataFrame(d, columns=h)
        except: return None

    # Step 1: Design
    def do_sampling(self):
        v = self.get_vars_config()
        if not v: return
        df = self.engine.generate_lhs(v, self.sp_samp.value(), self.sp_dec.value())
        self.update_data_table(df)
        self.statusBar().showMessage("实验设计完成")
        self.update_recommendation_ui()

    def update_data_table(self, df):
        self.data_tbl.clear()
        self.data_tbl.setRowCount(len(df)); self.data_tbl.setColumnCount(len(df.columns))
        self.data_tbl.setHorizontalHeaderLabels(df.columns)
        for i in range(len(df)):
            for j, val in enumerate(df.iloc[i]):
                it = QTableWidgetItem(str(val))
                if j==len(df.columns)-1: it.setBackground(QColor(220, 245, 255))
                self.data_tbl.setItem(i, j, it)

    # 数据导入导出
    def import_data(self):
        f, _ = QFileDialog.getOpenFileName(self, "导入数据", "", "Excel/CSV Files (*.xlsx *.xls *.csv)")
        if not f: return
        try:
            if f.endswith('.csv'):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
            
            if df.empty: return QMessageBox.warning(self, "Err", "文件为空")
            self.update_data_table(df)
            QMessageBox.information(self, "Success", f"成功导入 {len(df)} 行数据")
            
            vars_in_file = df.columns[:-1] 
            self.var_table.setRowCount(len(vars_in_file))
            for i, col in enumerate(vars_in_file):
                self.var_table.setItem(i, 0, QTableWidgetItem(str(col)))
                self.var_table.setItem(i, 1, QTableWidgetItem(str(df[col].min())))
                self.var_table.setItem(i, 2, QTableWidgetItem(str(df[col].max())))
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"导入失败: {str(e)}")

    def export_data(self):
        df = self.get_table_data(self.data_tbl)
        if df is None: return
        f, _ = QFileDialog.getSaveFileName(self, "导出数据", "DOE_Data.xlsx", "Excel Files (*.xlsx);;CSV Files (*.csv)")
        if not f: return
        try:
            if f.endswith('.csv'):
                df.to_csv(f, index=False)
            else:
                df.to_excel(f, index=False)
            QMessageBox.information(self, "Success", "导出成功")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"导出失败: {str(e)}")

    def show_lhs_dist(self):
        df = self.get_table_data(self.data_tbl)
        if df is None: return QMessageBox.warning(self, "Err", "请先生成数据")
        dlg = LHSPlotDialog(df, self)
        dlg.exec()

    # Step 2: Training
    def update_recommendation_ui(self, txt=None):
        df = self.get_table_data(self.data_tbl)
        if df is None: return
        algo_name, reason = self.engine.recommend_algorithm(df)
        self.lbl_recommend.setText(f"💡 推荐: {reason}")
        
    def do_train(self):
        df = self.get_table_data(self.data_tbl)
        if df is None: return QMessageBox.warning(self, "Err", "无数据")
        
        self.pdlg = QProgressDialog("正在训练模型...", "取消", 0, 0, self)
        self.pdlg.setWindowTitle("任务执行中")
        self.pdlg.setWindowModality(Qt.WindowModality.WindowModal)
        self.pdlg.show()
        
        self.train_start_time = time.time()
        
        model_type = self.combo_model.currentText()
        self.worker = TrainingWorker(self.engine, df, model_type)
        self.worker.finished_signal.connect(self.on_train_done)
        self.worker.start()

    def on_train_done(self, success, msg):
        self.pdlg.close()
        if not success: return QMessageBox.warning(self, "Err", msg)
        
        duration = time.time() - self.train_start_time
        m = self.engine.metrics
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] 训练完成! (耗时: {duration:.2f}s)\n" \
                  f"=====================\n" \
                  f"模型: {m['Model']}\n" \
                  f"R2 : {m['R2']:.4f}\n" \
                  f"RMSE: {m['RMSE']:.4f}\n" \
                  f"CV Folds: {m['CV_Folds']}"
        
        self.txt_log.setText(log_msg)
        self.statusBar().showMessage(f"模型训练完成 (耗时 {duration:.2f}s)")
        self.update_plot_options()
        self.show_diagnostics()
        
        self.cb_x.clear(); self.cb_x.addItems(self.engine.X_cols)
        self.cb_y.clear(); self.cb_y.addItems(self.engine.X_cols)
        if len(self.engine.X_cols)>=2: self.cb_y.setCurrentIndex(1)
        self.init_pred_tables()
        # 刷新模拟器
        if hasattr(self, 'realtime_predictor'):
            self.realtime_predictor.setup_inputs()

    # 模型保存/加载
    def save_model_file(self):
        f, _ = QFileDialog.getSaveFileName(self, "保存模型", "DOE_Model.pkl", "Pickle Files (*.pkl)")
        if f:
            ok, msg = self.engine.save_model(f)
            if ok: QMessageBox.information(self, "Success", msg)
            else: QMessageBox.warning(self, "Error", msg)

    def load_model_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "加载模型", "", "Pickle Files (*.pkl)")
        if f:
            ok, msg = self.engine.load_model(f)
            if ok: 
                QMessageBox.information(self, "Success", msg)
                self.update_plot_options()
                self.show_diagnostics()
                self.txt_log.setText(f"模型已加载: {f}\nType: {self.engine.model_type}")
                self.init_pred_tables()
                if hasattr(self, 'realtime_predictor'):
                    self.realtime_predictor.setup_inputs()
            else: QMessageBox.warning(self, "Error", msg)

    def show_diagnostics(self):
        if not self.engine.model_predictive:
            self.cv_diagnostics.fig.clf()
            ax = self.cv_diagnostics.fig.add_subplot(111)
            ax.text(0.5, 0.5, "请先训练模型", ha='center', fontsize=12, color='gray')
            self.cv_diagnostics.draw()
            return

        y_true, y_pred = self.engine.get_diagnostic_data()
        if len(y_true) > 0:
            self.cv_diagnostics.plot_diagnostics(y_true, y_pred, self.engine.metrics)
        else:
            self.cv_diagnostics.fig.clf(); self.cv_diagnostics.draw()

    # Step 3: Analysis
    def update_plot_options(self):
        if not self.engine.model_predictive: return
        
        m_type = self.engine.model_type
        self.combo_plot_type.blockSignals(True)
        self.combo_plot_type.clear()
        
        if m_type == "RandomForest":
            self.combo_plot_type.addItems(["SHAP蜂群图", "关键因子(Pareto/Sobol)", "主效应图", "交互效应图"])
            self.lbl_plot_hint.setText("(RF模型: 推荐查看SHAP蜂群图)")
        elif m_type == "Polynomial":
            self.combo_plot_type.addItems(["关键因子(Pareto/Sobol)", "响应面(Surface)", "主效应图", "交互效应图"])
            self.lbl_plot_hint.setText("(Polynomial: 推荐查看Pareto系数或响应面)")
        else:
            self.combo_plot_type.addItems(["响应面(Surface)", "关键因子(Pareto/Sobol)", "主效应图", "交互效应图"])
            self.lbl_plot_hint.setText("(Kriging模型: 推荐查看响应面)")
            
        self.combo_plot_type.blockSignals(False)
        self.combo_plot_type.setCurrentIndex(0)

    def update_analysis_plot(self):
        if not self.engine.model_predictive: return
        
        ptype = self.combo_plot_type.currentText()
        
        self.ctl_surf.setVisible("响应面" in ptype)
        self.ctl_inter.setVisible("交互" in ptype)
        
        self.stack_analysis.setCurrentWidget(self.cv_analysis)
        
        if "关键因子" in ptype:
            if self.engine.model_type == "Kriging":
                self.stack_analysis.setCurrentWidget(self.pg_sobol)
                return
            
            ef, tc = self.engine.get_pareto_data()
            self.cv_analysis.plot_pareto(ef, tc)
            
        elif "蜂群图" in ptype:
            self.cv_analysis.plot_shap_beeswarm(self.engine)
        elif "主效应" in ptype:
            self.cv_analysis.plot_main_effects(self.engine)
        elif "交互" in ptype:
            ef, _ = self.engine.get_pareto_data()
            sort_col = 'Importance' if 'Importance' in ef.columns else 'Abs_t_value'
            if 'Abs_Coefficient' in ef.columns: sort_col = 'Abs_Coefficient'
            
            top = ef.sort_values(sort_col, ascending=False)['Variable'].head(self.sp_top.value()).tolist()
            self.cv_analysis.plot_interaction_matrix(self.engine, top)
        elif "响应面" in ptype:
            x, y = self.cb_x.currentText(), self.cb_y.currentText()
            if x and y and x!=y:
                self.cv_analysis.plot_surface_3d(self.engine, x, y)

    def run_sobol_sort(self):
        pdlg = QProgressDialog("正在进行 Sobol 全局敏感度分析 (蒙特卡洛模拟)...", "取消", 0, 0, self)
        pdlg.setWindowModality(Qt.WindowModality.WindowModal)
        pdlg.show()
        QApplication.processEvents()
        
        n_custom = self.sp_sobol_n.value()
        df_sobol, df_samples = self.engine.perform_sobol_analysis(n_samples=n_custom)
        pdlg.close()
        
        if df_sobol is None:
            QMessageBox.warning(self, "Err", "计算失败")
            return

        self.stack_analysis.setCurrentWidget(self.cv_analysis)
        self.cv_analysis.plot_pareto(df_sobol, 0, title=f"Sobol Global Sensitivity (N={n_custom})")
        
        if self.chk_export_sobol.isChecked() and df_samples is not None:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            default_name = f"Sobol_sample_{timestamp}.csv"
            f, _ = QFileDialog.getSaveFileName(self, "导出 Sobol 样本", default_name, "CSV Files (*.csv)")
            if f:
                try:
                    df_samples.to_csv(f, index=False)
                    QMessageBox.information(self, "Success", f"样本已保存")
                except Exception as e:
                    QMessageBox.warning(self, "Export Error", f"导出失败: {str(e)}")

    # Step 4: Optimization
    def init_pred_tables(self):
        cols = self.engine.X_cols + ["预测Y", "Std (Uncertainty)"]
        self.opt_tbl.clear(); self.opt_tbl.setColumnCount(len(cols)); self.opt_tbl.setHorizontalHeaderLabels(cols); self.opt_tbl.setRowCount(1)
        self.al_tbl.clear(); self.al_tbl.setColumnCount(len(cols)); self.al_tbl.setHorizontalHeaderLabels(cols); self.al_tbl.setRowCount(1)
        self.pred_tbl.clear(); self.pred_tbl.setColumnCount(len(cols)); self.pred_tbl.setHorizontalHeaderLabels(cols)
        if self.pred_tbl.rowCount()==0: self.add_scenario()

    def add_scenario(self):
        r = self.pred_tbl.rowCount(); self.pred_tbl.insertRow(r)
        for j, col in enumerate(self.engine.X_cols):
            val = self.engine.bounds_info[col][2] if col in self.engine.bounds_info else 0.0
            self.pred_tbl.setItem(r, j, QTableWidgetItem(f"{val:.2f}"))

    def del_scenario(self):
        if self.pred_tbl.currentRow()>=0: self.pred_tbl.removeRow(self.pred_tbl.currentRow())

    def calc_scenarios(self):
        if not self.engine.model_predictive: return
        r = self.pred_tbl.rowCount(); c = len(self.engine.X_cols)
        for i in range(r):
            inp = {}
            try:
                for j in range(c):
                    col = self.engine.X_cols[j]
                    inp[col] = float(self.pred_tbl.item(i, j).text())
                y, std = self.engine.predict_value(inp)
                self.pred_tbl.setItem(i, c, QTableWidgetItem(f"{y:.4f}"))
                self.pred_tbl.setItem(i, c+1, QTableWidgetItem(f"{std:.4f}"))
            except: pass
        self.statusBar().showMessage("预测完成")

    def run_optimizer(self):
        if not self.engine.model_predictive: return QMessageBox.warning(self, "Err", "请先训练模型")
        
        goal = 'max' if self.rb_max.isChecked() else 'min'
        
        self.pdlg_opt = QProgressDialog("正在全局寻优...", "取消", 0, 50, self)
        self.pdlg_opt.setWindowTitle("任务执行中")
        self.pdlg_opt.setWindowModality(Qt.WindowModality.WindowModal)
        self.pdlg_opt.show()
        
        self.opt_worker = OptimizationWorker(self.engine, goal)
        self.opt_worker.progress_signal.connect(lambda v: self.pdlg_opt.setValue(self.pdlg_opt.value()+1))
        self.opt_worker.finished_signal.connect(self.on_opt_done)
        self.opt_worker.start()

    def on_opt_done(self, best_X, best_Y, std):
        self.pdlg_opt.close()
        if not best_X: return QMessageBox.warning(self, "Err", "优化失败")
        
        for j, col in enumerate(self.engine.X_cols):
            it = QTableWidgetItem(f"{best_X[col]:.4f}")
            self.opt_tbl.setItem(0, j, it)
        self.opt_tbl.setItem(0, len(self.engine.X_cols), QTableWidgetItem(f"{best_Y:.4f}"))
        self.opt_tbl.setItem(0, len(self.engine.X_cols)+1, QTableWidgetItem(f"{std:.4f}"))
        self.statusBar().showMessage("寻优完成")

    # === Modified: 批量自适应采样执行 ===
    def run_adaptive_sampling(self):
        if not self.engine.model_predictive: 
            return QMessageBox.warning(self, "Err", "请先训练模型")
        if self.engine.model_type != "Kriging":
            return QMessageBox.warning(self, "Err", "自适应采样仅支持 Kriging 模型 (需预测方差)")

        goal = 'max' if self.rb_max.isChecked() else 'min'
        n_batch = self.sp_al_batch.value() # 获取数量
        
        self.pdlg_al = QProgressDialog(f"正在计算 EI (推荐 {n_batch} 组)...", "取消", 0, 0, self)
        self.pdlg_al.setWindowModality(Qt.WindowModality.WindowModal)
        self.pdlg_al.show()
        
        # 传递 n_batch
        self.al_worker = AdaptiveSamplingWorker(self.engine, goal, n_batch)
        self.al_worker.finished_signal.connect(self.on_al_done)
        self.al_worker.start()

    def on_al_done(self, recommendations, info_str):
        self.pdlg_al.close()
        if not recommendations: 
            return QMessageBox.warning(self, "Err", f"计算失败: {info_str}")

        # 清空并重新设置行数
        self.al_tbl.setRowCount(0)
        self.al_tbl.setRowCount(len(recommendations))
        
        for i, item in enumerate(recommendations):
            best_X = item['vars']
            pred_y = item['pred_y']
            note = item['note']
            
            for j, col in enumerate(self.engine.X_cols):
                val = best_X.get(col, 0.0)
                it = QTableWidgetItem(f"{val:.4f}")
                it.setBackground(QColor("#fab1a0"))
                self.al_tbl.setItem(i, j, it)
                
            self.al_tbl.setItem(i, len(self.engine.X_cols), QTableWidgetItem(f"{pred_y:.4f}"))
            self.al_tbl.setItem(i, len(self.engine.X_cols)+1, QTableWidgetItem(note))
        
        QMessageBox.information(self, "推荐结果", f"成功生成 {len(recommendations)} 组实验建议！")

    def save_img(self, canvas, name):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        model = self.engine.model_type if self.engine.model_predictive else "UnknownModel"
        
        ptype_raw = self.combo_plot_type.currentText()
        if "Pareto" in ptype_raw: ptype = "Pareto"
        elif "蜂群图" in ptype_raw: ptype = "SHAP_Beeswarm"
        elif "主效应" in ptype_raw: ptype = "MainEffects"
        elif "交互" in ptype_raw: ptype = "Interaction"
        elif "Surface" in ptype_raw: ptype = "Surface"
        else: ptype = "Plot"
        
        default_name = f"{model}_{ptype}_{timestamp}.png"
        f, _ = QFileDialog.getSaveFileName(self, "保存图片", default_name, "Images (*.png)")
        if f: 
            canvas.figure.savefig(f, dpi=300, bbox_inches='tight')