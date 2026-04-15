# 只使用 PyQt6
from PyQt6.QtCore import QThread, pyqtSignal

class TrainingWorker(QThread):
    finished_signal = pyqtSignal(bool, str)
    def __init__(self, engine, df, model_type):
        super().__init__()
        self.engine = engine; self.df = df; self.model_type = model_type
    def run(self):
        try:
            success, msg = self.engine.train_models(self.df, self.model_type)
            self.finished_signal.emit(success, msg)
        except Exception as e: self.finished_signal.emit(False, str(e))

class OptimizationWorker(QThread):
    finished_signal = pyqtSignal(dict, float, float)
    progress_signal = pyqtSignal(int)
    def __init__(self, engine, goal):
        super().__init__()
        self.engine = engine; self.goal = goal
    def run(self):
        def optim_callback(xk, convergence=None): self.progress_signal.emit(1)
        try:
            best_X, best_Y = self.engine.optimize_response(self.goal, callback=optim_callback)
            _, std = self.engine.predict_value(best_X)
            self.finished_signal.emit(best_X, best_Y, std)
        except Exception: self.finished_signal.emit({}, 0.0, 0.0)

# === Modified: 批量自适应采样 Worker ===
class AdaptiveSamplingWorker(QThread):
    finished_signal = pyqtSignal(list, str) # 改为 list
    def __init__(self, engine, goal, n_batch=1):
        super().__init__()
        self.engine = engine; self.goal = goal; self.n_batch = n_batch
    def run(self):
        try:
            # 调用新的批量推荐方法
            recommendations, info = self.engine.recommend_next_batch(self.goal, self.n_batch)
            self.finished_signal.emit(recommendations, info)
        except Exception as e:
            self.finished_signal.emit([], str(e))