import numpy as np
import pandas as pd
import joblib 
from scipy.stats import qmc, t, norm 
from scipy.optimize import differential_evolution
import copy # 新增: 用于深拷贝数据

# Sklearn 核心库
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.inspection import permutation_importance

try:
    import shap
except ImportError:
    shap = None

class AnalysisEngine:
    def __init__(self):
        self.raw_df = None
        self.X_cols = []
        self.y_col = ""
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        
        self.model_interpretable = LinearRegression() 
        self.model_predictive = None                  
        self.model_type = "Kriging"                    
        
        self.metrics = {}
        self.bounds_info = {}
        self.feature_importance = None 

    def save_model(self, filepath):
        if self.model_predictive is None:
            return False, "模型未训练"
        try:
            data = {
                'model': self.model_predictive,
                'model_type': self.model_type,
                'scaler': self.scaler,
                'X_cols': self.X_cols,
                'y_col': self.y_col,
                'bounds_info': self.bounds_info,
                'metrics': self.metrics,
                'raw_df': self.raw_df,
                'feature_importance': self.feature_importance
            }
            joblib.dump(data, filepath)
            return True, "保存成功"
        except Exception as e:
            return False, str(e)

    def load_model(self, filepath):
        try:
            data = joblib.load(filepath)
            self.model_predictive = data['model']
            self.model_type = data['model_type']
            self.scaler = data['scaler']
            self.X_cols = data['X_cols']
            self.y_col = data['y_col']
            self.bounds_info = data['bounds_info']
            self.metrics = data['metrics']
            self.raw_df = data['raw_df']
            self.feature_importance = data.get('feature_importance')
            
            if self.raw_df is not None:
                X = self.raw_df[self.X_cols].values
                y = self.raw_df[self.y_col].values
                X_scaled = self.scaler.transform(X)
                X_poly_base = self.poly_features.fit_transform(X_scaled)
                self.model_interpretable.fit(X_poly_base, y)
                
            return True, "加载成功"
        except Exception as e:
            return False, str(e)

    def generate_lhs(self, variables, n_samples, decimals=2):
        if not variables: return None
        col_names = [v['name'] for v in variables]
        l_bounds = [v['low'] for v in variables]
        u_bounds = [v['up'] for v in variables]
        
        sampler = qmc.LatinHypercube(d=len(col_names))
        sample = qmc.scale(sampler.random(n=n_samples), l_bounds, u_bounds)
        
        df = pd.DataFrame(sample, columns=col_names)
        df = df.round(decimals)
        df['Response_Y'] = 0.0
        return df

    def recommend_algorithm(self, df):
        n_samples = len(df)
        if n_samples < 50:
            return "Kriging", "样本量较少 (<50)，推荐 Kriging 或 Polynomial 以获得平滑曲面。"
        else:
            return "RandomForest", "样本量充足 (>=50)，推荐 Random Forest 以捕捉复杂非线性关系。"

    def train_models(self, df, model_type="Kriging"):
        self.raw_df = df
        self.X_cols = list(df.columns[:-1])
        self.y_col = df.columns[-1]
        self.model_type = model_type
        
        X = df[self.X_cols].values
        y = df[self.y_col].values

        if np.std(y) == 0: return False, "Y值无变化，无法建模"

        for col in self.X_cols:
            self.bounds_info[col] = (df[col].min(), df[col].max(), df[col].mean())

        X_scaled = self.scaler.fit_transform(X)

        X_poly_base = self.poly_features.fit_transform(X_scaled)
        self.model_interpretable.fit(X_poly_base, y)

        if model_type == "Kriging":
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-5)
            self.model_predictive = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
            self.model_predictive.fit(X_scaled, y)
            self.feature_importance = None
            
        elif model_type == "RandomForest":
            self.model_predictive = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model_predictive.fit(X_scaled, y)
            r = permutation_importance(self.model_predictive, X_scaled, y, n_repeats=10, random_state=42)
            self.feature_importance = r.importances_mean

        elif model_type == "Polynomial":
            X_poly = self.poly_features.fit_transform(X_scaled)
            self.model_predictive = LinearRegression()
            self.model_predictive.fit(X_poly, y)
            self.feature_importance = None

        else:
            return False, f"未知的模型类型: {model_type}"

        n_splits = min(5, len(df))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        if model_type == "Polynomial":
            X_for_cv = self.poly_features.transform(X_scaled)
        else:
            X_for_cv = X_scaled

        scores_r2 = cross_val_score(self.model_predictive, X_for_cv, y, cv=kf, scoring='r2')
        scores_neg_mse = cross_val_score(self.model_predictive, X_for_cv, y, cv=kf, scoring='neg_mean_squared_error')
        
        self.metrics = {
            'R2': np.mean(scores_r2), 
            'RMSE': np.sqrt(-np.mean(scores_neg_mse)), 
            'CV_Folds': n_splits,
            'Model': model_type
        }
        return True, "Success"

    def predict_value(self, input_dict):
        if self.model_predictive is None: return 0.0, 0.0
        vec = [input_dict.get(col, 0.0) for col in self.X_cols]
        vec_scaled = self.scaler.transform([vec])
        
        if self.model_type == "Kriging":
            y_pred, y_std = self.model_predictive.predict(vec_scaled, return_std=True)
            return y_pred[0], y_std[0]
            
        elif self.model_type == "RandomForest":
            y_pred = self.model_predictive.predict(vec_scaled)
            per_tree_pred = [tree.predict(vec_scaled)[0] for tree in self.model_predictive.estimators_]
            y_std = np.std(per_tree_pred)
            return y_pred[0], y_std

        elif self.model_type == "Polynomial":
            vec_poly = self.poly_features.transform(vec_scaled)
            y_pred = self.model_predictive.predict(vec_poly)
            return y_pred[0], 0.0

        return 0.0, 0.0

    def get_diagnostic_data(self):
        if self.model_predictive is None: return [], []
        X = self.raw_df[self.X_cols].values
        y = self.raw_df[self.y_col].values
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == "Polynomial":
            X_input = self.poly_features.transform(X_scaled)
        else:
            X_input = X_scaled

        cv_n = min(5, len(y))
        try:
            y_pred_cv = cross_val_predict(self.model_predictive, X_input, y, cv=cv_n)
        except:
            y_pred_cv = self.model_predictive.predict(X_input)
        return y, y_pred_cv

    def optimize_response(self, goal='max', callback=None):
        if self.model_predictive is None: return None, 0.0
        bounds = [(self.bounds_info[col][0], self.bounds_info[col][1]) for col in self.X_cols]
        
        def objective(x):
            x_reshaped = x.reshape(1, -1)
            x_scaled = self.scaler.transform(x_reshaped)
            
            if self.model_type == "Polynomial":
                x_input = self.poly_features.transform(x_scaled)
            else:
                x_input = x_scaled
                
            y_pred = self.model_predictive.predict(x_input)[0]
            return -y_pred if goal == 'max' else y_pred
        
        result = differential_evolution(objective, bounds, seed=42, maxiter=50, popsize=15, callback=callback)
        best_X = {col: val for col, val in zip(self.X_cols, result.x)}
        best_Y = -result.fun if goal == 'max' else result.fun
        return best_X, best_Y

    # === Modified: 批量自适应采样 (Batch Active Learning) ===
    def recommend_next_batch(self, goal='max', n_recommend=1):
        """
        使用 'Kriging Believer' 策略批量推荐 n_recommend 个点
        """
        if self.model_predictive is None: return [], "模型未训练"
        if self.model_type != "Kriging": return [], "仅支持 Kriging 模型"

        recommendations = []
        
        # 1. 备份原始数据和模型状态 (非常重要！)
        original_df = self.raw_df.copy()
        
        # 临时工作数据
        temp_df = self.raw_df.copy()
        
        bounds = [(self.bounds_info[col][0], self.bounds_info[col][1]) for col in self.X_cols]

        for i in range(n_recommend):
            # 获取当前最优值 (基于临时数据)
            y_curr = temp_df[self.y_col].values
            current_best = np.max(y_curr) if goal == 'max' else np.min(y_curr)

            # 定义 EI 函数 (闭包引用当前的 model_predictive)
            def expected_improvement(x):
                x_reshaped = x.reshape(1, -1)
                x_scaled = self.scaler.transform(x_reshaped)
                mu, sigma = self.model_predictive.predict(x_scaled, return_std=True)
                mu = mu[0]; sigma = sigma[0]
                if sigma == 0.0: return 0.0
                
                xi = 0.01 
                if goal == 'max': imp = mu - current_best - xi
                else: imp = current_best - mu - xi
                    
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                return -ei

            # 寻找最佳 EI 点
            res = differential_evolution(expected_improvement, bounds, seed=None, maxiter=20, popsize=10)
            
            best_dict = {col: val for col, val in zip(self.X_cols, res.x)}
            
            # 预测该点的 Mean 和 Std
            pred_y, pred_std = self.predict_value(best_dict)
            
            recommendations.append({
                'vars': best_dict,
                'pred_y': pred_y,
                'pred_std': pred_std,
                'note': f"Rank {i+1} (EI)"
            })
            
            # === Kriging Believer 策略 ===
            # 将预测值作为"真实值"加入临时数据，重新训练模型
            # 这样模型就会认为这个区域已经探索过了，下次循环就会去别的地方
            new_row = best_dict.copy()
            new_row[self.y_col] = pred_y # 关键：相信模型预测的均值
            
            temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # 快速重训模型 (复用 train_models 逻辑)
            self.train_models(temp_df, "Kriging")

        # 3. 循环结束后，恢复原始状态
        self.train_models(original_df, "Kriging")
        
        return recommendations, "Success"

    def get_pareto_data(self):
        if self.model_type == "RandomForest" and self.feature_importance is not None:
            return pd.DataFrame({'Variable': self.X_cols, 'Importance': self.feature_importance}).sort_values('Importance', ascending=True), None
        if self.model_type == "Polynomial":
            feat_names = self.poly_features.get_feature_names_out(self.X_cols)
            coefs = self.model_predictive.coef_
            effects = pd.DataFrame({'Variable': feat_names, 'Coefficient': coefs, 'Abs_Coefficient': np.abs(coefs)})
            effects = effects[effects['Abs_Coefficient'] > 1e-4].sort_values('Abs_Coefficient', ascending=True)
            effects = effects.rename(columns={'Abs_Coefficient': 'Abs_t_value'})
            return effects, 0.0
        lin_temp = LinearRegression()
        X = self.raw_df[self.X_cols].values
        X_scaled = self.scaler.transform(X)
        y = self.raw_df[self.y_col].values
        lin_temp.fit(X_scaled, y)
        predictions = lin_temp.predict(X_scaled)
        mse = np.sum((y - predictions)**2) / max(1, len(y) - len(self.X_cols) - 1)
        X_aug = np.column_stack([np.ones(len(y)), X_scaled])
        try:
            XTX_inv = np.linalg.inv(np.dot(X_aug.T, X_aug))
            se = np.sqrt(np.diagonal(mse * XTX_inv))
            t_vals = np.concatenate(([lin_temp.intercept_], lin_temp.coef_)) / se
        except: t_vals = np.zeros(len(self.X_cols)+1)
        dof = len(y) - len(self.X_cols) - 1
        t_crit = t.ppf(0.975, dof) if dof > 0 else 2.0
        effects = pd.DataFrame({'Variable': self.X_cols, 'Coefficient': lin_temp.coef_, 't_value': t_vals[1:], 'Abs_t_value': np.abs(t_vals[1:])}).sort_values('Abs_t_value', ascending=True)
        return effects, t_crit

    def perform_sobol_analysis(self, n_samples=2000):
        if self.model_predictive is None: return None, None
        d = len(self.X_cols)
        sampler = qmc.LatinHypercube(d=d)
        A = qmc.scale(sampler.random(n=n_samples), [self.bounds_info[c][0] for c in self.X_cols], [self.bounds_info[c][1] for c in self.X_cols])
        B = qmc.scale(sampler.random(n=n_samples), [self.bounds_info[c][0] for c in self.X_cols], [self.bounds_info[c][1] for c in self.X_cols])
        df_samples = pd.DataFrame(A, columns=self.X_cols)
        y_A, _ = self.predict_value_batch(A)
        var_Y = np.var(y_A)
        if var_Y < 1e-9: return pd.DataFrame({'Variable':self.X_cols, 'Importance':0}), df_samples
        mean_y_A = np.mean(y_A)
        sobol_indices = []
        for i in range(d):
            AB_i = B.copy(); AB_i[:, i] = A[:, i] 
            y_AB_i, _ = self.predict_value_batch(AB_i)
            S1 = (np.mean(y_A * y_AB_i) - mean_y_A**2) / var_Y
            sobol_indices.append(max(0.0, S1))
        return pd.DataFrame({'Variable': self.X_cols, 'Importance': sobol_indices}).sort_values('Importance', ascending=True), df_samples

    def get_shap_data(self):
        if self.model_type != "RandomForest" or self.model_predictive is None: return None, None
        try:
            if shap is None: return "MISSING_LIB", None
            X = self.raw_df[self.X_cols].values
            X_scaled = self.scaler.transform(X)
            explainer = shap.TreeExplainer(self.model_predictive)
            return explainer.shap_values(X_scaled), X_scaled
        except Exception: return None, None

    def predict_value_batch(self, input_matrix):
        if self.model_predictive is None: return np.zeros(len(input_matrix)), np.zeros(len(input_matrix))
        mat_scaled = self.scaler.transform(input_matrix)
        if self.model_type == "Kriging": return self.model_predictive.predict(mat_scaled, return_std=True)
        elif self.model_type == "RandomForest": return self.model_predictive.predict(mat_scaled), np.zeros(len(input_matrix))
        elif self.model_type == "Polynomial": return self.model_predictive.predict(self.poly_features.transform(mat_scaled)), np.zeros(len(input_matrix))
        return np.zeros(len(input_matrix)), np.zeros(len(input_matrix))