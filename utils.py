import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler('doe_app.log', encoding='utf-8'),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger('DOEApp')

def save_dataframe(df, filename, format='excel'):
    """
    保存DataFrame到文件
    
    Parameters:
    df: pandas.DataFrame - 要保存的数据
    filename: str - 文件名
    format: str - 保存格式，支持'excel'和'csv'
    
    Returns:
    bool - 保存是否成功
    """
    try:
        if format == 'excel':
            df.to_excel(filename, index=False)
        elif format == 'csv':
            df.to_csv(filename, index=False, encoding='utf-8-sig')
        else:
            logger.error(f"不支持的文件格式: {format}")
            return False
        logger.info(f"数据已保存到 {filename}")
        return True
    except Exception as e:
        logger.error(f"保存数据失败: {e}")
        return False

def load_dataframe(filename):
    """
    从文件加载DataFrame
    
    Parameters:
    filename: str - 文件名
    
    Returns:
    pandas.DataFrame or None - 加载的数据，失败返回None
    """
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.xlsx', '.xls']:
            return pd.read_excel(filename)
        elif ext == '.csv':
            return pd.read_csv(filename, encoding='utf-8-sig')
        else:
            logger.error(f"不支持的文件格式: {ext}")
            return None
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return None

def validate_dataframe(df):
    """
    验证数据是否符合DOE要求
    
    Parameters:
    df: pandas.DataFrame - 要验证的数据
    
    Returns:
    tuple - (bool, str) 验证结果和错误信息
    """
    if df is None:
        return False, "数据为空"
    
    if df.empty:
        return False, "数据框为空"
    
    if len(df.columns) < 2:
        return False, "数据至少需要包含一个自变量和一个因变量"
    
    # 检查因变量是否为数值型
    y_col = df.columns[-1]
    if not np.issubdtype(df[y_col].dtype, np.number):
        return False, f"因变量 '{y_col}' 必须为数值型"
    
    # 检查自变量是否为数值型
    for col in df.columns[:-1]:
        if not np.issubdtype(df[col].dtype, np.number):
            return False, f"自变量 '{col}' 必须为数值型"
    
    return True, "数据验证通过"

def generate_filename(prefix='DOE', suffix=''):
    """
    生成带时间戳的文件名
    
    Parameters:
    prefix: str - 文件名前缀
    suffix: str - 文件名后缀
    
    Returns:
    str - 生成的文件名
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if suffix:
        return f"{prefix}_{timestamp}_{suffix}"
    return f"{prefix}_{timestamp}"

def calculate_vif(df, variables):
    """
    计算方差膨胀因子(VIF)，用于检测多重共线性
    
    Parameters:
    df: pandas.DataFrame - 数据
    variables: list - 自变量列表
    
    Returns:
    dict - 各变量的VIF值
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = variables
    vif_data["VIF"] = [variance_inflation_factor(df[variables].values, i) for i in range(len(variables))]
    
    return vif_data.set_index("Variable").to_dict()['VIF']
