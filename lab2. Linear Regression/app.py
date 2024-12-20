import ipywidgets as widgets
from ipywidgets import FloatSlider, Checkbox
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import product
import pandas as pd
import numpy as np
import warnings
from scipy.stats import shapiro, normaltest, t

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error as rmse
t.sf
# from scipy import stats
# import statsmodels.api as sm
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
sns.set_style("whitegrid")
sns.color_palette("pastel")
warnings.filterwarnings("ignore")


# from scipy.fft import fft, fftfreq

# https://www.kaggle.com/datasets/elikplim/eergy-efficiency-dataset?select=ENB2012_data.csv
df = None
x_train = None
x_test = None
y_train = None
y_test = None
x = None
y = None


def init_df(input_data_path):
    global df
    global x_train
    global x_test
    global y_train
    global y_test
    global y
    global x
    df = pd.read_csv(input_data_path)
    df = df.drop('Y2', axis=1)
    df["y"] = df["Y1"]
    y = df.y
    df = df.drop('Y1', axis=1)
    x = df.drop('y', axis=1)
    train_size = round(len(x)*0.7)
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]


"""
X1 Relative Compactness
X2 Surface Area
X3 Wall Area
X4 Roof Area
X5 Overall Height
X6 Orientation
X7 Glazing Area
X8 Glazing Area Distribution
y1 Heating Load
y2 Cooling Load
"""

def get_feat_checkboxes():
    return [Checkbox(value=True, description=f) for f in x.columns]

def plot_corr():
    plt.figure(figsize=(8, 6))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True,
                cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title("Матрица корреляций")
    plt.show()


def get_conf_df(alpha):
    regression_results = sm.OLS(y, x, missing="drop").fit()
    conf_intervals = regression_results.conf_int(alpha=alpha)
    conf_df = pd.DataFrame(conf_intervals.values, index=conf_intervals.index, columns=[
        "left border", "right border"])
    conf_df["t_statistic"] = regression_results.tvalues
    conf_df = conf_df.round(2)
    return conf_df


alpha_slider = FloatSlider(
    value=0.05,  # Первоначальное значение
    min=0,     # Минимум
    max=0.2,  # Максимум
    step=0.001,  # Шаг изменения
    description='Уровень α :',
    # True - событие observe возникает для каждого шага при изменении значения
    continuous_update=False,
    orientation='horizontal'  # Горизонтальное или вертикальное расположение
)

# def on_value_change(b):
#     global alpha
#     alpha = b["new"]
#     conf_df = get_conf_df(alpha)
#     display(conf_df)


# alpha_slider.observe(on_value_change, names='value')


def train_on_chosen(chosen_features):
    res = sm.OLS(
        y_train, x_train.loc[:, chosen_features], missing="drop").fit()
    return res


def get_rmse_on_test(res):
    pred = res.predict(x_test)
    return rmse(y_test, pred)


def get_predict(res):
    return res.predict(x_test)
