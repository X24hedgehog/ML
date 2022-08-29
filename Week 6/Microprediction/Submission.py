from microprediction import new_key, MicroWriter
import collections
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math
from river import dummy
from river import tree
from river import time_series
from river import model_selection
from river import facto
from river import optim
from river import stats
from river import feature_extraction
from river import preprocessing
from river import evaluate
from river import metrics
from river import linear_model
from river import compose
from river import stream
import pandas as pd


dataset = pd.read_csv("hospital_wait.csv")

dataset.head()
print(type(dataset))

params = {'converters': {'value': float},
          'parse_dates': {'time': "%Y-%m-%d %H:%M:%S"}}

dataset = dict()
# for x, y in stream.iter_csv('hospital_wait.csv', target = 'value', **params):
#     print(x,y)


hour_list = [str(i) for i in range(0, 24)]
minute_list = [f'{str(i)} min' for i in range(0, 60)]


def get_hour(x):
    x['h'] = x['time'].hour
    return x


print(get_hour({'time': datetime.datetime(2022, 7, 22, 15, 3, 37)}))


def get_minute(x):
    x['m'] = x['time'].minute//15 + 1
    return x


def get_day(x):

    return {'d': x['time'].day}


def get_hour_sin_and_cos(x):
    x['sin_h'] = np.sin(np.pi*(x['time'].hour)/12)
    x['cos_h'] = np.cos(np.pi*(x['time'].hour)/12)
    return {'sin_h': np.sin(np.pi*(x['time'].hour)/12), 'cos_h': np.cos(np.pi*(x['time'].hour)/12)}


print(get_hour_sin_and_cos({'time': datetime.datetime(2022, 7, 22, 6, 3, 37)}))


def get_minute_distances(x):
    x['sin_m'] = np.sin(np.pi*(x['time'].minute)/30)
    x['cos_m'] = np.cos(np.pi*(x['time'].minute)/30)
    return {'sin_m': np.sin(np.pi*(x['time'].minute)/30), 'cos_m': np.cos(np.pi*(x['time'].minute)/30)}


def get_date_progress(x):
    return {'date': x['time'].toordinal() - datetime.datetime(2022, 1, 1, 0, 0).toordinal()}


i = 0
temp = [323, 323, 323, 323]
cache = [temp]
my_dict = {}
# print(cache)
for x, y in stream.iter_csv('hospital_wait.csv', target='value', **params):
    # print(x)
    if i < 4:
        t = temp.copy()
        t[i] = y
        cache.append(t)
        temp = t
        my_dict[x['time']] = t
    else:
        t = temp.copy()
        t.pop(0)
        t.append(y)
        cache.append(t)
        temp = t
        my_dict[x['time']] = t
    i += 1

# print(my_dict)


def get_lag(x):
    lag_values = my_dict[x['time']]
    return {'lag_1': lag_values[0], 'lag_2': lag_values[1], 'lag_3': lag_values[2], 'lag_4': lag_values[3]}


models = [linear_model.LinearRegression(optimizer=optim.SGD(lr=lr)) for lr in [
    0.05, 0.02, 0.01, 0.005, 0.002, 0.0001]]


model = compose.Pipeline(
    ('features', compose.TransformerUnion(
        ('date_progress', compose.FuncTransformer(get_date_progress))
        #         ('lags', compose.FuncTransformer(get_lag))
    )))

model += (
    get_hour |
    feature_extraction.TargetAgg(
        by=['h'], how=stats.Mean()


    ))
model += (
    get_minute |
    feature_extraction.TargetAgg(
        by=['m'], how=stats.Mean()


    ))

model |= preprocessing.StandardScaler()
model |= preprocessing.TargetStandardScaler(
    model_selection.UCBRegressor(
        models +
        [
            tree.HoeffdingTreeRegressor(grace_period=20),
            linear_model.PARegressor(C=0.012, eps=0.05),
        ],
        delta=0.01, burn_in=100, seed=1
    )
)


metric = metrics.MAE() + metrics.R2()
evaluate.progressive_val_score(stream.iter_csv(
    'hospital_wait.csv', target='value', **params), model, metric, print_every=50)
# evaluate.progressive_val_score(stream.iter_csv('hospital_wait.csv', target = 'value', **params), dummy.StatisticRegressor(stats.Shift(1)), metric, print_every=50)
model.transform_one(x)


queue = collections.deque([], 4)


def evaluate_model(model):

    metric = metrics.Rolling(metrics.MAE(), 10)
    metric_b = metrics.Rolling(metrics.MAE(), 10)

    dates = []
    y_trues = []
    y_preds = []

    baseline = 0
    y_baseline = []
    for x, y in stream.iter_csv('hospital_wait.csv', target='value', **params):

        new_feats = {f"lag_{i}": v for i, v in enumerate(queue)}

        # copy of x
        x_ = dict(x)
        x_.update(new_feats)

        y_pred = model.predict_one(x_)
        model.learn_one(x_, y)

        queue.append(y)

        # Obtain the prior prediction and update the model in one go
        y_pred = model.predict_one(x)
        model.learn_one(x, y)

        # Update the error metric
        metric.update(y, y_pred)
        metric_b.update(y, baseline)

        # Store the true value and the prediction
        dates.append(x['time'])
        y_trues.append(y)
        y_preds.append(y_pred)
        y_baseline.append(baseline)
        baseline = y

    print(metric, metric_b)

    # Plot the results
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.grid(alpha=0.75)
    ax.plot(dates, y_trues, lw=3, color='#2ecc71',
            alpha=0.8, label='Ground truth')
    ax.plot(dates, y_preds, lw=3, color='#e74c3c',
            alpha=0.8, label='Prediction')
    ax.plot(dates, y_baseline, lw=3, color='#e74c3c',
            alpha=0.8, label='Baseline')
    ax.legend()
    ax.set_title(metric)


evaluate_model(model)


def make_model(alpha):
    models = [linear_model.LinearRegression(optimizer=optim.SGD(lr=lr), loss=optim.losses.Quantile(
        alpha=alpha)) for lr in [0.05, 0.02, 0.01, 0.005, 0.002, 0.0001]]

    model = compose.Pipeline(
        ('features', compose.TransformerUnion(
            ('date_progress', compose.FuncTransformer(get_date_progress)),
            ('lags', compose.FuncTransformer(get_lag))
        )))

    model += (
        get_hour |
        feature_extraction.TargetAgg(
            by=['h'], how=stats.Mean()


        ))

    model |= preprocessing.StandardScaler()
    model |= preprocessing.TargetStandardScaler(
        model_selection.UCBRegressor(
            models,
            delta=0.01, burn_in=100, seed=1
        )
    )
    return model


models = {f"ans_{i}": make_model(i/225) for i in range(1, 226)}


dates = []
y_trues = []
y_preds = {
    f"ans_{i}": [] for i in range(1, 226)
}

for x, y in stream.iter_csv('hospital_wait.csv', target='value', **params):

    y_trues.append(y)
    dates.append(x['time'])

    for name, model in models.items():
        y_preds[name].append(model.predict_one(x))
        model.learn_one(x, y)

    # Update the error metric
    metric.update(y, y_preds['ans_112'][-1])


xs = [arr[-1] for arr in list(y_preds.values())]

# Plot the results
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.grid(alpha=0.75)
# ax.plot(dates, y_trues, lw=3, color='#2ecc71', alpha=0.8, label='Truth')
# ax.plot(dates, y_preds['center'], lw=3,
#         color='#e74c3c', alpha=0.8, label='Prediction')
# ax.fill_between(dates, y_preds['lower'], y_preds['upper'],
#                 color='#e74c3c', alpha=0.3, label='Prediction interval')
# ax.legend()
# ax.set_title(metric)


write_key = new_key(difficulty=9)
mw = MicroWriter(write_key=write_key)
print(mw.shash(write_key))
print(mw.animal_from_key(write_key))

name = 'hospital-er-wait-minutes-piedmont_henry'
mw.num_predictions
print(xs)
plt.hist(xs)

res = mw.submit(name=name, values=list(xs), delay=70)
print(res)


print(write_key)
