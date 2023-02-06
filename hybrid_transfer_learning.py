#%% Init
import time
start = time.time()

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from pylab import rcParams
from whitebox import Reactor
from blackbox import ML
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

plt.style.use('seaborn-white')
rcParams['figure.figsize'] = 15, 8
rcParams["savefig.dpi"] = 300

#%% Get data
df = pd.read_csv('datasets/dataset1/csv_measurements.csv')
df = df.iloc[:-1:60, :]
df2 = pd.read_csv('datasets/dataset2/csv_measurements.csv')
df2 = df2.iloc[:-1:60, :]

#%% Define predictor and response
t = df['t/s']
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]

ts = t[:-720]
Xs = X.iloc[:-720, :]
ys = y[:-720]
us = np.array([Xs.iloc[:, 1], Xs.iloc[:, 4], Xs.iloc[:, 5]+273.15, \
    Xs.iloc[:, 3]+273.15])

tt = df2['t/s']
Xt = df2.iloc[:, 1:-1]
yt = df2.iloc[:, -1]
ut = np.array([Xt.iloc[:, 1], Xt.iloc[:, 4], Xt.iloc[:, 5]+273.15, \
    Xt.iloc[:, 3]+273.15])

tt1 = tt.iloc[720+5:2*720]
Xt1 = Xt.iloc[720+5:2*720, :]
yt1 = yt.iloc[720+5:2*720]
ut1 = np.array([Xt1.iloc[:, 1], Xt1.iloc[:, 4], Xt1.iloc[:, 5]+273.15, \
    Xt1.iloc[:, 3]+273.15])

tt2 = tt.iloc[2*720+10:3*720]
Xt2 = Xt.iloc[2*720+10:3*720, :]
yt2 = yt.iloc[2*720+10:3*720]
ut2 = np.array([Xt2.iloc[:, 1], Xt2.iloc[:, 4], Xt2.iloc[:, 5]+273.15, \
    Xt2.iloc[:, 3]+273.15])

tt3 = tt.iloc[3*720+10:4*720]
Xt3 = Xt.iloc[3*720+10:4*720, :]
yt3 = yt.iloc[3*720+10:4*720]
ut3 = np.array([Xt3.iloc[:, 1], Xt3.iloc[:, 4], Xt3.iloc[:, 5]+273.15, \
    Xt3.iloc[:, 3]+273.15])

tt4 = tt.iloc[4*720+10:5*720]
Xt4 = Xt.iloc[4*720+10:5*720, :]
yt4 = yt.iloc[4*720+10:5*720]
ut4 = np.array([Xt4.iloc[:, 1], Xt4.iloc[:, 4], Xt4.iloc[:, 5]+273.15, \
    Xt4.iloc[:, 3]+273.15])

#%% White-Box Model
wb_model = Reactor()
par = wb_model.train(ts, us, ys.values)

svR_test1 = wb_model.predict(tt1[:7*24], par, ut1[:7*24])
df_test1 = pd.DataFrame(data=[])
df_test1['WB'] = svR_test1[:, 3]

svR_test2 = wb_model.predict(tt2[:7*24], par, ut2[:7*24])
df_test2 = pd.DataFrame(data=[])
df_test2['WB'] = svR_test2[:, 3]

svR_test3 = wb_model.predict(tt3[:7*24], par, ut3[:7*24])
df_test3 = pd.DataFrame(data=[])
df_test3['WB'] = svR_test3[:, 3]

svR_test4 = wb_model.predict(tt4[:7*24], par, ut4[:7*24])
df_test4 = pd.DataFrame(data=[])
df_test4['WB'] = svR_test4[:, 3]

#%% Data-driven model
candidates = ['lasso', 'pls', 'mlp',]

for candidate in candidates:
    model = ML()
    model.train(Xs, ys, method=candidate)
    df_test1['DD_'+candidate.upper()] = model.predict(Xt1[:7*24])
    df_test2['DD_'+candidate.upper()] = model.predict(Xt2[:7*24])
    df_test3['DD_'+candidate.upper()] = model.predict(Xt3[:7*24])
    df_test4['DD_'+candidate.upper()] = model.predict(Xt4[:7*24])

fig, ax = plt.subplots(nrows=4, ncols=1)
ax[0].plot(tt1/(3600*24), yt1, 'k', label='Actual')
ax[0].plot(tt1[:7*24]/(3600*24), df_test1['WB'], label='WB')
for candidate in candidates:
    ax[0].plot(tt1[:7*24]/(3600*24), df_test1['DD_'+candidate.upper()], label='DD_'+candidate.upper())
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[1].plot(tt2/(3600*24), yt2, 'k', label='Actual')
ax[1].plot(tt2[:7*24]/(3600*24), df_test2['WB'], label='WB')
for candidate in candidates:
    ax[1].plot(tt2[:7*24]/(3600*24), df_test2['DD_'+candidate.upper()], label='DD_'+candidate.upper())
ax[1].tick_params(axis='both', which='major', labelsize=16)
ax[2].plot(tt3/(3600*24), yt3, 'k', label='Actual')
ax[2].plot(tt3[:7*24]/(3600*24), df_test3['WB'], label='WB')
for candidate in candidates:
    ax[2].plot(tt3[:7*24]/(3600*24), df_test3['DD_'+candidate.upper()], label='DD_'+candidate.upper())
ax[2].tick_params(axis='both', which='major', labelsize=16)
ax[3].plot(tt4/(3600*24), yt4, 'k', label='Actual')
ax[3].plot(tt4[:7*24]/(3600*24), df_test4['WB'], label='WB')
for candidate in candidates:
    ax[3].plot(tt4[:7*24]/(3600*24), df_test4['DD_'+candidate.upper()], label='DD_'+candidate.upper())
ax[3].tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='best')
ax[3].set_xlabel('Time / days', fontsize=18)
ax[0].set_ylabel('KPI / -', fontsize=18)
ax[1].set_ylabel('KPI / -', fontsize=18)
ax[2].set_ylabel('KPI / -', fontsize=18)
ax[3].set_ylabel('KPI / -', fontsize=18)
plt.savefig('figures/DD_profiles.png')

# FAZER UM SUBPLOT PARA CADA ZONA DE TESTE

#%% Parallel Cooperative Hybrid model (WB + BB)
svR = wb_model.predict(ts, par, us)

for candidate in candidates:
    model = ML()
    model.train(Xs, ys - svR[:, 3], method=candidate)
    df_test1['PCoopH_'+candidate.upper()] = df_test1['WB'].values + model.predict(Xt1[:7*24]).reshape(-1)
    df_test2['PCoopH_'+candidate.upper()] = df_test2['WB'].values + model.predict(Xt2[:7*24]).reshape(-1)
    df_test3['PCoopH_'+candidate.upper()] = df_test3['WB'].values + model.predict(Xt3[:7*24]).reshape(-1)
    df_test4['PCoopH_'+candidate.upper()] = df_test4['WB'].values + model.predict(Xt4[:7*24]).reshape(-1)

fig, ax = plt.subplots(nrows=4, ncols=1)
ax[0].plot(tt1/(3600*24), yt1, 'k', label='Actual')
for candidate in candidates:
    ax[0].plot(tt1[:7*24]/(3600*24), df_test1['PCoopH_'+candidate.upper()], label='PCoopH_'+candidate.upper())
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[1].plot(tt2/(3600*24), yt2, 'k', label='Actual')
for candidate in candidates:
    ax[1].plot(tt2[:7*24]/(3600*24), df_test2['PCoopH_'+candidate.upper()], label='PCoopH_'+candidate.upper())
ax[1].tick_params(axis='both', which='major', labelsize=16)
ax[2].plot(tt3/(3600*24), yt3, 'k', label='Actual')
for candidate in candidates:
    ax[2].plot(tt3[:7*24]/(3600*24), df_test3['PCoopH_'+candidate.upper()], label='PCoopH_'+candidate.upper())
ax[2].tick_params(axis='both', which='major', labelsize=16)
ax[3].plot(tt4/(3600*24), yt4, 'k', label='Actual')
for candidate in candidates:
    ax[3].plot(tt4[:7*24]/(3600*24), df_test4['PCoopH_'+candidate.upper()], label='PCoopH_'+candidate.upper())
ax[3].tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='best')
ax[3].set_xlabel('Time / days', fontsize=18)
ax[0].set_ylabel('KPI / -', fontsize=18)
ax[1].set_ylabel('KPI / -', fontsize=18)
ax[2].set_ylabel('KPI / -', fontsize=18)
ax[3].set_ylabel('KPI / -', fontsize=18)
plt.savefig('figures/PCoopH_profiles.png')

#%% Knowledge-Augmented Hybrid model (WB -> BB)
X_train = np.concatenate([Xs.values, svR], axis=1)
X_test1 = np.concatenate([Xt1[:7*24].values, svR_test1], axis=1)
X_test2 = np.concatenate([Xt2[:7*24].values, svR_test2], axis=1)
X_test3 = np.concatenate([Xt3[:7*24].values, svR_test3], axis=1)
X_test4 = np.concatenate([Xt4[:7*24].values, svR_test4], axis=1)

for candidate in candidates:
    model = ML()
    model.train(X_train, ys, method=candidate)
    df_test1['KAH_'+candidate.upper()] = model.predict(X_test1)
    df_test2['KAH_'+candidate.upper()] = model.predict(X_test2)
    df_test3['KAH_'+candidate.upper()] = model.predict(X_test3)
    df_test4['KAH_'+candidate.upper()] = model.predict(X_test4)

fig, ax = plt.subplots(nrows=4, ncols=1)
ax[0].plot(tt1/(3600*24), yt1, 'k', label='Actual')
for candidate in candidates:
    ax[0].plot(tt1[:7*24]/(3600*24), df_test1['KAH_'+candidate.upper()], label='KAH_'+candidate.upper())
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[1].plot(tt2/(3600*24), yt2, 'k', label='Actual')
for candidate in candidates:
    ax[1].plot(tt2[:7*24]/(3600*24), df_test2['KAH_'+candidate.upper()], label='KAH_'+candidate.upper())
ax[1].tick_params(axis='both', which='major', labelsize=16)
ax[2].plot(tt3/(3600*24), yt3, 'k', label='Actual')
for candidate in candidates:
    ax[2].plot(tt3[:7*24]/(3600*24), df_test3['KAH_'+candidate.upper()], label='KAH_'+candidate.upper())
ax[2].tick_params(axis='both', which='major', labelsize=16)
ax[3].plot(tt4/(3600*24), yt4, 'k', label='Actual')
for candidate in candidates:
    ax[3].plot(tt4[:7*24]/(3600*24), df_test4['KAH_'+candidate.upper()], label='KAH_'+candidate.upper())
ax[3].tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='best')
ax[3].set_xlabel('Time / days', fontsize=18)
ax[0].set_ylabel('KPI / -', fontsize=18)
ax[1].set_ylabel('KPI / -', fontsize=18)
ax[2].set_ylabel('KPI / -', fontsize=18)
ax[3].set_ylabel('KPI / -', fontsize=18)
plt.savefig('figures/KAH_profiles.png')

#%% Serial Hybrid model (BB -> WB)
rates = wb_model.dyn_rates(ts, us, ys.values, par0=par)

def serial_predict(time, X, u, model_wb, model_ml):
    samples = len(time)
    tspan = time.values
    x0 = np.array([0.0031, 0.4235, 0.1432, 0.4302, 333.5500,])

    sol = model_wb.ode_eval([tspan[0], tspan[-1]], model_ml.predict(X.values[0, :].reshape(1, -1)), u[:, 0], x0)
    x0 = sol.y[:, -1]
    states = np.array([])
    for i in range(samples-1):
        states = np.append(states, x0)
        rate = model_ml.predict(X.values[i, :].reshape(1, -1))
        sol = model_wb.ode_eval(tspan[i:i+2], rate, u[:, i], x0)
        x0 = sol.y[:, -1]
    states = np.append(states, sol.y[:, -1])
    states = states.reshape(samples, len(x0))
    return states[:, 3]

for candidate in candidates:
    model = ML()
    model.train(Xs.values[:-1, :], rates, method=candidate)
    df_test1['SH_'+candidate.upper()] = serial_predict(tt1[:7*24], Xt1[:7*24], ut1[:7*24], wb_model, model)
    df_test2['SH_'+candidate.upper()] = serial_predict(tt2[:7*24], Xt2[:7*24], ut2[:7*24], wb_model, model)
    df_test3['SH_'+candidate.upper()] = serial_predict(tt3[:7*24], Xt3[:7*24], ut3[:7*24], wb_model, model)
    df_test4['SH_'+candidate.upper()] = serial_predict(tt4[:7*24], Xt4[:7*24], ut4[:7*24], wb_model, model)

fig, ax = plt.subplots(nrows=4, ncols=1)
ax[0].plot(tt1/(3600*24), yt1, 'k', label='Actual')
for candidate in candidates:
    ax[0].plot(tt1[:7*24]/(3600*24), df_test1['SH_'+candidate.upper()], label='SH_'+candidate.upper())
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[1].plot(tt2/(3600*24), yt2, 'k', label='Actual')
for candidate in candidates:
    ax[1].plot(tt2[:7*24]/(3600*24), df_test2['SH_'+candidate.upper()], label='SH_'+candidate.upper())
ax[1].tick_params(axis='both', which='major', labelsize=16)
ax[2].plot(tt3/(3600*24), yt3, 'k', label='Actual')
for candidate in candidates:
    ax[2].plot(tt3[:7*24]/(3600*24), df_test3['SH_'+candidate.upper()], label='SH_'+candidate.upper())
ax[2].tick_params(axis='both', which='major', labelsize=16)
ax[3].plot(tt4/(3600*24), yt4, 'k', label='Actual')
for candidate in candidates:
    ax[3].plot(tt4[:7*24]/(3600*24), df_test4['SH_'+candidate.upper()], label='SH_'+candidate.upper())
ax[3].tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='best')
ax[3].set_xlabel('Time / days', fontsize=18)
ax[0].set_ylabel('KPI / -', fontsize=18)
ax[1].set_ylabel('KPI / -', fontsize=18)
ax[2].set_ylabel('KPI / -', fontsize=18)
ax[3].set_ylabel('KPI / -', fontsize=18)
plt.savefig('figures/SH_profiles.png')

#%% Metrics
names = []
rmse_1 = np.array([])
rmse_2 = np.array([])
rmse_3 = np.array([])
rmse_4 = np.array([])
for col in df_test1.columns:
    rmse_1 = np.append(
        rmse_1, 
        mean_squared_error(yt1[:7*24], df_test1[col], squared=False)
        )
    rmse_2 = np.append(
        rmse_2, 
        mean_squared_error(yt2[:7*24], df_test2[col], squared=False)
        )
    rmse_3 = np.append(
        rmse_3, 
        mean_squared_error(yt3[:7*24], df_test3[col], squared=False)
        )
    rmse_4 = np.append(
        rmse_4, 
        mean_squared_error(yt4[:7*24], df_test4[col], squared=False)
        )

df_rmse_1 = pd.DataFrame(
    data=rmse_1.T,
    index=df_test1.columns,
    columns=['Test 1']
    ).T.sort_index(axis=1)
df_rmse_2 = pd.DataFrame(
    data=rmse_2.T,
    index=df_test1.columns,
    columns=['Test 2']
    ).T.sort_index(axis=1)
df_rmse_3 = pd.DataFrame(
    data=rmse_3.T,
    index=df_test1.columns,
    columns=['Test 3']
    ).T.sort_index(axis=1)
df_rmse_4 = pd.DataFrame(
    data=rmse_4.T,
    index=df_test1.columns,
    columns=['Test 4']
    ).T.sort_index(axis=1)

#%% Plot Metrics
import numpy.matlib
cores = np.matlib.repmat(['Data-driven', 'Knowledge-Augmented', 'Parallel Cooperative Hybrid', 'Serial Hybrid',], 3, 1).T.reshape(-1,)
cores = np.append(cores, 'White-box')
models = np.matlib.repmat(['LASSO', 'MLP', 'PLS',], 4, 1).reshape(-1,)
models = np.append(models, 'WB')

fig, ax = plt.subplots(nrows=2, ncols=2)
sns.barplot(x=models, y=df_rmse_1.values.reshape(-1,), hue=cores, ax=ax[0,0])
ax[0,0].set_xticklabels(ax[0,0].get_xticklabels(), fontsize=14)
ax[0,0].set_ylabel('RMSE', fontsize=18)
ax[0,0].set_title('Test 1', fontsize=18)

sns.barplot(x=models, y=df_rmse_2.values.reshape(-1,), hue=cores, ax=ax[0,1])
ax[0,1].set_xticklabels(ax[0,1].get_xticklabels(), fontsize=14)
ax[0,1].set_ylabel('RMSE', fontsize=18)
ax[0,1].set_title('Test 2', fontsize=18)

sns.barplot(x=models, y=df_rmse_3.values.reshape(-1,), hue=cores, ax=ax[1,0])
ax[1,0].set_xticklabels(ax[1,0].get_xticklabels(), fontsize=14)
ax[1,0].set_ylabel('RMSE', fontsize=18)
ax[1,0].set_title('Test 3', fontsize=18)

sns.barplot(x=models, y=df_rmse_4.values.reshape(-1,), hue=cores, ax=ax[1,1])
ax[1,1].set_xticklabels(ax[1,1].get_xticklabels(), fontsize=14)
ax[1,1].set_ylabel('RMSE', fontsize=18)
ax[1,1].set_title('Test 4', fontsize=18)
plt.savefig('figures/HM_RMSE.png')

#%% Transfer Learning with KAH
X_train = np.concatenate([Xs.values, svR], axis=1)

KAH = ML()
KAH.train(X_train, ys, method='lasso')

# LASSO Benchmark
L1 = ML()
L1.train(Xt1[:7*24], yt1[:7*24], method='lasso')
L2 = ML()
L2.train(Xt2[:7*24], yt2[:7*24], method='lasso')
L3 = ML()
L3.train(Xt3[:7*24], yt3[:7*24], method='lasso')
L4 = ML()
L4.train(Xt4[:7*24], yt4[:7*24], method='lasso')

# KAH TL
X_test1 = np.concatenate([Xt1[:7*24].values, svR_test1], axis=1)
X_test2 = np.concatenate([Xt2[:7*24].values, svR_test2], axis=1)
X_test3 = np.concatenate([Xt3[:7*24].values, svR_test3], axis=1)
X_test4 = np.concatenate([Xt4[:7*24].values, svR_test4], axis=1)

X_train_TL1 = np.concatenate([X_test1, df_test1['KAH_LASSO'].values.reshape(-1, 1)], axis=1)
X_train_TL2 = np.concatenate([X_test2, df_test2['KAH_LASSO'].values.reshape(-1, 1)], axis=1)
X_train_TL3 = np.concatenate([X_test3, df_test3['KAH_LASSO'].values.reshape(-1, 1)], axis=1)
X_train_TL4 = np.concatenate([X_test4, df_test4['KAH_LASSO'].values.reshape(-1, 1)], axis=1)

TL_KAH1 = ML()
TL_KAH1.train(X_train_TL1, yt1[:7*24], method='lasso')
TL_KAH2 = ML()
TL_KAH2.train(X_train_TL2, yt2[:7*24], method='lasso')
TL_KAH3 = ML()
TL_KAH3.train(X_train_TL3, yt3[:7*24], method='lasso')
TL_KAH4 = ML()
TL_KAH4.train(X_train_TL4, yt4[:7*24], method='lasso')

svR_1 = wb_model.predict(tt1[7*24:], par, ut1[:, 7*24:])
svR_2 = wb_model.predict(tt2[7*24:], par, ut2[:, 7*24:])
svR_3 = wb_model.predict(tt3[7*24:], par, ut3[:, 7*24:])
svR_4 = wb_model.predict(tt4[7*24:], par, ut4[:, 7*24:])

X_tl1 = np.concatenate([Xt1[7*24:].values, svR_1], axis=1)
X_tl2 = np.concatenate([Xt2[7*24:].values, svR_2], axis=1)
X_tl3 = np.concatenate([Xt3[7*24:].values, svR_3], axis=1)
X_tl4 = np.concatenate([Xt4[7*24:].values, svR_4], axis=1)

df_tl1 = pd.DataFrame(data=KAH.predict(X_tl1), columns=['KAH_LASSO'])
df_tl2 = pd.DataFrame(data=KAH.predict(X_tl2), columns=['KAH_LASSO'])
df_tl3 = pd.DataFrame(data=KAH.predict(X_tl3), columns=['KAH_LASSO'])
df_tl4 = pd.DataFrame(data=KAH.predict(X_tl4), columns=['KAH_LASSO'])

X_test_TL1 = np.concatenate([X_tl1, df_tl1['KAH_LASSO'].values.reshape(-1, 1)], axis=1)
X_test_TL2 = np.concatenate([X_tl2, df_tl2['KAH_LASSO'].values.reshape(-1, 1)], axis=1)
X_test_TL3 = np.concatenate([X_tl3, df_tl3['KAH_LASSO'].values.reshape(-1, 1)], axis=1)
X_test_TL4 = np.concatenate([X_tl4, df_tl4['KAH_LASSO'].values.reshape(-1, 1)], axis=1)

df_tl1['TL-KAH'] = TL_KAH1.predict(X_test_TL1)
df_tl2['TL-KAH'] = TL_KAH2.predict(X_test_TL2)
df_tl3['TL-KAH'] = TL_KAH3.predict(X_test_TL3)
df_tl4['TL-KAH'] = TL_KAH4.predict(X_test_TL4)

df_tl1['LASSO'] = L1.predict(Xt1[7*24:])
df_tl2['LASSO'] = L2.predict(Xt2[7*24:])
df_tl3['LASSO'] = L3.predict(Xt3[7*24:])
df_tl4['LASSO'] = L4.predict(Xt4[7*24:])

#%% Transfer Learning with PH
TL_PH1 = ML()
TL_PH1.train(Xt1[:7*24], yt1[:7*24] - df_test1['KAH_LASSO'].values, method='lasso')
TL_PH2 = ML()
TL_PH2.train(Xt2[:7*24], yt2[:7*24] - df_test2['KAH_LASSO'].values, method='lasso')
TL_PH3 = ML()
TL_PH3.train(Xt3[:7*24], yt3[:7*24] - df_test3['KAH_LASSO'].values, method='lasso')
TL_PH4 = ML()
TL_PH4.train(Xt4[:7*24], yt4[:7*24] - df_test4['KAH_LASSO'].values, method='lasso')

df_tl1['TL-PH'] = df_tl1['KAH_LASSO'].values + TL_PH1.predict(Xt1[7*24:]).reshape(-1)
df_tl2['TL-PH'] = df_tl2['KAH_LASSO'].values + TL_PH2.predict(Xt2[7*24:]).reshape(-1)
df_tl3['TL-PH'] = df_tl3['KAH_LASSO'].values + TL_PH3.predict(Xt3[7*24:]).reshape(-1)
df_tl4['TL-PH'] = df_tl4['KAH_LASSO'].values + TL_PH4.predict(Xt4[7*24:]).reshape(-1)

#%% Results
fig, ax = plt.subplots(nrows=2, ncols=2)
for col in df_tl1.columns:
    rmse = mean_squared_error(yt1[7*24:], df_tl1[col], squared=False)
    ax[0,0].bar(col, rmse)
    ax[0,0].set_title('Test 1', fontsize=18)
for col in df_tl2.columns:
    rmse = mean_squared_error(yt2[7*24:], df_tl2[col], squared=False)
    ax[0,1].bar(col, rmse)
    ax[0,1].set_title('Test 2', fontsize=18)
for col in df_tl3.columns:
    rmse = mean_squared_error(yt3[7*24:], df_tl3[col], squared=False)
    ax[1,0].bar(col, rmse)
    ax[1,0].set_title('Test 3', fontsize=18)
for col in df_tl4.columns:
    rmse = mean_squared_error(yt4[7*24:], df_tl4[col], squared=False)
    ax[1,1].bar(col, rmse)
    ax[1,1].set_title('Test 4', fontsize=18)
plt.savefig('figures/TL_RMSE.png')

fig, ax = plt.subplots(nrows=2, ncols=2)
for col in df_tl1.columns:
    mape = mean_absolute_percentage_error(yt1[7*24:], df_tl1[col])*100
    ax[0,0].bar(col, mape)
    ax[0,0].set_title('MAPE - Test 1', fontsize=18)
for col in df_tl2.columns:
    mape = mean_absolute_percentage_error(yt2[7*24:], df_tl2[col])*100
    ax[0,1].bar(col, mape)
    ax[0,1].set_title('MAPE - Test 2', fontsize=18)
for col in df_tl3.columns:
    mape = mean_absolute_percentage_error(yt3[7*24:], df_tl3[col])*100
    ax[1,0].bar(col, mape)
    ax[1,0].set_title('MAPE - Test 3', fontsize=18)
for col in df_tl4.columns:
    mape = mean_absolute_percentage_error(yt4[7*24:], df_tl4[col])*100
    ax[1,1].bar(col, mape)
    ax[1,1].set_title('MAPE - Test 4', fontsize=18)
plt.savefig('figures/TL_MAPE.png')

rmse11 = mean_squared_error(yt1[7*24:], df_tl1['LASSO'], squared=False)
rmse12 = mean_squared_error(yt1[7*24:], df_tl1['KAH_LASSO'], squared=False)
rmse13 = mean_squared_error(yt1[7*24:], df_tl1['TL-KAH'], squared=False)
rmse41 = mean_squared_error(yt4[7*24:], df_tl4['LASSO'], squared=False)
rmse42 = mean_squared_error(yt4[7*24:], df_tl4['KAH_LASSO'], squared=False)
rmse43 = mean_squared_error(yt4[7*24:], df_tl4['TL-KAH'], squared=False)

fig, ax = plt.subplots()
ax.bar('Benchmark 1', rmse12, color='tab:orange', label='Knowledge-augmented')
ax.bar('Benchmark 2', rmse11, color='tab:blue', label='LASSO')
ax.bar('Transfer learning', rmse13, color='tab:olive', label='Knowledge-augmented + TL')
ax.set_ylim(bottom=0, top=0.011)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('Test 1', fontsize=22)
plt.legend(loc='best', fontsize=20)
plt.savefig('figures/TL_1_RMSE.png')

fig, ax = plt.subplots()
ax.bar('Benchmark 1', rmse42, color='tab:orange', label='Knowledge-augmented')
ax.bar('Benchmark 2', rmse41, color='tab:blue', label='LASSO')
ax.bar('Transfer learning', rmse43, color='tab:olive', label='Knowledge-augmented + TL')
ax.set_ylim(bottom=0, top=0.011)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.set_title('Test 4', fontsize=22)
plt.legend(loc='best', fontsize=20)
plt.savefig('figures/TL_4_RMSE.png')

fig, ax = plt.subplots(nrows=4, ncols=1)
ax[0].plot(tt1/(3600*24), yt1, 'k', label='Actual')
for col in df_tl1.columns:
    ax[0].plot(tt1[7*24:]/(3600*24), df_tl1[col], label=col)
ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[1].plot(tt2/(3600*24), yt2, 'k', label='Actual')
for col in df_tl1.columns:
    ax[1].plot(tt2[7*24:]/(3600*24), df_tl2[col], label=col)
ax[1].tick_params(axis='both', which='major', labelsize=16)
ax[2].plot(tt3/(3600*24), yt3, 'k', label='Actual')
for col in df_tl1.columns:
    ax[2].plot(tt3[7*24:]/(3600*24), df_tl3[col], label=col)
ax[2].tick_params(axis='both', which='major', labelsize=16)
ax[3].plot(tt4/(3600*24), yt4, 'k', label='Actual')
for col in df_tl1.columns:
    ax[3].plot(tt4[7*24:]/(3600*24), df_tl4[col], label=col)
ax[3].tick_params(axis='both', which='major', labelsize=16)
plt.legend(loc='best')
ax[3].set_xlabel('Time / days', fontsize=18)
ax[0].set_ylabel('KPI / -', fontsize=18)
ax[1].set_ylabel('KPI / -', fontsize=18)
ax[2].set_ylabel('KPI / -', fontsize=18)
ax[3].set_ylabel('KPI / -', fontsize=18)
plt.savefig('figures/TL_profiles.png')

end = time.time()
print('Run time: {:.0f} seconds'.format(end-start))
plt.show()
