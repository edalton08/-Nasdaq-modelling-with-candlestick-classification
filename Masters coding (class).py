import pandas as pd
import math as math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import KernelPCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

###

#  *** 'G4G' indicates code was inspired by GeeksforGeeks, link in report ***
#  *** 'KT' indicates code was inspired by Keras Tutorial website, link in report ***
#  *** Any other code was own code ***
###

###Constructing the dataset

df_1 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '71-'76.csv")
df_2 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '76-'81.csv")
df_3 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '81-'86.csv")
df_4 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '86-'91.csv")
df_5 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '91-'96.csv")
df_6 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '96-'01.csv")
df_7 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '01-'06.csv")
df_8 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '06-'11.csv")
df_9 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '11-'16.csv")
df_10 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '16-'21.csv")
df_11 = pd.read_csv("/Users/edward/Documents/Documents – Edward’s MacBook Pro/Year 1/Year 2/Masters Project /Nasdaq comp csv data/Nasdaq Comp Index Data '21-'25.csv")

df = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10, df_11], ignore_index=True)

df.shape

###Designing the classification, calculating relevant variables

def candlestick_classification(df, body_pct_bins=None, location_pct_bins=None):
    classifications = {
        'Date':[],
        'Body_pct':[],
        'Location_pct':[],
        'Body_label':[],
        'Location_label':[],
        'Colour':[],
        'Open':[],
        'High':[],
        'Low':[],
        'Close':[],
    }

    for index, row in df.iterrows():
        date = row['Date']
        open_values = row['Open']
        high_values = row['High']
        low_values = row['Low']
        close_values = row['Close']

        
        if high_values == low_values:
            continue
        

        if close_values > open_values:
            color = 'green'
        elif close_values < open_values:
            color = 'red'
        else:
            color = 'doji'

        body_pct = ((max(open_values, close_values) - min(open_values, close_values)) / (high_values - low_values)) * 100

        mid_point = (open_values + close_values) / 2
        upper_shadow = high_values - max(open_values, close_values)
        lower_shadow = min(open_values, close_values) - low_values
        location_pct = ((mid_point - low_values) / (high_values - low_values)) * 100

        body_label = int(round(body_pct / 10.0) * 10)

        if 0 <= location_pct <10:
            location_label = "0-10%"
        
        elif 10 <= location_pct < 20:
            location_label = "10-20%"

        elif 20 <= location_pct < 30:
            location_label = "20-30%"

        elif 30 <= location_pct < 40:
            location_label = "30-40%"

        elif 40 <= location_pct < 50:
            location_label = "40-50%"

        elif 50 <= location_pct < 60:
            location_label = "50-60%"
        
        elif 60 <= location_pct < 70:
            location_label = "60-70%"

        elif 70 <= location_pct < 80:
            location_label = "70-80%"

        elif 80 <= location_pct < 90:
            location_label = "80-90%"

        elif 90 <= location_pct < 100:
            location_label = "90-100%"
        
        else: location_label = "Outside of candle"

        classifications['Date'].append(date)
        classifications['Body_pct'].append(body_pct)
        classifications['Location_pct'].append(location_pct)
        classifications['Body_label'].append(body_label)
        classifications['Location_label'].append(location_label)
        classifications['Colour'].append(color)
        classifications['Open'].append(open_values)
        classifications['High'].append(high_values)
        classifications['Low'].append(low_values)
        classifications['Close'].append(close_values)
    
    classifications_df = pd.DataFrame(classifications)

    return classifications_df

classification_dataframe = candlestick_classification(df)

print(classification_dataframe.head())

classification_dataframe.shape

###EDA

classification_dataframe.describe()
print(classification_dataframe.dtypes)

classification_dataframe['Date'] = pd.to_datetime(classification_dataframe['Date'], format= '%d/%m/%Y')

print(classification_dataframe.isnull().sum())

print(sns.boxplot(y = classification_dataframe['Body_pct'], color= 'red'))
plt.show()


print(sns.boxplot(y = classification_dataframe['Location_pct'], color= 'coral'))
plt.show()

print(sns.histplot(data=classification_dataframe, x='Location_label', color='turquoise'))
plt.show()

target = -2468.75
tol = 0.0001

outliers = classification_dataframe.loc[(classification_dataframe['Location_pct'] >= target - tol) & (classification_dataframe['Location_pct'] <= target + tol)]

print(outliers)

# Outliers were rows with the body being outside the candle. In this case this was any value with the Location label, 'Outside of candle'.

print(classification_dataframe.loc[(classification_dataframe['Location_label'] == 'Outside of candle')])

valid_candles = classification_dataframe[classification_dataframe['Location_label'] != 'Outside of candle'].copy()

valid_candles.shape

#Boxplot for Body size percentage
print(sns.boxplot(y = valid_candles['Body_pct'], color= 'red'))
plt.title('Body Pct Boxplot')
plt.xlabel('Body Pct')
plt.show()

#Boxplot for Location percentage
print(sns.boxplot(y = valid_candles['Location_pct'], color= 'coral'))
plt.title('Location Pct Boxplot')
plt.xlabel('Location Pct')
plt.show()

print(valid_candles['Location_pct'].describe())
print(valid_candles['Location_pct'].head(10))
print(valid_candles['Location_pct'].dtype)

sns.countplot(data=valid_candles, x='Location_label', color='turquoise')
plt.show()

#Count plot to analyse the distribution of Location percentages
ordered_labels = ['0-10', '10-20', '20-30', '30-40', '40-50', 
                  '50-60', '60-70', '70-80', '80-90', '90-100']
valid_candles['Location_label'] = pd.Categorical(
    valid_candles['Location_label'],
    categories=ordered_labels,
    ordered=True
)

sns.countplot(data=valid_candles, x='Location_label', color='turquoise')
plt.title('Location Pct Labels Distribution')
plt.xlabel('Location Pct Labels')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

##Feature Correlation
corr = valid_candles[['Body_pct', 'Location_pct', 'Open', 'High', 'Low', 'Close']].corr()
corr

sns.heatmap(corr, cmap='RdBu', vmin = -1, vmax = 1,annot=True, annot_kws={'fontsize':7})
plt.show()

## Autocorrelation

pd.plotting.lag_plot(valid_candles['Body_pct'], lag=1)
plt.title('Autocorrelation for Body Pct')
plt.show()

pd.plotting.lag_plot(valid_candles['Location_pct'], lag=1)
plt.title('Autocorrelation for Location Pct')
plt.show()

##Markov Matrix

#Making red and green candlesticks into integers
coloured_candles = classification_dataframe['Colour'].map({'green': 1, 'red': 0})

coloured_candles.shape

coloured_candles = coloured_candles.dropna().astype(int).tolist()

#Creation of the matrix
matrix = [[0,0], [0,0]]

#Looping throught the dataset, pairing the candlestick colours tonwhich colour follows the previosu one.
for i, j in zip(coloured_candles, coloured_candles[1:]):
    matrix[i][j] += 1

#Calulate total number of  state (colour) changes for each current state. Divide by total number of observatiopns for each state to give probabilities.
for row in matrix:
    s = sum(row)
    if s > 0:
        for k in range(len(row)):
            row[k] = row[k]/s

#Produce table
for row in matrix:
    print(row)

### Strategy Feature engineering
    
#Mean Reversion
valid_candles['rolling_mean'] = valid_candles['Close'].rolling(window= 7).mean()
valid_candles['rolling_std'] = valid_candles['Close'].rolling(window=7).std()
valid_candles['mean_reversion'] = (valid_candles['Close'] - valid_candles['rolling_mean']) / valid_candles['rolling_std']

#Breakout

valid_candles['recent_high'] = valid_candles['Close'].rolling(window= 7).max()
valid_candles['breakout'] = np.where(valid_candles['Close'] > valid_candles['recent_high'].shift(1), 1, 0)

#Momentum

valid_candles['momentum'] = valid_candles['Close'].pct_change(periods=3)

#Gap

valid_candles['gap'] = valid_candles['Open'] - valid_candles['Close'].shift(1)


valid_candles.drop(['rolling_mean', 'rolling_std', 'recent_high'], axis= 1, inplace=True)




### Model Preprocessing
# Making the 'Colour' data ternary data so it can be inputed in to the models and be ivovled with the scaling
print(valid_candles['Colour'].unique())
print(valid_candles.columns)
valid_candles['Colour'] = valid_candles['Colour'].map({'green': 2, 'red':1, 'doji': 0})
model_data = valid_candles
model_data = model_data.dropna(subset= ['mean_reversion', 'breakout', 'momentum', 'gap'])

location_label_values = {
    '0-10%' : 0,
    '10-20%' : 1,
    '20-30%' : 2,
    '30-40%' : 3,
    '40-50%' : 4,
    '50-60%' : 5,
    '60-70%' : 6,
    '70-80%' : 7,
    '80-90%' : 8,
    '90-100%' : 9 
}

model_data['Location_label'] = model_data['Location_label'].map(location_label_values)


##Scaling

model_data.drop(columns= ['Date'], inplace=True)
model_data =model_data.reset_index(drop=True)

ns_features = model_data[['Body_label', 'Location_label']]
mean_rev_features = model_data[['Body_label', 'Location_label', 'mean_reversion']]
breakout_features = model_data[['Body_label', 'Location_label', 'Colour', 'breakout']]
momentum_features = model_data[['Body_label', 'Location_label', 'Colour', 'momentum']]
gap_features = model_data[['Body_label', 'Location_label', 'Colour', 'gap']]

ns_model_data = ns_features
mr_model_data = mean_rev_features
br_model_data = breakout_features
mom_model_data = momentum_features
gap_model_data = gap_features


mr_scaler = MinMaxScaler()
mr_model_data_scaled = pd.DataFrame(mr_scaler.fit_transform(mr_model_data), columns= mr_model_data.columns)
mr_model_data_scaled.min()


br_scaler = MinMaxScaler()
br_model_data_scaled = pd.DataFrame(br_scaler.fit_transform(br_model_data), columns= br_model_data.columns)



mom_scaler = MinMaxScaler()
mom_model_data_scaled = pd.DataFrame(mom_scaler.fit_transform(mom_model_data), columns= mom_model_data.columns)


gap_scaler = MinMaxScaler()
gap_model_data_scaled = pd.DataFrame(gap_scaler.fit_transform(gap_model_data), columns= gap_model_data.columns)



ns_scaler= MinMaxScaler()
ns_model_data_scaled = pd.DataFrame(ns_scaler.fit_transform(ns_model_data), columns=ns_model_data.columns)




##Sampling
# G4G
def produce_samples(model_data, history_size, future_size):

    x,y = [], []

    for i in range(len(model_data) - history_size - future_size +1):
        x.append(model_data[i : i + history_size, :])
        y.append(model_data[i + history_size : i + history_size + future_size, :])
    return np.array(x), np.array(y)

##Splitting
#G4G
split = 0.85

(x_ns_Val, y_ns_Val) = produce_samples(ns_model_data_scaled.values, 8000, 30)
ns_x_train = x_ns_Val[:int(split * len(x_ns_Val))]
ns_y_train = y_ns_Val[:int(split * len(y_ns_Val))]
ns_x_test = x_ns_Val[int(split * len(x_ns_Val)):]
ns_y_test = y_ns_Val[int(split * len(y_ns_Val)):]


(x_mr_Val, y_mr_Val) = produce_samples(mr_model_data_scaled.values, 8000, 30)
mr_x_train = x_mr_Val[:int(split * len(x_mr_Val))]
mr_y_train = y_mr_Val[:int(split * len(y_mr_Val))]
mr_x_test = x_mr_Val[int(split * len(x_mr_Val)):]
mr_y_test = y_mr_Val[int(split * len(y_mr_Val)):]



(x_br_Val, y_br_Val) = produce_samples(br_model_data_scaled.values, 8000, 30)
br_x_train = x_br_Val[:int(split * len(x_br_Val))]
br_y_train = y_br_Val[:int(split * len(y_br_Val))]
br_x_test = x_br_Val[int(split * len(x_br_Val)):]
br_y_test = y_br_Val[int(split * len(y_br_Val)):]


(x_mom_Val, y_mom_Val) = produce_samples(mom_model_data_scaled.values, 8000, 30)
mom_x_train = x_mom_Val[:int(split * len(x_mom_Val))]
mom_y_train = y_mom_Val[:int(split * len(y_mom_Val))]
mom_x_test = x_mom_Val[int(split * len(x_mom_Val)):]
mom_y_test = y_mom_Val[int(split * len(y_mom_Val)):]

(x_gap_Val, y_gap_Val) = produce_samples(gap_model_data_scaled.values, 8000, 30)
gap_x_train = x_gap_Val[:int(split * len(x_gap_Val))]
gap_y_train = y_gap_Val[:int(split * len(y_gap_Val))]
gap_x_test = x_gap_Val[int(split * len(x_gap_Val)):]
gap_y_test = y_gap_Val[int(split * len(y_gap_Val)):]



###Model training
# G4G
##Control model- NS

ns_multivariate_lstm = keras.Sequential()
ns_multivariate_lstm.add(keras.layers.LSTM(200, input_shape=(ns_x_train.shape[1], ns_x_train.shape[2])))
ns_multivariate_lstm.add(keras.layers.Dropout(0.2))
ns_multivariate_lstm.add(keras.layers.Dense(30 * 2, activation='sigmoid'))
ns_multivariate_lstm.add(keras.layers.Reshape((30, 2)))
ns_multivariate_lstm.compile(loss='mean_squared_error', metrics=['MAE'], optimizer='Adam')
ns_multivariate_lstm.summary()



##Mean reversion model

mr_multivariate_lstm = keras.Sequential()
mr_multivariate_lstm.add(keras.layers.LSTM(200, input_shape=(mr_x_train.shape[1], mr_x_train.shape[2])))
mr_multivariate_lstm.add(keras.layers.Dropout(0.2))
mr_multivariate_lstm.add(keras.layers.Dense(30 * 3, activation='sigmoid'))
mr_multivariate_lstm.add(keras.layers.Reshape((30, 3)))
mr_multivariate_lstm.compile(loss= 'mean_squared_error', metrics=['MAE'], optimizer='Adam')
mr_multivariate_lstm.summary()



##Breakout model

br_multivariate_lstm = keras.Sequential()
br_multivariate_lstm.add(keras.layers.LSTM(200, input_shape=(br_x_train.shape[1], br_x_train.shape[2])))
br_multivariate_lstm.add(keras.layers.Dropout(0.2))
br_multivariate_lstm.add(keras.layers.Dense(30 * 4, activation='sigmoid'))
br_multivariate_lstm.add(keras.layers.Reshape((30, 4)))
br_multivariate_lstm.compile(loss= 'mean_squared_error', metrics=['MAE'], optimizer='Adam')
br_multivariate_lstm.summary()



##Momentum model

mom_multivariate_lstm = keras.Sequential()
mom_multivariate_lstm.add(keras.layers.LSTM(200, input_shape=(mom_x_train.shape[1], mom_x_train.shape[2])))
mom_multivariate_lstm.add(keras.layers.Dropout(0.2))
mom_multivariate_lstm.add(keras.layers.Dense(30 * 4, activation='sigmoid'))
mom_multivariate_lstm.add(keras.layers.Reshape((30, 4)))
mom_multivariate_lstm.compile(loss= 'mean_squared_error', metrics=['MAE'], optimizer='Adam')
mom_multivariate_lstm.summary()



##Gap model

gap_multivariate_lstm = keras.Sequential()
gap_multivariate_lstm.add(keras.layers.LSTM(200, input_shape=(gap_x_train.shape[1], gap_x_train.shape[2])))
gap_multivariate_lstm.add(keras.layers.Dropout(0.2))
gap_multivariate_lstm.add(keras.layers.Dense(30 * 4, activation='sigmoid'))
gap_multivariate_lstm.add(keras.layers.Reshape((30, 4)))
gap_multivariate_lstm.compile(loss= 'mean_squared_error', metrics=['MAE'], optimizer='Adam')
gap_multivariate_lstm.summary()



###Model fitting and early stops

##No Strategy - Control
# KT
ns_early_stop = EarlyStopping(monitor= 'val_loss', patience= 25, min_delta= 0.0001, restore_best_weights=True)

#G4G
ns_model_history = ns_multivariate_lstm.fit(ns_x_train, ns_y_train, epochs=130, batch_size=20, validation_split=0.2, callbacks=[ns_early_stop])


##mean reversion model
#KT
early_stop = EarlyStopping(monitor= 'val_loss', patience= 25, min_delta=0.0001, restore_best_weights=True)

#G4G
mr_model_history = mr_multivariate_lstm.fit(mr_x_train, mr_y_train, epochs=130, batch_size=20, validation_split=0.2, callbacks=[early_stop])


##breakout model training
#KT
br_early_stop = EarlyStopping(monitor= 'val_loss', patience= 25, min_delta=0.0001, restore_best_weights=True)

#G4G
br_model_history = br_multivariate_lstm.fit(br_x_train, br_y_train, epochs=130, batch_size=20, validation_split=0.2, callbacks=[br_early_stop])


##Momentum model training
#KT
mom_early_stop = EarlyStopping(monitor= 'val_loss', patience= 25, min_delta= 0.0001, restore_best_weights=True)

#G4G
mom_model_history = mom_multivariate_lstm.fit(mom_x_train, mom_y_train, epochs=130, batch_size=20, validation_split=0.2, callbacks=[mom_early_stop])


##Gap model training
#KT
gap_early_stop = EarlyStopping(monitor= 'val_loss', patience= 25, min_delta=0.0001, restore_best_weights=True)

#G4G
gap_model_history = gap_multivariate_lstm.fit(gap_x_train, gap_y_train, epochs=130, batch_size=20, validation_split=0.2, callbacks=[gap_early_stop])




###Model evaluation

#G4G
##No Strategy

ns_eval = ns_multivariate_lstm.evaluate(ns_x_test, ns_y_test, batch_size=64)
print(ns_eval)


##Mean reversion model

mr_eval = mr_multivariate_lstm.evaluate(mr_x_test, mr_y_test, batch_size=64)
print(mr_eval)


##Brerakout model

br_eval = br_multivariate_lstm.evaluate(br_x_test, br_y_test, batch_size= 64)
print(br_eval)


##Momentum model

mom_eval = mom_multivariate_lstm.evaluate(mom_x_test, mom_y_test, batch_size= 64)
print(mom_eval)


##Gap model

gap_eval = gap_multivariate_lstm.evaluate(gap_x_test, gap_y_test, batch_size= 64)
print(gap_eval)




###Model Predictions

##NS model (Control)

ns_preds = ns_multivariate_lstm.predict(ns_x_test[:1])

scaled_ns_preds_2d = ns_preds.reshape(-1, 2)

inversed_ns_preds = ns_scaler.inverse_transform(scaled_ns_preds_2d)


ns_y_test_outputs = ns_y_test[:1]

ns_y_test_outputs = ns_y_test_outputs.reshape(-1, 2)
ns_y_test_inverse = ns_scaler.inverse_transform(ns_y_test_outputs)

ns_results_df = {

    "Predicted Body Label": inversed_ns_preds[:, 0],
    "Predicted Location Label": inversed_ns_preds[:, 1],
    "Actual Body Label": ns_y_test_inverse[:, 0],
    "Actual Location Label": ns_y_test_inverse[:, 1],
}

ns_results_df = pd.DataFrame(ns_results_df)

ns_results_df['Date'] = valid_candles['Date'].iloc[-len(ns_results_df):].reset_index(drop=True)

ns_results_df.head(30)

ns_r2 = r2_score(ns_y_test_inverse, inversed_ns_preds)
print(f"R² Score: {ns_r2:.4f}")





##Mean reversion 

mean_rev_preds = mr_multivariate_lstm.predict(mr_x_test[:1])

scaled_mr_preds_2d = mean_rev_preds.reshape(-1, 3)

inversed_mr_preds = mr_scaler.inverse_transform(scaled_mr_preds_2d)

mr_y_test_outputs = mr_y_test[:1]

mr_y_test_outputs = mr_y_test_outputs.reshape(-1, 3)
mr_y_test_inverse = mr_scaler.inverse_transform(mr_y_test_outputs)

mr_results_df = {

    "Predicted Body Label": inversed_mr_preds[:, 0],
    "Predicted Location Label": inversed_mr_preds[:, 1],
    "Actual Body Label": mr_y_test_inverse[:, 0],
    "Actual Location Label": mr_y_test_inverse[:, 1],
}

mr_results_df = pd.DataFrame(mr_results_df)

mr_results_df['Date'] = valid_candles['Date'].iloc[-len(mr_results_df):].reset_index(drop=True)

mr_results_df.head(30)

plt.hist(mr_results_df['Predicted Body Label'], bins=10, edgecolor='black')
plt.show()

mr_r2 = r2_score(mr_y_test_inverse, inversed_mr_preds)
print(f"R² Score: {mr_r2:.4f}")


## Breakout model
br_preds = br_multivariate_lstm.predict(br_x_test[:1])

scaled_br_preds_2d = br_preds.reshape(-1, 4)

inversed_br_preds = br_scaler.inverse_transform(scaled_br_preds_2d)

br_y_test_outputs = br_y_test[:1]

br_y_test_outputs = br_y_test_outputs.reshape(-1, 4)
br_y_test_inverse = br_scaler.inverse_transform(br_y_test_outputs)

br_results_df = {

    "Predicted Body Label": inversed_br_preds[:, 0],
    "Predicted Location Label": inversed_br_preds[:, 1],
    "Actual Body Label": br_y_test_inverse[:, 0],
    "Actual Location Label": br_y_test_inverse[:, 1],
}

br_results_df = pd.DataFrame(br_results_df)

br_results_df['Date'] = valid_candles['Date'].iloc[-len(br_results_df):].reset_index(drop=True)

br_results_df.head(5)

br_r2 = r2_score(br_y_test_inverse, inversed_br_preds)
print(f"R² Score: {br_r2:.4f}")

## Momentum model
mom_preds = mom_multivariate_lstm.predict(mom_x_test[:1])

scaled_mom_preds_2d = mom_preds.reshape(-1, 4)

inversed_mom_preds = mom_scaler.inverse_transform(scaled_mom_preds_2d)


mom_y_test_outputs = mom_y_test[:1]

mom_y_test_outputs = mom_y_test_outputs.reshape(-1, 4)
mom_y_test_inverse = mom_scaler.inverse_transform(mom_y_test_outputs)

mom_results_df = {

    "Predicted Body Label": inversed_mom_preds[:, 0],
    "Predicted Location Label": inversed_mom_preds[:, 1],
    "Actual Body Label": mom_y_test_inverse[:, 0],
    "Actual Location Label": mom_y_test_inverse[:, 1],
}

mom_results_df = pd.DataFrame(mom_results_df)

mom_results_df['Date'] = valid_candles['Date'].iloc[-len(mom_results_df):].reset_index(drop=True)


mom_r2 = r2_score(mom_y_test_inverse, inversed_mom_preds)
print(f"R² Score: {mom_r2:.4f}")

## Gap model
gap_preds = gap_multivariate_lstm.predict(gap_x_test[:1])

scaled_gap_preds_2d = gap_preds.reshape(-1, 4)

inversed_gap_preds = gap_scaler.inverse_transform(scaled_gap_preds_2d)


gap_y_test_outputs = gap_y_test[:1]

gap_y_test_outputs = gap_y_test_outputs.reshape(-1, 4)
gap_y_test_inverse = gap_scaler.inverse_transform(gap_y_test_outputs)

gap_results_df = {

    "Predicted Body Label": inversed_gap_preds[:, 0],
    "Predicted Location Label": inversed_gap_preds[:, 1],
    "Actual Body Label": gap_y_test_inverse[:, 0],
    "Actual Location Label": gap_y_test_inverse[:, 1],
}

gap_results_df = pd.DataFrame(gap_results_df)

gap_results_df['Date'] = valid_candles['Date'].iloc[-len(gap_results_df):].reset_index(drop=True)

gap_r2 = r2_score(gap_y_test_inverse, inversed_gap_preds)
print(f"R² Score: {gap_r2:.4f}")

### PCT results



### Visualisations

#G4G

#Control model figure
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(ns_results_df.index, gap_results_df['Predicted Body Label'], label='Predicted Body Label', color= 'blue')
ax.plot(ns_results_df.index, gap_results_df['Actual Body Label'], label = "Actual Body Label", linestyle= '--', color = 'blue')

ax.plot(ns_results_df.index, gap_results_df['Predicted Location Label'], label='Predicted Location Label', color= 'green')
ax.plot(ns_results_df.index, gap_results_df['Actual Location Label'], label = "Actual Location Label", linestyle= '--', color = 'green')

ax.set_title('Predicted vs Actual results for No Strategy model with Label')
ax.set_xlabel('Date')
ax.set_ylabel('Label values')
ax.legend()
plt.tight_layout()
plt.show()


#Mean reversion figure

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(mr_results_df.index, gap_results_df['Predicted Body Label'], label='Predicted Body Label', color= 'blue')
ax.plot(mr_results_df.index, gap_results_df['Actual Body Label'], label = "Actual Body Label", linestyle= '--', color = 'blue')

ax.plot(mr_results_df.index, gap_results_df['Predicted Location Label'], label='Predicted Location Label', color= 'green')
ax.plot(mr_results_df.index, gap_results_df['Actual Location Label'], label = "Actual Location Label", linestyle= '--', color = 'green')

ax.set_title('Predicted vs Actual results for Mean Reversion strategy model with Label')
ax.set_xlabel('Date')
ax.set_ylabel('Label values')
ax.legend()
plt.tight_layout()
plt.show()

#Breakout model figure

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(br_results_df.index, gap_results_df['Predicted Body Label'], label='Predicted Body Label', color= 'blue')
ax.plot(br_results_df.index, gap_results_df['Actual Body Label'], label = "Actual Body Label", linestyle= '--', color = 'blue')

ax.plot(br_results_df.index, gap_results_df['Predicted Location Label'], label='Predicted Location Label', color= 'green')
ax.plot(br_results_df.index, gap_results_df['Actual Location Label'], label = "Actual Location Label", linestyle= '--', color = 'green')

ax.set_title('Predicted vs Actual results for Breakout strategy model with Label')
ax.set_xlabel('Date')
ax.set_ylabel('Label values')
ax.legend()
plt.tight_layout()
plt.show()

##Momentum model figures

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(mom_results_df.index, gap_results_df['Predicted Body Label'], label='Predicted Body Label', color= 'blue')
ax.plot(mom_results_df.index, gap_results_df['Actual Body Label'], label = "Actual Body Label", linestyle= '--', color = 'blue')

ax.plot(mom_results_df.index, gap_results_df['Predicted Location Label'], label='Predicted Location Label', color= 'green')
ax.plot(mom_results_df.index, gap_results_df['Actual Location Label'], label = "Actual Location Label", linestyle= '--', color = 'green')

ax.set_title('Predicted vs Actual results for Momentum strategy model with Label')
ax.set_xlabel('Date')
ax.set_ylabel('Label values')
ax.legend()
plt.tight_layout()
plt.show()



#Gap model figures

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(gap_results_df.index, gap_results_df['Predicted Body Label'], label='Predicted Body Label', color= 'blue')
ax.plot(gap_results_df.index, gap_results_df['Actual Body Label'], label = "Actual Body Label", linestyle= '--', color = 'blue')

ax.plot(gap_results_df.index, gap_results_df['Predicted Location Label'], label='Predicted Location Label', color= 'green')
ax.plot(gap_results_df.index, gap_results_df['Actual Location Label'], label = "Actual Location Label", linestyle= '--', color = 'green')

ax.set_title('Predicted vs Actual results for Gap strategy model with Label')
ax.set_xlabel('Date')
ax.set_ylabel('Label values')
ax.legend()
plt.tight_layout()
plt.show()





###Model analysis with MDA

def mda(df, actual, predictions):

    actual_direction = df[actual].diff()>0
    predcition_direction = df[predictions].diff()>0
    correct_direction = (actual_direction == predcition_direction)
    
    return correct_direction.mean()

## No Strategy (Control)

#Labels

ns_mda_body_label = mda(ns_results_df, 'Actual Body Label', 'Predicted Body Label')
ns_mda_location_label = mda(ns_results_df, 'Actual Location Label', 'Predicted Location Label')

print(f"Mean Directional Accuracy of Body Label: {ns_mda_body_label * 100:.3f}%")
print(f"Mean Directional Accuracy of Location Label): {ns_mda_location_label * 100:.3f}%")


##Mean Reversion

#Labels

mr_mda_body_label = mda(mr_results_df, 'Actual Body Label', 'Predicted Body Label')
mr_mda_location_label = mda(mr_results_df, 'Actual Location Label', 'Predicted Location Label')

print(f"Mean Directional Accuracy of Body Label: {mr_mda_body_label * 100:.3f}%")
print(f"Mean Directional Accuracy of Location Label): {mr_mda_location_label * 100:.3f}%")

#Pct



##Breakout

#Labels

br_mda_body_label = mda(br_results_df, 'Actual Body Label', 'Predicted Body Label')
br_mda_location_label = mda(br_results_df, 'Actual Location Label', 'Predicted Location Label')

print(f"Mean Directional Accuracy of Body Label: {br_mda_body_label * 100:.3f}%")
print(f"Mean Directional Accuracy of Location Label): {br_mda_location_label * 100:.3f}%")



##Momentum

#Labels

mom_mda_body_label = mda(mom_results_df, 'Actual Body Label', 'Predicted Body Label')
mom_mda_location_label = mda(mom_results_df, 'Actual Location Label', 'Predicted Location Label')

print(f"Mean Directional Accuracy of Body Label: {mom_mda_body_label * 100:.3f}%")
print(f"Mean Directional Accuracy of Location Label): {mom_mda_location_label * 100:.3f}%")



##Gap

gap_mda_body_label = mda(gap_results_df, 'Actual Body Label', 'Predicted Body Label')
gap_mda_location_label = mda(gap_results_df, 'Actual Location Label', 'Predicted Location Label')

print(f"Mean Directional Accuracy of Body Label: {gap_mda_body_label * 100:.3f}%")
print(f"Mean Directional Accuracy of Location Label): {gap_mda_location_label * 100:.3f}%")





