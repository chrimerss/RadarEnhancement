# Radar Forecast Model – Deep Learning

## Long Short-term Memory
LSTM is good at processing time series data.

The model structure can be decomposed by two networks, one is encoder, and the other is predictor.


Model configurations:
Model Name|hidden layers|# training samples|tsize|# epochs|training process|Date
--------|--------|----------|----------|--------------|--------|------|----
newest-5_8| 16 | 400| 10| 500| Fig.1|2019.5.8

<p align="center">
<img src="images/training-5_8.PNG" style="width:60%"><br>
Fig.1 Training loss on 5.8
<img src="images/201811230600-predict.gif" style="width: 80%"><br>
Fig.2 Comparison of predction and observation
</p>

### Wind Effect

## GRU

## Optical Flow

## Kalman Filter 


To be continued...
