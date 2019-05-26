# Radar Forecast Model – Deep Learning

## Long Short-term Memory
LSTM is good at processing time series data.

The model structure can be decomposed by two networks, one is encoder, and the other is predictor.


Model configurations:

Model Name|hidden layers|loss function|# training samples|tsize|# epochs|training process|Date
--------|--------|----------|----------|-------|--------------|--------|------|
newest-5_8| 16 |built-in RSME| 400| 10| 500| Fig.1|2019.5.8
newest-5-13|16|RMSE+FAR|400|10|500|Fig.2|2019.5.13

<p align="center">
<img src="images/training-5_8.PNG" style="width:60%"><br>
Fig.1 Training loss on 5.8
<img src="images/201811230600-predict.gif" style="width: 80%"><br>
Fig.2 Comparison of predction and observation
</p>

### Wind Effect
To add wind effect in, one way is to customize the loss function by comparing LSTM modeled image at t and predicted image with Lagrangian plus wind at t. In this way, we drive the LSTM to the correct advection direction.

## GAN


## GRU

We select one event from our radar data shown below
<p align="center">
    <img src="demo.gif" style="width:40%">
</p>

## Kalman Filter

## Semi-Lagrangian movement

we create several wind field to test the robustness of this algorithm with radar movement, the first goes with uniform speed gradient wind vector.

The basic idea of semi-lagrangian goes here ...
the stability analysis of semi-lagrangian algorithm goes here...


<p align="center">
    <img src="lagrangian/demo_uniform.gif" style="width: 40%">
</p>

## Optical Flow

The results are like this ...

Left panel is the ground truth, while the right is prediction.
<p align="center">
<img src="opticalflow/optical_flow.gif" style="width:80%">
</p>
This is pretty messy because the optical flow assumes that all pixels have the same intensity which is not the case in radar images because localized growth and decays may be generated.

Here is one implementation with optical flow [Motion Vectors in Weather Radars](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.539.99&rep=rep1&type=pdf)

## FlowNet

<p align="center">
    <img src="images/flownet.png"
</p>

## Kalman Filter



To be continued...
