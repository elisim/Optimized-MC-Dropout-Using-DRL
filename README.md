# Optimized-MC-Dropout

MC-dropout estimates uncertainty at test time using the variance statistics extracted from several dropout-enabled forward passes. Unfortunately, the prediction cost of an effective MC-dropout can reach hundreds of feed-forward iterations for each prediction.
In this repository, I model MC-Dropout in a DRL framework, to find the optimial passes needed for producing predefined confidence level. 
My code followed by [Making a custom environment in gym](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa) for creating an Open-AI gym environment.
