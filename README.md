**Status:** Archive (code is provided as-is, no updates expected)

# Optimized-MC-Dropout-Using-DRL

MC-dropout estimates uncertainty at test time using the variance statistics extracted from several dropout-enabled forward passes. Unfortunately, the prediction cost of an effective MC-dropout can reach hundreds of feed-forward iterations for each prediction.
In this repository, I model MC-dropout in a Deep Reinforcement Learning (DRL) framework, to find the optimial passes needed for producing predefined confidence level. 

# Acknowledges

* For creating the Open-AI gym environment, I followed the code from [Making a custom environment in gym](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa) 
* For MC-dropout expriments, I used the code from Yarin Gal - [DropoutUncertaintyExps](https://github.com/yaringal/DropoutUncertaintyExps)
