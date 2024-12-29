# MLX Image Models

> Work in progress...

This repo is an attempt to design a MLX Deep Learning Model library equivalent to timm, also known as the Pytorch Image Models library.

Contrary to the MLX-image / MLX-vision packages, the goal is to get a *fully compatible* library with timm and models weights stored on the Hugging Face Hub.

The main design goals are:

* 100% coverage of the timm models
* Fully-tested models for exactness w.r.t timm

Things that won't be supported:

* Model training
* Optimizers, losses, etc.
* Export to ONNX, JIT, etc.
* Other niche things from timm.
