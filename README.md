## DynaVis

This is the official repository for DynaVis, a visualization solution for accurately capturing the high-dimensional embedding dynamics of a model training process.

### Structure

* `/singleVis` Core function and modules of the framework ( including visualization model, spatial and temporal complex, dataloader etc. )

* `/data` manage dataset and checkpoint for the model training process

### Setup

#### 1. setup the environment

Follow the instructions in the official documentation [TimeVis](https://github.com/code-philia/TimeVis) to setup the environment

#### 2. prepare dataset

prepare dataset of your own model training process follow the [instructions](./data/data.md) in the `/data` 

or you can download the sample dataset in our paper from the [link](https://sites.google.com/view/dynavis/home?authuser=0)

#### 3. run experiment

~~~python
activate dynavis
bash run.sh