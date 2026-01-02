# Notes Repository

This repository contains notes, experiments, and code for various topics. See the readme's inside for more details

## Table of Contents

- [probability/](probability/)
    - [notes/](probability/notes/): Notes on probability basics
	- [estimating_low_head_prob_coinflips/](probability/estimating_low_head_prob_coinflips/): A bit of estimation theory for coin flips with low head probability
	

- [pytorch_fx/](pytorch_fx/)
	- [basic_intro/](pytorch_fx/basic_intro/): Introductory FX usage, pattern matching, and graph manipulation
	- [optmizations/](pytorch_fx/optmizations/): Model optimization experiments
		- [conv_bn_fold/](pytorch_fx/optmizations/conv_bn_fold/): BatchNorm folding into Conv layers
		- [gelu_quickgelu_replacement/](pytorch_fx/optmizations/gelu_quickgelu_replacement/): Training and evaluating GELU/QuickGELU variants on vision models
    - [abstract_interpret/](pytorch_fx/abstract_interpret/): a simple abstract interpretation example in the odd/even domain by subclassing `fx.Interpreter`.
    - [quantization/](pytorch_fx/quantization/): A framework to do quantization simulation. WIP
- [quantization/](quantization/): Quantization notes and experiments
    - [basics/](quantization/basics/): General foundational notes and concepts
    - [typed_quant/](quantization/typed_quant/): Some expts in C++ for a type system for quantized numbers
- [queueing/](queueing/): Notes/experiments on queueing processes
    - [batching/](queueing/batching/): Analysis/simulation of a process with restricted batch sizes, and an introduction to SimPy
