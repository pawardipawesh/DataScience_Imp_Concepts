# Deep Learning Important Concepts
- Deep Neural Networks
- Activation Fuctions[Derivative Graph](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) , [Comaprison](https://en.wikipedia.org/wiki/Activation_function) , [sigmoid Vs Tanh Vs Relu](https://towardsdatascience.com/exploring-activation-functions-for-neural-networks-73498da59b02)
  - Sigmoid/Logit
  - Tanh
  - Relu
  - Leky Relu
  - softmax [Blog1](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax) [Blog2](https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d)
- Optimization Algorithms[Very Nice Blog](https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a)
  - Gradient Descent
  - Stochastic Gradient Descent
  - GD with Momentum
  - RProp (Does not work well with mini batches)
  - RMSProp(Divide with explonential weighted avraged gradient and multiple with current gradient)
  - Adam
- Loss functions [Nice Blog](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0)
  - Regression
    - Mean,ordinal,Least Square loss/L1 loss/Quadratic loss (Less robust to outliers)
    - Mean Absolute loss/L2 Loss (More robut to outliers)
    - Huber loss/Smooth Mean Absolute Error (Combination of MSE and MAE with a thresold)
    - Log cosh loss
  - Classification [Blog](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/)
    - Cross Entropy [Nice Blog](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
    
     Distance between two distributions. Entropy is measure of uncertainty in a distribution.
      - binary cross entropy loss
      - Categorical cross entropy
- Batch Normalization(Normalize hidden layer input)
  - How : standard Normalization with Learnable parameters
  - Why : if input distribution changes a lot(coveriate shift), retraining is needed. To avooid this and make netwrk more robust Batch Normalization.
- Regularization
  - Dropout
  - Batch Normalization 
  - Early Stopping
  - Weight constraint
- Sequential learning
  - Recurrent Neural Networks
    - BPTT, TBPTT [Nice answer](https://stats.stackexchange.com/questions/219914/rnns-when-to-apply-bptt-and-or-update-weights)
  - Gated Recurrent Unit (GRU)
  - Long Short Term Memory Networks (LSTMs)
  
    GRU and LSTMs helps in capturing long term dependency(vanishing gradient problem) which RNN can not.
  - Concepts cum Applications
    - Word2Vec
      - CBOW
      - Skipgram
      - Implemented using Hierarchial Softmax, Negative Sampling
    - Glove
- Vanishing Gradient Problem
  - why : In sigmoid, Tanh activations, drivative ranges from 0-0.25 and 0-1 with normal distribution, derivative is very small after some -ve and +ve values. Also as we go in earlier layers gradients become even smaller due to multiple multiplications.
  - how to handle : Relu Activation....TBPTT....Careful Weight initialization
- Exploading gradient problem
  - why: gradient accumalate, must be happening mostly with Relu, and explode as they pass towards input layers. causes overflow and hence NANs
  - How to handle: gradients clipping, careful weight initialization
  
- Encoder Decoder Architecture
- Attention Mechanism [Nice Blog](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
- AutoEncoders (Feature Reduction, Selection) [Blog](https://towardsdatascience.com/autoencoders-bits-and-bytes-of-deep-learning-eaba376f23ad)
- Can we train LSTMs without padding or tructating?
  - Yes. However, to do so we will have to have batch size=1. This is because, when batch size>1, forward propagation happens in paralell. With diffent timesteps for samples in same batch we won't be able tp process in paralell. So, we can train LSTMs with varying timesteps samples, However we will have to keep batch size=1. This will hamper training time a lot. Hence, most of the time padding with masking is preferred.
