# Machine Learning Important concepts
- Feature Selection techniques
  - Forward selection
  - Backward Selection
  - Bidirectional Selection
- Feature Reduction Techniques
  - PCA [Nice Blog](https://towardsdatascience.com/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)
- Feature Scaling [what](https://en.wikipedia.org/wiki/Feature_scaling) , [Effects of Diffrent Scaling methods on various algos and Normalization vs Standardization](https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf)
  - Min-Max
  - Mean Normalization
  - Standardization (Z-score Normalization)
  - Scaling to unit length
- Sequential Modelling Algorithms
  - HMMs : Generative models - 
  - CRFs: Discriminative models - : (Sequential Version of Logistic Regression) [Link](https://blog.echen.me/2012/01/03/introduction-to-conditional-random-fields/)
    - Linear chain CRFs : Consider only i-1 lable
    - General CRFs
- [Generative Models vs Discriminative Models](https://medium.com/@mlengineer/generative-and-discriminative-models-af5637a66a3)
- [joint-probability-vs-conditional-probability](https://medium.com/@mlengineer/joint-probability-vs-conditional-probability-fa2d47d95c4a)
  - RNN 
  - LSTM
  - GRU
- Classical Classification Algorithms
  - Logistic Regression (Its Regression as it outputs probability of class labels)
  - Naive Bayes [Very Nice Blg](https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/)
    
    It takes less time in training as just calculates joint probabilities during training
  - SVM (can be used only for classification and it directly predicts class labels on the basis of hyperplanes and support vectors)
    - Linear Kernel
    - RBF Kernel
    - Polynomial Kernel
  - Decision Trees (Regression as well as classification)
  - Bag of classifiers (Bagging)
    - Random Forests
  - Boosting Based classifiers
    - how samples are weighted
      - weighted sampling
      - rejection sampling
    - how decision is made 
      - weighted avergaing (classifiers are weighted on the basis of their performance)
    - XGBoost [Framework known as Xtreme Gradient Boosting](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/)
      - AdaBoost
      - Gradient Boosting
      - Stochastic Gradient Boosting
      - Regularized Gradient Boosting
  - Bagging Vs Boosting (https://quantdare.com/what-is-the-difference-between-bagging-and-boosting/)
 - Clustering Techniques
   - K-Means
   - K-medoid
   - DBSCAN
   - Agglomerative
   - Divisive
   - Genetic Algorithms
- Regularization
  - L1 regularization
  - L2 Norm
- Similarity measures [Nice blog](https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms)
  - Euclidian distance(L2 Norm)
  - Manhattan Distance(L1 Norm)
- MultiLable classification[Nice blog](https://towardsdatascience.com/journey-to-the-center-of-multi-label-classification-384c40229bff)
- Handling Imbalanced Datasets
  - Undersampling
    - Random
    - Informative
      - Easy Ensemble(learn multiple classifiers such that each of them is exposed to all minority samples but undersampled majority ones)
      - Balance cascade
    
    Disadvantage: Information loss
  - Oversampling
    - Random
    - Informative
    - Synthetic minority oversampling Technique (SMOTE)
    
    Advantage: No Information loss
    
    Disadvantage : Overfitting as we replicate
  - Cost sensitive learning ( More cost for misspredictions of minority class)
- Evaluation Metrics
  - Accuracy
  - Precision(Specificity), Recall(Sensitivity), F-Measure
  - AUC ROC Curve [Blog](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)
  - AUC PR Curve
    - [AUC ROC vs AUC PR](http://www.chioka.in/differences-between-roc-auc-and-pr-auc/)
  - R2 (regression scoring) [check](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score)
  - bias vs variance [Blog](https://towardsdatascience.com/understanding-the-bias-variance-tradeoff-165e6942b229)
- How to evaluate
  - cross validation
    - Hold out cv
    - k-fold cv
    - leave one out cv
    
    LOOCV model will have high variance meaning results of LOOCV mzy vary alot if you retrain the model wih same hyperparameter on some other data points for same problem. This happens because train set will have hghest overlap as compared to other CV techniques in LOOCV. [captain_ahab answer](https://stats.stackexchange.com/a/244112/129463)
- Calibration
  - [Blog](https://medium.com/analytics-vidhya/calibration-in-machine-learning-e7972ac93555)
# Deep Learning Important Concepts
- Deep Neural Networks
- Activation Fuctions[Derivative Graph](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6) , [Comaprison](https://en.wikipedia.org/wiki/Activation_function) , [sigmoid Vs Tanh Vs Relu](https://towardsdatascience.com/exploring-activation-functions-for-neural-networks-73498da59b02)
  - Sigmoid/Logit
  - Tanh
  - Relu
  - Leky Relu
  - softmax [Blog1](https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax) [Blog2](https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d)
- Optimization Algorithms
  - Gradient Descent
  - Stochastic Gradient Descent
  - Momentum
  - RMSProp
  - Adam
- Loss functions [Nice Blog](https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0)
  - Regression
    - Mean,ordinal,Least Square loss/L1 loss/Quadratic loss (Less robust to outliers)
    - Mean Absolute loss/L2 Loss (More robut to outliers)
    - Huber loss/Smooth Mean Absolute Error (Combination of MSE and MAE with a thresold)
    - Log cosh loss
  - Classification [Blog](https://machinelearningmastery.com/loss-and-loss-functions-for-training-deep-learning-neural-networks/)
    - Cross Entropy [Nice Blog](https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a)
    - Why cross entropy is always greater than entropy [link](https://stats.stackexchange.com/questions/370428/why-is-the-cross-entropy-always-more-than-the-entropy)
    -  Intutive explanation: We can reduce uncertainty in predicted distribution to make it equivalent to true distribution but we can not further reduce it. i.e. uncertainty in predicted distribution is atleast as much as in true distribution
    
     Distance between two distributions. Entropy is measure of uncertainty in a distribution.
      - binary cross entropy loss
      - Categorical cross entropy
- Regularization
  - Dropout
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
- Encoder Decoder Architecture
- Attention Mechanism [Nice Blog](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3)
- AutoEncoders (Feature Reduction, Selection) [Blog](https://towardsdatascience.com/autoencoders-bits-and-bytes-of-deep-learning-eaba376f23ad)
      
# Important Questions
- Why not approach classification through regression [Link](https://stats.stackexchange.com/questions/22381/why-not-approach-classification-through-regression)
- Linear vs Logistic regression ?
- CRF Vs HMMs?
- [Underfitting, Overfitting and ways to handle](https://towardsdatascience.com/underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6fe4a8a49dbf)
- What is maximum likelihood estimates?
- What is Expectation-Maximization algorithm (EM Algorithm)? Can it be used for regressiona or classification ? Does it always converge?
- How SVM work? What are kernels in SVM
- Decision Tree globally or locally optimal? How to avoid overfitting in decision trees[good blog](https://www.edupristine.com/blog/decision-trees-development-and-scoring)
- MultiCollinearity ? Why bad ? How to detect? How to handle? [Link](http://www.sfu.ca/~dsignori/buec333/lecture%2016.pdf)
  - We update weights(coeficients) using derivative of loss with respect to weight. Now if two features(independent variables) are related, then we can not independently estimate weight change and update.
- L1 vs L2 regularization. What are differences? Which is better ? Can they be termed as feature selection or reduction techniques? [Nice Blog](https://towardsdatascience.com/intuitions-on-l1-and-l2-regularisation-235f2db4c261#15c2, https://medium.com/@mukulranjan/how-does-lasso-regression-l1-encourage-zero-coefficients-but-not-the-l2-20e4893cba5d)
- Optimizations functions in Deep learning ?
- When to use which feature scaling ?
- Similarity measures in Machine learning [Nice blog](https://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/)
- Need of Negative sampling and hierarchial softmax in skip gram based word2vec [Nice Video1](https://www.coursera.org/lecture/nlp-sequence-models/word2vec-8CZiw?authMode=login) [Nice Video2](https://www.coursera.org/lecture/nlp-sequence-models/negative-sampling-Iwx0e)
- Why Weight initialization ways are important in DNN? Vanishing Gradients and Exploding gradients ? [Blog](https://www.dlology.com/blog/how-to-deal-with-vanishingexploding-gradients-in-keras/)
- Why high dimensionality is a curse? [Blog](https://towardsdatascience.com/the-curse-of-dimensionality-50dc6e49aa1e)
- what is advantage of higher grams and what is their disadvantage? probability of finding higher grams in test set is less.
- Are features also sampled in Random Forest ? what makes them better than Decision trees?
- How to reduce overfitting without regularization(by processing data)
- startegies to handle imbalanced data
- What happens when no. of features are greter than no. of samples?
- Is random weight assignment better than assigning same weights to the units in the hidden layer?
- Integer encoding vs one hot encoding vs dumy encoding. When to use which?[Blog](https://towardsdatascience.com/one-hot-encoding-multicollinearity-and-the-dummy-variable-trap-b5840be3c41a)
- Why we use exponentiation in softmax activation funcion instead can we not just divide logit  by sum of logits?[check why](https://medium.com/data-science-bootcamp/understand-the-softmax-function-in-minutes-f3a59641e86d)
- Why softmax is computationally complex?
- Which is easy Regression or Classification? Why?
- Decision boundry in decision tree?
- Where we can not apply K-means ?
- How to approach classification problem where target variable is ordinal? [Blog](https://towardsdatascience.com/simple-trick-to-train-an-ordinal-regression-with-any-classifier-6911183d2a3c)
- when to choose which optimizer?
- How we take final decision in Boosting. How we weight classifiers and trainig samples ?
- ROC and PR curve ?
- Discriminant models vs Generative models?
- why L1 normalization reduces some weights to zero why not L2?
- How to detect and handle outliers ? Z-score and Inter quartile range(IQR) [Blog](https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)
- Why we divide by + v in add-1 smoothening in n-gram language models ?
  - When we add 1 to numerator, we mean class is occuring with that word atleast once. Then we assume there are at max V such unknown words for which we need to add one. Hence now in dataset
  - there will be V such words occuring with  that class. Hence divide by + V

# Links To nice resources
- [Google guide] https://developers.google.com/machine-learning/guides
- [Machine Learning Glossary] https://developers.google.com/machine-learning/glossary
