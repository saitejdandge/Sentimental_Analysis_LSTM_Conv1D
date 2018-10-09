# Sentimental_Analysis of Tweet Emotions
Sentimental Analysis using Long Short Term Memory Recurrent Neural Networks (DeepLearning)
<br/>
(40000,)
<br/>
(40000,)
<br/>
<br/>
Excluding stopwords ...
<br/>
Tokenized to Word indices as
<br/>
(40000,)
<br/>
After padding data
(40000, 20)
<br/>
Loading Glove Vectors ...
<br/>
Loaded GloVe Vectors Successfully
<br/>
Embedding Matrix Generated :  (32855, 50)
<br/>
Label Encoding Classes as
        { 0: 'anger', 
        <br/>
          1: 'boredom',
          <br/>
          2: 'empty', 
          <br/>
          3: 'enthusiasm',
          <br/>
          4: 'fun',
          <br/>
          5: 'happiness', 
          <br/>
          6: 'hate',
          <br/>
          7: 'love', 
          <br/>
          8: 'neutral',
          <br/>
          9: 'relief', 
          <br/>
          10: 'sadness', 
          <br/>
          11: 'surprise',
          <br/>
          12: 'worry'}
          <br/>
<br/>
One Hot Encoded class shape :
<br/>
(40000, 13)
<br/><br/>
2018-10-09 00:56:56.582717: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2

<br/>
x shape (4323, 300, 300, 3)
<br/>
y shape (4323, 5)
<br/>

| Layer(type)       	        | Output Shape           |  Param #  |
| ------------- 		          |:-------------:	       | -----:    |
| embedding_1 (Embedding)    	|  (None, 20, 50)        |   1642750 |
| conv1d_1 (Conv1D)           |  (None, 20, 30)        |   1530    |
| max_pooling1d_1 (MaxPooling1|  (None, 5, 30)         |   0       |
| lstm_1 (LSTM)               |  (None, 5, 100)        |   52400   |
| flatten_1 (Flatten)         |  ((None, 500)          |   0       |
| dense_1 (Dense)             |   (None, 500)          |   250500  |
| dense_2 (Dense)             |   (None, 300)          |   150300  |
| dense_3 (Dense)             |   (None, 13)           |   3913    |



Total params: <b>2,101,393</b><br/>
Trainable params: 458,643<br/>
Non-trainable params: 1,642,750<br/>
<br/>
Train on 29936 samples, validate on 64 samples
<br/><br/>
Finished Preprocessing data ...<br/>
x_data shape :  (40000, 20)<br/>
y_data shape :  (40000, 13)
<br/>
spliting data into training, testing set
<br/>

<h1>Yeah I think, I need more data</h1>


<br/>
Epoch 00096: val_acc did not improve from 0.40625
Epoch 97/100
29936/29936 [==============================] - 8s 277us/step - loss: 1.8189 - acc: 0.3791 - val_loss: 1.9288 - val_acc: 0.2969
<br/><br/>
Epoch 00097: val_acc did not improve from 0.40625
Epoch 98/100
29936/29936 [==============================] - 8s 259us/step - loss: 1.8162 - acc: 0.3803 - val_loss: 1.9488 - val_acc: 0.3281
<br/><br/>
Epoch 00098: val_acc did not improve from 0.40625
Epoch 99/100
29936/29936 [==============================] - 7s 249us/step - loss: 1.8131 - acc: 0.3824 - val_loss: 1.9878 - val_acc: 0.3438
<br/><br/>
Epoch 00099: val_acc did not improve from 0.40625
Epoch 100/100
29936/29936 [==============================] - 7s 249us/step - loss: 1.8122 - acc: 0.3822 - val_loss: 1.9144 - val_acc: 0.3281
<br/><br/>
Epoch 00100: val_acc did not improve from 0.40625
<br/><br/>
<h2>Test accuracy: 0.337</h2>

<br/><br/>
![alt text](Figure_1.png)
