# DeepVO

This project is an attemption to combine CNN with LSTM to solve SLAM (simultaneous localizing and mapping) problem.
CNN here is used to 'extract' features, which are subsequently feed into a convolutional 2D LSTM network for time-sequence
movement prediction. Tensorflow 1.4 has support for 2D-LSTM so I used its built-in function rather than building it from 
the scratch. I am still trying to figure a way to stack two LSTM together as it is not currently supported(?).

A typical process to realize the model includes:
1/ generate stacked images; I hope by stacking two consecutive images together, CNN can learn to extract features between two
   images;
2/ generate corresponding labels from EUROC ground truth; although most of the image timesteps are covered in the provided
   ground truth, there are chances where not, if you read the code, you can see for cases where timestamps are not included
   in the grountruth, I tryied to get a guessed values from two nearesrt neighbors. To illustrate, if current image timestamp
   is t, the closed timestamp before t is defiend as t_{-1} and that after t is defined as t_{1}, f(t) represents the postion 
   of the timestamp, in this case, f(t) = (t-t_{-1}/(t_{1}-t_{-1}) )(f_{1}-f_{-1}). I havent found out a good way to get a 
   good estimation of the quaternion part(becasue of the none linear property), so only postion, including px,py,pz are considered in this case.
   
2/ generate tfrecord file from the stacked image and above-metnioned method generated labels;
3/ train and test.

If you have any questions or suggestions, please feel free to let me know.

It is inspired by the recent work nicely done by Sen Wang at Oxford Uni.
