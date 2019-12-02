# brain


# Week 8 - update:

1. **Most important:** changed the learning rate.

The more conservative learning rate of 1e-5 finally brings some good results, especially when training __big models__ (e.g. with trainable conv. base) or with __big input size__ (e.g. (512,512,3)).

I couldn't understand why my models wouldn't train well just after changing the shape to (512,512). The reason was that this bigger shape forced smaller batch size (8 instead of 16). Smaller batches call for more conservative learning rates.

*Learning rate maters in ML, duh...*

2. changed windowing slightly to include a bone window. 
The 3 channels used to be: (brain, subdural, soft tissue)

I changed it to be: (brain, subdural, bone)

3. started experimenting with ResNet50
Getting some better results out-of-the box over VGG16, using ResNet50 as feature selector. See [here](https://github.com/tomek-l/brain/blob/master/Week%208%20-%20tlewicki%20-%20notebook%205.1%20-%20%E2%9C%94%EF%B8%8F%20trying%20ResNet%20with%20right%20learning%20rate.ipynb). Will probably stick to ResNet for now.

 # Week 7 - update:
 1. Created a script for long-running training on HPC. This is much more reliable to run than notebooks.
 2. Further experimenting with full dataset. __Bottom line__: if it doesn't work well on 10k examples, it probably won't on 670k... 😞
 
 
 # Week 6 - update:
 1. Experimenting with weighted loss
 2. Experimenting with full dataset
 3. I'm moving to HPC, since the experiments are becoming longer and longer and kaggle usually crashes sooner or later 🤷.
 4. I downloaded the dataset and placed it in ```/data/cmpe257-02-fa2019/team-1-meerkats``` on HPC. No more kaggle limitations 🎉

 

# Week 5 - update:
Added 6 notebooks with lots of cool things to the repo:
1. Started training on 10k training samples.

2. New metric of CPD _Correct Positive Diagnoses_. Accuracy is clearly a poor metric for the task of medical diagnoses.
Guessing all diagnoses to be 0 (negative) yields 80% accuracy. I feel that the metric of CPD is more representative, but not perfect.
Most importantly, it lets me track progress when log loss gets really low.

3. Started experimenting with weighted losses and finally got some successful results.
I think that's the way to go forward, since we cannot leave the heavy _dataset imbalance_
(both w.r.t. positive/negative as well as imbalance among labels) not addressed.

4. I'm still only training the dense head and using convolutional base from VGG16 trained on imagenet as feature extractor.
I'm sure there's a lot of potential in training an end-to-end net.




