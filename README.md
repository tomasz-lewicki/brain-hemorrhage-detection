# brain


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

5. I'm moving to HPC, since the experiments are becoming longer and longer and kaggle usually crashes sooner or later 🤷.

