# Algorithm project source code
### Group 41
### Group member
Liu Sichen, Lu Guandong

### **Requirements**

- python == 3.7.0

- numpy >= 1.13.3

- pandas >= 0.23.0

- pytroch == 1.0.0

- torchvision >= 0.2.1


### **Idea**

We proposed a method we called layed simplification method to simplify the AlexNet.

### **result**

Compared to the original AlexNet, we simplified to 40 percent parameters and the accuracy changed from 87.53 to 88.97.

### **Quick Start**
- Install pytorch and torchvision first.

- Run 'dataProcess.py' to train an original AlexNet.

`py dataProcess.py`

Note that it will automatically download the MNIST dataset into the data dir.

- Run 'prunning.py' to get every 4 kernel's response to a certain class. The result will be stored in 'Evaluation.csv'.

`py prunning.py`

- Run 'labelsplit.py' to get the 0-1 response to every class. 1 means reserved kernels and 0 means deleted kernels.

`py labelsplit.py`

- Run 'myway.py' to check the group of classes.

`py myway.py`

- Run 'firstLayerTraining.py' to train the upper layer network. Note that the converting_dictionary may be changed by the output of the myway.py.

`py firstLayerTraining.py`

- Run 'secondLayerModel.py' and 'secondLayerRetraining.py' to build and retrain the second layer model.

```
py secondLayerModel.py
py secondLayerRetraining.py
```

- Run 'Evaluation.py' to check the final accuracy of the layered network.

`py Evaluation.py`
