---
layout: post
title: An Introduction to Knowledge Distillation
date: 2021-12-27 08:00:00
description: an introduction to knowledge distillation with additional Pytorch code
comments: true
categories: knowledge-distillation jsc-research
---

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/teacher-student.jpg" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 1. A teacher and her students.
</div>

Have you ever heard about knowledge distillation? If yes then you may skip this introduction and directly go on to the coding section. But if you haven't, then let's us dive in together to this topic. I know this topic when I was looking for optimization deep learning model for embedded devices. Well, it's actually not a new topic in deep learning and many researchers already developed it. But it's better late than nothing to know this topic, isn't it? Ok let's start this.

What is knowledge distillation? hmm maybe I should start with a simple first. When you open this post, at the beginning you see an image about a teacher and her students in a classroom. The teacher teaches and tranfers her knowledge to the students and the students grasp the information from the teacher. It's almost the same with knowledge distillation. In knowledge distillation, there are 2 models: a teacher and a student. A teacher basically is a big model where it has many parameters. Or in the knowledge distillation paper [[1](https://arxiv.org/abs/1503.02531)], it's called the cumbersome model. The teacher itself could be pretrained or not. Whereas a student is a non-pretrained small model with less parameters than the teacher model. To get the insight about this teacher and student, I put an image below.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/kd.jpg" title="kd image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Figure 2. Knowledge distillation model.
</div>

Actually we can have only a single model where it acts as teacher and student itself which it called as self-distillation. We will talk about this in another section. Ok back to the topic. Most of the model are not embedded devices friendly although they have high accuracy. They take a lot of storages and computations. Take an example of Resnet-50 [[2](https://arxiv.org/abs/1512.03385)] where it has over 23 million trainable parameters and has a size about 90 MB. With the knowledge distillation method, we want to transfer the knowledge from the big model to the small model so it is suitable for embedded devices deployment. It's hoped that the knowledge that is transferred from the teacher to the student will have similar in accuracy but less parameters and size. 

So, how does it work? the teacher predicts the probability distribution and transfers its knowledge to the student while at the same time also minimizing its loss function. Those probabilities are calculated using softmax function as in \eqref{eq:softmax-function}

\begin{equation}
\label{eq:softmax-function}
q_{i} = \frac{exp(z_{i}/T)}{\sum_{j}exp(z_{j}/T)}
\end{equation}

The student also returns some probability distribution. We want this distribution as close as the distributions from the teacher output. So, we apply Kullback-Leibler (KL) divergence between student and teacher as distillation loss. At the same time, we also want the student correctly predicts the classes. Here, we use categorical cross-entropy as student loss. Combining both, we get knowledge distillation loss function \eqref{eq:kd-loss}

\begin{equation}
\label{eq:kd-loss}
L_{KD} = \alpha * L_{Student} + (1 - \alpha) * L_{Teacher}
\end{equation}

Ok, until now I only talk about the theory behind it. So, let's continue to the coding step!

## Code
In this example code, I only use mnist dataset and fully-connected model.

Ok, first, let's import the library.

{% highlight python %}

import numpy as np
import random

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

{% endhighlight %}

Next, we load, split, and convert our mnist data into pytorch tensor.

{% highlight python %}

transform = transforms.Compose([transforms.ToTensor()])

trainset = datasets.MNIST(root='files_mnist/', download=False, train=True, transform=transform)
train, valid = random_split(trainset,[50000,10000])

testset = datasets.MNIST(root='files_mnist/', download=False, train=False, transform=transform)

trainloader = DataLoader(train, batch_size=32, shuffle=True, worker_init_fn=seed_worker, generator=g)
valloader = DataLoader(valid, batch_size=32, shuffle=True, worker_init_fn=seed_worker, generator=g)
testloader = DataLoader(testset, batch_size=32, shuffle=False, worker_init_fn=seed_worker, generator=g)

{% endhighlight %}

It's time to create teacher and student model. The teacher model only uses 2 hidden layers. You can try another option such as adding another layers or changing the input and output of each layers. The point is that the teacher model should be bigger than the student model.

{% highlight python %}
# Teacher model
class teacher_net(nn.Module):
    def __init__(self):
        super(teacher_net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Student model
class student_net(nn.Module):
    def __init__(self):
        super(student_net, self).__init__()
        self.fc1 = nn.Linear(784, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

{% endhighlight %}

Next we define our loss function as in equation \eqref{eq:kd-loss}.

{% highlight python %}
# Knowledge distillation loss
def loss_kd(scores, labels, targets, alpha=1, T=1):
    p_s = F.log_softmax(scores/T, dim=1)
    p_t = F.softmax(targets/T, dim=1)
    distill_loss = nn.KLDivLoss()(p_s, p_t) * (T**2) 
    student_loss = F.cross_entropy(scores, labels)
    
    loss = alpha * student_loss + (1. - alpha) * distill_loss
    
    return loss

{% endhighlight %}

Last, we train our student model using $$\alpha$$=0.5, temperature=2, and epoch=10. You can play with other numbers.

{% highlight python %}
# Training student with distillation loss
epochs = 10
temperature = 2
alpha=0.5

for epoch in range(epochs):
    train_loss = 0.0
    train_acc = 0.0
    train_total = 0
    
    valid_loss = 0.0
    valid_acc = 0.0
    val_total = 0
    
    student_model.train()
    for data, labels in trainloader:
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        scores = student_model(data)
        targets = teacher_model(data)
        
        _, preds = torch.max(scores, 1)
        
        loss = loss_kd(scores, labels, targets, alpha=alpha, T=temperature)
        
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
        train_acc += torch.sum(preds == labels)
        train_total += labels.size(0)
        
    student_model.eval()
    for data, labels in valloader:
        data = data.to(device)
        labels = labels.to(device)
        
        scores = student_model(data)
        targets = teacher_model(data)
        
        _, preds = torch.max(scores, 1)
        
        loss = loss_kd(scores, labels, targets, alpha=alpha, T=temperature)
        
        valid_loss += loss.item()
        valid_acc += torch.sum(preds == labels)
        val_total += labels.size(0)
    
    training_loss = train_loss / len(trainloader)
    training_acc = (100 * train_acc) / (train_total)
    validation_loss = valid_loss / len(valloader)
    validation_acc = (100 * valid_acc) / (val_total)
    
    print("Epoch: {}/{} \nTraining Loss:{:.4f} Training Acc:{:.1f} \nValidation Loss:{:.4f} Validation Acc:{:.1f}\n".format(
    epoch, epochs, training_loss, training_acc, validation_loss, validation_acc))

{% endhighlight %}

Ok, that's all for the introduction of knowledge distillation. Thank you for your time. I hope you like it. Stay tuned for other posts related to computer vision and deep learning!.

P.S: If you want to see the result, just open [my github page](https://github.com/bhaswara/knowledge-distillation). I put all there.

## References
[1] Hinton, G., Vinyals, O., and Dean, J., “Distilling the Knowledge in a Neural Network”, <i>arXiv e-prints</i>, 2015.

[2] He, K., Zhang, X., Ren, S., and Sun, J., “Deep Residual Learning for Image Recognition”, <i>arXiv e-prints</i>, 2015.
