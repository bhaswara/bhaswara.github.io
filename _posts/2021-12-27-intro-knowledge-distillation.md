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
L_{KD} = L_{Student} + L_{Teacher}
\end{equation}

Ok, until now we only talk about the theory behind it. So, let's continue to the coding step!

