---
title: "Bilinear Multimodal Fusion for Visual Question Answering"
excerpt: "Trained a model to tackle the VQA problem using a series of attention-based reasoning steps each performed by a recurrent Memory, Attention and Composition (MAC) cell 1<br/><img src='/images/500x300.png'>"
collection: portfolio
---

---
abstract: |
  Visual Question Answering (VQA) is a recent problem of interest in
  computer vision and natural language processing that has attracted
  much attention from academia and industry alike. Given an image and a
  natural language question about the image, the task is to provide an
  accurate natural language answer. This study aims to investigate the
  underlying notions of VQA and propose a compositional reasoning
  solution for the GQA dataset. The proposed approach decomposes the VQA
  problem into a series of reasoning steps - each handled by a
  specialized recurrent cell that exploits the principle of bilinear
  fusion to combine the two modalities (image and question). Results
  show that our approach is comparable to the state of the art MAC
  network on $\textbf{10\%}$ GQA dataset. It achieves 52.08% accuracy
  whereas the current state-of-the-art MAC network achieves 53.94%.\
author:
- |
  Department of Computer Science\
  NC State University\
  *aatriek\@ncsu.edu*
- |
  Department of Computer Science\
  NC State University\
  *rpancho\@ncsu.edu*
- |
  Department of ECE\
  NC State University\
  *nvimale\@ncsu.edu*
title: |
  Bilinear Multimodal Fusion for\
  Visual Question Answering
---

Shell *et al.*: Bare Demo of IEEEtran.cls for IEEE Journals

Machine Reasoning, Visual Qustion Answering, Recurrent Neural Networks,
Attention networks, Bilinear Fusion.

Introduction
============

Through the breakthroughs in deep learning over the past decade, we have
enabled AI agents to rival humans in a number of computer vision tasks.
These include image classification, image restoration, optical character
and facial recognition, and motion analysis. These tasks are all trivial
in the sense that they do not require a deep, holistic understanding of
the input images. An intelligent agent, similar to humans, must also be
able to identify different objects in an image, recognize the spatial
relationships between objects, provide context to the objects and
entities, and draw other novel inferences based on this acquired
knowledge. To this end, this study will aim at designing a compositional
reasoning system that is capable of solving complex problems involving
images and pertinent natural language questions. The solutions are then
returned by the reasoning system as natural language answers.

Fig. [1](#fig:VQAexample){reference-type="ref"
reference="fig:VQAexample"} illustrates some typical examples of the
visual reasoning task. As is obvious from the example, the raw image can
contain more than just superficial information. It's also worth noting
that the questions are not only arbitrary, but they encompass many
sub-problems in computer vision and natural knowledge processing tasks.
The diversity of skills required to solve a VQA problem makes it
particularly challenging. For example, to be able to answer the question
\"*Is the bowl to the right of the green apple*\" in Fig.
[1](#fig:VQAexample){reference-type="ref" reference="fig:VQAexample"},
the agent first needs to recognize the objects *bowl* and *apple*. Then,
it needs to classify the attributes of the detected objects to learn
which apple is *green*. Finally, it must also be able to learn the
spatial relationship between the two objects.

![Example of VQA task [@b2]](Images/VQAexample.png){#fig:VQAexample
width="50%"}

As should be apparent through the above example, VQA is a multi-modal
problem. Common technical difficulties involve parsing of complex
questions written in natural language, image processing tasks such as
object recognition, scene classification and spatial reasoning, and
combining the features extracted from the questions and the images.
Furthermore, since the natural language questions can be compositional,
reasoning must be structured and iterative.

![image](Images/MACnetwork.png){width="75%"}

In this study, we combine the key ideas from two state of the art papers
to build a reasoning system for the VQA task. More specifically,
bilinear fusion as discussed in [@b5] is integrated with the MAC network
proposed in [@b3]. The MAC network also works as our baseline. The rest
of the work is organized as follows. Section II puts forward an
elaborate description of the proposed approach. Section III describes in
detail the procedure for hyper-parameter tuning. Finally, Section IV
quantitatively demonstrates the results on $10\%$ GQA dataset and
compares them against the baseline, i.e., the original MAC network.

Methodology
===========

The quintessential goal of VQA is to extract question-relevant semantic
information from the images, which ranges from the detection of minute
details to the inference of abstract scene attributes for the whole
image, based on the question. The most common approach available is to
treat VQA as a classification problem. All methods proposed in the
literature consist of three steps: (i) extracting image features; (ii)
extracting question features; and (iii) combining these features to
produce a final answer. Most of these methods combine the features from
the image and question using simplistic mechanisms like concatenation,
element-wise multiplication, element-wise addition, etc., and then pass
this resultant to a classifier. Methods that compute spatial attention
maps for the visual features using the question features have also been
studied.

Our method takes inspiration from the works of \[Hudson and
Manning(2018)\], and [@b5]. The proposed approach decomposes problems
into a series of attention-based reasoning steps, each performed by a
special recurrent cell which maintains two separated hidden states:
control and memory. The control state indicates the reasoning operation
the cell should realize at a given step, while the memory state
maintains the long-term information learnt through the reasoning process
up to that step. By stacking together similar recurrent cells and
exploiting the interactions between the dual hidden states, the network
learns to perform iterative reasoning that is inferred directly from the
data.

The implementation was done in Python using the PyTorch framework [@b6].
The code is available on github at
http://github.com/ronilp/mac-network-pytorch-gqa.

The MAC Network
---------------

As has been conveyed before, the network architecture is based on ideas
presented in the original MAC paper [@b3]. The MAC network is an
end-to-end completely differentiable architecture capable of performing
a multi-step reasoning process by clamping together *p* MAC recurrent
cells. Each MAC cell is individually responsible for performing exactly
one reasoning step in the process. The model decomposes a VQA problem
set into up to *p* reasoning operations that interact with the knowledge
base (image), and iteratively aggregate useful information into the
network's memory state. The network consists of three steps (also shown
in Fig. [\[fig:MACnetwork\]](#fig:MACnetwork){reference-type="ref"
reference="fig:MACnetwork"}):

### Input Unit

The input unit transforms the incoming raw data into their respective
vectorial representations. The question string is converted into a
sequence of learned word embedding, which is then processed by a
bi-directional LSTM to produce contextual words (with respect to the
context of the question) and the question representation. These
embeddings were initialized using GloVe Wikipedia pre-trained model
[@b7]. Subsequently, for each reasoning step, a linear transformation
remodels the question into a position-aware vector representation
$q_{i}$, symbolizing the parts of the question important for the
$i^{th}$ reasoning step. The raw input image is first processed by a
Faster-RCNN based ResNet-101 pre-trained model to obtain the image's
vector representation.

### MAC Cell

The MAC cell is a recurrent cell that captures the fundamental reasoning
operation within the MAC network. Each cell maintains dual hidden
states, control and memory, to enforce attention and store intermediate
reasoning result, respectively. Much like as in any general computer
architecture system, each MAC cell has three operational units (also
shown in Fig. [2](#fig:MACcell){reference-type="ref"
reference="fig:MACcell"}):

![The MAC cell [@b3]](Images/MACcell.png){#fig:MACcell width="45%"}

1.  The control unit successively attends to different parts of the
    question, updating the control state at each reasoning step to
    characterize the reasoning operation a particular cell needs to
    perform.

2.  The read unit extracts the useful information out of the knowledge
    base (or, the image), while being supervised by the control state.

3.  The write unit integrates the information retrieved at the read unit
    into the memory state, thus yielding the new intermediate result.

The most important decision in the reasoning process is how to merge
together the visual and control features within a MAC cell. This
operation is carried out by the read unit and can be accomplished in a
number of ways suggested in the literature. Our approach uses a bilinear
fusion module (as shown in Fig.
[3](#fig:BilinearFusion){reference-type="ref"
reference="fig:BilinearFusion"}) for this operation. This module learns
extra parameters in form of $W$ and $b$ matrices, thus ensuring a rich
vectorial representation that can encode even the most complex
correlations between the question and the image.

![Bilinear Fusion](Images/BilinearFusion.png){#fig:BilinearFusion
width="45%"}

### Output Unit

The output unit predicts the final answer to the question by combining
together the question representation $q$ and the final memory state
$m_{p}$. Since there is a fixed set of possible answers for the GQA
dataset, the output unit passes the concatenation of $q$ and $m_{p}$
through a 2-layer fully-connected software classifier to obtain a
distribution over the candidate answers.

Dataset
-------

The data used for this study is the GQA dataset
(https://cs.stanford.edu/people/dorarad/gqa/index.html) which is a part
of the visual reasoning challenge for CVPR 2019. The GQA dataset
features compositional questions over real-world images and leverages
semantic representations of both the scenes and questions to mitigate
language priors and conditional biases. It consists of 113K images and
scene graphs, and 22M questions of assorted types and varying
compositionality degrees, measuring performance on an array of reasoning
skills such as object and attribute recognition, transitive relation
tracking, spatial reasoning, logical inference and comparisons. It has a
vocabulary size of 3097 words and 1878 possible answers [@b2]. Due to
computational limitations, only a $10\%$ subset of the GQA dataset is
used in this study.

Model Training and Hyperparameter Selection
===========================================

GQA provides a balanced train, test and val split of the 10% dataset and
we use the same to conduct our experiments. Due to the compute intensive
nature of this project, the model was trained on cloud GPUs on Google
Cloud Platform. We used servers with four Nvidia Tesla P100 GPUs, 16
CPUs and 60 GB of RAM. The total dataset size was 120 GB after
pre-processing images and extracting object features for 100 proposals
with ResNet-101 using Faster-RCNN.\
The training time is proportional to the number of reasoning steps we
make in the model. Each reasoning step means addition of another MAC
unit in the core-recurrent part. We treated the number of reasoning
steps as a hyper-parameter and tried several values for it. The results
were better for higher values (10-12 steps) but the training time was
significantly higher. We found that for the questions in GQA dataset,
even four reasoning steps can provide decent results while keeping the
training time reasonable.

::: {#tab:hyper}
       **Hyper-parameter**            **Range of Values**       **Final Value**
  ------------------------------ ----------------------------- -----------------
      No. of Reasoning Steps            2, 4, 6, 8, 12                 4
    Size of embeddings for q/a           50, 100, 300                 300
       Dropout probability                   0.15                    0.15
          Learning Rate           $10^{-5}, 10^{-4}, 10^{-3}$      $10^{-4}$
              Epochs                      10, 20, 25                  25
            Batch Size                       1024                    1024
   Multi-modal hidden dimension          50, 150, 300                 50

  : Model Structure and Hyperparameters
:::

[\[tab:hyper\]]{#tab:hyper label="tab:hyper"}

                     **Question**                            **Ground truth answer**           **Single word answer**   **Model Prediction**
  -------------------------------------------------- ---------------------------------------- ------------------------ ----------------------
       Which part of the photo is the bike in?        The bike is on the right of the image.           right                   right
     Are there both bikes and bags in the image?         No, there is a bike but no bags.                no                     yes
           Is there a truck on the street?             Yes, there is a truck on the street.             yes                     yes
   On which side of the image is the white vehicle?   The truck is on the left of the image.            left                    left

[\[tab:pred\]]{#tab:pred label="tab:pred"}

We had to introduce another important hyper-parameter because of our
implementation of Bilinear Fusion. The bilinear fusion module in our
network requires learning a weight matrix. Since the length of the
question vector and object feature vector are both 2048, the weight
matrix to be learnt becomes huge introducing 2048 x 2048 extra
parameters. To reduce the number of extra hyper-parameters, we
introduced two linear layers for down-scaling and up-scaling the
vectors. The hidden dimension to which we scale down the vectors is an
important hyper-parameter for learning the multi-modal fusion. We tried
several values of the hidden multi-modal dimension and values over 300
resulted in GPU going out of memory. 50 hidden dimensions worked well
for our case.

We trained the model for 25 epochs. It took about 24 hrs to train the
model on this setup.

Table [1](#tab:hyper){reference-type="ref" reference="tab:hyper"}
illustrates the final selected values for each hyperparameter. Figures 5
and 6 display the accuracy and loss curves for the proposed model.

![Plot showing Accuracy
values](Images/Accuracy_Curve.jpg){#fig:AccCurves width="45%"}

![Plot showing Loss values](Images/Loss_Curve.jpg){#fig:LearningCurves
width="45%"}

Evaluation
==========

Fig. 8 shows a sample image from the GQA dataset. Table III shows the
relevant questions from GQA for the image. A comparison of Ground truth
single-word answers and predictions made by Bilinear Fusion model is
shown in the table.

The standard accuracy metric is not sufficient to evaluate Visual
Question Answering models. Accuracy alone does not account for a range
of anomalous behaviors that models demonstrate, such as ignoring key
question words or attending to irrelevant image regions. Researchers
have argued for the need to devise new evaluation measures and
techniques to shed more light on systems' inner workings.

![A real world image from GQA
dataset](Images/GroundVsPred.jpg){#fig:GroundPred width="45%"}

The model was evaluated using CVPR 2019's GQA challenge evaluation
criterion. The following metrics were used for evaluation:

-   **Accuracy** - Standard accuracy. For each question-answer pair
    $(q,a)$, we give $1$ point if the predicted answer $p$ matches $a$
    and $0$ otherwise, and average over all questions in the dataset.

-   **Consistency** - A metric for the level of consistency in responses
    across different questions. For each question-answer pair $(q,a)$,
    we define a set $E_q={q_1, q_2, ..., q_n}$ of entailed questions,
    the answers to which can be unambiguously inferred given $(q,a)$.

-   **Validity** - Measures whether the model gives valid answers, ones
    that can be theoretically correct for the question. For each
    question $q$ we define an answers scope $V$, with all answers
    possibly correct for some image.

-   **Plausibility** - Measures whether the model responses are
    reasonable in the real world or not making sense. For each open
    question $q$ about an attribute $a$ of an object o we check whether
    the model's prediction $p$ occurs at least once with the object o
    over the whole dataset scene graphs. We give $1$ point if it occurs
    and $0$ otherwise.

-   **Distribution** - Measures the overall match between the true
    answer distribution and the model predicted distribution. We
    partition the questions in the dataset into groups based on their
    subject, e.g. (apple,color) for all color questions about apples,
    etc. For each group $G$, we look at the true answer distribution
    across the dataset and the distribution of the model's predictions
    over the same questions. We then compare these distributions using
    *Chi-Square statistic*, and average the scores across all groups.

More details on these metrics can be found in [@b2].

Figure [7](#fig:TestCurves){reference-type="ref"
reference="fig:TestCurves"} shows the accuracy curves for our model and
the state-of-the-art MAC network on the test dataset. Table
[2](#tab:metrics){reference-type="ref" reference="tab:metrics"} displays
the comparison between the two models as well as humans using the
above-defined metrics.

![Test Accuracy for Bilinear Fusion Model and the current
state-of-the-art Model](Images/MAC_vs_Ours.jpg){#fig:TestCurves
width="45%"}

::: {#tab:metrics}
    **Metric**    **Bilinear Fusion**   **MAC Network**   **Humans**
  -------------- --------------------- ----------------- ------------
     Accuracy            52.08               53.94           89.3
   Consistency           85.90               81.59           98.4
     Validity            92.12               96.16           98.9
   Plausibility          87.56               84.48           97.2
   Distribution          6.22                5.34             \-

  : Test results on various evaluation metrics
:::

[\[tab:metrics\]]{#tab:metrics label="tab:metrics"}

As is apparent from the [2](#tab:metrics){reference-type="ref"
reference="tab:metrics"}, our model nearly matches the performance of
the state of the art MAC model on 10% GQA data. It is believed that our
model could outperform the MAC following an extensive ablation study. We
further plan to extend the proposed model as follows:

1.  Bilinear fusion at every reasoning step leads to a steep increase in
    the number of parameters, which might cause overfitting. This can be
    prevented by using some form of regularization such as dropout.

2.  Skip connections can be used in order to prevent detrimental loss of
    input image and question information.

00 Kafle, K., Kanan, C. 2016. Visual Question Answering: Datasets,
Algorithms, and Future Challenges. arXiv e-prints arXiv:1610.01465.
Hudson, D. A., Manning, C. D. 2019. GQA: a new dataset for compositional
question answering over real-world images. arXiv e-prints
arXiv:1902.09506. Hudson, D. A., Manning, C. D. 2018. Compositional
Attention Networks for Machine Reasoning. arXiv e-prints
arXiv:1803.03067. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J.,
Jones, L., Gomez, A.N., Kaiser, L., and Polosukhin, 2017.  Attention Is
All You Need. arXiv e-prints arXiv:1706.03762. Cadene, R., Ben-younes,
H., Cord, M., Thome, N. 2019. MUREL: Multimodal Relational Reasoning for
Visual Question Answering. arXiv e-prints arXiv:1902.09487. Paszke,
Adam, Sam Gross, Soumith Chintala, Gregory Chanan, Edward Yang, Zachary
DeVito, Zeming Lin, Alban Desmaison, Luca Antiga and Adam Lerer.
"Automatic differentiation in PyTorch." (2017). Pennington, Jeffrey &
Socher, Richard & Manning, Christoper. (2014). Glove: Global Vectors for
Word Representation. EMNLP. 14. 1532-1543. 10.3115/v1/D14-1162.