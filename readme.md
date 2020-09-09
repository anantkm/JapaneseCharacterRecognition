
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<HTML>
<HEAD>
<TITLE>COMP9444 Project 1</TITLE>
</HEAD>
<BODY LANG="EN">
<h2 align=center>COMP9444 Neural Networks and Deep Learning</h2>
<h2 align=center>Term 2, 2020</h2>
<H3 align=center>Project 1 - Japanese Characters and Intertwined Spirals</H3>
<p align=center>
Due: Sunday 12 July, 23:59 pm
<br align=center>
Marks: 30% of final assessment
<p>
In this assignment, you will be implementing and
training various neural network models for
two different classification tasks, and analysing the results.
<p>
You are to submit two Python files
<code>kuzu.py</code> and <code>spiral.py</code>,
as well as a written report <code>hw1.pdf</code>
(in <code>pdf</code> format).
<p>
<h3 id="provided-files">Provided Files</h3>
<p>Copy the archive <a href="./hw1.zip"><code>hw1.zip</code></a> into your own filespace and unzip it. This should create a directory <code>hw1</code>
with the data file <code>spirals.csv</code>
as well as four Python files
<code>kuzu.py</code>, <code>spiral.py</code>,
<code>kuzu_main.py</code> and <code>spiral_main.py</code>.
<p>Your task is to complete the skeleton files 
<code>kuzu.py</code>, <code>spiral.py</code> and submit them,
along with your report.
<p>

<h3 id="marking-scheme">Part 1: Japanese Character Recognition</h3>

For Part 1 of the assignment you will be implementing networks to recognize handwritten
Hiragana symbols. The dataset to be used is Kuzushiji-MNIST or KMNIST
for short. The paper describing the dataset is available
<a href="https://arxiv.org/pdf/1812.01718.pdf">here</a>.
It is worth reading, but in short: significant changes occurred to the
language when Japan reformed their education system in 1868, and the
majority of Japanese today cannot read texts published over 150 years
ago. This paper presents a dataset of handwritten, labeled examples of
this old-style script (Kuzushiji). Along with this dataset, however,
they also provide a much simpler one, containing 10 Hiragana
characters with 7000 samples per class. This is the dataset we will be
using.
<figure>
<img src="modern_vs_old.jpg" alt="" />
<figcaption>
Text from 1772 (left) compared to 1900 showing the standardization of written Japanese.
</figcaption>
</figure>
<ol>
<li> [1 mark] Implement a model <code>NetLin</code>
which computes a linear function of the pixels in the image,
followed by log softmax.
Run the code by typing:
<pre>
python3 kuzu_main.py --net lin
</pre>
Copy the final accuracy and confusion matrix into your report.
Note that the <strong>columns</strong> of the confusion matrix indicate the target character,
while the <strong>rows</strong> indicate the one chosen by the network.
(0="o", 1="ki", 2="su", 3="tsu", 4="na", 5="ha", 6="ma", 7="ya", 8="re", 9="wo").
More examples of each character can be found
<a href="http://codh.rois.ac.jp/kmnist/index.html.en">here</a>.
<p>
<li> [2 marks] Implement a fully connected 2-layer network
<code>NetFull</code>,
using tanh at the hidden nodes and log softmax at the output node.
Run the code by typing:
<pre>
python3 kuzu_main.py --net full
</pre>
Try different values (multiples of 10) for the number of
hidden nodes and try to determine a value that achieves
high accuracy on the test set.
Copy the final accuracy and confusion matrix into your report.
<p>
<li> [2 marks] Implement a convolutional network called
<code>NetConv</code>, with two convolutional layers
plus one fully connected layer, all using relu activation function,
followed by the output layer.
You are free to choose for yourself the number and size
of the filters, metaparameter values,
and whether to use max pooling or
a fully convolutional architecture.
Run the code by typing:
<pre>
python3 kuzu_main.py --net conv
</pre>
Your network should consistently achieve at least 93% accuracy
on the test set after 10 training epochs.
Copy the final accuracy and confusion matrix into your report.
<p>
<li> [7 marks]
Discuss what you have learned from this exercise,
including the following points:
<ol type="a">
<li> the relative accuracy of the three models,
<li> the confusion matrix for each model: which characters
are most likely to be mistaken for which other characters, and why?
<li> you may wish to experiment with other architectures and/or
metaparameters for this dataset, and report on your results;
the aim of this exercise is not only to achieve high accuracy
but also to understand the effect of different choices on the
final accuracy.
</ol>
</ol>

<h3 id="marking-scheme">Part 2: Twin Spirals Task</h3>
<figure>
<img src="spirals.jpg" height="384">
</figure>
<p>
For Part 2 you will be training on the famous
Two Spirals Problem (Lang and Witbrock, 1988).
The supplied code
<code>spiral_main.py</code>
loads the training data from
<code>spirals.csv</code>,
applies the specified model and produces a graph of the resulting function,
along with the data.
For this task there is no test set as such,
but we instead judge the generalization
by plotting the function computed by the network
and making a visual assessment.
<ol>
<li> [2 marks]
Provide code for a Pytorch Module called
<code>PolarNet</code>
which operates as follows:
First, the input <code>(x,y)</code> is converted
to polar co-ordinates <code>(r,a)</code> with
<code>r=sqrt(x*x + y*y)</code>, <code>a=atan2(y,x)</code>.
Next, <code>(r,a)</code> is fed into a 
fully connected neural network with one hidden layer using <code>tanh</code>
activation, followed by a single output using <code>sigmoid</code>
activation. The conversion to polar coordinates should be
included in your <code>forward()</code> method, so that the
Module performs the entire task of conversion followed by
network layers.
<p>
<li> [1 mark]
Run the code by typing
<pre>
python3 spiral_main.py --net polar --hid 10
</pre>
Try to find the minimum number of hidden nodes required
so that this PolarNet
learns to correctly classify all of the training data
within 20000 epochs, on almost all runs.
The <code>graph_output()</code> method will generate a picture of the function
computed by your PolarNet called <code>polar_out.png</code>,
which you should include in your report.
<p>
<li> [1 mark]
Provide code for a Pytorch Module called
<code>RawNet</code>
which operates on the raw input <code>(x,y)</code>
without converting to polar coordinates.
Your network should consist of two fully connected hidden layers
with tanh activation, plus the output layer, with sigmoid activation.
You should <b>not</b> use <code>Sequential</code> but should instead
build the network from individual components as shown
in the program <code>xor.py</code> from Exercises 5
(repeated in slide 4 of lecture slides 3b on PyTorch).
The number of neurons in both
hidden layers should be determined by the parameter <code>num_hid</code>.
<p>
<li> [1 mark]
Run the code by typing
<pre>
python3 spiral_main.py --net raw
</pre>
Keeping the number of hidden nodes in each layer fixed at 10,
try to find a value for the size of the initial weights (--init)
such that this RawNet learns to correctly classify all of the training data
within 20000 epochs, on almost all runs.
Include in your report the number of hidden nodes,
and the values of any other metaparameters.
The <code>graph_output()</code> method will generate a picture of the function
computed by your RawNet called <code>raw_out.png</code>,
which you should include in your report.
<p>
<li> [1 mark]
Provide code for a Pytorch Module called
<code>ShortNet</code>
which again operates on the raw input <code>(x,y)</code>
without converting to polar coordinates.
This network should again consist of two hidden layers (with tanh activation)
plus the output layer (with sigmoid activation),
but this time should include short-cut connections
between every pair of layers (<code>input, hid1, hid2</code> and <code>output</code>)
as depicted on slide 10 of lecture slides 3a on Hidden Unit Dynamics.
Note, however that this diagram shows only two hidden nodes in each
layer, which is not enough to learn the task;
in your code the number of neurons in both
hidden layers should be determined by the parameter <code>num_hid</code>.
<p>
<li> [1 mark]
Run the code by typing
<pre>
python3 spiral_main.py --net short
</pre>
You should experiment to find a good value for the initial weight size,
and try to find the mininum number of hidden nodes per layer
so that this ShortNet
learns to correctly classify all of the training data
within 20000 epochs, on almost all runs.
Include in your report the number of hidden nodes per layer,
as well as the initial weight size and any other metaparameters.
The <code>graph_output()</code> method will generate a picture of the function
computed by your ShortNet called <code>short_out.png</code>,
which you should include in your report.
<p>
<li> [2 marks]
Using <code>graph_output()</code> as a guide, write a method called
<code>graph_hidden(net, layer, node)</code>
which plots the activation
(after applying the <code>tanh</code> function) of
the hidden node with the specified number <code>(node)</code>
in the specified <code>layer</code> (1 or 2).
(Note: if <code>net</code> is of type <code>PolarNet</code>,
<code>graph_output()</code> only needs to behave correctly when layer is 1).
<p>
Hint: you might need to modify <code>forward()</code>
so that the hidden unit activations are retained, i.e.
replace <code>hid1 = torch.tanh(...)</code> with
<code>self.hid1 = torch.tanh(...)</code>
<p>
Use this code to generate plots of all the hidden nodes
in PolarNet, and all the hidden nodes in both layers of
RawNet and ShortNet,
and include them in your report.
<p>
<li> [9 marks]
Discuss what you have learned from this exercise,
including the following points:
<ol type="a">
<li> the qualitative difference between the functions
computed by the hidden layer nodes of
the three models, and a brief description of how
the network uses these functions to achieve the classification
<li> the effect of different values for initial weight size
on the speed and success of learning, for both RawNet and ShortNet
<li> the relative "naturalness" of the output function computed
by the three networks, and the importance of representation
for deep learning tasks in general
<li> you may like to also experiment with other changes
and comment on the result - for example,
changing batch size from 97 to 194,
using SGD instead of Adam,  
changing tanh to relu, adding a third hidden layer, etc.
</ol>

</ol>

<h3 id="submission">Submission</h3>
<p>You should submit by typing</p>
<pre>
give cs9444 hw1 kuzu.py spiral.py hw1.pdf
</pre>
<p>You can submit as many times as you like - later submissions will overwrite earlier ones. You can check that your submission has been received by using the following command:</p>
<p>9444 classrun -check</p>
<p>The submission deadline is Sunday 12 July, 23:59. 15% penalty will be applied to the (maximum) mark for every 24 hours late after the deadline.</p>
<p>Additional information may be found in the <a href=./faq.shtml>FAQ</a> and will be considered as part of the specification for the project. You should check this page regularly.</p>

<h3 id="plagiarism-policy">Plagiarism Policy</h3>
<p>Group submissions will not be allowed for this assignment. Your program must be entirely your own work. Plagiarism detection software will be used to compare all submissions pairwise and serious penalties will be applied, particularly in the case of repeat offences.</p>
<p>
<b>DO NOT COPY FROM OTHERS; DO NOT ALLOW ANYONE TO SEE YOUR CODE</b>
<p>
Please refer to the
<a href="https://student.unsw.edu.au/plagiarism">UNSW Policy on Academic Integrity and Plagiarism</a>
if you require further clarification on this matter.
<P>
<P>
Good luck!
<br>
<HR>
</BODY>
</HTML>
