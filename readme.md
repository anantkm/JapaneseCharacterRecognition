
<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">
<HTML>
<BODY LANG="EN">
<h2 align=center>COMP9444 Neural Networks and Deep Learning</h2>
<h2 align=center>Project 1</h2>
<H3 align=center>Japanese Characters and Intertwined Spirals</H3>
In this assignment, you will be implementing and
training various neural network models for
two different classification tasks, and analysing the results.
<p>
You are to submit two Python files
<code>kuzu.py</code> and <code>spiral.py</code>,
as well as a written report <code>hw1.pdf</code>
(in <code>pdf</code> format).

<h3 id="marking-scheme">Japanese Character Recognition</h3>

In this part of the assignment you will be implementing networks to recognize handwritten
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
<ol>
<li>Implement a model <code>NetLin</code>
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
<li>Implement a fully connected 2-layer network
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
<li>Implement a convolutional network called
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
<br>
<HR>
</BODY>
</HTML>
