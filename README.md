# English-To-German Neural Machine Translation Using Transformer


 <div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/neural-machine-translation/blob/master/EN_DE_Neural_Machine_Translation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
 </div>


Neural Machine Translation (NMT) is a family model or an approach to solving machine translation problems through an artificial neural network, typically deep learning. In other words, the model is dispatched to translate a sequence of words from the source language to the target language. In this case, the source language would be English and the target would be German. To fabricate the model, the Transformer layers are leveraged. The NMT model is trained on the Multi30K dataset. The model is then assessed on a subset of the dataset, which is the Flickr 2016 test dataset.


## Experiment


Follow this [link](https://github.com/reshalfahsi/neural-machine-translation/blob/master/EN-DE_Neural_Machine_Translation.ipynb) to play along and explore the NMT model.


## Result

## Quantitative Result

The performance of the model in terms of cross-entropy loss and translation edit rate (TER) on the test dataset.

Metrics | Score |
------------ | ------------- |
Loss | 1.951 |
TER | 0.811 |


## Loss Curve

<p align="center"> <img src="https://github.com/reshalfahsi/neural-machine-translation/blob/master/assets/loss_curve.png" alt="loss_curve" > <br /> Loss of training and validation versus the epochs. </p>

## Qualitative Result

Here, the NMT model's qualitative performance is associated with the Transformer's attention maps.

<p align="center"> <img src="https://github.com/reshalfahsi/neural-machine-translation/blob/master/assets/qualitative_result.png" alt="qualitative_result"> <br /> The attention maps from each of the Transformer's heads. Almost every corresponding word pair (English-German) at each head pays attention mutually. </p>


## Credit

- [Language Translation With TorchText](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html)
- [6 - Attention is All You Need Notebook Tutorial](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)
- [Multi30K Dataset](https://github.com/multi30k/dataset)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Self-Attention and Positional Encoding](https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/)
