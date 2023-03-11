<img src="./img/dsd.png" hspace="10%" width="80%">

## DSD: Dense-Sparse-Dense Training for Deep Neural Networks

Unofficial implementation of <a href="https://arxiv.org/abs/1607.04381">DSD: Dense-Sparse-Dense Training for Deep Neural Networks</a> in Pytorch.

<img src="./img/weight-distribution.png" hspace="3%" width="90%">

> Dense-sparse-dense training framework regularizes neural networks by pruning and then restoring connections. The paper method learns which connections are important during the initial dense solution. Then it regularizes the network by pruning the unimportant connections and retraining to a sparser and more robust solution with same or better accuracy. Finally, the pruned connections are restored and the entire network is retrained again. This increases the dimensionality of parameters, and thus model capacity, from the sparser model.

## VGG-Face fine tuning with DSD for Facial Expression Recognition on FER2013
> In `./src/fer2013_dsd_pytorch.ipynb` notebook is unofficial implementation of fine tunning of [VGG-Face](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/) model for facial expression recognition problem, described in [Local Learning with Deep and Handcrafted Features for Facial Expression Recognition](https://arxiv.org/pdf/1804.10892v7.pdf), which uses DSD trainig method. 

### Weights and Dataset
> You can download VGG-Face model (converted to .pth) [here](https://drive.google.com/file/d/1XUbKVw9QGkzyu4wQe6M-pLehTssLRHqP/view?usp=sharing). Locate it in `./src/pretrained/` like `./src/pretrained/VGG_FACE_converted.pth`.
> Splitted [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) dataset can be downloaded [here](https://drive.google.com/file/d/1PSSPHgJ6Te1_eyvYv8i1JpkZnU-Bhe8h/view?usp=sharing). Extract it and locate `train.csv`, `test.csv` and `val.csv` files in `./src/data`. 
## Citations

To cite authors of papers:

```bibtex
@misc{1607.04381,
    Author = {Song Han and Jeff Pool and Sharan Narang and Huizi Mao and Enhao Gong and Shijian Tang and Erich Elsen and Peter Vajda and Manohar Paluri and John Tran and Bryan Catanzaro and William J. Dally},
    Title = {DSD: Dense-Sparse-Dense Training for Deep Neural Networks,
    Year = {2016},
    Eprint = {arXiv:1607.04381},
}

@InProceedings{Parkhi15,
  author       = "Omkar M. Parkhi and Andrea Vedaldi and Andrew Zisserman",
  title        = "Deep Face Recognition",
  booktitle    = "British Machine Vision Conference",
  year         = "2015",
}

@article{DBLP:journals/corr/abs-1804-10892,
  author    = {Mariana{-}Iuliana Georgescu and Radu Tudor Ionescu and Marius Popescu},
  title     = {Local Learning with Deep and Handcrafted Features for Facial Expression Recognition},
  journal   = {CoRR},
  volume    = {abs/1804.10892},
  year      = {2018},
  url       = {http://arxiv.org/abs/1804.10892},
  eprinttype = {arXiv},
  eprint    = {1804.10892},
  timestamp = {Mon, 13 Aug 2018 16:48:55 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-1804-10892.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
