# Universal linguistic inductive biases via meta-learning

This is the code for the paper "Universal linguistic inductive biases via meta-learning" by [Tom McCoy](http://rtmccoy.com/), [Erin Grant](https://eringrant.github.io/), [Paul Smolensky](https://www.microsoft.com/en-us/research/people/psmo/), [Tom Griffiths](http://cocosci.princeton.edu/tom/index.php), and [Tal Linzen](https://tallinzen.net/). The paper can be found here, with an accompanying demo [here](http://rtmccoy.com/meta-learning-linguistic-biases.html).

Below are instructions for running our code. All scripts should be run in the `src/` directory.

## Dependencies

This code requires [PyTorch](https://pytorch.org/get-started/locally/). We used version 1.0.0 for our experiments, but the code should also be compatible with newer versions.

## Mini example

To run a small, simple example of meta-learning, run the following two lines of code:

```
python make_tasks_cv.py
python cv_trainer.py
```

The first line creates the data to use for meta-training, then the second line executes the meta-training. The second line will output the model's few-shot accuracy on the meta-dev set after every 100 meta-training examples; if everything is working correctly, you should see such an output roughly every 5 or 10 minutes, and the accuracy should be increasing as time goes on. 


## Overview of this repo

The models that we evaluate in the paper are in `saved_weights/`, while keys that can be used to generate our meta-training and evaluation sets are in `data/`. All of our code is in `src/`; here is a description of the files there:
- `cv_trainer.py`: For training a simple example
- `evaluation.py`: Functions for evaluation
- `input_output_correspondences.py`: Determines what outputs should be given for given types of inputs
- `load_data.py`: Functions for loading data
- `main.py`: For meta-training models
- `make_tasks.py`: For generating meta-training, meta-validation, and meta-test tasks from keys.
- `make_tasks_cv.py`: For generating tasks to use in the mini example
- `make_tasks_imp.py`: For generating certain evaluation sets
- `mix_input_outputs.py`: Tools for generating some of the evaluation sets
- `models.py`: Definitions of the PyTorch models we use
- `phonology_task_creation.py`: Functions for use in generating tasks
- `train_dev_test_split.py`: Creating meta-training, meta-validation, and meta-test splits that do not overlap
- `training.py`: Training and meta-training models
- `utils.py`: Miscellaneous functions

The rest of this README describes how to reproduce the experiments described in our paper. Some of the code is based on [this blog post](https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0).

## Part 1: Dataset generation

The first step is to generate the sets of languages that will be used for meta-training, meta-validation, and meta-evaluation. These scripts have already been run for you: their results are in `io_correspondences/` and `data/`. We provide these scripts in case you wish to replicate our experiments from scratch, but otherwise you can skip Step 1.

1. Create files specifying input-output correspondences for each constraint set. For "YesOnset, NoCoda" we further generate an additional file expanding out to input length 6. These files specify the shape of the output for a given input shape; e.g., CVC -> .CV.CV.
```
python input_output_correspondences.py --onset yes --coda yes --prefix yo_yc_ --max_input_length 5
python input_output_correspondences.py --onset yes --coda no --prefix yo_nc_ --max_input_length 5
python input_output_correspondences.py --onset no --coda yes --prefix no_yc_ --max_input_length 5
python input_output_correspondences.py --onset no --coda no --prefix no_nc_ --max_input_length 5
python input_output_correspondences.py --onset yes --coda no --prefix yo_nc_ --max_input_length 6
```

2. Create keys that will be used to generate non-overlapping meta-training languages, meta-dev languages, and meta-test languages. Each key specifies on language, and they are generated such that all generated languages are unique both within and across the meta-training, meta-development, and meta-test set. Saving these keys is much more memory-efficient than saving the full datasets. We generate one set of keys for each of the four constraint sets.
```
python train_dev_test_split.py --output_prefix yonc --constraints yonc
python train_dev_test_split.py --output_prefix yoyc --constraints yoyc
python train_dev_test_split.py --output_prefix noyc --constraints noyc
python train_dev_test_split.py --output_prefix nonc --constraints nonc
```

## Part 2: Expanding keys out into datasets
Step 1 generated keys for datasets; in this step, we expand those keys into datasets. Each of these lines of code should generate a meta-training set in `data/PREFIX.train`, a meta-dev set in `data/PREFIX.dev`, and a meta-test set in `data/PREFIX.test`, where `PREFIX` is given by the `--output_prefix` argument. This works by expanding out the keys in the `data/` directory into full datasets.

### (A) Tasks used for meta-training and meta-testing:
```
python make_tasks.py --ranking_prefix yonc --output_prefix yonc --constraints yonc
```

### (B) Tasks used for ease-of-learning evaluations with different constraint sets:
```
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --ranking_prefix yonc --output_prefix yonc_10per --constraints yonc
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --ranking_prefix yoyc --output_prefix yoyc_10per --constraints yoyc
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --ranking_prefix noyc --output_prefix noyc_10per --constraints noyc
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --ranking_prefix nonc --output_prefix nonc_10per --constraints nonc
```

### (C) Tasks used for ease-of-learning evaluations with inconsistent constraint rankings or constraint sets
```
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --ranking_prefix yonc --output_prefix yonc_shuffle_aio --constraints yonc --aio_shuffle yo_nc_io_correspondences.txt
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --output_prefix yoyc_shuffle_aio --ranking_prefix yoyc --constraints yoyc --aio_shuffle yo_yc_io_correspondences.txt
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --output_prefix noyc_shuffle_aio --ranking_prefix noyc --constraints noyc --aio_shuffle no_yc_io_correspondences.txt
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --output_prefix nonc_shuffle_aio --ranking_prefix nonc --constraints nonc --aio_shuffle no_nc_io_correspondences.txt
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --ranking_prefix yonc --output_prefix all_shuffle_aio --constraints yonc --aio_shuffle no_nc_io_correspondences.txt,no_yc_io_correspondences.txt,yo_nc_io_correspondences.txt,yo_yc_io_correspondences.txt
```

### (D) Tasks used for the poverty-of-the-stimulus evaluation with all new phonemes
```
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --ranking_prefix yonc --constraints yonc --output_prefix pos_new_phonemes --test_all_new True --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10
```

### (E) Tasks used for the poverty-of-the-stimulus evaluation of generalization to a novel length
```
python make_tasks_imp.py --n_train 20000 --n_dev 500 --n_test 1000 --output_prefix pos_length --constraints yonc --implication_type length
python make_tasks_imp.py --n_train 20000 --n_dev 500 --n_test 1000 --output_prefix pos_length6 --constraints yonc --implication_type length6
```

### (F) Tasks for the poverty-of-the-stimulus evaluation with implicational universals
```
python make_tasks_imp.py --n_train 20000 --n_dev 500 --n_test 1000 --output_prefix pos_imp --constraints yonc --implication_type one
```


## Part 3: Create model initializations randomly or with meta-learning

In this step, we create the two model initializations that we perform our evaluations on. You can skip this step if you want, because the exact initializations that we used in the paper have been provided under `models/`. Be aware that running meta-training can take a long time (several days).

1. Meta-train a model using `yonc.train` as the meta-training set, `yonc.dev` as the meta-dev set, and `yonc.test` as the meta-test set, then save its weights to `models/maml_yonc_256_5.weights`.
```
python main.py --data_prefix yonc --method maml --save_prefix maml_yonc_256_5
```

2. Save a randomly-initialized model's weights to `models/random_yonc_256_5`.
```
python main.py --data_prefix yonc --method random --save_prefix random_yonc_256_5
```

## Part 4: Evaluation

Finally, we evaluate our two models. You may not get exactly the same numbers as what we report here due to the fact that we ran our evaluations on GPUs whose operations are nondeterministic; but the results should be close.

### 100-shot learning evaluation
These are the results discussed on pg. 4 of the paper, second column, paragraph headed "meta-learning results".

1. 100-shot results for meta-initialized model (paper reports 98.8%):
```
python evaluation.py --data_prefix yonc --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 1.0 --inner_batch_size 100 --save_prefix maml_yonc_256_5
```

2. 100-shot results for randomly-initialized model (paper reports 6.5%):
```
python evaluation.py --data_prefix yonc --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 1.0 --inner_batch_size 100 --save_prefix random_yonc_256_5
```

### Ease of learning: Set of constraints
These are the results discussed pg. 4 of the paper, second column, first 4 paragraphs under "Ease of Learning"; and also in Figure 4 (top), Figure 5, and Figure 6a.

1. Meta-initialized model, average number of examples needed to learn YesOnset/NoCoda: 203.75 examples
```
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

2. Meta-initialized model, average number of examples needed to learn NoOnset/NoCoda: 791.11 examples
```
python evaluation.py --data_prefix nonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

3. Meta-initialized model, average number of examples needed to learn YesOnset/YesCoda: 1112.0 examples
```
python evaluation.py --data_prefix yoyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

4. Meta-initialized model, average number of examples needed to learn NoOnset/YesCoda: 1505.0 examples
```
python evaluation.py --data_prefix noyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

5. Randomly-initialized model, average number of examples needed to learn YesOnset/NoCoda: 20642.5 examples
```
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95 
```

6. Randomly-initialized model, average number of examples needed to learn NoOnset/YesCoda: 25577.67 examples
```
python evaluation.py --data_prefix nonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

7. Randomly-initialized model, average number of examples needed to learn YesOnset/YesCoda: 25447.8 examples
```
python evaluation.py --data_prefix yoyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

8. Randomly-initialized model, average number of examples needed to learn NoOnset/YesCoda: 24647.38 examples
```
python evaluation.py --data_prefix noyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

### Ease of learning: Consistent vs. inconsistent constraint ranking
This is the experiment discussed on pg. 5 of the paper, first column, first 2 full paragraphs; and also in Fig. 4 (bottom left) and Fig. 6b.

1. Meta-initialized model, average number of examples needed to learn a consistent constraint ranking: 203.75 examples (this is the same line of code as the first one in the previous section):
```
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

2. Meta-initialized model, average number of examples needed to learn an inconsistent constraint ranking: 19803.0 examples
```
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

3. Randomly-initialized model, average number of examples needed to learn a consistent constraint ranking: 20642.5 examples (this is the same line of code as one in the previous section)
```
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

4. Randomly-initialized model, average number of examples needed to learn an inconsistent constraint ranking: 74918.5 examples
```
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

### Ease of learning: Consistent vs. inconsistent set of constraints 
This is the experiment discussed on pg. 5 of the paper, last paragraph in first column continuing into second column; and also in Fig. 4 (bottom right) and Fig. 6c.

1. Meta-initialized model, consistent constraint set: The number reported in the paper is the average of the 4 numbers below (which is 23,248 examples).

Average number of examples needed to learn a language with the consistent constraints YesOnset and NoCoda: 19803.0 examples
```
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

Average number of examples needed to learn a language with the consistent constraints NoOnset and NoCoda: 8721.0 examples
```
python evaluation.py --data_prefix nonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

Average number of examples needed to learn a language with the consistent constraints YesOnset and YesCoda: 29866.0 examples
```
python evaluation.py --data_prefix yoyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

Average number of examples needed to learn a language with the consistent constraints NoOnset and YesCoda: 34603.875 examples
```
python evaluation.py --data_prefix noyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

2. Meta-initialized model, average number of examples needed to learn a language with an inconsistent constraint set: 33135.125 examples
```
python evaluation.py --data_prefix all_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

3. Randomly-initialized model, consistent constraint set: The number reported in the paper is the average of the 4 numbers below (which is 78,282 examples).

Average number of examples needed to learn a language with the consistent constraints YesOnset and NoCoda: 74918.5 examples
```
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

Average number of examples needed to learn a language with the consistent constraints NoOnset and NoCoda: 72906.56 examples
```
python evaluation.py --data_prefix nonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
Average number of examples needed to learn a language with the consistent constraints YesOnset and YesCoda: 84942.1 examples
```
python evaluation.py --data_prefix yoyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
Average number of examples needed to learn a language with the consistent constraints NoOnset and YesCoda: 80360.75 examples
```
python evaluation.py --data_prefix yoyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

4. Randomly-initialized model, average number of examples needed to learn a language with an inconsistent constraint set: 87370.63 examples
```
python evaluation.py --data_prefix all_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

### Poverty of the stimulus: All new phonemes 
This is the experiment reported on pg. 5 of the paper, 2nd column; and also in Fig. 7, left.

1. Meta-initialized model: Accuracy on unseen types: 0.837
```
python evaluation.py --data_prefix pos_new_phonemes --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

2. Randomly-initialized model: Accuracy on unseen types: 0.045 
```
python evaluation.py --data_prefix pos_new_phonemes --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
### Poverty of the stimulus: Length 5 
This is the experiment reported on pg. 6 of the paper, first full paragraph; and also in Fig 7, middle.

1. Meta-initialized model: Accuracy on length 5: 0.897
```
python evaluation.py --data_prefix pos_length --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

2. Meta-initialized model: Accuracy on length 6: 0.504
```
python evaluation.py --data_prefix pos_length6 --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

3. Randomly-initialized model: Accuracy on length 5: 0.59
```
python evaluation.py --data_prefix pos_length --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

### Poverty of the stimulus: Implicational universals 
This is the experiment reported on pg. 6 of the paper, last paragraph before the conclusion; and also in Fig. 7, right.

1. Meta-initialized model: Accuracy on unseen example type: 0.926
```
python evaluation.py --data_prefix pos_imp --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100 --eval_technique converge --threshold 0.95
```
2. Randomly-initialized model: Accuracy on unseen example type" 0.100
```
python evaluation.py --data_prefix pos_imp --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100 --eval_technique converge --threshold 0.95
```  



## Abbreviations and shorthand used in the code
Numerical codes for constraints:
- 0: Onset
- 1: NoCoda
- 2: NoDeletion
- 3: No Insertion

Constraint sets:
- nonc: NoOnset, NoCoda
- noyc: NoOnset, YesCoda
- yonc: YesOnset, NoCoda (default constraints)
- yoyc: YesOnset, YesCoda

## License

This code is licensed under an [MIT license](https://github.com/tommccoy1/meta-learning-linguistic-biases/blob/master/LICENSE).

## Citing

If you make use of this code, please cite the following ([bibtex](https://tommccoy1.github.io/metaug_js/metaug_bib.html)):

R. Thomas McCoy, Erin Grant, Paul Smolensky, Thomas L. Griffiths, and Tal Linzen.  2020. Universal linguistic inductive biases via meta-learning.  In *Proceedings of the 42nd Annual Conference of the Cognitive Science Society*. 

*Questions? Comments? Email [tom.mccoy@jhu.edu](mailto:tom.mccoy@jhu.edu).*


