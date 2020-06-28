# MAML for Universal Grammar


## Saved models

Are in saved_weights


Important files:
- models.py: Defines the models
- training.py: Defines MAML for PyTorch
- cv.train, cv.dev, cv.test: Datasets for the simple task I used to debug
- cv_trainer.py: Script to run MAML on this simple task

Most of the other stuff in here is for generating the datasets.


## Scripts
input_output_correspondences.py: Computes optimal output for each type of input and saves that (much faster than computing OT outputs on the fly for every word).
load_data.py: functions for loading saved data
cv_trainer.py: Simple quickstart
main.py: Code for meta-training a model (or saving a random set of weights)
phonology_task_creation.py: Generating tasks based on a constraint ranking
models.py: Definitions for our PyTorch models
training.py: Code for training or meta-training models
utils.py: Miscellaneous functions
train_dev_test_split.py: Generating a set of meta-training languages, a set of meta-test languages, and a set of meta-dev languages.
mix_input_outputs.py: Creating input/output correspondeces that draw randomly from multiple different constraint rankings, or from multiple constraint sets and constraint rankings
make_tasks_cv.py: Expand out a list of keys into full datasets for the simple CV case
make_tasks_imp.py: Create tasks that test for some implicational universal
make_tasks.py: Expand out a list of keys into full datasets for our real case

Some of the code is based on this blog post: https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0

## Abbreviations and shorthand used in the code"
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

## Replicating dataset generation
1. Create files specifying input-output correspondences for each constraint set. For "YesOnset, NoCoda" we further generate an additional file expanding out to input length 6.
python input_output_correspondences.py --onset yes --coda yes --prefix yo_yc_ --max_input_length 5
python input_output_correspondences.py --onset yes --coda no --prefix yo_nc_ --max_input_length 5
python input_output_correspondences.py --onset no --coda yes --prefix no_yc_ --max_input_length 5
python input_output_correspondences.py --onset no --coda no --prefix no_nc_ --max_input_length 5
python input_output_correspondences.py --onset yes --coda no --prefix yo_nc_ --max_input_length 6

2. Create keys that will be used to generate non-overlapping meta-training languages, meta-dev languages, and meta-test languages. Again, we do this once for each of the four constraint sets.
python train_dev_test_split.py --output_prefix yonc --constraints yonc
python train_dev_test_split.py --output_prefix yoyc --constraints yoyc
python train_dev_test_split.py --output_prefix noyc --constraints noyc
python train_dev_test_split.py --output_prefix nonc --constraints nonc

## Expanding keys out into datasets
1. Tasks used for meta-training and meta-testing:
python make_tasks.py --ranking_prefix yonc --constraints yonc

2. Tasks used for convergence testing:
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --ranking_prefix yonc --output_prefix yonc_10per --constraints yonc
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --ranking_prefix yoyc --output_prefix yoyc_10per --constraints yoyc
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --ranking_prefix noyc --output_prefix noyc_10per --constraints noyc
python make_tasks.py --n_train 20000 --n_dev 500 --n_test 1000 --n_train_tasks_per_ranking 0 --n_dev_tasks_per_ranking 0 --n_test_tasks_per_ranking 10 --ranking_prefix nonc --output_prefix nonc_10per --constraints nonc

## Meta-train a model
python main.py --data_prefix yonc --method maml --save_prefix maml_yonc_256_5
python main.py --data_prefix yonc --method random --save_prefix random_yonc_256_5

## Results of experiments:
1. 100-shot learning results for meta-initialized and randomly-initialized model (pg. 4 of the paper, second column, paragraph headed "meta-learning results"):
100-shot results for meta-initialized model (paper reports 98.8%):
python evaluation.py --data_prefix yonc --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 1.0 --inner_batch_size 100 --save_prefix maml_yonc_256_5
100-shot results for randomly-initialized model (paper reports 6.5%):
python evaluation.py --data_prefix yonc --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 1.0 --inner_batch_size 100 --save_prefix random_yonc_256_5

2. Ease of learning: Set of constraints: (pg. 4 of the paper, second column, first 4 paragraphs under "Ease of Learning"; also, Figure 4 top, Figure 5, and Figure 6a):
MAML, yonc: 203.75 
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
MAML, nonc: 791.11
python evaluation.py --data_prefix nonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
MAML, yoyc: 1112.0 
python evaluation.py --data_prefix yoyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
MAML, noyc: 1505.0
python evaluation.py --data_prefix noyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95

Random, yonc: 20642.5
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95 
Random, nonc: 25577.67
python evaluation.py --data_prefix nonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
Random, yoyc: 25447.8
python evaluation.py --data_prefix yoyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
Random, noyc: 24647.38  
python evaluation.py --data_prefix noyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95

3. Ease of learning: Consistent constraint ranking (pg. 5 of the paper, first column, first 2 full paragraphs; also Fig. 4, bottom left, and Fig. 6b):
MAML, consistent ranking: 203.75 (same as first line in Ease of Learning):
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
MAML, inconsistent ranking: 19803.0
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95


Random, consistent ranking: 20642.5 (same as above)
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
Random, inconsistent ranking: 74918.5
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95

4. Ease of learning: Consistent set of constraints (pg. 5 of the paper, last paragraph in first column continuing into second column; also Fig. 4, bottom right, and Fig. 6c):
MAML, consistent constraint set:
YONC: 19803.0
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
NONC: 8721.0
python evaluation.py --data_prefix nonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
YOYC: 29866.0 
python evaluation.py --data_prefix yoyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
NOYC: 34603.875
python evaluation.py --data_prefix noyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
Average: 23,248

MAML, inconsistent constraint set: 33135.125
python evaluation.py --data_prefix all_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95

Random, consistent constraint set:
YONC: 74918.5
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
NONC: 72906.56
python evaluation.py --data_prefix nonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
YOYC: 84942.1
python evaluation.py --data_prefix yoyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
NOYC: 80360.75 
python evaluation.py --data_prefix yoyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
Average: 78,282

Random, inconsistent constraint set: 87370.63
python evaluation.py --data_prefix all_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95


5. Poverty of the stimulus: All new phonemes (pg. 5 of the paper, 2nd column; also Fig. 7, left):
MAML: Unseen types: 0.837
python evaluation.py --data_prefix yonc_test_all_new_conv --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95

Random: Unseen types: 0.045 
python evaluation.py --data_prefix yonc_test_all_new_conv --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95

6. Poverty of the stimulus: Length 5 (pg. 6 of the paper, first full paragraph; also Fig 7, middle):
MAML: Length 5: 0.897
python evaluation.py --data_prefix imp_length --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
MAML: Length 6:

