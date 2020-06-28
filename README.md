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



## Step 1: Dataset generation

The first step is to generate the sets of languages that will be used for meta-training, meta-validation, and meta-evaluation. These scripts have already been run for you: their results are in `io_correspondences/` and `data/`. We provide these scripts in case you wish to replicate our experiments from scratch, but otherwise you can skip Step 1.

1. Create files specifying input-output correspondences for each constraint set. For "YesOnset, NoCoda" we further generate an additional file expanding out to input length 6. These files specify the shape of the output for a given input shape; e.g., CVC -> .CV.CV.
```
python input_output_correspondences.py --onset yes --coda yes --prefix yo_yc_ --max_input_length 5
python input_output_correspondences.py --onset yes --coda no --prefix yo_nc_ --max_input_length 5
python input_output_correspondences.py --onset no --coda yes --prefix no_yc_ --max_input_length 5
python input_output_correspondences.py --onset no --coda no --prefix no_nc_ --max_input_length 5
python input_output_correspondences.py --onset yes --coda no --prefix yo_nc_ --max_input_length 6
```

2. Create keys that will be used to generate non-overlapping meta-training languages, meta-dev languages, and meta-test languages. Each key specifies on language, and they are generated such that all generated languages are unique both within and across the meta-training, meta-development, and meta-test set. Saving these keys is much more memory-efficient than saving the full datasets; from the key (as long as we specify the random seed) each language's dataset can then be generated exactly as it was generated for use in the paper. We generate one set of keys for each of the four constraint sets.
```
python train_dev_test_split.py --output_prefix yonc --constraints yonc
python train_dev_test_split.py --output_prefix yoyc --constraints yoyc
python train_dev_test_split.py --output_prefix noyc --constraints noyc
python train_dev_test_split.py --output_prefix nonc --constraints nonc
```

## Step 2: Expanding keys out into datasets
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
### Verifying that the datasets were generated properly

The above lines of code should exactly recreate the datasets that we used in the paper because the random seed has been set to ensure replicability. To verify that this is working properly, here are some checks you can perform on `yonc.train`, `yonc.dev`, and `yonc.test`, which are the outputs of (A):

First, the first line of`yonc.train` should be the following:
```
oppp,.op. kqkpo,.po. zoqaz,.zo.qaz. qopao,.qo.pa.o. oqazk,.o.qak. opkoa,.op.ko.a. qqzoq,.zoq. okzka,.oz.ka. kzopa,.zo.pa. zpoko,.po.ko. zoqoa,.zo.qo.a. aaoaa,.a.a.o.a.a. aokoq,.a.o.koq. ooo,.o.o.o. oopo,.o.o.po. oqzpa,.oz.pa. aaozp,.a.a.op. kaqaq,.ka.qaq. pako,.pa.ko. aaoqq,.a.a.oq. aaza,.a.a.za. akaza,.a.ka.za. okak,.o.kak. qaqp,.qap. akkko,.ak.ko. oz,.oz. zoaoo,.zo.a.o.o. aooop,.a.o.o.op. aozoz,.a.o.zoz. azoaq,.a.zo.aq. zooao,.zo.o.a.o. aazpp,.a.ap. qzaaq,.za.aq. opkpo,.ok.po. azooa,.a.zo.o.a. akkz,.az. aao,.a.a.o. pqqok,.qok. ookop,.o.o.kop. aaaz,.a.a.az. qpozp,.pop. zaoa,.za.o.a. okoqq,.o.koq. ozka,.oz.ka. zzo,.zo. ookpz,.o.oz. okoqp,.o.kop. aao,.a.a.o. kopo,.ko.po. azkka,.ak.ka. zaoaq,.za.o.aq. pzooa,.zo.o.a. aozqa,.a.oz.qa. paaaz,.pa.a.az. aako,.a.a.ko. qokpo,.qok.po. pakka,.pak.ka. pa,.pa. kazo,.ka.zo. pqqz, oqooa,.o.qo.o.a. ooaap,.o.o.a.ap. aapp,.a.ap. kkoo,.ko.o. zazz,.zaz. aaoq,.a.a.oq. oooaq,.o.o.o.aq. apaa,.a.pa.a. opkao,.op.ka.o. azooz,.a.zo.oz. zazpz,.zaz. oaoap,.o.a.o.ap. okako,.o.ka.ko. ozq,.oq. az,.az. p, aak,.a.ak. pkko,.ko. opoqo,.o.po.qo. pooa,.po.o.a. azqp,.ap. aqkoo,.aq.ko.o. poa,.po.a. akpqo,.ap.qo. p, qqpo,.po. oooa,.o.o.o.a. kqqaz,.qaz. qpzaa,.za.a. pkzqp, qpqzq, ook,.o.ok. zkqa,.qa. oooza,.o.o.o.za. aoo,.a.o.o. kppaa,.pa.a. zoaq,.zo.aq. azp,.ap. oqokk,.o.qok. kzkpq,        oqpkz,.oz. aqzzp,.ap. kzaa,.za.a. aok,.a.ok. kokzq,.koq. qakkk,.qak. qaoko,.qa.o.ko. qpzq, zoao,.zo.a.o. qakko,.qak.ko. ao,.a.o. a,.a. kp, poaoo,.po.a.o.o. oozoo,.o.o.zo.o. aqpa,.aq.pa. azkzp,.ap. pz, kozkq,.koq. qqaz,.qaz. koqo,.ko.qo. pkzao,.za.o. qpq, azkq,.aq. oqaza,.o.qa.za. aaokq,.a.a.oq. opoqq,.o.poq. poqop,.po.qop. appz,.az. ppo,.po. qaqoo,.qa.qo.o. zppoo,.po.o. za,.za. oozoo,.o.o.zo.o. qqapk,.qak. pazao,.pa.za.o. qookz,.qo.oz. qqap,.qap. q, aapo,.a.a.po. kao,.ka.o. qoko,.qo.ko. pkzak,.zak. po,.po. oqao,.o.qa.o. oqoo,.o.qo.o. koa,.ko.a. zzopq,.zoq. okoao,.o.ko.a.o. paaaq,.pa.a.aq. oazko,.o.az.ko. aaaqo,.a.a.a.qo. zpk, qopap,.qo.pap. pqkoz,.koz. zaq,.zaq. qaak,.qa.ak. pq, aaqoz,.a.a.qoz. apkaq,.ap.kaq. pz, oaap,.o.a.ap. aoaqa,.a.o.a.qa. oopzq,.o.oq. paoao,.pa.o.a.o. aaaa,.a.a.a.a. qaaa,.qa.a.a. zkkk, aaoo,.a.a.o.o. kqook,.qo.ok. oaaaa,.o.a.a.a.a. aazp,.a.ap. paak,.pa.ak. azaza,.a.za.za. kako,.ka.ko. oakoo,.o.a.ko.o. aopz,.a.oz. pqaqk,.qak. oqoz,.o.qoz. akqo,.ak.qo. zo,.zo. aaaza,.a.a.a.za. , oqozo,.o.qo.zo. z, oqapz,.o.qaz. qoopo,.qo.o.po. qzppa,.pa. apz,.az. aopo,.a.o.po. azaoa,.a.za.o.a. zoopp,.zo.op. opqqa,.oq.qa. qkako,.ka.ko. pp, qoa,.qo.a. azqo,.az.qo. qqooo,.qo.o.o. aoopz,.a.o.oz. zz,   a e i o u A E I O U b c d f g h j k l m n p q r s t v w x z .   o a,z k q p,3 2 1 0
```

While the last line of `yonc.train` should be the following:
```
juEf,.ju.Ef. Oufu,.O.u.fu. EuEOz,.E.u.E.Oz. , ujOz,.u.jOz. OufOO,.O.u.fO.O. fEu,.fE.u. , , EEOdE,.E.E.O.dE. fEujz,.fE.uz. uzfuO,.uz.fu.O. fddfO,.fO. uud,.u.ud. Ouudz,.O.u.uz. EEEfu,.E.E.E.fu. zfudu,.fu.du. OOzfz,.O.Oz. OzOEd,.O.zO.Ed. dujdf,.duf. fdfE,.fE. zEdu,.zE.du. jf, zuOuf,.zu.O.uf. O,.O. OEdEE,.O.E.dE.E. Ej,.Ej. jjzEO,.zE.O. dEfz,.dEz. jff, jOdd,.jOd. uOuE,.u.O.u.E. u,.u. Ouudd,.O.u.ud. ujjj,.uj. Edz,.Ez. jOuzf,.jO.uf. uOfO,.u.O.fO. uuOdu,.u.u.O.du. fOfj,.fOj. OzuOE,.O.zu.O.E. OfE,.O.fE. dOj,.dOj. EEEdd,.E.E.Ed. zjdd, jzEjj,.zEj. uuzfu,.u.uz.fu. OzjOE,.Oz.jO.E. OOfzu,.O.Of.zu. EEOuu,.E.E.O.u.u. OOd,.O.Od. uOOOz,.u.O.O.Oz. EOEE,.E.O.E.E. ujE,.u.jE. OOufj,.O.O.uj. uddE,.ud.dE. dzjEu,.jE.u. zuuuO,.zu.u.u.O. zfdd, j, zudjE,.zud.jE. EEEO,.E.E.E.O. dffzd, Ozf,.Of. zO,.zO. djj, Ezuuj,.E.zu.uj. E,.E. dOfuO,.dO.fu.O. OfEE,.O.fE.E. OEufj,.O.E.uj. Ejzj,.Ej. EjOOz,.E.jO.Oz. uOEf,.u.O.Ef. Edzdu,.Ez.du. Oju,.O.ju. djudj,.juj. OfdEf,.Of.dEf. fuOuO,.fu.O.u.O. ddfE,.fE. dOOd,.dO.Od. zOOO,.zO.O.O. dOzu,.dO.zu. EOfzu,.E.Of.zu. fd, fuEdz,.fu.Ez. uzju,.uz.ju. OufO,.O.u.fO. , jOOOf,.jO.O.Of. fddE,.dE. ujEz,.u.jEz. zOEz,.zO.Ez. jfOuE,.fO.u.E. duudz,.du.uz. OfEEf,.O.fE.Ef. udOEE,.u.dO.E.E. dzEEf,.zE.Ef. , jOOEd,.jO.O.Ed.     uOjuE,.u.O.ju.E. Ezzu,.Ez.zu. zdfOu,.fO.u. zEO,.zE.O. uuOOd,.u.u.O.Od. fj, EEdud,.E.E.dud. EffjO,.Ef.jO. EzOOu,.E.zO.O.u. zduju,.du.ju. juuu,.ju.u.u. Ojjfd,.Od. jOdz,.jOz. fufdO,.fuf.dO. f, OuuO,.O.u.u.O. zzu,.zu. EuE,.E.u.E. fuuOE,.fu.u.O.E. jOOEz,.jO.O.Ez. jfju,.ju. fjO,.jO. fzEuj,.zE.uj. zujOd,.zu.jOd. OOE,.O.O.E. zOdE,.zO.dE. EdOEu,.E.dO.E.u. jfjfO,.fO. EuOuE,.E.u.O.u.E. ujjjz,.uz. jud,.jud. djuff,.juf. ufjuf,.uf.juf. OEfEO,.O.E.fE.O. OzuOf,.O.zu.Of. uz,.uz. EEu,.E.E.u. EfzzE,.Ez.zE. uf,.uf. fOz,.fOz. uuOf,.u.u.Of. OEuzu,.O.E.u.zu. Ojjjz,.Oz. jjEdE,.jE.dE. fuuO,.fu.u.O. OfOfE,.O.fO.fE. OuzOd,.O.u.zOd. EEufj,.E.E.uj. jduuz,.du.uz. zzjjE,.jE. jEOd,.jE.Od. jjzE,.zE. EddEd,.Ed.dEd. fEuO,.fE.u.O. OfEu,.O.fE.u. EOOd,.E.O.Od. OEujE,.O.E.u.jE. EEE,.E.E.E. zfEf,.fEf. dzf, dOjdO,.dOj.dO. ujzj,.uj. uEz,.u.Ez. EzuOj,.E.zu.Oj. jjfzd, EOuOu,.E.O.u.O.u. jjfd, EzEOj,.E.zE.Oj. ujOOz,.u.jO.Oz. zzffd, z, uzdO,.uz.dO. zufj,.zuj. fzuEd,.zu.Ed. zE,.zE. duOE,.du.O.E. Ozuf,.O.zuf. jzfOu,.fO.u. dEOEf,.dE.O.Ef. dfEE,.fE.E. uzuOu,.u.zu.O.u. zuEEz,.zu.E.Ez. uOE,.u.O.E. fuuf,.fu.uf. Ozdu,.Oz.du. Ofjd,.Od. uu,.u.u. dzzEj,.zEj. Ejfu,.Ej.fu. uuduz,.u.u.duz. uEuzE,.u.E.u.zE. ddj, EzOdE,.E.zO.dE. djuff,.juf. udEd,.u.dEd. Oufjd,.O.ud. dfOEO,.fO.E.O. zuzzu,.zuz.zu. fzEzE,.zE.zE. EE,.E.E.    a e i o u A E I O U b c d f g h j k l m n p q r s t v w x z .   E u O,d j f z,3 2 1 0
```

In addition, the exact `yonc.dev` and `yonc.test` files have been provided in `data/`, as `yonc_original.dev` and `yonc_original.test`, so that you can check if your generated `yonc.dev` matches `yonc_original.dev` and that your generated `yonc.test` matches `yonc_original.test`.


## Step 3: Create model initializations randomly or with meta-learning

In this step, we create the two model initializations that we perform our evaluations on. You can skip this step if you want, because the exact initializations that we used in the paper have been provided under `models/`. Be aware that running meta-training can take a long time (several days).

1. Meta-train a model using `yonc.train` as the meta-training set, `yonc.dev` as the meta-dev set, and `yonc.test` as the meta-test set, then save its weights to `models/maml_yonc_256_5.weights`.
```
python main.py --data_prefix yonc --method maml --save_prefix maml_yonc_256_5
```

2. Save a randomly-initialized model's weights to `models/random_yonc_256_5`.
```
python main.py --data_prefix yonc --method random --save_prefix random_yonc_256_5
```

## Results of experiments:
1. 100-shot learning results for meta-initialized and randomly-initialized model (pg. 4 of the paper, second column, paragraph headed "meta-learning results"):
100-shot results for meta-initialized model (paper reports 98.8%):
```
python evaluation.py --data_prefix yonc --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 1.0 --inner_batch_size 100 --save_prefix maml_yonc_256_5
```

100-shot results for randomly-initialized model (paper reports 6.5%):
```
python evaluation.py --data_prefix yonc --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 1.0 --inner_batch_size 100 --save_prefix random_yonc_256_5
```

2. Ease of learning: Set of constraints: (pg. 4 of the paper, second column, first 4 paragraphs under "Ease of Learning"; also, Figure 4 top, Figure 5, and Figure 6a):
MAML, yonc: 203.75 
```
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
MAML, nonc: 791.11
```
python evaluation.py --data_prefix nonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
MAML, yoyc: 1112.0 
```
python evaluation.py --data_prefix yoyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
MAML, noyc: 1505.0
```
python evaluation.py --data_prefix noyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
Random, yonc: 20642.5
```
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95 
```
Random, nonc: 25577.67
```
python evaluation.py --data_prefix nonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
Random, yoyc: 25447.8
```
python evaluation.py --data_prefix yoyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
Random, noyc: 24647.38  
```
python evaluation.py --data_prefix noyc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
3. Ease of learning: Consistent constraint ranking (pg. 5 of the paper, first column, first 2 full paragraphs; also Fig. 4, bottom left, and Fig. 6b):
MAML, consistent ranking: 203.75 (same as first line in Ease of Learning):
```
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
MAML, inconsistent ranking: 19803.0
```
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

Random, consistent ranking: 20642.5 (same as above)
```
python evaluation.py --data_prefix yonc_10per --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
Random, inconsistent ranking: 74918.5
```
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
4. Ease of learning: Consistent set of constraints (pg. 5 of the paper, last paragraph in first column continuing into second column; also Fig. 4, bottom right, and Fig. 6c):
MAML, consistent constraint set:
YONC: 19803.0
```
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
NONC: 8721.0
```
python evaluation.py --data_prefix nonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
YOYC: 29866.0 
```
python evaluation.py --data_prefix yoyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
NOYC: 34603.875
```
python evaluation.py --data_prefix noyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
Average: 23,248

MAML, inconsistent constraint set: 33135.125
```
python evaluation.py --data_prefix all_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
Random, consistent constraint set:
YONC: 74918.5
```
python evaluation.py --data_prefix yonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
NONC: 72906.56
```
python evaluation.py --data_prefix nonc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
YOYC: 84942.1
```
python evaluation.py --data_prefix yoyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
NOYC: 80360.75 
```
python evaluation.py --data_prefix yoyc_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
Average: 78,282

Random, inconsistent constraint set: 87370.63
```
python evaluation.py --data_prefix all_shuffle_aio --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

5. Poverty of the stimulus: All new phonemes (pg. 5 of the paper, 2nd column; also Fig. 7, left):
MAML: Unseen types: 0.837
```
python evaluation.py --data_prefix pos_new_phonemes --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

Random: Unseen types: 0.045 
```
python evaluation.py --data_prefix pos_new_phonemes --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
6. Poverty of the stimulus: Length 5 (pg. 6 of the paper, first full paragraph; also Fig 7, middle):
MAML: Length 5: 0.897
```
python evaluation.py --data_prefix pos_length --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
MAML: Length 6: 0.504 NOTE: Different from paper (different seed)
```
python evaluation.py --data_prefix pos_length6 --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```

Random: Length 5: 0.59
```
python evaluation.py --data_prefix pos_length --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100000000 --eval_technique converge --threshold 0.95
```
7. Poverty of the stimulus: Implicational universals (pg. 6 of the paper, last paragraph before the conclusion; also Fig. 6, right):
MAML: 0.926
```
python evaluation.py --data_prefix pos_imp --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix maml_yonc_256_5 --print_every 10 --patience 100 --eval_technique converge --threshold 0.95
```
Random: 0.100
```
python evaluation.py --data_prefix pos_imp --vocab_size 34 --emb_size 10 --hidden_size 256 --lr_inner 0.001 --inner_batch_size 10 --save_prefix random_yonc_256_5 --print_every 10 --patience 100 --eval_technique converge --threshold 0.95
```  
All code should be run in the `src` directory.



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




