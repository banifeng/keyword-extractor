# **BERT for Keyphrase Extraction** (Pytorch)

This repository provides the code of the paper [**Joint Keyphrase Chunking and Salience Ranking with BERT**](https://arxiv.org/pdf/2004.13639.pdf).

In this paper, we conduct an empirical study of **<u>5 keyphrase extraction models</u>** with **<u>3 BERT variants</u>**, and then propose a multi-task model BERT-JointKPE. Experiments on two KPE benchmarks, [OpenKP](https://www.aclweb.org/anthology/D19-1521.pdf) with Bing web pages and [KP20K](https://www.aclweb.org/anthology/P17-1054.pdf) demonstrate JointKPE’s state-of-the-art and robust effectiveness. Our further analyses also show that JointKPE has advantages in predicting **<u>long keyphrases</u>** and **<u>non-entity keyphrases</u>**, which were challenging for previous KPE techniques.

Please cite our paper if our experimental results, analysis conclusions or the code are helpful to you ~ :)

```
@misc{sun2020joint,
    title={Joint Keyphrase Chunking and Salience Ranking with BERT},
    author={Si Sun and Chenyan Xiong and Zhenghao Liu and Zhiyuan Liu and Jie Bao},
    year={2020},
    eprint={2004.13639},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


#### * Model Classes

|Index|Model|Descriptions|
|:---:|:---|:-----------|
|1|**BERT-JointKPE** (Bert2Joint)|A **<u>multi-task</u>** model is trained jointly on the chunking task and the ranking task, balancing the estimation of keyphrase quality and salience. |
|2|**BERT-RankKPE** (Bert2Rank)|Learn the salience phrases in the documents using a **<u>ranking</u>** network. |
|3|**BERT-ChunkKPE** (Bert2Chunk)|Classify high quality keyphrases using a **<u>chunking</u>** network. |
|4|**BERT-TagKPE** (Bert2Tag)|We modified the **<u>sequence tagging</u>** model to generate enough candidate keyphrases for a document. |
|5|**BERT-SpanKPE** (Bert2Span)|We modified the **<u>span extraction</u>** model to extract multiple keyphrases from a document. |


#### * BERT Variants

- [BERT](https://arxiv.org/abs/1810.04805)
- [SpanBERT](https://arxiv.org/abs/1907.10529)
- [RoBERTa](https://arxiv.org/abs/1907.11692)


## QUICKSTART

```
python 3.5
Pytorch 1.3.0
Tensorflow (tested on 1.14.0, only for tensorboardX)
```

### 1/ Download

- First download and decompress our data folder to this repo, the folder includes benchmark datasets and pre-trained BERT variants.

  - [Data Download Link](https://drive.google.com/open?id=1UugkRsKM8GXPPrrZxWa8HvGe1nyWdd6F)

- We also provide 15 checkpoints (5 KPE models * 3 BERT variants) trained on OpenKP training dataset.

  - [Checkpoint Download Link](https://drive.google.com/open?id=13FvONBTM4NZZCR-I7LVypkFa0xihxWnM)

### 2/ Preprocess

- To preprocess the source datasets using `preprocess.sh` in the `preprocess` folder:

  ```
  source preprocess.sh
  ```

- Optional arguments:

  ```
  --dataset_class         choices=['openkp', 'kp20k']
  --source_dataset_dir    The path to the source dataset
  --output_path           The dir to save preprocess data; default: ../data/prepro_dataset
  ```

### 3/ Train Models

- To train a new model from scratch using `train.sh` in the `scripts` folder:

  ```
  source train.sh
  ```
  PS. Running the training script for the first time will take some time to perform preprocess such as tokenization, and by default, the processed features will be saved under ../data/cached_features, which can be directly loaded next time.

- Optional arguments:

  ```
  --dataset_class         choices=['openkp', 'kp20k']
  --model_class           choices=['bert2span', 'bert2tag', 'bert2chunk', 'bert2rank', 'bert2joint']
  --pretrain_model_type   choices=['bert-base-cased', 'spanbert-base-cased', 'roberta-base']
  ```
  Complete optional arguments can be seen in `config.py` in the `scripts` folder.

- Training Parameters:

  We always keep the following settings in all our experiments:
  ```
  args.warmup_proportion = 0.1
  args.max_train_steps = 20810 (openkp) , 73430 (kp20k)
  args.per_gpu_train_batch_size * max(1, args.n_gpu) * args.gradient_accumulation_steps = 64
  ```

- Distributed Training

  We recommend using `DistributedDataParallel` to train models on multiple GPUs (It's faster than `DataParallel`, but it will take up more memory)

  ```
  CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train.py
  # if you use DataParallel rather than DistributedDataParallel, remember to set --local_rank=-1
  ```

### 4/ Inference

- To evaluate models using trained checkpoints using `test.sh` in the `scripts` folder:

  ```
  source test.sh
  ```

- Optional arguments:

  ```
  --dataset_class         choices=['openkp', 'kp20k']
  --model_class           choices=['bert2span', 'bert2tag', 'bert2chunk', 'bert2rank', 'bert2joint']
  --pretrain_model_type   choices=['bert-base-cased', 'spanbert-base-cased', 'roberta-base']
  --eval_checkpoint       The checkpoint file to be evaluated
  ```

### 5/ Re-produce evaluation results using our checkpoints

  - Run `test.sh`, and change the `eval_checkpoint` to the checkpoint files we provided to reproduce the following results.

    ```
    --dataset_class         openkp
    --eval_checkpoint       The filepath of our provided checkpoint
    ```

## * RESULTS

  The following results are ranked by F1@3 on OpenKP Dev dataset, the eval results can be seen in the [OpenKP Leaderboard](https://microsoft.github.io/msmarco/).

#### * BERT (Base)

|Rank|Method|F1 @1,**@3**,@5|Precision @1,@3,@5|Recall @1,@3,@5|
|:--:|:----:|:--------:|:---------------:|:------------:|
|1|Bert2Joint|0.371, **0.384**, 0.326|0.504, 0.313, 0.227|0.315, 0.555, 0.657|
|2|Bert2Rank|0.369, **0.381**, 0.325|0.502, 0.311, 0.227|0.315, 0.551, 0.655|
|3|Bert2Tag|0.370, **0.374**, 0.318|0.502, 0.305, 0.222 | 0.315, 0.541, 0.642|
|4|Bert2Chunk|0.370, **0.370**, 0.311|0.504, 0.302, 0.217|0.314, 0.533, 0.627|
|5|Bert2Span|0.341, **0.340**, 0.293|0.466, 0.277, 0.203|0.289, 0.492, 0.593|


#### * SpanBERT (Base)

|Rank|Method|F1 @1,@3,@5|Precision @1,@3,@5|Recall @1,@3,@5|
|:--:|:----:|:--------:|:---------------:|:------------:|
|1|Bert2Joint|0.388, **0.393**, 0.333|0.527, 0.321, 0.232|0.331, 0.567, 0.671|
|2|Bert2Rank|0.385, **0.390**, 0.332|0.521, 0.319, 0.232|0.328, 0.564, 0.666|
|3|Bert2Tag|0.384, **0.385**, 0.327|0.520, 0.315, 0.228|0.327, 0.555, 0.657|
|4|Bert2Chunk|0.378, **0.385**, 0.326|0.514, 0.314, 0.228|0.322, 0.555, 0.656|
|5|Bert2Span|0.347, **0.359**, 0.304|0.477, 0.294, 0.212|0.293, 0.518, 0.613|


#### * RoBERTa (Base)

|Rank|Method|F1 @1,@3,@5|Precision @1,@3,@5|Recall @1,@3,@5|
|:--:|:----:|:--------:|:---------------:|:------------:|
|1|Bert2Joint|0.391, **0.398**, 0.338|0.532, 0.325, 0.235|0.334, 0.577, 0.681|
|2|Bert2Rank|0.388, **0.395**, 0.335|0.526, 0.322, 0.233|0.330, 0.570, 0.677|
|3|Bert2Tag|0.387, **0.389**, 0.330|0.525, 0.318, 0.230|0.329, 0.562, 0.666|
|4|Bert2Chunk|0.380, **0.382**, 0.327|0.518, 0.312, 0.228|0.324, 0.551, 0.660|
|5|Bert2Span|0.358, **0.355**, 0.306|0.487, 0.289, 0.213|0.304, 0.513, 0.619|


## MODEL OVERVIEW

### * BERT-JointKPE, RankKPE, ChunkKPE (See Paper)

### * BERT-TagKPE (See Code)

- **Word-Level Representations :**   We encode an input document into a sequence of WordPiece tokens' vectors with a pretrained BERT (or its variants), and then we pick up the first sub-token vector of each word to represent the input in word-level.

- **Phrase-Level Representations :** We perform a **soft-select** method to decode phrase from word-level vector instead of hard-select used in the standard sequence tagging task .

  The word-level representation is feed into an classification layer to obtain the tag probabilities of each word on 5 classes  (O, B, I, E, U) , and then we employ different tag patterns for extracting different n-grams ( 1 ≤ n ≤ 5 ) over the whole sequence.

  Last there are a collect of n-gram candidates, each word of the n-gram just has one score.

  **Soft-select Example :** considering all 3-grams (B I E) on the L-length document, we can extract (L-3+1)  3-grams sequentially like sliding window. In each 3-gram, we only keep **B score** for the first word, **I score** for the middle word, and **E score** for the last word, etc.

  > **O** : Non Keyphrase ;  **B** : Begin word of the keyprase ;  **I** : Middle word of the keyphrase ;  **E** : End word of keyprhase ;  **U** : Uni-word keyphrase

- **Document-Level Keyphrase :** At the Last stage, the recovering from phrase-level n-grams to document-level keyphrases can be naturally formulated as a ranking task.

  Incorporating with term frequency, we employ **Min Pooling** to get the final score of each n-gram (we called it **Buckets Effect**: No matter how high a bucket, it depends on the height of the water in which the lowest piece of wood) . Based on the final scores, we extract 5 top ranked keyprhase candidates for each document.

### * BERT-SpanKPE (See Code)

- **Word-Level Representations :** Same as BERT-TagKPE
- **Phrase-Level Representations :** Traditional span extraction model could not extract multiple important keyphrase spans for the same document. Therefore, we propose an self-attention span extraction model.

  Given the token representations \{t1, t2, ..., tn\}, we first calculate the probability that the token is the starting word Ps(ti), and then apply the single-head self-attention layer to calculate the ending word probability of all j>=i tokens Pe(tj).

- **Document-Level Keyphrase :** We select the spans with the highest probability P = Ps(ti) * Pe(tj) as the keyphrase spans.



## CONTACT

For any question, please contact **Si Sun** by email s-sun17@mails.tsinghua.edu.cn , we will try our best to solve.


I0810 09:33:07.858009 140185854891840 utils.py:174] @1
08/10/2020 09:33:07 AM: [ @1 ]
I0810 09:33:07.859859 140185854891840 utils.py:175] F1:0.2886450475362608
08/10/2020 09:33:07 AM: [ F1:0.2886450475362608 ]
I0810 09:33:07.861207 140185854891840 utils.py:176] P:0.5584269028945438
08/10/2020 09:33:07 AM: [ P:0.5584269028945438 ]
I0810 09:33:07.862335 140185854891840 utils.py:177] R:0.21173793237655755
08/10/2020 09:33:07 AM: [ R:0.21173793237655755 ]
I0810 09:33:07.862462 140185854891840 utils.py:174] @3
08/10/2020 09:33:07 AM: [ @3 ]
I0810 09:33:07.863540 140185854891840 utils.py:175] F1:0.40249089014705836
08/10/2020 09:33:07 AM: [ F1:0.40249089014705836 ]
I0810 09:33:07.864751 140185854891840 utils.py:176] P:0.42216329063928226
08/10/2020 09:33:07 AM: [ P:0.42216329063928226 ]
I0810 09:33:07.865867 140185854891840 utils.py:177] R:0.44801040733306835
08/10/2020 09:33:07 AM: [ R:0.44801040733306835 ]
I0810 09:33:07.865987 140185854891840 utils.py:174] @5
08/10/2020 09:33:07 AM: [ @5 ]
I0810 09:33:07.867043 140185854891840 utils.py:175] F1:0.39609502244278966
08/10/2020 09:33:07 AM: [ F1:0.39609502244278966 ]
I0810 09:33:07.868148 140185854891840 utils.py:176] P:0.33995373243807486
08/10/2020 09:33:07 AM: [ P:0.33995373243807486 ]
I0810 09:33:07.869250 140185854891840 utils.py:177] R:0.5800876953310422
08/10/2020 09:33:07 AM: [ R:0.5800876953310422 ]
I0810 09:33:07.869369 140185854891840 utils.py:174] @10
08/10/2020 09:33:07 AM: [ @10 ]
I0810 09:33:07.870418 140185854891840 utils.py:175] F1:0.3299407035824225
08/10/2020 09:33:07 AM: [ F1:0.3299407035824225 ]
I0810 09:33:07.871493 140185854891840 utils.py:176] P:0.23430570445184226
08/10/2020 09:33:07 AM: [ P:0.23430570445184226 ]
I0810 09:33:07.872566 140185854891840 utils.py:177] R:0.753294534651239




I0828 23:56:39.047868 139640222074688 utils.py:174] @1
08/28/2020 11:56:39 PM: [ @1 ]
I0828 23:56:39.051323 139640222074688 utils.py:175] F1:0.2961541917580941
08/28/2020 11:56:39 PM: [ F1:0.2961541917580941 ]
I0828 23:56:39.053117 139640222074688 utils.py:176] P:0.567454719855555
08/28/2020 11:56:39 PM: [ P:0.567454719855555 ]
I0828 23:56:39.054636 139640222074688 utils.py:177] R:0.21786366798240112
08/28/2020 11:56:39 PM: [ R:0.21786366798240112 ]
I0828 23:56:39.054772 139640222074688 utils.py:174] @3
08/28/2020 11:56:39 PM: [ @3 ]
I0828 23:56:39.056096 139640222074688 utils.py:175] F1:0.41971445187945977
08/28/2020 11:56:39 PM: [ F1:0.41971445187945977 ]
I0828 23:56:39.057499 139640222074688 utils.py:176] P:0.4351219695687336
08/28/2020 11:56:39 PM: [ P:0.4351219695687336 ]
I0828 23:56:39.058888 139640222074688 utils.py:177] R:0.470426712986902
08/28/2020 11:56:39 PM: [ R:0.470426712986902 ]
I0828 23:56:39.058998 139640222074688 utils.py:174] @5
08/28/2020 11:56:39 PM: [ @5 ]
I0828 23:56:39.060267 139640222074688 utils.py:175] F1:0.41535263873317707
08/28/2020 11:56:39 PM: [ F1:0.41535263873317707 ]
I0828 23:56:39.061602 139640222074688 utils.py:176] P:0.3527619477515094
08/28/2020 11:56:39 PM: [ P:0.3527619477515094 ]
I0828 23:56:39.062935 139640222074688 utils.py:177] R:0.6121054837386465
08/28/2020 11:56:39 PM: [ R:0.6121054837386465 ]
I0828 23:56:39.063044 139640222074688 utils.py:174] @10
08/28/2020 11:56:39 PM: [ @10 ]
I0828 23:56:39.064313 139640222074688 utils.py:175] F1:0.3425826365610231
08/28/2020 11:56:39 PM: [ F1:0.3425826365610231 ]
I0828 23:56:39.065616 139640222074688 utils.py:176] P:0.24068160018055637
08/28/2020 11:56:39 PM: [ P:0.24068160018055637 ]
I0828 23:56:39.066942 139640222074688 utils.py:177] R:0.7886851720466819
08/28/2020 11:56:39 PM: [ R:0.7886851720466819 ]



I0903 11:39:40.850990 139942839408448 utils.py:167] Start Evaluatng : Mode = eval || Epoch = 6
09/03/2020 11:39:40 AM: [ Start Evaluatng : Mode = eval || Epoch = 6 ]
I0903 11:39:53.908466 139942839408448 utils.py:174] @1
09/03/2020 11:39:53 AM: [ @1 ]
I0903 11:39:53.913002 139942839408448 utils.py:175] F1:0.30657614714027964
09/03/2020 11:39:53 AM: [ F1:0.30657614714027964 ]
I0903 11:39:53.914926 139942839408448 utils.py:176] P:0.5879365795858489
09/03/2020 11:39:53 AM: [ P:0.5879365795858489 ]
I0903 11:39:53.916518 139942839408448 utils.py:177] R:0.22575080164121447
09/03/2020 11:39:53 AM: [ R:0.22575080164121447 ]
I0903 11:39:53.916656 139942839408448 utils.py:174] @3
09/03/2020 11:39:53 AM: [ @3 ]
I0903 11:39:53.917992 139942839408448 utils.py:175] F1:0.42484717480993683
09/03/2020 11:39:53 AM: [ F1:0.42484717480993683 ]
I0903 11:39:53.919395 139942839408448 utils.py:176] P:0.442062103857511
09/03/2020 11:39:53 AM: [ P:0.442062103857511 ]
I0903 11:39:53.920792 139942839408448 utils.py:177] R:0.4753979745713736
09/03/2020 11:39:53 AM: [ R:0.4753979745713736 ]
I0903 11:39:53.920908 139942839408448 utils.py:174] @5
09/03/2020 11:39:53 AM: [ @5 ]
I0903 11:39:53.922249 139942839408448 utils.py:175] F1:0.41771067507253845
09/03/2020 11:39:53 AM: [ F1:0.41771067507253845 ]
I0903 11:39:53.923646 139942839408448 utils.py:176] P:0.35577498166224686
09/03/2020 11:39:53 AM: [ P:0.35577498166224686 ]
I0903 11:39:53.925035 139942839408448 utils.py:177] R:0.6146666452025151
09/03/2020 11:39:53 AM: [ R:0.6146666452025151 ]
I0903 11:39:53.925140 139942839408448 utils.py:174] @10
09/03/2020 11:39:53 AM: [ @10 ]
I0903 11:39:53.926441 139942839408448 utils.py:175] F1:0.344184059698228
09/03/2020 11:39:53 AM: [ F1:0.344184059698228 ]
I0903 11:39:53.927770 139942839408448 utils.py:176] P:0.2424363821023529
09/03/2020 11:39:53 AM: [ P:0.2424363821023529 ]
I0903 11:39:53.929100 139942839408448 utils.py:177] R:0.7904029540974541
09/03/2020 11:39:53 AM: [ R:0.7904029540974541 ]
I0903 11:39:53.930365 139942839408448 utils.py:181] ------------------------------------------------------------
09/03/2020 11:39:53 AM: [ ------------------------------------------------------------ ]
I0903 11:39:53.930477 139942839408448 utils.py:184] Update ! Update ! Update ! ||  Mode = eval || Max f1_score5 = 0.4177 (epoch = 6, local_rank = -1)
09/03/2020 11:39:53 AM: [ Update ! Update ! Update ! ||  Mode = eval || Max f1_score5 = 0.4177 (epoch = 6, local_rank = -1) ]
I0903 11:39:53.930548 139942839408448 utils.py:185] ------------------------------------------------------------
09/03/2020 11:39:53 AM: [ ------------------------------------------------------------ ]
I0903 11:39:53.930624 139942839408448 utils.py:187] Local Rank = -1 || End Evaluatng : Mode = eval || Epoch = 6 (Time: 13.08 (s))
09/03/2020 11:39:53 AM: [ Local Rank = -1 || End Evaluatng : Mode = eval || Epoch = 6 (Time: 13.08 (s))  ]
I0903 11:39:53.930692 139942839408448 utils.py:188] ********************************************************************************
09/03/2020 11:39:53 AM: [ ******************************************************************************** ]