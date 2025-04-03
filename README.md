# ComCon

This is the code of paper 
**AnEffectiveRegularizationApproachviaComplementary-Contradictory FeaturesforMultimodalKnowledgeGraphCompletion**.

## Dependencies

- Python 3.8
- PyTorch 1.7
- NumPy 1.17.2+
- tqdm 4.41.1+

### 1. Preprocess the Datasets

First preprocess the datasets.

```shell script
cd code
python process_datasets.py
```
Now, the processed datasets are in the `data` directory.

### 2. Reproduce the Results
python learn.py --dataset DB15K --rank 256 --optimizer Adam \
--learning_rate 1e-3 --batch_size 2048 --regularizer N3 --reg 1e-3 --max_epochs 220 \
--valid 5 -train -id 0 -save -weight

python learn.py --dataset MKG-W --rank 256 --optimizer Adam \
--learning_rate 5e-3 --batch_size 2048 --regularizer N3 --reg 1e-3 --max_epochs 170 \
--valid 5 -train -id 0 -save -weight
