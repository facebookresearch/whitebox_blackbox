


# Splitting data


The following script will create four splits: public (29K), private (15K), test (15K) and val (1K).
The val split is used to cross-validate the performance of the all the models.
The private model is trained on the private set; the attacker can then use the public set to prepare its attack, and the attack is evaluated on the private and test splits, using the private model.

```
python create_splits.py
```


# Training a model

```
python train.py  \
--architecture smallnet \
--dataset cifar10 \
--dump_path path/to/checkpoint \
--exp_name bypass \
--optimizer sgd,lr=0.001,momentum=0.9 \
--split_train public_0 \
--transform center
```

## Citation

If you use this code, please cite the paper

```
@article{sablayrolles2019white,
  title={White-box vs black-box: Bayes optimal strategies for membership inference},
  author={Sablayrolles, Alexandre and Douze, Matthijs and Ollivier, Yann and Schmid, Cordelia and J{\'e}gou, Herv{\'e}},
  journal={ICML},
  year={2019}
}
```


## License

This repository is licensed under the CC BY-NC 4.0.
