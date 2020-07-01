Scripts to evaluate different drug repurposing methods

```python
cd ..

# test the auc of the diffusion method
python evaluate_auc.py -c benchmark-scripts/config_eval_diffusion.json

# test the num of clinical trials 100 of the diffusion method
# first change the metric in config to 'clinical-trial'
python evaluate_auc.py -c benchmark-scripts/config_eval_diffusion_clintrial.json


# test the auc of the Z-score method
```
