# Biased rulers

## A Survey on Quantifying Biases in Pretrained Language Models

## Metrics

- `metrics.seat`: Implementations of SEAT, and variations of SEAT by Lauscher et al. (2021) and Tan et al. (2019).
- `metrics.disco`: Implementation of DisCo.
- `metrics.lpbs`: Implementation of the _log probability bias score_.

## Notebooks
-The notebooks folder contains subfolders LM_correlations, sublist correlations, winobias
- LM_correlations: contains correlation results on how different template types and embedding types correlate based on scores from different language models.
- sublist_correlations: contains correlation results on how different template types and embedding types correlate based on scores from different sublist of jobs.
- correlation_analysis*.ipynb in each folder contains the results from the correlation experiments
- All codes used for Winobias experiments are directly adopted from Manela et al. (2021) https://github.com/12kleingordon34/NLP_masters_project
