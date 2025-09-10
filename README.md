# Ground Rruth Lexicon Learning
6 lexicon learning systems built using different methods of representing input speech and clustering word segments, given the ground truth word boundaries.

Get the boundaries from the provided ground-truth alignment files:
```
python boundaries.py $language $dataset 
```

For mandarin, get the "VAD" alignment which just cuts the utterance into segments that are short enough to process.
```
python vad.py 
```

## Graph clustering:
For all these there are the following options for model_name: ``wavlm-large``, ``hubert-large``, ``hubert-soft``, ``mhubert``, ``mandarin-hubert``

1. Continuous features + Averaging + Cosine distance 
```
python graph.py $language $dataset $model_name $layer cos $threshold
```

2. Continuous features  + DTW distance 
```
python graph.py $language $dataset $model_name $layer dtw $threshold
```

3. Discrete units + Edit distance 
```
python graph.py $language $dataset $model_name $layer ed $threshold --k $k --lmbda $lmbda
```

## Traditional clustering
1. Continuous features + Averaging + K-means
```
python avg.py $language $dataset $model_name $layer kmeans 
```

2. Continuous features + Averaging + BIRCH
```
python avg.py $language $dataset $model_name $layer birch

```

3. Continuous features + Averaging + Agglomerative Hierarchical 
```
python avg.py $language $dataset $model_name $layer agg 
```