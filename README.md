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

