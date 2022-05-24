# active_learning-w2v2_asr


## Run Repo
1. Clone Repo

```
git clone https://github.com/fauxneticien/active_learning-w2v2_asr.git
```

2. Create virtual environment and intall requirements specified in `https://github.com/CoEDL/vad-sli-asr/blob/master/Dockerfile`

3. Upload cgn dataset to `./data/datasets`. Change second column's title from "text" to "sentence" in the tsv files.

4. Copy `./data/datasets/cgn/wav` to `./wav`

5. Run one of the following commands at the root level.

```
./scripts/_run-exps.sh cgn facebook/wav2vec2-large random   
```

```
./scripts/_run-exps.sh cgn facebook/wav2vec2-large entropy   
```

```
./scripts/_run-exps.sh cgn GroNLP/wav2vec2-dutch-large random   
```

```
./scripts/_run-exps.sh cgn GroNLP/wav2vec2-dutch-large entropy   
```
