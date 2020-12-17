## Required tools

```
git clone -b project https://github.com/CSTR-Edinburgh/ophelia.git
git clone https://github.com/AvashnaGovender/Merlyn.git
git clone https://github.com/AvashnaGovender/Tacotron.git
```

## DCTTS + WaveRNN

See recipe [here.](recipe_WaveRNN.md)

## Attention experiments

### Obtaining forced alignment labels:

Step 5a - Run forced alignment in:  
https://github.com/AvashnaGovender/Tacotron/blob/master/PAG_recipe.md

### FA as target

Convert forced alignment labels to the forced alignment matrix:  

Step 6 - Get durations and create guides:  
https://github.com/AvashnaGovender/Tacotron/blob/master/PAG_recipe.md

To use FA as target in DCTTS see config file:  
[fa_as_target.cfg](../config/project/fa_as_target.cfg)

### FA as guides

Create attention guides from forced alignment matrix

```
cd ophelia/
python convert_alignment_to_guide.py fa_matrix.npy fa_guide.npy 
```

To use FA as guide in DCTTS see config file:  
[fa_as_guide.cfg](../config/project/fa_as_guide.cfg)

### FA as attention

Add phone level duration to transcript.csv using forced alignment matrix

```
cd ophelia/
python add_duration_to_transcript.py fa_matrix_dir transcript_file new_transcript_file
```

To use FA as attention in DCTTS see config file:  
[fa_as_attention.cfg](../config/project/fa_as_attention.cfg)

## Text Encoder experiments

### Labels -/+ TE

Convert state labels to 416 normalised label features (needs state labels and question file)

```
cd Merlyn/
python scripts/prepare_inputs.py
```

To use Labels-TE in DCTTS see config file:  
[labels_minus_te.cfg](../config/project/labels_minus_te.cfg)

To use Labels+TE in DCTTS see config file:  
[labels_plus_te.cfg](../config/project/labels_plus_te.cfg)

To use C-Labels+TE in DCTTS see config file:  
[c-labels_plus_te.cfg](../config/project/c-labels_plus_te.cfg)

### PE&Labels + TE

Create new transcription file with phoneme sequence from labels by replace phone sequence of transcript file with phone sequence from HTS style labels
```
cd ophelia/
./labels2tra.sh labels_dir transcript_file new_transcript_file
```

To use PE&Labels+TE set MerlinTextEncWithPhoneEmbedding to True in the config file.

## Gross error detection experiments

To calculate CDP, Ain and Aout:
```
cd ophelia/
python calculate_CDP_Ain_Aout.py attention_matrix.npy
```

## FIA experiments

To generate without FIA (forcibly incremental attention) set turn_off_monotonic_for_synthesis to True in the config file.

## Tacotron experiments

See readme in Tacotron repository: https://github.com/AvashnaGovender/Tacotron
