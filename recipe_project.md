# Required tools

```
git clone -b project https://github.com/CSTR-Edinburgh/ophelia.git
git clone Merlyn
git clone Tacotron
```

# Tacotron experiments

Link to Tacotron repo README

# Attention experiments

## Obtaining forced alignment labels:

Add commands on how to obtain the forced alignment labels (state level labels - 5 states per phone)

## FA as target

Add commands on how to convert the forced alignment labels to forced alignment matrix (matrix with zeros and ones with FA patch)

Set attention_guide_fa=True to use attention guides as target.

## FA as guides

Create attention guides from forced alignment matrix

```
cd ophelia/
python convert_alignment_to_guide.py fa_matrix.npy fa_guide.npy 
```

## FA as attention

Add phone level duration to transcript.csv using forced alignment matrix

```
cd ophelia/
python add_duration_to_transcript.py fa_matrix_dir transcript_file new_transcript_file
```

# Text Encoder experiments


## Labels -/+ TE

Convert state labels to 416 normalised label features (needs state labels and question file)

```
cd Merlyn/
python scripts/prepare_inputs.py
```

Set select_central to True in the config file to select centre phone.

## PE&Labels + TE

Create new transcription file with phoneme sequence from labels by replace phone sequence of transcript file with phone sequence from HTS style labels
```
cd ophelia/
./labels2tra.sh labels_dir transcript_file new_transcript_file
```

Set MerlinTextEncWithPhoneEmbedding to True in the config file.

# FIA experiments

To generate without FIA set turn_off_monotonic_for_synthesis to True in the config file.
