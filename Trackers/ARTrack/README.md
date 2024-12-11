download the pytracking source code at https://github.com/MIV-XJTU/ARTrack

download weighted file as well on their github

put the weighted file in arbitrary path 

rename artrack_seq_attn_param as artrack_seq_attn and move to {project root}/lib/test/parameter

rename artrack_seq_attn_tracker as artrack_seq_attn and move to {project root}/lib/test/tracker

by the way, you may need to correct some content in the line 26 of artrack_seq_attn_param

command:
```python
cd {project root}
python tracking/test.py artrack_seq artrack_seq_large_384_full --dataset {your_dataset_name}
```
{pytracking root} shall be replaced by your own root

{your_dataset_name} shall be the name of the dataset that you wanna run your experiment

you may need to train the large model on you own
