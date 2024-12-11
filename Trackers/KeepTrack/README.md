download the pytracking source code at https://github.com/visionml/pytracking

download weighted file as well on their github

put these two files in {pytracking root}/pytracking/networks

rename keep_track_attn_param as keep_track_attn and move to {pytracking root}/pytracking/parameter

rename keep_track_attn_tracker as keep_track_attn and move to {pytracking root}/pytracking/tracker

command:
```
cd {pytracking root}
python pytracking/run_tracker.py keeptrack_attn default--dataset_name {your_dataset_name}
```
{pytracking root} shall be replaced by your own root

{your_dataset_name} shall be the name of the dataset that you wanna run your experiment
