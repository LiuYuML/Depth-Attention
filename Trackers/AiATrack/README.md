download the pytracking source code at https://github.com/Little-Podi/AiATrack

download weighted file as well on their github

put the weighted file in arbitrary path 

rename aiatrack_attn_params as aiatrack_attn and move to {project root}/lib/test/parameter

rename aiatrack_attn_tracker as aiatrack_attn and move to {project root}/lib/test/tracker

by the way,you may need to correct some content in the line 22 of aiatrack_attn_params

command:
```
cd {project root}
python tracking/test.py --dataset {your_dataset_name}
```

{project root} shall be replaced by your own root

{your_dataset_name} shall be the name of the dataset that you wanna run your experiment

you may need to train the large model on you own
