download the pytracking source code at https://github.com/MCG-NJU/MixFormer
download weighted file as well on their github
put the weighted file in arbitrary path 
rename xxx_params as xxx_attn and move to {project root}/lib/test/parameter
rename xxx_attn_tracker as xxx_attn and move to {project root}/lib/test/tracker
by the way,you may need to correct some content in the line 27 of aiatrack_attn_params
cd {project root}
python tracking/test.py xxxxxx (they have figure out all the params in {project root}/tracking/test_mixformer_convmae.sh,just check it out yourself)
# {project root} shall be replaced by your own root
# {your_dataset_name} shall be the name of the dataset that you wanna run your experiment
# you may need to train the large model on you own