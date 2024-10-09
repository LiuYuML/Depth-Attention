download the pytracking source code at https://github.com/franktpmvu/NeighborTrack
download weighted file as well on their github
put the weighted file in arbitrary path 
rename xxx_params as "xxx_attn_" and move to {project root}/trackers/ostrack/lib/test/parameter
rename xxx_attn_tracker as "xxx_attn_" and move to {project root}/trackers/ostrack/lib/test/tracker
by the way,you may need to correct some content in the line 24 of xxx_attn_params
cd {project root}
and run the python script