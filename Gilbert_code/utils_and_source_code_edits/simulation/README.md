## File Descriptions:
- #### traci.py 
    Source Location: edited from ~/flow/flow/core/kernel/simulation\
    This file contains code/methods that are shared amongst environments in FLOW source code. The lines of interest are 119 to 122. These lines enable SUMO to output an xml containing trip infos that we collect travel times from.
    
    ##### Note (TODO) : this file file edition overwrites emission output path with trip-info output. (line ~121). Adding a seperate trip-info output path to flow would be difficult (without changed flow's source code) at this time.
