# Folder for required data in training and testing

## The following files are required under subfolders of each city:

- prob.json: array in size **[336]\[number of regions]\[number of regions]**, recording the probability for people in the i-th region to travel to the j-th during the t-th time step, during one week with the time resolution of 30 mins.
- flow.json: array in size **[number of regions]**, recording the relative strength of population flow in each region.
- dense.json: array in size **[number of regions]**, recording the relative population density in each region.
- pop_region.json: array in size **[number of regions]**, recording the population size in each region.
- start.json: array in size **[number of regions]\[8]**, recording the initial number of people in 8 states, each region.

*The folders 'city_sample' in 'data\', 'model\' and 'result\' are empty and are used only for an example. The training and testing program can be applied to any city once the according data are prepared as above.