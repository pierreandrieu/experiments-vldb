# experiments-vldb
# to reproduce experiments of research vldb paper 

# import project: 
git pull https://github.com/pierreandrieu/experiments-vldb

# then,  go to project

# build container: 
docker build . --tag name_of_container

# run docker:
docker container run -it --rm name_of_container args

#note about arguments:
Note: args are the arguments for the main script. 
Example: docker container run -it --rm name_of_container exp=3,5,6 
will run the experiments 3, 5 and 6 of the paper
To reproduce all the experiments, argument is "all"

A user guide is returned if no arguments are empty

# remark
Note: the experiments are run in increasing time of time computation




