# Reproduce experiments of vldb2022 research paper on rank aggregation

## How to run it :

1. import project:
`git pull https://github.com/pierreandrieu/experiments-vldb`

2. then, go to the project folder

3. build container:
`docker build . --tag name_of_container`

4. run docker:
`docker container run -it --rm name_of_container args`

## Remarks
Arguments use:

- args are the arguments for the main script.  
- Example:
`docker container run -it --rm name_of_container exp=3,5,6`  
will run the experiments 3, 5 and 6 of the paper (integers between 1 and 6, separated with a coma, no space).
- To reproduce all the experiments, argument is "all".
- A user guide is returned if no arguments are given

The experiments are run in increasing order of time computation
