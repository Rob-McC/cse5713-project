##################################################################################################
#                                                                                                #
#            the code implemented in pytorch, pytorch does not support python 2                  #
#                                                                                                #
##################################################################################################
REQUIRED PYTHON LIBRARY
natsort
# pip3 isntall natsort            for python 3

re
# pip3 install regex              for python 3

pytorch --- without cuda\does not support python 2
# pip3 install torch torchvision                                                                                        for linux/MAC
# pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html                  for windows

numpy
# pip3 install numpy

REQUIRED SYSTEM ARGUMENTS

The code run with 2 arguments
argv[1] = training folder contain data file  # it read from train input files from a folder
argv[2] = testing folder contain test data file   # it read from test input files from a folder

e.g
train\
   0input.txt
   1input.txt
   ....
   10000input.txt

test\
   0input.txt
   1input.txt
   ....
   10000input.txt

the code will generate the new position file in the created folder
run evaluation RMSE, the new generated position folder and test folder should be in the same directory

