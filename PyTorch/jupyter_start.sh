#!/bin/bash
conda info -e > jupyter_start.tmp.txt

flag=false
#if there is env "pytorch"

while read line
do
    for word in $line
    do
        if [ "$word" == "pytorch" ]; then
            flag=true
        fi
        break
    done
done < jupyter_start.tmp.txt

rm jupyter_start.tmp.txt 2>nul

if [ "$flag" == "true" ]; then
    # conda init bash
    source activate pytorch
    python ./test_torch.py
    # python -m ipykernel install --user --name pytorch --display-name "pytorch"
    jupyter notebook
    conda deactivate
    # jupyter kernelspec remove pytorch
else
    echo ERROR: env "pytorch" does not exist.
fi