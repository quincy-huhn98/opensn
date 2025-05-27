#!/bin/bash

scatt=(0 0.25 0.5 0.75 0.9 0.95 0.99)

for i in "${!scatt[@]}"; do
    value="${scatt[$i]}"

    ../build/python/opensn -i offline_reed.py -p scatt=$value -p id=$i

done

echo "Merge"
../build/python/opensn -i merge_reed.py -p id=$i