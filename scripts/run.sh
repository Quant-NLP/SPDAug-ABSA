#!/bin/bash

if [ "$1" = "ACSC" ]
then
    python ./model/asc.py $1 $2 $3 $4
elif [ "$1" = "ATSC" ]
then
    python ./model/asc.py $1 $2 $3 $4
elif [ "$1" = "ATE" ]
then
    python ./model/ate.py $1 $2 $3 $4
elif [ "$1" = "SC" ]
then
    python ./model/sc.py $1 $2 $3 $4
else
    echo "Nonexistent Task !"
fi