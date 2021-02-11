#!/bin/bash

if [ "$1" = "ACSC" ]
then
    python ./model/acsc.py $1 $2 $3 $4
elif [ "$1" = "ATSC" ]
then
    python ./model/atsc.py $1 $2 $3 $4
elif [ "$1" = "ATE" ]
then
    python ./model/ate.py $1 $2 $3 $4
else
    echo "Nonexistent Task !"
fi