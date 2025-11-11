#!/bin/bash

base="$1"

if [ ! -d "$base" ]; then
    echo "$base not found"
    exit 1
fi


homedir=$(pwd)

for dir in $base/sparc_* ; do
    if [ -d "$dir" ]; then
	echo $dir
	if [ -f "$dir/queued" ]; then
	    echo "Already queued.  Skipping."
	elif [ -f "$dir/C1.h5" ]; then
	    echo "Output exists.  Skipping."
	else
	    echo "Launching $dir"
	    cd $dir
	    sbatch singlebatchjob.perlmutter
	    touch queued
	    cd $homedir
	fi
    fi
done
