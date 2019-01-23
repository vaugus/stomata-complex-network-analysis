#!/bin/bash
SECONDS=0

if [ $1 -eq 1 ]
then

    for a in F1 F2 F3 F4 F5 F6 F7 F8 F9 F10
    do
        for b in 1 2 3 4
        do
            python3 main.py < input/segunda/controle/$a/$b.in > output/segunda/controle/$a/$b.out
        done
    done
fi

if [ $1 -eq 2 ]
then 
    for a in terca quarta quinta
    do
        for b in controle quente
        do
            for c in F1 F2 F3 F4 F5
            do
                for d in 1 2 3 4
                do  
                    python3 main.py < input/$a/$b/$c/$d.in > output/$a/$b/$c/$d.out
                done
            done
        done
    done
fi

ELAPSED="Elapsed: $(($SECONDS / 3600))hrs $((($SECONDS / 60) % 60))min $(($SECONDS % 60))sec"
echo $ELAPSED