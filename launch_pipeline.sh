#!/bin/bash

# 1,1
#NFULL=20
#GRIDSPARSE=10
#GRIDFULL=30
#NEPOCHS=1000
#LR=0.01

# 2,1
NFULL=20
#GRIDSPARSE=5
#GRIDFULL=10
#NEPOCHS=3000
NEPOCHS=750
#LR=0.1
#HD=0
CONVRES=5
CONVWIND=2
CONVBAND=0.1

#for DATASET in 'Coffee' 'DistalPhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'ProximalPhalanxOutlineAgeGroup' 'ProximalPhalanxOutlineCorrect' 'ProximalPhalanxTW' 'ItalyPowerDemand' 'GunPoint' 'GunPointAgeSpan' 'GunPointMaleVersusFemale' 'PowerCons'
for DATASET in 'Coffee' 'DistalPhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'ProximalPhalanxOutlineAgeGroup' 'ProximalPhalanxOutlineCorrect' 'ProximalPhalanxTW' 'ECG200' 'ItalyPowerDemand' 'MedicalImages' 'Plane' 'SwedishLeaf' 'GunPoint' 'GunPointAgeSpan' 'GunPointMaleVersusFemale' 'GunPointOldVersusYoung' 'PowerCons' 'SyntheticControl'
do
  for GRIDSPARSE in 5
  do
    for GRIDFULL in 10
    do
      for LR in 0.001 0.01 0.1 1 10
      do
        for HD in 0-1
        do
          DATAPATH="./UCR/$DATASET/$DATASET"
          oarsub -p "cputype='xeon'" -l /nodes=1,walltime=20:00:00 "./pipeline.sh $DATAPATH $NFULL $GRIDSPARSE $GRIDFULL $NEPOCHS $LR $CONVRES $CONVWIND $CONVBAND $HD"
#          oarsub -p "cputype='xeon' and mem > 500000" -l /nodes=1,walltime=20:00:00 "./pipeline.sh $DATAPATH $NFULL $GRIDSPARSE $GRIDFULL $NEPOCHS $LR $CONVRES $CONVWIND $CONVBAND $HD"
#          ./pipeline.sh $DATAPATH $NFULL $GRIDSPARSE $GRIDFULL $NEPOCHS $LR $CONVRES $CONVWIND $CONVBAND $HD
        done
      done
    done
  done
done
