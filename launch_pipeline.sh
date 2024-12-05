#!/bin/bash

NFULL=20
NEPOCHS=750
CONVRES=5
CONVWIND=2
CONVBAND=0.1

for DATASET in 'Coffee' 'DistalPhalanxOutlineAgeGroup' 'DistalPhalanxOutlineCorrect' 'DistalPhalanxTW' 'ProximalPhalanxOutlineAgeGroup' 'ProximalPhalanxOutlineCorrect' 'ProximalPhalanxTW' 'ECG200' 'ItalyPowerDemand' 'MedicalImages' 'Plane' 'SwedishLeaf' 'GunPoint' 'GunPointAgeSpan' 'GunPointMaleVersusFemale' 'GunPointOldVersusYoung' 'PowerCons' 'SyntheticControl'
do
  for GRIDSPARSE in 5
  do
    for GRIDFULL in 10
    do
      for LR in 0.001
      do
        for HD in 0-1
        do
          DATAPATH="./UCR/$DATASET/$DATASET"
          ./pipeline.sh $DATAPATH $NFULL $GRIDSPARSE $GRIDFULL $NEPOCHS $LR $CONVRES $CONVWIND $CONVBAND $HD
        done
      done
    done
  done
done
