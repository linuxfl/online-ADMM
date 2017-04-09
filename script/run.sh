#!/bin/bash
mpirun -machinefile hostfile -np 2 ./bin/admm ./conf/admm.conf
