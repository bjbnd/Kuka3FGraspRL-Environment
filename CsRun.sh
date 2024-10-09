#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 <num_loops>"
    exit 1
fi

# Evaluation env
terminator -e "/home/major/Desktop/Simulation/CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu22_04/./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23001 -h" &
sleep 5
# Extract the number of loops from the argument  -GwsRemoteApi.autoStart=false
last_port=$(( $1 + 23001 ))
for ((i = 23002; i <= last_port; i++)); do
    # Open Terminator window and run command
    terminator -e "/home/major/Desktop/Simulation/CoppeliaSim_Edu_V4_6_0_rev18_Ubuntu22_04/./coppeliaSim.sh -GzmqRemoteApi.rpcPort=$i -h" &
    sleep 5
done


