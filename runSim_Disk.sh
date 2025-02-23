#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100GB 
#SBATCH --time=23:59:00

cd ~/Desktop/oommf

tclsh oommf.tcl boxsi "Projects/Trilayer/Simulation/Sim_Trilayer.mif" -parameters "SimType 1 pt $1 qt $2" -restart 0 -threads 20

cd Projects/Trilayer/Simulation

mv -i SimTrilayerTopM0*.omf m0filetop.omf

cd ~/Desktop/oommf

mkdir Projects/Trilayer/Energy\ Data\ Total

mkdir Projects/Trilayer/Magnetization\ Data\ Total

tclsh oommf.tcl boxsi "Projects/Trilayer/Simulation/Sim_Trilayer.mif" -parameters "SimType 4" -restart 0 -threads 20

tclsh oommf.tcl avf2odt -average "point" -headers "none" -ipat "Projects/Trilayer/Simulation/SimTrilayer-Oxs_RungeKuttaEvolve-evolver-Total_energy_density-*-*.oef"

tclsh oommf.tcl avf2odt -average "point" -headers "none" -ipat "Projects/Trilayer/Simulation/SimTrilayer-Oxs_TimeDriver-Magnetization-*-*.omf" 

cd Projects/Trilayer/Simulation

mv -i SimTrilayer-*.odt ../

cd ../

mv -i SimTrilayer-Oxs_RungeKuttaEvolve-*.odt Energy\ Data\ Total

mv -i SimTrilayer-Oxs_TimeDriver-*.odt Magnetization\ Data\ Total

cd ~

done