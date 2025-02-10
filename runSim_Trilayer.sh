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

tclsh oommf.tcl boxsi "Projects/Trilayer/Simulation/Sim_Trilayer.mif" -parameters "SimType 2 pb $3 qb $4" -restart 0 -threads 20

cd Projects/Trilayer/Simulation

mv -i SimTrilayerBottomM0*.omf m0filebottom.omf

cd ~/Desktop/oommf

tclsh oommf.tcl boxsi "Projects/Trilayer/Simulation/Sim_Trilayer.mif" -parameters "SimType 0" -restart 0 -threads 20

mkdir Projects/Trilayer/Energy\ Data\ Top

mkdir Projects/Trilayer/Magnetization\ Data\ Top

mkdir Projects/Trilayer/Energy\ Data\ Bottom

mkdir Projects/Trilayer/Magnetization\ Data\ Bottom

mkdir Projects/Trilayer/Energy\ Data\ Total

mkdir Projects/Trilayer/Magnetization\ Data\ Total

tclsh oommf.tcl avf2odt -average "line" -axis "z" -headers "none" -region -5.0e-8 -5.0e-8 0.0 5.0e-8 5.0e-8 1.0e-9 -ipat "Projects/Trilayer/Simulation/SimTrilayer-Oxs_RungeKuttaEvolve-evolver-Total_energy_density-*-*.oef"

tclsh oommf.tcl avf2odt -average "line" -axis "z" -headers "none" -region -5.0e-8 -5.0e-8 0.0 5.0e-8 5.0e-8 1.0e-9 -ipat "Projects/Trilayer/Simulation/SimTrilayer-Oxs_TimeDriver-Magnetization-*-*.omf" 

cd Projects/Trilayer/Simulation

mv -i SimTrilayer-*.odt ../

cd ../

mv -i SimTrilayer-Oxs_RungeKuttaEvolve-*.odt Energy\ Data\ Top

mv -i SimTrilayer-Oxs_TimeDriver-*.odt Magnetization\ Data\ Top

cd ~/Desktop/oommf

tclsh oommf.tcl avf2odt -average "line" -axis "z" -headers "none" -region -5.0e-8 -5.0e-8 0.0 5.0e-8 5.0e-8 -1.0e-9 -ipat "Projects/Trilayer/Simulation/SimTrilayer-Oxs_RungeKuttaEvolve-evolver-Total_energy_density-*-*.oef"

tclsh oommf.tcl avf2odt -average "line" -axis "z" -headers "none" -region -5.0e-8 -5.0e-8 0.0 5.0e-8 5.0e-8 -1.0e-9 -ipat "Projects/Trilayer/Simulation/SimTrilayer-Oxs_TimeDriver-Magnetization-*-*.omf" 

cd Projects/Trilayer/Simulation

mv -i SimTrilayer-*.odt ../

cd ../

mv -i SimTrilayer-Oxs_RungeKuttaEvolve-*.odt Energy\ Data\ Bottom

mv -i SimTrilayer-Oxs_TimeDriver-*.odt Magnetization\ Data\ Bottom

cd ~/Desktop/oommf

tclsh oommf.tcl avf2odt -average "point" -headers "none" -ipat "Projects/Trilayer/Simulation/SimTrilayer-Oxs_RungeKuttaEvolve-evolver-Total_energy_density-*-*.oef"

tclsh oommf.tcl avf2odt -average "point" -headers "none" -ipat "Projects/Trilayer/Simulation/SimTrilayer-Oxs_TimeDriver-Magnetization-*-*.omf" 

cd Projects/Trilayer/Simulation

mv -i SimTrilayer-*.odt ../

cd ../

mv -i SimTrilayer-Oxs_RungeKuttaEvolve-*.odt Energy\ Data\ Total

mv -i SimTrilayer-Oxs_TimeDriver-*.odt Magnetization\ Data\ Total

cd ~

done