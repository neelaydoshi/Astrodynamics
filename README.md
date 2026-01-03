# Astrodynamics
This repo contains some personal projects for learning astrodynamics.


## Satellite Tracking
The [satellite_tracking.ipynb](https://github.com/neelaydoshi/Astrodynamics/blob/main/Satellite_Tracking/satellite_tracking.ipynb) shows a simple example of tracks a satellite in LEO, given its two-line elements (TLE) data and the location of the ground-based tracker. [Skyfield](https://github.com/skyfielders/python-skyfield) library is used for the simulation.

<p align="center">
  <img src="./Satellite_Tracking/satellite_trajectory_1.6h.png" width="1000" />
  <br>
  <b>Figure 1: Satellite ground track and position coordinates. </b>
</p>


## Orbit Transfer
The [satellite_transfer_1.ipynb](https://github.com/neelaydoshi/Astrodynamics/blob/main/Orbit_Transfer/satellite_transfer_1.ipynb) runs a simple [code](https://github.com/neelaydoshi/Astrodynamics/blob/main/Orbit_Transfer/utils.py) for visualizing a satellite being launched from Earth and arriving to Mars.

It is a simple 3-body problem code wherein trajectory of the satellite is determined by the gravitational pull of both the Sun and Mars.

You can play around with the satellite trajectory by:
- changing the initial position of Earth and Mars.
- changing the launch velocity of the satellite.
- changing the duration of the simulation.

<p align="center">
  <img src="./Orbit_Transfer/satellite_transfer_1.gif" width="600" />
  <br>
  <b>Figure 1: Satellite transfer from Earth to Mars. </b>
</p>
