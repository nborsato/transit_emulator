# Transit Emulator

A simple transit emulator used to test detection significances of the cross-correlation function. This emulator was used in Borsato et al. 2023 to estimate the feasibility of using 2m telescopes for transit studies of ultra-hot Jupiters.

## Installation and Setup

This repository uses standard Python libraries, along with the cross-correlation library [tayph](https://github.com/Hoeijmakers/tayph), which needs to be installed. Once all relevant libraries are installed, the program should run right off the bat. You can control the emulator's parameters in the `0_parameters.json` file. Use these to change the emulator's configuration. The example spectra come from the Mantis Network templates, which are also used as cross-correlation templates, however one could easily use other transmission spectra.

## Usage
To run the code, just run the numbered python files one after the other. 

The primary purpose of this program is to simulate transits of exoplanets using pre-generated transmission spectra and convert them into spectra one might observe from instruments such as HARPS-North or the FOCES spectrograph. It currently uses the estimated S/N of the Wendelstein telescopes; however, this can easily be changed to reflect a different observing location.

## Features

The code is built in a modular way, allowing for upgrades or alterations to use more complex S/N profiles and theoretical stellar spectra if needed.

## Contributions

If you are interested in contributing to or improving the repository, I would be happy to add you to the project. Please feel free to reach out if you're interested.
