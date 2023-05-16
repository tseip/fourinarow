# Authorial Credit
Please note that this is a rewrite of Bas van Opheusden's [Four-in-a-row implementation](https://github.com/basvanopheusden/fourinarow). Consider him the author of this code for citation purposes.

# Dependencies
This repository requires Python 3 to be installed on your machine.

The build system used by this repo is cmake:
sudo apt-get install cmake

Additionally, this repo requires SWIG to build:
sudo apt-get install swig 

If on Windows, install cmake using the Windows installer, and then download and unzip SWIG to a directory. Add the following environment variables for all users:
SWIG_DIR <path to unzipped SWIG directory>
SWIG_EXECUTABLE <path to swig.exe in unzipped SWIG directory>
Finally, add the swig directory (SWIG_DIR above) to your PATH system variable.

# Four-in-a-row build instructions
From a fresh checkout, create a build directory:  
mkdir build  

Then from the build directory (cd build):  
cmake ..  
cmake --build .  
Note, if on Windows, depending on your Python installation options, you may not have the debug Python libraries installed. If this is the case, specify the Release build at build-time:
cmake --build . --config Release

To run tests, simply execute ./tests in the build output.  

To install the required Python packages, from the model_fitting directory run:  
pip install -r requirements.txt  

To fit a model, from the model_fitting directory run:
python model_fit.py <path_to_game_csv>  