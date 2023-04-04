# Authorial Credit
Please note that this is a rewrite of Bas van Opheusden's [Four-in-a-row implementation](https://github.com/basvanopheusden/fourinarow). Consider him the author of this code for citation purposes.

# Dependencies
The build system used by this repo is cmake:
sudo apt-get install cmake

Additionally, this repo requires SWIG to build:
sudo apt-get install swig  

# Four-in-a-row build instructions
From a fresh checkout, create a build directory:  
mkdir build  

Then from the build directory (cd build):  
cmake ..  
cmake --build .  

To run tests, simply execute ./tests in the build output.  

To install the required Python packages, from the model_fitting directory run:  
pip install -r requirements.txt  

To fit a model, from the model_fitting directory run:
python model_fit.py <path_to_game_csv>  