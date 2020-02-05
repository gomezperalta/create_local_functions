# create_local_functions
This code computes the local functions needed as input data to train the Artificial Neural Networks. This is done with the program fijn_maker.py.

With fijn_maker.py you create a Python dictionary, where the key are the cif and the values are the local functions as a matrix.

fij_maker.py needs the next files as well:
*Wyckoff_finder.py 
*WyckoffSG_dict.npy
*datosrahm.csv

The pickle file "red_cod-db_computed.pkl" is an example of the pickle file asked by fijn_maker.py. 
The most important column of that pickle file is "WyckOcc", which is a dictionary of the Wyckoff sites occupied in the compound.

This code was developed in Linux and needs the Python module pymatgen.
