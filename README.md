### aRead - A package for opening AREPO Snapshot Files

This package provides classes to open, store and (in limited ways) manipulate snapshot data created by the moving-mesh code AREPO. 
Note - readAREPO opens .hdf5 snapshots only, i.e snapshot file type 3. 

#### readSinks

Function for reading in snapshots made by the SINK_PARTICLES module for AREPO. There are four functions available in this class:

- openSinkSnap: Opens one sink snapshot.

- readSinkSnaps: Reads a number of sink snapshots and calculates the total mass in sinks, number of sinks and time.

- openSinkEvolution: Opens one sink evolution file.

- readSinkEvolution: Reads all sink evolution files in a directory and combines all their data into arrays.

If you're using different modules that change the format of the sink snapshots, the dataHash and dataBytes attribures can be modified to reflect this. 