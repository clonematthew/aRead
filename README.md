### aRead - A package for opening AREPO Snapshot Files

This package provides classes to open, store and (in limited ways) manipulate snapshot data created by the moving-mesh code AREPO. 
Note - readAREPO opens .hdf5 snapshots only, i.e snapshot file type 3. 

#### readSinks

readSinks has 4 different modes of operation:

- 1 - Open a single sink snapshot, provide the path to the snapshot including the snapshot name.

- 2 - Open multiple sink snapshots, provide the path to the *directory* containing sink snapshots and the number of sink snapshots to read (from 0-n).

- 3 - Open a sink evolution file, provide the path to the evolution file including its name.

- 4 - Open multiple sink evolution files (and combine), provide the path to the *directory* containing the sink evolution files.

Note - mode 2 only provides the total mass in sinks, the number of sinks and the time, whereas modes 1, 3 and 4 provide info about the individal sinks. 
