# Importing libraries
import numpy as np
import struct
import os

# Class to hold and read sink attributes
class readSinks():
    def __init__(self, type, filepath, nStart=0, nEnd=100, dumpMax=5000, newSinks=100):
        # The types and length of the data in the dumps.
        # Change this if using a different format or extra variables
        self.dataHash = "ddddddddddddliiii"
        self.dataBytes = 120

        # Use the different read functions based on type
        if type == 1:
            self.openSinkSnap(filepath)
        if type == 2:
            self.readSinkSnaps(filepath, nStart, nEnd)
        if type == 3:
            self.openSinkEvolution(filepath, dumpMax, newSinks)
        if type == 4:
            self.readAllEvolution(filepath)

    # Function to open up a singular sink snapshot and assign variables
    def openSinkSnap(self, filepath):
        # Open the file
        f = open(filepath, "rb")

        # Get the time and number of sinks
        self.time = struct.unpack("d", f.read(8))[0]
        self.nSinks = struct.unpack("i", f.read(4))[0]

        # Repeat the data stencil for each sink
        dataHashAll = self.dataHash * self.nSinks
        dataBytesAll = self.dataBytes * self.nSinks
        dataHashLength = len(self.dataHash)

        # Open all the sink data at once
        sinkData = struct.unpack(dataHashAll, f.read(dataBytesAll))

        # Assign each value to an array
        self.sinkX          = sinkData[0::dataHashLength]
        self.sinkY          = sinkData[1::dataHashLength]
        self.sinkZ          = sinkData[2::dataHashLength]
        self.sinkVX         = sinkData[3::dataHashLength]
        self.sinkVY         = sinkData[4::dataHashLength]
        self.sinkVZ         = sinkData[5::dataHashLength]
        self.sinkAX         = sinkData[6::dataHashLength]
        self.sinkAY         = sinkData[7::dataHashLength]
        self.sinkAZ         = sinkData[8::dataHashLength]
        self.sinkMass       = sinkData[9::dataHashLength]
        self.formationMass  = sinkData[10::dataHashLength]
        self.formationTime  = sinkData[11::dataHashLength]
        self.sinkID         = sinkData[12::dataHashLength]
        self.formationOrder = sinkData[15::dataHashLength]

    # Function to read multiple sink snapshots and assign variables
    def readSinkSnaps(self, filepath, nStart, nEnd):
        snapPrefix = "sink_snap_"

        # Creating arrays to store data
        self.totSinkMass = np.zeros(nEnd-nStart)
        self.numSinks = np.zeros(nEnd-nStart)
        self.snapTime = np.zeros(nEnd-nStart)

        # Looping through every snap
        for i in range(nStart, nEnd-nStart):
            # Working out what the name of the file will be
            if i <= 9: 
                filename = filepath + snapPrefix + "00" + str(i)
            elif i <= 99:
                filename = filepath + snapPrefix + "0" + str(i)
            else:
                filename = filepath + snapPrefix + str(i)

            # Opening snap
            self.openSinkSnap(filename)

            # Updating array info
            self.totSinkMass[i] = np.sum(self.sinkMass)
            self.numSinks[i] = self.nSinks
            self.snapTime[i] = self.time

    # Function to open up a sink evolution file and assign variables
    def openSinkEvolution(self, filepath, dumpMax=5000, newSinks=1000, sizeOverride=0):
        # Open the file initially
        with open(filepath, "rb") as fInit:
            # We'll open up the first time and nSinks to tell us how big our arrays should be
            _ = struct.unpack("d", fInit.read(8))[0]
            nSinksInit = struct.unpack("i", fInit.read(4))[0]

        # Creating variables to store scalar attributes
        self.time = []
        self.nSinks = []

        # When using the read all evolution we want to keep arrays all the same length for combining, so need an override 
        if sizeOverride == 0:
            sinkSize = nSinksInit + newSinks
        else:
            sinkSize = sizeOverride

        # Creating variables to store vector attributes (dTypes are to save memory where we can)
        self.sinkX          = np.zeros((dumpMax, sinkSize), dtype=np.float32)
        self.sinkY          = np.zeros((dumpMax, sinkSize), dtype=np.float32)
        self.sinkZ          = np.zeros((dumpMax, sinkSize), dtype=np.float32)
        self.sinkVX         = np.zeros((dumpMax, sinkSize), dtype=np.float32)
        self.sinkVY         = np.zeros((dumpMax, sinkSize), dtype=np.float32)
        self.sinkVZ         = np.zeros((dumpMax, sinkSize), dtype=np.float32)
        self.sinkMass       = np.zeros((dumpMax, sinkSize), dtype=np.float32)
        self.formationMass  = np.zeros((dumpMax, sinkSize), dtype=np.float32)
        self.formationTime  = np.zeros((dumpMax, sinkSize), dtype=np.float64)
        self.formationOrder = np.zeros((dumpMax, sinkSize), dtype=np.int64)
        self.sinkID         = np.zeros((dumpMax, sinkSize), dtype=np.int64)

        # Loop through the file and keep reading until we can't anymore
        with open(filepath, "rb") as f:
            dumpCount = 0

            while dumpCount < dumpMax:
                try:
                    # Getting the time and the number of sinks
                    self.time.append(struct.unpack("d", f.read(8))[0])
                    self.nSinks.append(struct.unpack("i", f.read(4))[0]) 
                    nSinksHere = self.nSinks[dumpCount]

                    # Repeat the data stencil for each sink
                    dataHashAll = self.dataHash * nSinksHere
                    dataBytesAll = self.dataBytes * nSinksHere
                    dataHashLength = len(self.dataHash)

                    # Open all the sink data at once
                    sinkData = struct.unpack(dataHashAll, f.read(dataBytesAll))

                    # Check we haven't exceeded the max number of sinks we can store
                    if self.nSinks[dumpCount] >= nSinksInit+999:
                        # Switching to append mode
                        raise BufferError("Not enough space for all the sinks.")

                    # Assign each value to the arrays
                    self.sinkX[dumpCount, :nSinksHere]          = sinkData[0::dataHashLength]
                    self.sinkY[dumpCount, :nSinksHere]          = sinkData[1::dataHashLength]
                    self.sinkZ[dumpCount, :nSinksHere]          = sinkData[2::dataHashLength]
                    self.sinkVX[dumpCount, :nSinksHere]         = sinkData[3::dataHashLength]
                    self.sinkVY[dumpCount, :nSinksHere]         = sinkData[4::dataHashLength]
                    self.sinkVZ[dumpCount, :nSinksHere]         = sinkData[5::dataHashLength]
                    self.sinkMass[dumpCount, :nSinksHere]       = sinkData[9::dataHashLength]
                    self.formationMass[dumpCount, :nSinksHere]  = sinkData[10::dataHashLength]
                    self.formationTime[dumpCount, :nSinksHere]  = sinkData[11::dataHashLength]
                    self.sinkID[dumpCount, :nSinksHere]         = sinkData[12::dataHashLength]
                    self.formationOrder[dumpCount, :nSinksHere] = sinkData[15::dataHashLength]

                    # Increase the counter
                    dumpCount += 1

                except:
                    break

        # Re assignning time and nsinks to arrays
        self.time = np.array(self.time)
        self.nSinks = np.array(self.nSinks)

    # Function to read all the sink evolution files and combine them
    def readAllEvolution(self, filepath):
        # Get all the files we need to read and sort them
        allFiles = os.listdir(filepath)
        allFiles = np.sort(allFiles)

        # Open the final file and use this to find how big our arrays need to be
        with open(filepath+allFiles[-1], "rb") as fInit:
            _ = struct.unpack("d", fInit.read(8))[0]
            nSinksEnd = struct.unpack("i", fInit.read(4))[0]

        # Loop through each of the files
        for file in allFiles:
            # Open the file using our sink evolution function
            self.openSinkEvolution(filepath+file, sizeOverride=nSinksEnd+1000)

            # Find how much of the arrays we've filled
            nDumps = len(self.time[self.time != 0])

            # Assign these values to arrays
            if file == allFiles[0]:
                # We create new arrays for the first file as they don't exist yet
                self.allTime            = self.time[:nDumps]
                self.allSinks           = self.nSinks[:nDumps]
                self.allX               = self.sinkX[:nDumps]
                self.allY               = self.sinkY[:nDumps]
                self.allZ               = self.sinkZ[:nDumps]
                self.allVX              = self.sinkVX[:nDumps]
                self.allVY              = self.sinkVY[:nDumps]
                self.allVZ              = self.sinkVZ[:nDumps]
                self.allIDs             = self.sinkID[:nDumps]
                self.allMasses          = self.sinkMass[:nDumps]
                self.allFormationTimes  = self.formationTime[:nDumps]
                self.allFormationMass   = self.formationMass[:nDumps]
                self.allFormationOrder  = self.formationOrder[:nDumps]
            else:
                # Otherwise we simply add to the existing ones
                if len(self.time) == 0:
                    # Sometimes the evolution file is empty
                    pass
                else:
                    # We need to account for the files overlapping, like if we start from a snapshot file instead of restart file
                    newStart = self.time[0]

                    if newStart < self.allTime[-1]:
                        # Find where the new array fits into the old one
                        startIndex = np.where(self.allTime > newStart)[0][0]
                    else:
                        startIndex = -1

                # Now join everything together
                self.allTime            = np.concatenate((self.allTime[:startIndex], self.time[:nDumps]))
                self.allSinks           = np.concatenate((self.allSinks[:startIndex], self.nSinks[:nDumps]))
                self.allX               = np.concatenate((self.allX[:startIndex], self.sinkX[:nDumps]))
                self.allY               = np.concatenate((self.allY[:startIndex], self.sinkY[:nDumps]))
                self.allZ               = np.concatenate((self.allZ[:startIndex], self.sinkZ[:nDumps]))
                self.allVX              = np.concatenate((self.allVX[:startIndex], self.sinkVX[:nDumps]))
                self.allVY              = np.concatenate((self.allVY[:startIndex], self.sinkVZ[:nDumps]))
                self.allVZ              = np.concatenate((self.allVZ[:startIndex], self.sinkVZ[:nDumps]))
                self.allIDs             = np.concatenate((self.allIDs[:startIndex], self.sinkID[:nDumps]))
                self.allMasses          = np.concatenate((self.allMasses[:startIndex], self.sinkMass[:nDumps]))
                self.allFormationTimes  = np.concatenate((self.allFormationTimes[:startIndex], self.formationTime[:nDumps]))
                self.allFormationMass   = np.concatenate((self.allFormationMass[:startIndex], self.formationMass[:nDumps]))
                self.allFormationOrder  = np.concatenate((self.allFormationOrder[:startIndex], self.formationOrder[:nDumps]))

        # Re-assign to the normal array names
        self.time           = self.allTime
        self.nSinks         = self.allSinks
        self.sinkX          = self.allX
        self.sinkY          = self.allY
        self.sinkZ          = self.allZ
        self.sinkVX         = self.allVX
        self.sinkVY         = self.allVY
        self.sinkVZ         = self.allVZ
        self.sinkID         = self.allIDs
        self.sinkMass       = self.allMasses
        self.formationTime  = self.allFormationTimes
        self.formationMass  = self.allFormationMass
        self.formationOrder = self.allFormationOrder