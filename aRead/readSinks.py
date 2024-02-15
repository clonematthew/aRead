# Importing libraries
import numpy as np
import struct
import os

# Class to hold the sink attributes
class readSinks():
    # Initialisation function
    def __init__(self, type, filepath, nSnaps=1, sinkMax=1000, dumpMax=5000):
        # Initialise buffer size flag
        self.bufferSizeError = False

        # If we're only opening a single file 
        if type == 1:
            self.openSinkSnap(filepath)
        # If we're opening multiple sink particle files
        elif type == 2:
            self.readAllSnaps(filepath, nSnaps)
        # If we're opening the sink evolution files
        elif type == 3:
            self.readSinkEvolution(filepath, sinkMax, dumpMax)
        # If we have multiple sink evolution files to combine
        elif type== 4:
            self.readAllEvolution(filepath, sinkMax, dumpMax)
        else:
            pass

    # Function to read the sink buffer
    def readSinkBuffer(self, f):
        # Looping through each sink to get its data
        for i in range(self.nSinks):
            # Reading the doubles in the buffer
            sinkData = struct.unpack("dddddddddddd", f.read(8 * 12))

            # Positions
            self.sinkX[i] = sinkData[0]
            self.sinkY[i] = sinkData[1]
            self.sinkZ[i] = sinkData[2]

            # Velocites
            self.sinkVX[i] = sinkData[3]
            self.sinkVY[i] = sinkData[4]
            self.sinkVZ[i] = sinkData[5]

            # Accelerations
            self.sinkAX[i] = sinkData[6]
            self.sinkAY[i] = sinkData[7]
            self.sinkAZ[i] = sinkData[8]

            # Masses
            self.sinkMass[i] = sinkData[9]

            # Formation characteristics
            self.formationMass[i] = sinkData[10]
            self.formationTime[i] = sinkData[11]

            # Reading the sink ID
            sinkData = struct.unpack("l", f.read(8))
            self.sinkID[i] = sinkData[0]

            # Reading the rest of the dataset
            sinkData = struct.unpack("iiii", f.read(4* 4))

            self.sinkHomeTask[i] = sinkData[0]
            self.sinkIndex[i] = sinkData[1]
            self.formationOrder[i] = sinkData[2]

    # Function to open a sink snapshot file 
    def openSinkSnap(self, filepath):
        # Opening the file
        f = open(filepath, "rb")

        # Reading the header information
        self.time = struct.unpack("d", f.read(8))[0]
        self.nSinks = struct.unpack("i", f.read(4))[0]

        # Creating arrays to store everything
        self.sinkX = np.zeros(self.nSinks)
        self.sinkY = np.zeros(self.nSinks)
        self.sinkZ = np.zeros(self.nSinks)

        self.sinkVX = np.zeros(self.nSinks)
        self.sinkVY = np.zeros(self.nSinks)
        self.sinkVZ = np.zeros(self.nSinks)

        self.sinkAX = np.zeros(self.nSinks)
        self.sinkAY = np.zeros(self.nSinks)
        self.sinkAZ = np.zeros(self.nSinks)

        self.sinkMass = np.zeros(self.nSinks)
        self.sinkID = np.zeros(self.nSinks)

        self.sinkHomeTask = np.zeros(self.nSinks)
        self.sinkIndex = np.zeros(self.nSinks)

        self.formationMass = np.zeros(self.nSinks)
        self.formationTime = np.zeros(self.nSinks)
        self.formationOrder = np.zeros(self.nSinks)

        # Looping through each sink to get its data
        self.readSinkBuffer(f)

    # Function to read all the sink files in a given directory
    def readAllSnaps(self, directory, nSnaps):
        snapPrefix = "sink_snap_"

        # Creating arrays to store data
        self.totSinkMass = np.zeros(nSnaps)
        self.numSinks = np.zeros(nSnaps)
        self.snapTime = np.zeros(nSnaps)

        # Looping through every snap
        for i in range(nSnaps):
            # Working out what the name of the file will be
            if i <= 9: 
                filename = directory + snapPrefix + "00" + str(i)
            elif i <= 99:
                filename = directory + snapPrefix + "0" + str(i)
            else:
                filename = directory + snapPrefix + str(i)

            # Opening snap
            self.openSinkSnap(filename)

            # Updating array info
            self.totSinkMass[i] = np.sum(self.sinkMass)
            self.numSinks[i] = self.nSinks
            self.snapTime[i] = self.time

    # Function to read the sink evolution files
    def readSinkEvolution(self, filepath, sinkMax, dumpMax):
        # Creating arrays to store sink variables
        self.time = []
        self.nSinks = []
        self.sinkX = np.zeros((dumpMax, sinkMax))
        self.sinkY = np.zeros((dumpMax, sinkMax))
        self.sinkZ = np.zeros((dumpMax, sinkMax))
        self.sinkVX = np.zeros((dumpMax, sinkMax))
        self.sinkVY = np.zeros((dumpMax, sinkMax))
        self.sinkVZ = np.zeros((dumpMax, sinkMax))
        self.sinkMass = np.zeros((dumpMax, sinkMax))
        self.formationMass = np.zeros((dumpMax, sinkMax))
        self.formationTime = np.zeros((dumpMax, sinkMax))
        self.sinkID = np.zeros((dumpMax, sinkMax))
        self.formationOrder = np.zeros((dumpMax, sinkMax))

        # Creating counter variable
        dumpCount = 0

        # Opening the binary file
        with open(filepath, "rb") as f:
            while dumpCount < dumpMax and not self.bufferSizeError:
                try:
                    # Getting the time and the number of sinks
                    self.time.append(struct.unpack("d", f.read(8))[0])
                    self.nSinks.append(struct.unpack("i", f.read(4))[0])

                    # Check if we're going to hit the sink limit
                    if self.nSinks[dumpCount] >= (sinkMax-2):
                        # Return and error and break
                        print("Too many sinks, increase sink buffer size")
                        self.bufferSizeError = True
 
                    # Looping through every sink
                    for i in range(self.nSinks[dumpCount]):
                        # Reading sink info
                        sinkData = struct.unpack("dddddddddddd", f.read(8 * 12))

                        # Storing the sink properties we care about
                        self.sinkX[dumpCount, i] = sinkData[0]
                        self.sinkY[dumpCount, i] = sinkData[1]
                        self.sinkZ[dumpCount, i] = sinkData[2]
                        self.sinkVX[dumpCount, i] = sinkData[3]
                        self.sinkVY[dumpCount, i] = sinkData[4]
                        self.sinkVZ[dumpCount, i] = sinkData[5]
                        self.sinkMass[dumpCount, i] = sinkData[9]
                        self.formationMass[dumpCount, i] = sinkData[10]
                        self.formationTime[dumpCount, i] = sinkData[11]

                        # Reading the rest of the buffer
                        sinkData = struct.unpack("l", f.read(8))
                        self.sinkID[dumpCount, i] = sinkData[0]

                        sinkData = struct.unpack("iiii", f.read(4 * 4))
                        self.formationOrder[dumpCount, i] = sinkData[2]

                    # Counting up the dump counter
                    dumpCount += 1 
                except:
                    dumpCount += 1

        # Re assignning time and nsinks to arrays
        self.time = np.array(self.time)
        self.nSinks = np.array(self.nSinks)

    # Function to read all the evolution files in a directory
    def readAllEvolution(self, filepath, sinkMax, dumpMax):
        # List the files in this directory
        allFiles = os.listdir(filepath)

        # Sort the evolution files
        vals = []
        for file in allFiles:
            val = int(file[-2:])
            vals.append(val)

        inds = np.argsort(vals)
        allFiles = np.array(allFiles)
        allFiles = allFiles[inds]

        # Loop through each file
        while not self.bufferSizeError:
            for file in allFiles:
                # Open file and extract masses
                sinkData = self.readSinkEvolution(filepath + file, sinkMax=sinkMax, dumpMax=dumpMax)
                index = len(self.time[self.time != 0])

                # If first file, make new arrays
                if file == allFiles[0]:
                    self.allTime = self.time[:index]
                    self.allSinks = self.nSinks[:index]
                    self.allIDs = self.sinkID[:index]
                    self.allOrders = self.formationOrder[:index]
                    self.allFormationTimes = self.formationTime[:index]
                    self.allFormationMass = self.formationMass[:index]
                    self.allMasses = self.sinkMass[:index]
                    self.allX = self.sinkX[:index]
                    self.allY = self.sinkY[:index]
                    self.allZ = self.sinkZ[:index]
                    self.allVX = self.sinkVX[:index]
                    self.allVY = self.sinkVY[:index]
                    self.allVZ = self.sinkVZ[:index]

                # Otherwise, add to existing
                else:
                    if len(self.time) == 0:
                        pass
                    else:
                        # Find where this new file starts
                        newStart = self.time[0]

                        if newStart < self.allTime[-1]:
                            # Find where we got to this time
                            startIndex = np.where(self.allTime > newStart)
                            startIndex = startIndex[0][0]
                        else:
                            startIndex = -1

                        self.allTime = np.concatenate((self.allTime[:startIndex], self.time[:index]))
                        self.allSinks = np.concatenate((self.allSinks[:startIndex], self.nSinks[:index]))
                        self.allIDs = np.concatenate((self.allIDs[:startIndex], self.sinkID[:index]))
                        self.allOrders = np.concatenate((self.allOrders[:startIndex], self.formationOrder[:index]))
                        self.allFormationTimes = np.concatenate((self.allFormationTimes[:startIndex], self.formationTime[:index]))
                        self.allFormationMass = np.concatenate((self.allFormationMass[:startIndex], self.formationMass[:index]))
                        self.allMasses = np.concatenate((self.allMasses[:startIndex], self.sinkMass[:index]))
                        self.allX = np.concatenate((self.allX[:startIndex], self.sinkX[:index]))
                        self.allY = np.concatenate((self.allY[:startIndex], self.sinkY[:index]))
                        self.allZ = np.concatenate((self.allZ[:startIndex], self.sinkZ[:index]))
                        self.allVX = np.concatenate((self.allVX[:startindex], self.sinkVX[:index]))
                        self.allVY = np.concatenate((self.allVY[:startindex], self.sinkVZ[:index]))
                        self.allVZ = np.concatenate((self.allVZ[:startindex], self.sinkVZ[:index]))

            # End the while loop
            self.bufferSizeError = True

        # Re assign data
        self.time = self.allTime
        self.nSinks = self.allSinks
        self.sinkID = self.allIDs
        self.formationOrder = self.allOrders
        self.formationTime = self.allFormationTimes
        self.formationMass = self.allFormationMass
        self.sinkMass = self.allMasses
        self.sinkX = self.allX
        self.sinkY = self.allY
        self.sinkZ = self.allZ
        self.sinkVX = self.allVX
        self.sinkVY = self.allVY
        self.sinkVZ = self.allVZ