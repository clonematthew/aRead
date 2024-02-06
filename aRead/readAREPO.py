# Importing libraries
import numpy as np
import h5py

# Constants
mProt = 1.67e-24 # cgs units
kB =  1.38e-16   # cgs unit

# Function to open arepo data 
class readAREPO():
    def __init__(self, filename, chemistry=True):
        # Loading in hdf5 file and assigninig data
        self.dataDict = self.snapshotRead(filename)

        # The 3-D Position and Velocitiy arrays
        positions = self.dataDict["Coordinates"]
        splitPos = np.array_split(positions, 3, axis=1)

        self.x = splitPos[0].reshape(self.nParticles)
        self.y = splitPos[1].reshape(self.nParticles)
        self.z = splitPos[2].reshape(self.nParticles)

        velocities = self.dataDict["Velocities"]
        splitVels = np.array_split(velocities, 3, axis=1)

        self.vx = splitVels[0].reshape(self.nParticles)
        self.vy = splitVels[1].reshape(self.nParticles) 
        self.vz = splitVels[2].reshape(self.nParticles) 

        accelerations = self.dataDict["Acceleration"]
        splitAcc = np.array_split(accelerations, 3, axis=1)

        self.ax = splitAcc[0].reshape(self.nParticles)
        self.ay = splitAcc[1].reshape(self.nParticles) 
        self.az = splitAcc[2].reshape(self.nParticles) 

        # The rest of the scalar attributes
        self.rho = self.dataDict["Density"]
        self.mass = self.dataDict["Masses"] 
        self.u = self.dataDict["InternalEnergy"]
        self.ids = self.dataDict["ParticleIDs"]
        self.dustTemp = self.dataDict["DustTemperature"]
        self.chem = self.dataDict["ChemicalAbundances"]
        self.potential = self.dataDict["Potential"]
        self.maxPotential = self.dataDict["PotentialPeak"]
        self.numberDensity = self.rho / (1.4 * mProt) 
        self.velocityDivergence = self.dataDict["VelocityDivergence"]

        # Calculating the temperature of the gas
        yn = self.rho / ((1 + 4 * 0.1) * mProt)
        ynTot = (1 + 0.1 - self.chem[:,0] + self.chem[:,1]) * yn
        energy = self.u * self.rho
        self.gasTemp = 2 * energy  / (3 * ynTot * kB())

        # Loading chemistry if desired
        if chemistry:
            self.extractChemistry()

    # Function to extract all the chemistry information
    def extractChemistry(self, totC=1.4e-4, totO=3.2e-4):
        # Getting the different abundances from the chemistry data
        self.H2 = self.chem[:,0]
        self.HI = self.chem[:,1]
        self.CI = self.chem[:,2]
        self.CHx = self.chem[:,3]
        self.OHx = self.chem[:,4]
        self.CO = self.chem[:,5]
        self.HCOI = self.chem[:,6]
        self.HeI = self.chem[:,7]
        self.MI = self.chem[:,8]

        # Calculating the H and C abundances
        self.H = 1 - 2 * self.H2 - self.HI - self.CHx - self.OHx - self.HCOI
        self.C = totC - self.CI - self.CHx - self.CO - self.HCOI
        self.O = totO - self.CO - self.HCOI - self.OHx

    # Function to read the header atrributes
    def readHeader(self, snapshotFile):
        # Getting the header
        header = snapshotFile["Header"]

        # Loading the attributes we want
        self.boxSize = header.attrs.get("BoxSize")
        self.time = header.attrs.get("Time")
        self.numPartTotal = header.attrs.get("NumPart_Total")

        self.nParticles = self.numPartTotal[0]
        self.nSinks = self.numPartTotal[5]

        # Only reading sink particle data if we have sinks
        if self.nSinks > 0:
            sinkDict = self.readSinks(snapshotFile)

            # Splitting and storing coordinates
            splitSinkPos = np.array_split(sinkDict["Coordinates"], 3, axis=1)
            self.sinkX = splitSinkPos[0].reshape(self.nSinks)
            self.sinkY = splitSinkPos[1].reshape(self.nSinks) 
            self.sinkZ = splitSinkPos[2].reshape(self.nSinks)

            # Splitting and storing velocities
            splitSinkVel = np.array_split(sinkDict["Velocities"], 3, axis=1)
            self.sinkVX = splitSinkVel[0].reshape(self.nSinks)
            self.sinkVY = splitSinkVel[1].reshape(self.nSinks)
            self.sinkVZ = splitSinkVel[2].reshape(self.nSinks)

            # Storing masses and potentials
            self.sinkMass = sinkDict["Masses"]
            self.sinkPotential = sinkDict["Potential"]
            self.sinkID = sinkDict["ParticleIDs"]

    # Function to read the sink data stored in the snapshot files
    def readSinks(self, snapshotFile):
        # Loading sink particles
        sinkData = snapshotFile["PartType5"]

        # Setting up sink particle dict
        sinkDict = {}

        # List of attributes to read
        attrs = ["Coordinates", "Masses", "Velocities", "Potential", "ParticleIDs"]

        # Looping through each attribute
        for att in attrs:
            # Extracting attribute
            dat = sinkData[att][:]

            if att == "ParticleIDs":
                sinkDict[att] = dat
            else:
                # Converting
                cgs = sinkData[att].attrs.get("to_cgs")
                sinkDict[att] = np.multiply(dat, cgs)

        return sinkDict

    # Function to read the hdf5 data and assign it to a data dictionary
    def snapshotRead(self, filename):
        # Reading file with h5py
        snapshotFile = h5py.File(filename, "r")

        # Reading the header information
        self.readHeader(snapshotFile)

        # Setting up data dictionary
        dataDict = {}

        # List of attributes to read
        attrs = ["Coordinates", "Velocities", "Density", "Masses", "InternalEnergy", "ParticleIDs", "DustTemperature", "ChemicalAbundances", "Potential", "PotentialPeak", "Acceleration", "VelocityDivergence"]

        # Selecting part type 0 data
        data = snapshotFile["PartType0"]

        # Looping through each attribute 
        for att in attrs:
            # Extracting the attribute
            dat = data[att][:]

            if len(dat) == 0:
                pass
            else:
                # If conversion factor, get it and scale
                if att != "DustTemperature" and att != "ChemicalAbundances" and att != "PotentialPeak" and att != "Acceleration" and att != "VelocityDivergence" and att != "ParticleIDs":
                    cgs = data[att].attrs.get("to_cgs")
                    dataDict[att] = np.multiply(dat, cgs)
                else:
                    dataDict[att] = dat
        
        return dataDict 