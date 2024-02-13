# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

# Constants
mProt = 1.67e-24 # cgs units
kB =  1.38e-16   # cgs unit

# Function to open arepo data 
class readAREPO():
    def __init__(self, filename, chemistry=True, rates=False):
        # Loading in hdf5 file and assigninig data
        self.dataDict = self.snapshotRead(filename, chemistry, rates)

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
        self.gasTemp = 2 * energy  / (3 * ynTot * kB)

        # Loading chemistry if desired
        if chemistry:
            self.extractChemistry()

        # Loading rates if required
        if rates:
            self.rates = self.dataDict["SGCHEM_HeatCoolRates"]

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
    def snapshotRead(self, filename, rates):
        # Reading file with h5py
        snapshotFile = h5py.File(filename, "r")

        # Reading the header information
        self.readHeader(snapshotFile)

        # Setting up data dictionary
        dataDict = {}

        # List of attributes to read
        attrs = ["Coordinates", "Velocities", "Density", "Masses", "InternalEnergy", "ParticleIDs", "ChemicalAbundances", "DustTemperature", "Potential", "PotentialPeak", "Acceleration", "VelocityDivergence"]

        # Extracting chemical variables if needed
        if rates:
            attrs.append("SGCHEM_HeatCoolRates")

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
                if att != "DustTemperature" and att != "ChemicalAbundances" and att != "PotentialPeak" and att != "Acceleration" and att != "VelocityDivergence" and att != "ParticleIDs" and att !="SGCHEM_HeatCoolRates":
                    cgs = data[att].attrs.get("to_cgs")
                    dataDict[att] = np.multiply(dat, cgs)
                else:
                    dataDict[att] = dat
        
        return dataDict 
    
    # Show a weighted histogram based on three variables
    def image(self, x, y, w, log=True, cmap="plasma"):
        # Logging the weights if required
        if log:
            w = np.log10(w)

        # Binning the data 
        weightedHist, xb, yb = np.histogram2d(y, x, weights=w, bins=(500, 500))
        histNumbers, xb, yb = np.histogram2d(y, x, bins=(500, 500))

        # Combining the histogrammed and non histogrammed data
        finalHist = weightedHist/histNumbers
        finalHist = np.ma.masked_where(histNumbers < 1, finalHist)

        # Checking if we need to log the cmap and plotting
        plt.imshow(finalHist, aspect="auto", cmap=cmap, origin="lower", extent=[yb[0], yb[-1], xb[0], xb[-1]])
        plt.colorbar()

    # Plot a temperature-density diagram
    def tempDensity(self, cmin=0.001):
        # Log the variables for the axes
        nDenst = np.log10(self.numberDensity)
        temp = np.log10(self.gasTemp)
        weight = self.mass/1.991e33

        # Calculating the normalisation
        norm = mpl.colors.Normalize(vmin=np.min(weight), vmax=np.max(weight), clip=False)

        # Make the plot
        hist = plt.hist2d(nDenst, temp, bins=1000, cmap="jet", weights=weight, cmin=cmin, norm=norm)
        plt.colorbar(label="Mass $\\rm [M_\odot]$")
        plt.xlabel("Number density, $\\rm [cm^{-3}]$")
        plt.ylabel("Temperature, $\\rm [K]$")

    # Plot a radial density profile
    def radialDensity(self):
        # Finding the maximum density
        maxDensity = np.max(self.rho)
        maxIndex = np.where(self.rho == maxDensity)

        # Finding the position of this 
        xcom = self.x[maxIndex]
        ycom = self.y[maxIndex]
        zcom = self.z[maxIndex]

        # Finding the minimum cell size to use as our min r
        minCellSize = (self.mass[maxIndex] / self.rho[maxIndex])**(1/3)

        # Getting every particle's distance from this
        r = np.sqrt((self.x - xcom)**2 + (self.y - ycom)**2 + (self.z - zcom)**2) + minCellSize[0]

        # Defining the limits of the bins
        rMin = minCellSize[0] * 2
        rMax = np.max(self.x) * 2

        # Creating the bins
        nBins = 50
        logDiff = (np.log10(rMax) - np.log10(rMin)) / nBins

        # Creating arrays for the binned data
        shellDensity = np.zeros(nBins)
        shellRadius = np.zeros(nBins)

        # Looping through each bin
        for i in range(nBins):
            # Setting the radii for the current shell
            rInner = 10**(np.log10(rMin) + i * logDiff)
            rOuter = 10**(np.log10(rMin) + (i+1) * logDiff)

            # Selecting cells inside this range
            inRadius = np.where((r > rInner) & (r < rOuter))[0]

            # Working out the shell's desnity
            shellMass = np.sum(self.mass[inRadius])
            shellVolume = (np.pi * 4 / 3) * (rOuter**3 - rInner**3)

            shellDensity[i] = shellMass / shellVolume
            shellRadius[i] = (rOuter + rInner) / 2

        return shellRadius, shellDensity

    # Function to check how the density spread looks
    def checkDensity(self):
        # Logging density
        logRho = np.log10(self.rho)

        # Creating the figure
        plt.figure(figsize=(8,8))
        plt.hist2d(self.x, logRho, bins=500, cmin=0.01)
        plt.xlabel("x Position, $\\rm cm$")
        plt.ylabel("Density, $\\rm \log{\\rho}$")

    # Function to see where the gas and dust are coupling
    def gasDustTemps(self, binNum=50):
        # Logging number denisty
        numDense = np.log10(self.numberDensity)

        # Defining density bins
        densityBins = np.linspace(np.min(numDense), np.max(numDense), binNum)

        # Creating storage arrays
        avgDust = np.zeros(binNum-1)
        stdDust = np.zeros(binNum-1)
        avgGas = np.zeros(binNum-1)
        stdGas = np.zeros(binNum-1)
        densityMid = np.zeros(binNum-1)

        # Binning the temperature data 
        for i in range(binNum-1):
            # Getting our bin ranges
            binMin = densityBins[i]
            binMax = densityBins[i+1]

            # Finding gas and temperture particles in this bin
            ind = np.where((numDense <= binMax) & (numDense >= binMin))
            g = self.gasTemp[ind]
            d = self.dustTemp[ind]

            # Logging
            g = np.log10(g)
            d = np.log10(d)

            # Calculating the average and spread
            avgGas[i] = np.mean(g)
            stdGas[i] = np.std(g)
            avgDust[i] = np.mean(d)
            stdDust[i] = np.std(d)

            densityMid[i] = (binMax + binMin) / 2

        # Plotting the gas and dust temperatures
        plt.errorbar(densityMid, avgGas, yerr=stdGas, fmt="g.", capsize=3, label="Gas", alpha=0.6)
        plt.ylabel("Temperature, $\\rm K$")
        plt.yticks([0, 1, 2, 3, 4], ["1", "10", "100", "1000", "10000"])
        plt.errorbar(densityMid, avgDust, yerr=stdDust, fmt="b.", capsize=3, label="Dust", alpha=0.6)
        plt.xlabel("Number Density, $\\rm {cm^{-3}}$")
        plt.xticks([-2, 0, 2, 4, 6, 8, 10], ["0.01", "1", "100", "10000", "$\\rm 10^6$", "$\\rm 10^8$", "$\\rm 10^{10}$"] )
        plt.legend(loc="upper right")

    # Plotting the snapshot's IMF
    def imfPlot(self, color="r", label="UV1", bins=20, density=False):
        # Converting the mass and logging
        mass = self.sinkMass / 1.991e33
        logMass = np.log10(mass)

        # Plotting the histogram
        hist = plt.hist(logMass, bins=bins, histtype="step", color=color, linestyle="-", label=label, density=density)

    # Find momentum 
    def calculateMomentum(self):
        self.momentum = np.sum(self.mass * np.sqrt(self.vx**2 + self.vy**2 + self.vz**2))

        if self.nSinks > 0:
            self.sinkMomentum = np.sum(self.sinkMass * np.sqrt(self.sinkVX**2 + self.sinkVY**2 + self.sinkVZ**2))
            self.totalMomentum = self.momentum + self.sinkMomentum
        else:
            self.sinkMomentum = 0
            self.totalMomentum = self.momentum

    # Find centre of mass
    def centreOfMass(self):
        # Calculate the centre of mass
        self.comX = np.sum(self.x * self.mass) / np.sum(self.mass)
        self.comY = np.sum(self.y * self.mass) / np.sum(self.mass)
        self.comZ = np.sum(self.z * self.mass) / np.sum(self.mass)  

        # Find distances to centre of mass
        self.rCOM = np.sqrt((self.x - self.comX)**2 + (self.y - self.comY)**2 + (self.z - self.comZ)**2)