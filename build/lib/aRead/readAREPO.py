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
    def __init__(self, filename, attrsType=0, chemistry=False, rates=False):
        if attrsType == 0:
            # Define basic set of attributes to read
            readList = ["Coordinates", "Masses", "Velocities", "Density", "ChemicalAbundances"]
        else:
            # Read everything
            readList = []

        # Loading in hdf5 file and assigninig data
        self.dataDict = self.snapshotRead(filename, readList)

        for i in range(len(self.snapshotAttributes)):
            # Assign the positions
            if self.snapshotAttributes[i] == "Coordinates":
                positions = self.dataDict["Coordinates"]
                splitPos = np.array_split(positions, 3, axis=1)

                self.x = splitPos[0].reshape(self.nParticles)
                self.y = splitPos[1].reshape(self.nParticles)
                self.z = splitPos[2].reshape(self.nParticles)
            
            # Assign the velocities
            if self.snapshotAttributes[i] == "Velocities":
                velocities = self.dataDict["Velocities"]
                splitVels = np.array_split(velocities, 3, axis=1)

                self.vx = splitVels[0].reshape(self.nParticles)
                self.vy = splitVels[1].reshape(self.nParticles) 
                self.vz = splitVels[2].reshape(self.nParticles) 

            # Assign the accelerations
            if self.snapshotAttributes[i] == "Acceleration":
                accelerations = self.dataDict["Acceleration"]
                splitAcc = np.array_split(accelerations, 3, axis=1)

                self.ax = splitAcc[0].reshape(self.nParticles)
                self.ay = splitAcc[1].reshape(self.nParticles) 
                self.az = splitAcc[2].reshape(self.nParticles) 

            # The rest of the scalar attributes
            if self.snapshotAttributes[i] == "Density":
                self.rho = self.dataDict["Density"]
                self.numberDensity = self.rho / (1.4 * mProt)

            if self.snapshotAttributes[i] == "Masses":
                self.mass = self.dataDict["Masses"] 

            if self.snapshotAttributes[i] == "InternalEnergy":
                self.u = self.dataDict["InternalEnergy"]

            if self.snapshotAttributes[i] == "ParticleIDs":
                self.ids = self.dataDict["ParticleIDs"]

            if self.snapshotAttributes[i] == "DustTemperature":
                self.dustTemp = self.dataDict["DustTemperature"]

            if self.snapshotAttributes[i] == "ChemicalAbundances":
                self.chem = self.dataDict["ChemicalAbundances"]
                if chemistry:
                    self.extractChemistry()

            if self.snapshotAttributes[i] == "Potential":
                self.potential = self.dataDict["Potential"]

            if self.snapshotAttributes[i] == "PotentialPeak":
                self.maxPotential = self.dataDict["PotentialPeak"]

            if self.snapshotAttributes[i] == "VelocityDivergence":
                self.velocityDivergence = self.dataDict["VelocityDivergence"]

            if self.snapshotAttributes[i] == "SGCHEM_HeatCoolRates":
                self.rates = self.dataDict["SGCHEM_HeatCoolRates"]
                self.extractRates()

        # Using proxy for not an IC snap 
        # TODO: Fix
        if len(self.snapshotAttributes) > 7:
            # Calculating the temperature of the gas
            yn = self.rho / ((1 + 4 * 0.1) * mProt)
            ynTot = (1 + 0.1 - self.chem[:,0] + self.chem[:,1]) * yn
            energy = self.u * self.rho
            self.gasTemp = 2 * energy  / (3 * ynTot * kB)

    # Function to extract all the chemistry information and assign to variables
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

    # Function to pull out all the different heating and cooling rates
    def extractRates(self):
        self.gasGrain = self.rates[:,0]             # Gas-grain cooling
        self.H2cool = self.rates[:,1]               # H2 cooling
        self.atomicCool = self.rates[:,2]           # Atomic cooling
        self.lymanAlpha = self.rates[:,3]           # HI electronic excitation cooling
        self.HeIcool = self.rates[:,4]              # HeI electronic excitation cooling
        self.bremsstrahlung = self.rates[:,5]       # Thermal bremsstrahlung
        self.cosmicRays = self.rates[:,6]           # Cosmic Ray Heating
        self.photoElectric = self.rates[:,7]        # Photoelectric heating
        self.OIfineStruc = self.rates[:,8]          # O Fine structure cooling
        self.CIIfineStruc = self.rates[:,9]         # C+ Fine structure cooling
        self.dustRecomb = self.rates[:,10]          # Dust recombination cooling
        self.highTfineStruc = self.rates[:,11]      # High temperature fine structure cooling (multiple)
        self.H2dissCollisional = self.rates[:,12]   # H2 Collisional Dissassociation
        self.H2dissPhoto = self.rates[:,13]         # H2 Photodissasociation
        self.UVpump = self.rates[:,14]              # UV Pumping of H2
        self.H2form = self.rates[:,15]              # H2 Formation Heating
        self.H2ionCollisional = self.rates[:,16]    # H2 Collisional Ionization
        self.HIIIrecomb = self.rates[:,17]          # H+ Recombination Cooling 
        self.COcool = self.rates[:,18] + self.rates[:,19] + self.rates[:,20] # CO cooling (multiple isotopes)
        self.CIfineStruc = self.rates[:,21]         # C Fine structure cooling

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
    
    # Read snapshot file (Part Type 0)
    def snapshotRead(self, filename, readList):
        # Read the snapshot with h5py
        snapshotFile = h5py.File(filename, "r")

        # Read the header information
        self.readHeader(snapshotFile)

        # Set up data dict and get kets
        dataDict = {}
        data = snapshotFile["PartType0"]
        attrs = list(data.keys())            
        self.snapshotAttributes = attrs

        # Use the readlist if we're given one
        if len(readList) > 0:
            attrs = readList
            self.snapshotAttributes = readList

        # Loop through each attribute key
        for att in attrs:
            dat = data[att][:]

            if len(dat) == 0:
                pass
            else:
                # Apply conversion factors if there is one
                cgs = data[att].attrs.get("to_cgs")
                if cgs == None or cgs == 0.0:
                    dataDict[att] = dat
                else:
                    dataDict[att] = np.multiply(dat, cgs)

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

    # Plot a temperature-density diagram
    def tempDensity(self, cmin=0.001, formatting=True):
        # Log the variables for the axes
        nDenst = np.log10(self.numberDensity)
        temp = np.log10(self.gasTemp)
        weight = self.mass/1.991e33

        # Calculating the normalisation
        norm = mpl.colors.Normalize(vmin=np.min(weight), vmax=np.max(weight), clip=False)

        # Make the plot
        hist = plt.hist2d(nDenst, temp, bins=1000, cmap="jet", weights=weight, cmin=cmin, norm=norm)

        if formatting:
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
    def gasDustTemps(self, binNum=50, formatting=True):
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
        plt.errorbar(densityMid, avgDust, yerr=stdDust, fmt="b.", capsize=3, label="Dust", alpha=0.6)

        if formatting:
            plt.ylabel("Temperature, $\\rm K$")
            plt.yticks([0, 1, 2, 3, 4], ["1", "10", "100", "1000", "10000"])
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
        self.xMomentum = np.sum(self.mass * self.vx)
        self.yMomentum = np.sum(self.mass * self.vy)
        self.zMomentum = np.sum(self.mass * self.vz)
        self.momentum = np.sqrt(self.xMomentum**2 + self.yMomentum**2 + self.zMomentum**2)

        if self.nSinks > 0:
            self.xMomentumS = np.sum(self.sinkMass * self.sinkVX)
            self.yMomentumS = np.sum(self.sinkMass * self.sinkVY)
            self.zMomentumS = np.sum(self.sinkMass * self.sinkVZ)

            self.sinkMomentum = np.sqrt(self.xMomentumS**2 + self.yMomentumS**2 + self.zMomentumS**2)
            
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

        # Consider the sinks
        if self.nSinks > 0:
            # Calculate their centre of mass
            self.comXs = np.sum(self.sinkX * self.sinkMass) / np.sum(self.sinkMass)
            self.comYs = np.sum(self.sinkY * self.sinkMass) / np.sum(self.sinkMass)
            self.comZs = np.sum(self.sinkZ * self.sinkMass) / np.sum(self.sinkMass)

            # Adjust particle CoM accordingly
            self.comX = (self.comX * np.sum(self.mass) + self.comXs * np.sum(self.sinkMass)) / (np.sum(self.mass) + np.sum(self.sinkMass))
            self.comY = (self.comY * np.sum(self.mass) + self.comYs * np.sum(self.sinkMass)) / (np.sum(self.mass) + np.sum(self.sinkMass))
            self.comZ = (self.comZ * np.sum(self.mass) + self.comZs * np.sum(self.sinkMass)) / (np.sum(self.mass) + np.sum(self.sinkMass))

        # Find distances to centre of mass
        self.rCOM = np.sqrt((self.x - self.comX)**2 + (self.y - self.comY)**2 + (self.z - self.comZ)**2)

    # Plot the fractional heating and cooling rates
    def plotHealCoolRates(self, nBins=100, normalised=False):
        # Create bins 
        rates = np.zeros((nBins, np.shape(self.rates)[1]+1))
        densityBins = 10**np.linspace(np.log10(np.min(self.numberDensity))+0.0001, np.log10(np.max(self..numberDensity))-0.0001, nBins+1)

        # Loop through each density bin
        for i in range(nBins):
            inBin = np.where((self.numberDensity > densityBins[i]) & (self.numberDensity <= densityBins[i+1]))

            # Pass if we've got an empty bin
            if len(inBin[0]) == 0:
                rates[i] = rates[i-1]
            else:
                # Extract each of the rates from the rates array
                for j in range(np.shape(self.rates)[1]):
                    rates[i,j] = np.abs(np.median(self.rates[:,j][inBin]))
    
                # Add in the pdV work term
                rates[i,np.shape(self.rates)[1]] = abs(np.median((5/3 - 1) * self.velocityDivergence[inBin] * (36447.2682 / 1e17) * self.rho[inBin] * self.u[inBin]))

            if normalised:
                rates[i] = rates[i] / np.sum(rates[i])    

        # Plot the data
        labels = ["Gas Grain", "pdV Work", "$\\rm H_2$ Formation", "$\\rm H_2$ Dissassociation", "Cosmic Rays", "Photoelectric","Bremsstrahlung", "O Cooling", "H+ Recombination", "$\\rm H_2$ Cooling", "C Cooling", "C+ Cooling", "CO Cooling", "Atomic Cooling", "High T Fine Structure", "Collisional Ionization", "Lyman-$\\rm \\alpha$ Cooling", "Dust Recombination"]
        colours = ["green", "lightgreen", "gold", "darkorange", "orange", "red", "darkred", "lightblue",  "cornflowerblue", "royalblue", "blue", "mediumblue", "darkblue", "slateblue", "mediumpurple", "purple", "gray", "black"]

        ratArr = [rates[:,0], rates[:,-1], rates[:,15], (rates[:,12]+rates[:,13]), rates[:,6], rates[:,7], rates[:,5], rates[:,8], rates[:,17], rates[:,1], rates[:,9], rates[:,-2], (rates[:,18]+rates[:,19]+rates[:,20]), rates[:,2],  rates[:,11], rates[:,16], rates[:,3], rates[:,10]]
        a = plt.stackplot(densityBins, ratArr, labels=labels, colors=colours)