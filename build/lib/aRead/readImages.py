# Library imports
import numpy as np

def readImage(path):
   with open(path,"rb") as file:
      class imageDataStruct():
         pass
      
      image = imageDataStruct()

      # Get the time
      image.time = np.fromfile(file, dtype=np.double, count=1)[0]

      # Get the dimensions of the image
      image.nx = np.fromfile(file, dtype=np.int32, count=1)[0]
      image.ny = np.fromfile(file, dtype=np.int32, count=1)[0]
      image.nz = np.fromfile(file, dtype=np.int32, count=1)[0]

      # Get the image limits
      image.x0 = np.fromfile(file,dtype=np.double,count=1)[0]
      image.x1 = np.fromfile(file,dtype=np.double,count=1)[0]
      image.y0 = np.fromfile(file,dtype=np.double,count=1)[0]
      image.y1 = np.fromfile(file,dtype=np.double,count=1)[0]
      image.z0 = np.fromfile(file,dtype=np.double,count=1)[0]
      image.z1 = np.fromfile(file,dtype=np.double,count=1)[0]

      # Get unit conversion factors
      image.umass_g = np.fromfile(file,dtype=np.double,count=1)[0]
      image.ulength_cm = np.fromfile(file,dtype=np.double,count=1)[0]
      image.utime_s = np.fromfile(file,dtype=np.double,count=1)[0]

      # Load the image data
      image.data = np.fromfile(file, dtype=np.double, count=image.nx*image.ny)

      # Reshape the data 
      image.image = np.reshape(image.data, (image.nx, image.ny), order = 'F')
      image.image = np.rot90(image.image)

   return image