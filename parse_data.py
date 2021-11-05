import struct
import numpy as np
import matplotlib.pyplot as plt

def parse_vol_file(file_name):
    with open(file_name, "rb") as f:
        numX = int.from_bytes(f.read(4), "big")
        numY = int.from_bytes(f.read(4), "big")
        numZ = int.from_bytes(f.read(4), "big")
        _ = int.from_bytes(f.read(4), "big")
        x=struct.unpack('f', f.read(4))
        y=struct.unpack('f', f.read(4))
        z=struct.unpack('f', f.read(4))
        aspect_ratio = (x, y, z)
        data=np.zeros((numX,numY,numZ),dtype=np.uint8)
        for i in range(numX):
            for j in range(numY):
                for k in range(numZ):
                    data[i][j][k]=int.from_bytes(f.read(1),"big")
    return aspect_ratio, data

def analyze_data(data):
    unique, counts = np.unique(data, return_counts=True)
    plt.plot(unique, counts)
    plt.show()
    
def show_data_slice(slice):
    plt.imshow(slice.T, vmin=0, vmax=255, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    #file_name = r"data/C60.vol"
    #file_name = r"data/Foot.vol"
    file_name = r"data/Frog.vol"
    aspect_ratio, data = parse_vol_file(file_name)
    analyze_data(data)
    show_data_slice(data[data.shape[0]//2])
    #show_data_slice(data[:,data.shape[1]//2])
    #show_data_slice(data[:,:,90].T)
