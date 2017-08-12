import pyedflib
import numpy as np

def getSeizureTimes(filename):
    #Open the summary file that contains the given file
    chb = filename[0:5]
    try:
        summary = open("mit_data/" + chb + "-summary.txt", "r")
    except IOError:
        print("Error: File not found.")
        return -1

    done = False
    while not done:
        if filename in summary.readline():
            #Throw away the next two lines
            line = summary.readline() #file start time
            if ('Number' in line):
                numSeizures = int(line[28])
            else:
                summary.readline() #file end time
                numSeizures = int(summary.readline()[28])

            #return 0 if no seizures
            if int(numSeizures) == 0:
                return [0,0,0]
            startTimes, endTimes = [], []
            for i in range(numSeizures):
                startTime = summary.readline()
                startTime = startTime[startTime.index(':')+1:startTime.index('seconds')-1]
                endTime = summary.readline()
                endTime = endTime[endTime.index(':')+1:endTime.index('seconds')-1]

                startTimes.append(int(startTime))
                endTimes.append(int(endTime))

            summary.close()
            done = True
    return [int(numSeizures), startTimes, endTimes]

def getSignals(filename):
    #get signals (recalculate seizure times for concat. channels)
    f = pyedflib.EdfReader("mit_data/" + filename)
    nChannels = len(f.getNSamples())
    signals = np.array([f.readSignal(i) for i in range(nChannels)])
    f._close()

    return signals

def getFreqSpec(signal):
    #returns frequency spectrum via fourier transform
    #sampled at 256Hz

    nsignal = len(signal)/10
    fs = 256 #sample rate
    T = 1./fs # sample time
    y = np.fft.fft(signal)
    return y

def getFreqVec():
    nsignal = 512 #256*2, 2 seconds
    return 256*np.linspace(0, nsignal/10, int(nsignal/10))/nsignal

def generateFreqSamples(signal):
    nSignal = 512#256*2
    nSamples = int(len(signal)/(512))
    if(nSamples == 0):
        raise ValueError('signal is too short (less than 2 seconds). check generateFreqSamples')
    samples = np.zeros((nSamples, 51))#51 is the length of the freq spectrum array
    energy = np.zeros((nSamples))#51 is the length of the freq spectrum array
    for k in range(0, nSamples-1):
        samples[k] = getFreqSpec(signal[k*256:k*256 + 2*256])[:int(nSignal/10)]
        energy[k] = getEnergy(signal[k*256:k*256 + 2*256])
    return samples, energy

def findNearest(array,value):
    return np.abs(array-value).argmin()

def getEnergy(signal):
    return np.sum(np.square(signal))

def getSlice(signals, index):
    return [signals[i][index] for i in range(signals.shape[0])]


def getFreqVals(signal, seizureTimes):
    x_valsChannel = []
    y_valsChannel = []
    energyChannel = []
    for j in range(signal.shape[0]):
        x_vals = []
        y_vals = []
        energy = []
        seizureFreqSamples,nonSeizFreqSamples = [], []
        #No seizure
        if (seizureTimes[0] == 0):
            x_val, energies = generateFreqSamples(signal[j])
            energy.append(energies)
            y_vals.append(np.full(len(x_val), -1))
            x_vals.append(x_val)
        #1 seizure
        elif (seizureTimes[0] == 1):
            start, end = seizureTimes[1][0]*256, seizureTimes[2][0]*256

            #before seizure
            x_val, energies = generateFreqSamples(signal[j][0:start])
            energy.append(energies)
            y_vals.append(np.full(len(x_val), -1))
            x_vals.append(x_val)

            #seizure
            x_val, energies = generateFreqSamples(signal[j][start:end])
            energy.append(energies)
            y_vals.append(np.full(len(x_val), 1))
            x_vals.append(x_val)

            #after seizure
            x_val, energies = generateFreqSamples(signal[j][end:])
            energy.append(energies)
            y_vals.append(np.full(len(x_val), -1))
            x_vals.append(x_val)
        #More than 1 seizure
        else:
            prevEnd = seizureTimes[2][0]
            for k in range(seizureTimes[0]):
                start, end = seizureTimes[1][k]*256, seizureTimes[2][k]*256

                #before 1st seizure
                if (k == 0):
                    x_val, energies = generateFreqSamples(signal[j][0:start])
                    energy.append(energies)
                    y_vals.append(np.full(len(x_val), -1))
                    x_vals.append(x_val)
                #after seizure before
                else:
                    x_val, energies = generateFreqSamples(signal[j][prevEnd:start])
                    energy.append(energies)
                    y_vals.append(np.full(len(x_val), -1))
                    x_vals.append(x_val)
                    prevEnd = end

                #seizure
                x_val, energies = generateFreqSamples(signal[j][start:end])
                energy.append(energies)
                y_vals.append(np.full(len(x_val), 1))
                x_vals.append(x_val)

                #after last seizure
                if(k == seizureTimes[0]-1):
                    x_val, energies = generateFreqSamples(signal[j][end:])
                    energy.append(energies)
                    y_vals.append(np.full(len(x_val), -1))
                    x_vals.append(x_val)


        nx_val = 0
        for xval in x_vals:
            nx_val+= len(xval)
        #1800 is the typical amount of 2 second chunks we can take out of the signal
        #this is just to make the shape fo the data uniform
        if nx_val < 1800:
            diff = 1800 - nx_val
            x_vals.append(np.full((diff,51), np.nan))
            y_vals.append(np.full(diff, 0))
            energy.append(energies)

        x_valsChannel.append(np.concatenate(x_vals, 0))
        y_valsChannel.append(np.concatenate(y_vals, 0))
        energyChannel.append(np.concatenate(energy, 0))
        del x_vals
        del y_vals
        del energy
    x_valsEEG = np.array(x_valsChannel)
    y_valsEEG = np.array(y_valsChannel)
    energyEEG = np.array(energyChannel)
    del x_valsChannel
    del y_valsChannel
    del energyChannel
    #return np.concatenate(x_valsEEG, 0), np.concatenate(y_valsEEG, 0)
    return x_valsEEG, y_valsEEG, energyEEG
