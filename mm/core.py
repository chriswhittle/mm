import numpy as np
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt

### CONSTANTS

# colours to use for different wavelength beams
BEAM_COLORS = [(0, 500e-9, 'blue', 'royalblue', 'slateblue', 'blueviolet', 'midnightblue'),
                (500e-9, 550e-9, 'darkgreen', 'olivedrab', 'yellowgreen', 'limegreen', 'darkgreen'),
                (550e-9, 600e-9, 'goldenrod', 'gold', 'orange', 'darkkhaki', 'olive'),
                (600e-9, np.inf, 'firebrick', 'red', 'orangered', 'lightsalmon', 'maroon')]

# standard SI prefixes for distance
SI_UNITS = [(1e-15, 'fm'), (1e-12, 'pm'), (1e-9, 'nm'), (1e-6, 'um'), (1e-3, 'mm'), (1, 'm'), (1e3, 'km')]

### FUNCTIONS

def determinePrefix(lengths):
    '''
    Given a length (or list of lengths), 
    return a tuple with the appropriate unit multiplier 
    and a string of the unit symbol.
    '''
    minLength = np.array(lengths, ndmin=1).min()
    i = 0
    while minLength > SI_UNITS[i][0]:
        i += 1
    return SI_UNITS[max(i-1,0)]

def applyBuffer(value, buffer, left=False):
    '''
    Apply buffer to given value. If right side of domain,
    multiply by (1+buffer) for positive values and (1-buffer)
    for negative values (and vice versa for left side).
    '''
    return value*(1+buffer) if ((value > 0) != left) else value*(1-buffer)

def defaultExt(array, default=0, min=False):
    '''
    Compute maximum of an array. Empty arrays
    give the specified default value.
    '''
    if len(array) > 0:
        return np.amin(array) if min else np.amax(array)
    else:
        return default

def emptyMatrix(d):
    '''
    Given a list of distances, return 2x2 identity matrices
    repeated over a dimension with size equal to the list.
    '''
    return np.tile(np.eye(2)[np.newaxis,:,:], (len(d), 1, 1) )

def space(d, start, end):
    '''
    Given a N-length list of distances and numbers specifying the
    start and end of propagation, return a Nx2x2 matrix for 
    propagation along the specified length.
    '''
    startInd = np.argmax(d>=start)
    endInd = np.argmax(d>=end)

    matrix = emptyMatrix(d)
    matrix[startInd:endInd,0,1] = np.linspace(0, end-start, endInd-startInd)
    matrix[endInd:,0,1] = end-start

    return matrix

def buildTransfer(d, source, lenses):
    '''
    Given a N-length list of distances, the position from which
    the transfer matrix should be built and a list of Lens objects,
    return a Nx2x2 matrix for propagation through the lens system.
    '''
    d = np.array(d, ndmin=1)
    for i in range(0, len(d)-1):
        if d[i] > d[i+1]:
            raise ValueError('List of positions should be sorted')
            ################### TODO: add reverse logic
    i = 0
    matrix = emptyMatrix(d)
    currentPosition = source
    while i < len(lenses) and lenses[i].pos < d[-1]:
        matrix = np.matmul(space(d, currentPosition, lenses[i].pos), matrix)
        matrix = np.matmul(lenses[i].matrix(d), matrix)
        currentPosition = lenses[i].pos
        i += 1
    matrix = np.matmul(space(d, currentPosition, d[-1]), matrix)
    return matrix

def applyTransfer(q, matrix):
    '''
    Given a complex beam parameter q and a Nx2x2 ray transfer matrix,
    calculate the complex beam parameter at each of the N positions.
    '''
    return (matrix[:,0,0]*q + matrix[:,0,1])/(matrix[:,1,0]*q + matrix[:,1,1])

def getRadius(q, wavelength):
    '''
    Given a list of complex beam parameters and a wavelength,
    calculate the beam radius from each q.
    '''
    waistSize = np.sqrt(wavelength * np.imag(q) / np.pi)
    return waistSize*np.sqrt(1 + (np.real(q) / np.imag(q))**2)

def getPhase(q):
    '''
    Given a list of complex beam parameters,
    calculate the Gouy phase from each q.
    '''
    return np.degrees(np.arctan(np.real(q)/np.imag(q)))

def plot(*beams, **kwargs):
    '''
    Plot the beam radius and Guoy phase over some distance
    (determined by either the measurement datapoints, waist position,
    Rayleigh range or lens positions).
    '''
    BUFFER = kwargs.get('BUFFER', 0.1)
    N = kwargs.get('N', 2000)
    includeData = kwargs.get('includeData', False)

    # pre-process beams
    minSize = np.inf
    maxSize = 0
    beamDomains = [None]*len(beams)
    beamSizes = [None]*len(beams)
    beamPhases = [None]*len(beams)
    plotData = [False]*len(beams)
    for i in range(len(beams)):
        beam = beams[i]

        if hasattr(beam, 'positionArray') and includeData:
            dMin = np.min(beam.positionArray)
            dMax = np.max(beam.positionArray)
            bufferDistance = BUFFER*(dMax - dMin)
            beamDomains[i] = np.linspace(dMin-bufferDistance, dMax+bufferDistance, N)

            plotData[i] = True
        else:
            if beam.reverse:
                domains = [-2*np.real(beam.sourceQ),
                            beam.source - np.imag(beam.sourceQ),
                            defaultExt([x.pos for x in beam.lenses], beam.source, min=True)]
                domainStart = applyBuffer(min(domains), BUFFER, left=True)
                domainEnd = beam.source
            else:
                domains = [-2*np.real(beam.sourceQ),
                            beam.source + np.imag(beam.sourceQ),
                            defaultExt([x.pos for x in beam.lenses], beam.source)]
                domainStart = beam.source
                domainEnd = applyBuffer(max(domains), BUFFER)
            beamDomains[i] = np.linspace(domainStart, domainEnd, N)

        beamSizes[i], beamPhases[i] = beam.propagateBeam(beamDomains[i])

        curMin = np.min(beamSizes[i])
        if curMin < minSize:
            minSize = curMin

        curMax = np.max(beamSizes[i])
        if curMax > maxSize:
            maxSize = curMax

    waistUnit = determinePrefix(minSize)

    # plot beams
    plt.figure(figsize=(9, 6))
    plt.subplot(211)
    for i in range(len(beams)):
        beam = beams[i]
        plt.plot(beamDomains[i], beamSizes[i]/waistUnit[0], '-', color=beam.profile[3+i])
        if plotData[i]:
            plt.plot(beam.positionArray, beam.sizeArray/waistUnit[0], '.', markersize=10, color=beam.profile[2])
    plt.grid(which='both')
    plt.gca().autoscale(enable=True, axis='x', tight=True)
    plt.ylabel('Beam size [{}]'.format(waistUnit[1]))
    plt.ylim([0, (1+BUFFER)*maxSize/waistUnit[0]])
    plt.subplot(212)
    for i in range(len(beams)):
        plt.plot(beamDomains[i], beamPhases[i], '-', color=beam.profile[3+i])
    plt.ylabel('Gouy phase [deg]')
    plt.xlabel('Position [m]')
    plt.grid(which='both')
    plt.gca().autoscale(enable=True, axis='x', tight=True)
    plt.ylim([-90, 90])
    plt.yticks(np.arange(-90, 90+30, 30))
    plt.show()

def optimize(lenses, beam, target, position, tol=0.0001):
    '''
    Uses the given list of lenses to match the
    beam and the target (the latter of which may be
    another beam or a target waist size) at the given
    position.
    '''
    lenses = np.array(lenses, ndmin=1)
    def obj(positions, beam, target, position):
        for i in range(len(lenses)):
            lenses[i].pos = positions[i]

        q = beam.propagateBeamQ(position)

        isBeam = not isinstance(target, (int, long, float, complex))

        if isBeam and len(lenses) > 1:
            target = target.propagateBeamQ(position)
            return np.abs(q-target)
        else:
            if isBeam:
                target = target.propagateBeam(position, returnPhase=False)
            return (beam.getRadius(q) - target)**2

    baseBound = (beam.source, position)
    if beam.reverse:
        baseBound = baseBound[::-1]

    positions = minimize(obj, [x.pos for x in lenses], bounds=[baseBound]*len(lenses),
                            args=(beam, target, position), tol=(beam.getRadius()*tol)**2).x

    for i in range(len(lenses)):
        lenses[i].pos = positions[i]

    return positions

### CLASSES

class Beam(object):
    '''
    A Beam object is defined by a complex beam parameter given at a particular position,
    as well as a list of lenses acting on the beam.
    '''
    def __init__(self, source, sourceQ, wavelength, lenses=[], reverse=False):
        '''
        Instantiate a beam with a given source position,
        complex beam parameter at source, wavelength, and
        list of Lens objects.
        '''
        self.source, self.sourceQ, self.wavelength, self.lenses, self.reverse = source, sourceQ, wavelength, lenses, reverse

    @property
    def wavelength(self):
        '''
        Getter for wavelength.
        '''
        return self.__wavelength

    @wavelength.setter
    def wavelength(self, value):
        '''
        Setter for wavelength; updates color information
        for plotting.
        '''
        self.__wavelength = value
        for profile in BEAM_COLORS:
            if profile[0] < self.wavelength < profile[1]:
                self.profile = profile
                return

    @property
    def lenses(self):
        '''
        Getter for lens list.
        '''
        self.__lenses = sorted(self.__lenses, key=lambda x: x.pos)
        return self.__lenses

    @lenses.setter
    def lenses(self, value):
        '''
        Setter for lens list; ensures list is sorted.
        '''
        self.__lenses = value

    def setData(self, positionArray, sizeArray):
        '''
        Associate a set of measurements (list of measurement positions
        and beam size at those positions) with the beam.
        '''
        self.positionArray, self.sizeArray = positionArray, sizeArray

    def plot(self, BUFFER=0.1, N=2000, includeData=True):
        plot(self, BUFFER=BUFFER, N=N, includeData=includeData)

    def buildTransfer(self, d):
        '''
        Given a N-length list of positions, return a
        Nx2x2 ray transfer matrix from the lenses
        associated with this beam.
        '''
        return buildTransfer(d, self.source, self.lenses)

    def getRadius(self, q=None):
        '''
        Given a list of complex beam parameters,
        return the beam radii using the appropriate
        wavelength.
        '''
        if q is None:
            q = self.sourceQ
        return getRadius(q, self.wavelength)

    def propagateBeamQ(self, d):
        '''
        Given a list of distances, return the
        complex beam parameter at each position
        by propagating through the system of lenses.
        '''
        return applyTransfer(self.sourceQ, self.buildTransfer(d))

    def propagateBeam(self, d, returnPhase=True):
        '''
        Given a list of distances, return the
        beam radii (and Guoy phase) at each position
        by propagating through the system of lenses.
        '''
        qValues = self.propagateBeamQ(d)
        beamSizes = self.getRadius(qValues)
        if returnPhase:
            phases = getPhase(qValues)
            return beamSizes, phases
        else:
            return beamSizes

    def addLens(self, lens):
        '''
        Add the specified lens to this path.
        '''
        self.lenses = self.lenses + [lens]

    def removeLens(self, lens):
        '''
        Remove the specified lens from this path.
        '''
        self.lenses.remove(lens)


def fitBeam(positionArray, sizeArray, source, wavelength, lenses=[]):
    '''
    Given a list of positions, a list of beam radii, a source position,
    wavelength and a list of lenses (present during measurement),
    return a beam object defined by the source position and required
    source complex beam parameter to reproduce the measured values.
    '''
    i = np.argsort(positionArray)
    positionArray = np.array(positionArray)[i]
    sizeArray = np.array(sizeArray)[i]

    posMax = positionArray.max()
    if positionArray.min() < source < posMax:
        raise ValueError('Source of beam cannot be in the middle of measurements')

    def beamSize(d, sourceQReal, sourceQImag):
        newBeam = Beam(source, sourceQReal + sourceQImag * 1j, wavelength, lenses)
        return newBeam.propagateBeam(positionArray, returnPhase=False)

    fitParams, _ = curve_fit(beamSize, positionArray, sizeArray,
        bounds=([-np.inf, 0], [np.inf, np.inf]))

    newBeam = Beam(source, fitParams[0] + fitParams[1] * 1j,
        wavelength, lenses, reverse=(posMax < source))
    newBeam.setData(positionArray, sizeArray)
    return newBeam

class Lens(object):
    '''
    A lens object, defined by a position and focal length.
    '''
    def __init__(self, f, pos=0, beams=None):
        self.f, self.pos = f, pos
        if beams is not None:
            beams = np.array(beams, ndmin=1)
            for beam in beams:
                beam.addLens(self)

    def matrix(self, d):
        posInd = np.argmax(d>=self.pos)

        matrix = emptyMatrix(d)
        matrix[posInd:,1,0] = -1./self.f

        return matrix