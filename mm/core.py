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

def determine_prefix(lengths):
    '''
    Given a length (or list of lengths), 
    return a tuple with the appropriate unit multiplier 
    and a string of the unit symbol.
    '''
    min_length = np.array(lengths, ndmin=1).min()
    i = 0
    while min_length > SI_UNITS[i][0]:
        i += 1
    return SI_UNITS[max(i-1,0)]

def apply_buffer(value, buffer, left=False):
    '''
    Apply buffer to given value. If right side of domain,
    multiply by (1+buffer) for positive values and (1-buffer)
    for negative values (and vice versa for left side).
    '''
    return value*(1+buffer) if ((value > 0) != left) else value*(1-buffer)

def default_ext(array, default=0, min=False):
    '''
    Compute maximum of an array. Empty arrays
    give the specified default value.
    '''
    if len(array) > 0:
        return np.amin(array) if min else np.amax(array)
    else:
        return default

def nearest_index(array, value):
    return np.argmin(np.abs(array-value))

def empty_matrix(d):
    '''
    Given a list of distances, return 2x2 identity matrices
    repeated over a dimension with size equal to the list.
    '''
    return np.tile(np.eye(2)[np.newaxis,:,:], (len(d), 1, 1) )

def space(d, n, start=None, end=None, start_ind=None, end_ind=None):
    '''
    Given a N-length list of distances and numbers specifying the
    start and end of propagation, return a Nx2x2 matrix for 
    propagation along the specified length.
    '''
    if start_ind is None:
        start_ind = nearest_index(d, start)
    elif start is None:
        start = d[start_ind]
    
    if end_ind is None:
        end_ind = nearest_index(d, end)
    elif end is None:
        end = d[end_ind]

    matrix = empty_matrix(d)
    matrix[start_ind:end_ind,0,1] = d[start_ind:end_ind]-start
    matrix[end_ind:,0,1] = end-start

    matrix[:0,1] /= n
    
    return matrix

def curved_interface(d, n1, n2, r, pos=None, pos_ind=None):
    if pos_ind is None:
        pos_ind = nearest_index(d, pos)

    matrix = empty_matrix(d)
    matrix[pos_ind:,1,0] = (n1 - n2) / (r*n2)
    matrix[pos_ind:,1,1] = n1 / n2

    return matrix

def apply_transfer(q, matrix):
    '''
    Given a complex beam parameter q and a Nx2x2 ray transfer matrix,
    calculate the complex beam parameter at each of the N positions.
    '''
    return (matrix[:,0,0]*q + matrix[:,0,1])/(matrix[:,1,0]*q + matrix[:,1,1])

def get_radius(q, wavelength):
    '''
    Given a list of complex beam parameters and a wavelength,
    calculate the beam radius from each q.
    '''
    waist_size = np.sqrt(wavelength * np.imag(q) / np.pi)
    return waist_size*np.sqrt(1 + (np.real(q) / np.imag(q))**2)

def get_phase(q):
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
    include_data = kwargs.get('include_data', True)

    # pre-process beams
    min_size = np.inf
    max_size = 0
    beam_domains = [None]*len(beams)
    beam_sizes = [None]*len(beams)
    beam_phases = [None]*len(beams)
    plot_data = [False]*len(beams)
    for i, beam in enumerate(beams):
        beam = beams[i]

        if hasattr(beam, 'position_array') and include_data:
            d_min = np.min(beam.position_array)
            d_max = np.max(beam.position_array)
            buffer_distance = BUFFER*(d_max - d_min)
            beam_domains[i] = np.linspace(d_min-buffer_distance, d_max+buffer_distance, N)

            plot_data[i] = True
        else:
            if beam.reverse:
                last_lens = default_ext([x.pos for x in beam.lenses], beam.source, min=True)
                last_q = beam.propagate_beam_q(last_lens-BUFFER).item()
                domains = [last_lens-BUFFER-2*np.real(last_q),
                            beam.source - np.imag(last_q),
                            last_lens]
                domain_start = apply_buffer(min(domains), BUFFER, left=True)
                domain_end = beam.source
            else:
                last_lens = default_ext([x.pos for x in beam.lenses], beam.source)
                last_q = beam.propagate_beam_q(last_lens+BUFFER).item()
                domains = [last_lens+BUFFER-2*np.real(last_q),
                            beam.source + np.imag(last_q),
                            last_lens]

                domain_start = beam.source
                domain_end = apply_buffer(max(domains), BUFFER)
            beam_domains[i] = np.linspace(domain_start, domain_end, N)

        beam_sizes[i], beam_phases[i] = beam.propagate_beam(beam_domains[i])

        cur_min = np.min(beam_sizes[i])
        if cur_min < min_size:
            min_size = cur_min

        cur_max = np.max(beam_sizes[i])
        if cur_max > max_size:
            max_size = cur_max

    waist_unit = determine_prefix(min_size)
    show_legend = False

    # plot beams
    plt.figure(figsize=(7, 5))
    plt.subplot(211)
    for i in range(len(beams)):
        beam = beams[i]
        kwargs = {}
        if beam.label is not None:
            kwargs['label'] = beam.label
            show_legend = True
        plt.plot(beam_domains[i], beam_sizes[i]/waist_unit[0], '-', color=beam.profile[3+i], **kwargs)
        if plot_data[i]:
            plt.plot(beam.position_array, beam.size_array/waist_unit[0], '.', markersize=10, color=beam.profile[2])
            
        for j in range(len(beam.lenses)):
            plt.axvline(x=beam.lenses[j].pos, linestyle='--', color=beam.profile[2], alpha=0.5)
            
    plt.grid(which='both')
    plt.gca().autoscale(enable=True, axis='x', tight=True)
    plt.ylabel('Beam size [{}]'.format(waist_unit[1]))
    plt.ylim([0, (1+BUFFER)*max_size/waist_unit[0]])
    if show_legend:
        plt.legend(loc='best')
    plt.subplot(212)
    for i in range(len(beams)):
        plt.plot(beam_domains[i], beam_phases[i], '-', color=beam.profile[3+i])
    plt.ylabel('Gouy phase [$^\circ$]')
    plt.xlabel('Position [m]')
    plt.grid(which='both')
    plt.gca().autoscale(enable=True, axis='x', tight=True)
    plt.ylim([-90, 90])
    plt.yticks(np.arange(-90, 90+30, 30))
    plt.tight_layout()
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

        q = beam.propagate_beam_q(position)

        is_beam = not isinstance(target, (int, float, complex))

        if is_beam and len(lenses) > 1:
            target = target.propagate_beam_q(position)
            return np.abs(q-target)
        else:
            if is_beam:
                target = target.propagate_beam(position, return_phase=False)
            return (beam.get_radius(q, position) - target)**2

    base_bound = (beam.source, position)
    if beam.reverse:
        base_bound = base_bound[::-1]

    positions = minimize(obj, [x.pos for x in lenses], bounds=[base_bound]*len(lenses),
                            args=(beam, target, position), tol=(beam.get_radius()*tol)**2).x

    for i in range(len(lenses)):
        lenses[i].pos = positions[i]

    is_beam = not isinstance(target, (int, float, complex))
    q = beam.propagate_beam_q(position)
    if is_beam:
        q2 = target.propagate_beam_q(position)
        return ((q * np.conj(q2) / (np.conj(q2) - q)) * (2*beam.wavelength / (get_radius(q2, beam.wavelength) / get_radius(q, beam.wavelength) / np.pi)) * -1j)**4
    else:
        return np.abs(q-target)

### CLASSES

class Beam(object):
    '''
    A Beam object is defined by a complex beam parameter given at a particular position,
    as well as a list of lenses acting on the beam.
    '''
    def __init__(self, source, source_q, lenses=[], reverse=False, wavelength=None, label=None):
        '''
        Instantiate a beam with a given source position,
        complex beam parameter at source, wavelength, and
        list of Lens objects.
        '''
        self.source, self.source_q, self.lenses, self.reverse, self.wavelength, self.label = source, source_q, lenses, reverse, wavelength, label

    @property
    def system(self):
        '''
        Getter for system context.
        '''
        if not hasattr(self, '__system'):
            self.__system = DEFAULT_SYSTEM
        return self.__system

    @system.setter
    def system(self, system):
        '''
        Setter for system context.
        '''
        self.__system = system

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
        self.__wavelength = value if value is not None else self.system.wavelength
        for profile in BEAM_COLORS:
            if profile[0] < self.wavelength < profile[1]:
                self.profile = profile
                return

    @property
    def lenses(self):
        '''
        Getter for lens list.
        '''
        return self.__lenses

    @lenses.setter
    def lenses(self, value):
        '''
        Setter for lens list; ensures list is sorted.
        '''
        self.__lenses = sorted(value, key=lambda x: x.pos)

    def set_data(self, position_array, size_array):
        '''
        Associate a set of measurements (list of measurement positions
        and beam size at those positions) with the beam.
        '''
        self.position_array, self.size_array = position_array, size_array

    def plot(self, BUFFER=0.1, N=2000, include_data=True):
        plot(self, BUFFER=BUFFER, N=N, include_data=include_data)

    def build_transfer(self, d):
        '''
        Given a N-length list of positions, return a
        Nx2x2 ray transfer matrix from the lenses
        associated with this beam.
        '''
        wv = self.wavelength
        sub = self.system.ambient_substrate
        n0 = sub.n(wv)
        
        d = np.array(d, ndmin=1)
        for i in range(0, len(d)-1):
            if d[i] > d[i+1]:
                raise ValueError('List of positions should be sorted')
                ################### TODO: add reverse logic
        i = 0
        matrix = empty_matrix(d)
        current_position = self.source
        while i < len(self.lenses) and self.lenses[i].start() < d[-1]:
            matrix = np.matmul(space(d, n0, current_position, self.lenses[i].start()), matrix)
            matrix = np.matmul(self.lenses[i].matrix(d, wv, sub), matrix)
            current_position = self.lenses[i].end()
            i += 1
        matrix = np.matmul(space(d, n0, current_position, d[-1]), matrix)
        return matrix

    def path_wavelength(self, d):
        d = np.array(d, ndmin=1)
        n = np.ones(len(d)) * self.system.ambient_substrate.n(self.wavelength)
        for l in self.lenses:
            if hasattr(l, 'substrate'):
                start = nearest_index(d, l.start())
                end = nearest_index(d, l.end())
                n[start:end] = l.substrate.n(self.wavelength)
        return self.wavelength / n

    def get_radius(self, q=None, d=None):
        '''
        Given a list of complex beam parameters,
        return the beam radii using the appropriate
        wavelength.
        '''
        if q is None:
            q = self.source_q
        if d is None:
            wavelength = self.wavelength
        else:
            wavelength = self.path_wavelength(d)
        return get_radius(q, wavelength)

    def propagate_beam_q(self, d):
        '''
        Given a list of distances, return the
        complex beam parameter at each position
        by propagating through the system of lenses.
        '''
        return apply_transfer(self.source_q, self.build_transfer(d))

    def propagate_beam(self, d, return_phase=True):
        '''
        Given a list of distances, return the
        beam radii (and Guoy phase) at each position
        by propagating through the system of lenses.
        '''
        q_values = self.propagate_beam_q(d)
        beam_sizes = self.get_radius(q_values, d)
        if return_phase:
            phases = get_phase(q_values)
            return beam_sizes, phases
        else:
            return beam_sizes

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


def fit_beam(position_array, size_array, source, wavelength, lenses=[], label=None):
    '''
    Given a list of positions, a list of beam radii, a source position,
    wavelength and a list of lenses (present during measurement),
    return a beam object defined by the source position and required
    source complex beam parameter to reproduce the measured values.
    '''
    i = np.argsort(position_array)
    position_array = np.array(position_array)[i]
    size_array = np.array(size_array)[i]

    pos_max = position_array.max()
    if position_array.min() < source < pos_max:
        raise ValueError('Source of beam cannot be in the middle of measurements')

    def kinked_line(d, a1, b1, a2, b2):
        if a1 != a2:
            d0 = (b2 - b1)/(a1 - a2)
        else:
            d0 = 0
            
        size = np.zeros(d.shape)
        size[d<=d0] = a1*(d[d<=d0]-b1)
        size[d>d0] = a2*(d[d>d0]-b2)
        return size

    prefit_params, _ = curve_fit(kinked_line, position_array, size_array)
    a1, b1, a2, b2 = prefit_params

    if np.sign(a1) == np.sign(a2):
        def line(d, a, b):
            return a*(d-b)
        prefit_params, _ = curve_fit(line, position_array, size_array)
        a, b = prefit_params
        waist_pos = b
        waist = wavelength / np.pi / a
    else:
        waist_pos = (b2 - b1)/(a1 - a2)
        waist = np.min(size_array)
    
    def beam_size(d, source_q_real, source_q_imag):
        new_beam = Beam(source, source_q_real + source_q_imag * 1j, lenses,
                       (pos_max < source), wavelength)
        return new_beam.propagate_beam(position_array, return_phase=False)

    initial_guess = (source - waist_pos, waist)

    fit_params, _ = curve_fit(beam_size, position_array, size_array,
                              bounds=([-np.inf, 0], [np.inf, np.inf]),
                              p0=initial_guess)

    new_beam = Beam(source, fit_params[0] + fit_params[1] * 1j,
                   lenses, (pos_max < source), wavelength, label)
    new_beam.set_data(position_array, size_array)
    return new_beam

class Lens(object):
    '''
    A lens object, defined by a position and focal length.
    '''
    def __init__(self, f, pos=0, beams=None):
        self.f, self.pos = f, pos
        self.add_beams(beams)

    def add_beams(self, beams):
        if beams is not None:
            beams = np.array(beams, ndmin=1)
            for beam in beams:
                beam.addLens(self)

    def matrix(self, d, wavelength, ambient_substrate):
        pos_ind = nearest_index(d, self.pos)

        matrix = empty_matrix(d)
        matrix[pos_ind:,1,0] = -1./self.f

        return matrix

    def start(self):
        return self.pos

    def end(self):
        return self.pos

class ThickLens(Lens):
    def __init__(self, r, pos=0, beams=None, substrate=None):
        self.r, self.pos, self.substrate = r, pos, substrate
        self.add_beams(beams)

    def start(self):
        return self.pos - self.thickness/2

    def end(self):
        return self.pos + self.thickness/2

    def thickness_calc(self):
        if self.r1 > 0 and self.r2 > 0:
            thickness = 0.00635

            for r in [self.r1, self.r2]:
                if r != np.inf:
                    thickness += r - np.sqrt(r**2 - (self.diameter/2)**2)
        else:
            thickness = 0.0035

        return thickness
    
    @property
    def thickness(self):
        if not hasattr(self, '__thickness'):
            self.__thickness = self.thickness_calc()
        return self.__thickness

    @thickness.setter
    def thickness(self, value):
        self.__thickness = value

    @property
    def diameter(self):
        if not hasattr(self, '__diameter'):
            self.__diameter = 0.0254
        return self.__diameter

    @diameter.setter
    def diameter(self, value):
        self.__diameter = value
        
    @property
    def system(self):
        if not hasattr(self, '__system'):
            self.__system = DEFAULT_SYSTEM
        return self.__system

    @system.setter
    def system(self, value):
        self.__system = value
        
    @property
    def substrate(self):
        if not hasattr(self, '__substrate'):
            self.__substrate = self.system.optic_substrate
        return self.__substrate

    @substrate.setter
    def substrate(self, value):
        self.__substrate = value

    def matrix(self, d, wavelength=None, ambient_substrate=None):
        if wavelength is None:
            wavelength = self.system.wavelength
        if ambient_substrate is None:
            ambient_substrate = self.system.ambient_substrate
        
        n_lens = self.substrate.n(wavelength)
        nAmbient = ambient_substrate.n(wavelength)

        start = self.start()
        end = self.end()
        
        start_ind = nearest_index(d, start)
        end_ind = nearest_index(d, end)

        matrix = np.matmul(
            np.matmul(curved_interface(d, n_lens, nAmbient, self.r2, pos_ind=end_ind),
                      space(d, n_lens, start_ind=start_ind, end_ind=end_ind)),
            curved_interface(d, nAmbient, n_lens, self.r1, pos_ind=start_ind))
        
        return matrix
        
class PlanoConvex(ThickLens):
    '''
    A plano-convex(cave) lens, defined by a position,
    a radius of curvature and a substrate material.
    '''
    @property
    def r1(self):
        if not hasattr(self, '__r1'):
            self.__r1 = self.r
        return self.__r1

    @r1.setter
    def r1(self, value):
        self.__r1 = value
        
    @property
    def r2(self):
        if not hasattr(self, '__r1'):
            self.__r2 = np.inf
        return self.__r2

    @r2.setter
    def r2(self, value):
        self.__r2 = value
    
class BiConvex(ThickLens):
    '''
    A bi-convex(cave) lens, defined by a position,
    a radius (or radii) of curvature and a substrate
    material.
    '''        
    @property
    def r1(self):
        if not hasattr(self, '__r1'):
            self.__r1 = self.r
        return self.__r1

    @r1.setter
    def r1(self, value):
        self.__r1 = value
        
    @property
    def r2(self):
        if not hasattr(self, '__r1'):
            self.__r2 = -self.r
        return self.__r2

    @r2.setter
    def r2(self, value):
        self.__r2 = value

class Substrate(object):
    '''
    A substrate material, defined by a dispersion
    formula.
    '''
    def __init__(self, n, microns=True):
        '''
        Initialize substrate with refractive index.
        '''
        self.n = lambda l: n(l*1e6 if microns else l)

BK7 = Substrate(lambda l:
                np.sqrt(1 +
                        1.03961212*l**2/(l**2 - 0.00600069867) + 
                        0.231792344*l**2/(l**2 - 0.0200179144) +
                        1.01046945*l**2/(l**2 - 103.560653)
                ))
        
FUSED_SILICA = Substrate(lambda l:
                np.sqrt(1 +
                        0.6961663*l**2/(l**2 - 0.0684043**2) + 
                        0.4079426*l**2/(l**2 - 0.1162414**2) +
                        0.8974794*l**2/(l**2 - 9.896161**2)
                ))

VACUUM = Substrate(lambda l: 1)
AIR = Substrate(lambda l:
                1 + 0.05792105/(238.0185 - l**-2) +
                0.00167917/(57.362 - l**-2))

class System(object):
    def __init__(self, wavelength=1064e-9, optic_substrate=FUSED_SILICA, ambient_substrate=AIR):
        self.wavelength, self.optic_substrate, self.ambient_substrate = wavelength, optic_substrate, ambient_substrate

    def beam(self, *args, **kwargs):
        new_beam = Beam(*args, **kwargs)
        new_beam.system = self
        return new_beam
        
DEFAULT_SYSTEM = System()
