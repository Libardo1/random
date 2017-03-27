import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import pickle as pkl
from scipy.interpolate import interp1d

def dump_data(diff, length, game_id, player_id, folder_path='./'):
    """
        param diff: difference between pushing the button and game start
        param game_id : int
        param player_id : int
        param folder_path : assumed to be current location if not stated
        param length : length of game, either 30 or 120

        return : data object, dumps pickle file based on game_id and player_id
    """
    data = load_all(folder_path)
    data.player_id = player_id
    data.game_id = game_id

    if len(data.tags.tags.shape) > 0:
        tag = int(data.tags.tags[-1] + diff)
    else:
        tag = int(data.tags.tags)

    data.crop(tag, length)

    #data.resample(length * 10) # sample every 10th of a second as for the labels

    pkl.dump(data, open('data_game%d_player%d.pkl'% (game_id, player_id), 'wb'))
    return data




def load_all(folder_path):
    """
    :param folder_path: Path to folder containing Empatica data from one game (.csv files) (should end in \\)
    :return: data object containing data as numpy arrays along with methods for plotting, together or individually

     This is all that should be needed to be run outside this file
     You can plot data like:
     data.plot() for all data in a single plot
     data.acc.plot() for acceleration data in its own plot

     the plot methods also take filename arguments if you want to save the figures e.g.
     data.plot(filename='game_1.eps')
    """

    # Load accelerometer data
    acc_data = load_acc(folder_path)

    # Load blood volume pulse data
    bvp_data = load_bvp(folder_path)

    # Load electrodermal activity data
    eda_data = load_eda(folder_path)
    
    # Load heart rate data
    hr_data = load_hr(folder_path)

    # Load inter-beat interval data
    ibi_data = load_ibi(folder_path)

    # Load tags data
    tags_data = load_tags(folder_path)

    # Load temperature data
    temp_data = load_temp(folder_path)

    # Create data object with loaded data
    data = DataContainer(acc_data, bvp_data, eda_data, hr_data, ibi_data, tags_data, temp_data)

    return data


def load_acc(folder_path):
    file_path = folder_path + 'ACC.csv'

    # Load data from .csv
    raw = genfromtxt(file_path, delimiter=',')

    # Create container object
    acc_data = AccContainer(raw[0, 0], raw[1, 0], raw[2:, 0], raw[2:, 1], raw[2:, 2])

    return acc_data


def load_bvp(folder_path):
    file_path = folder_path + 'BVP.csv'

    # Load data from .csv
    raw = genfromtxt(file_path, delimiter=',')

    # Create container object
    bvp_data = BvpContainer(raw[0], raw[1], raw[2:])

    return bvp_data


def load_eda(folder_path):
    file_path = folder_path + 'EDA.csv'

    # Load data from .csv
    raw = genfromtxt(file_path, delimiter=',')

    # Create container object
    eda_data = EdaContainer(raw[0], raw[1], raw[2:])

    return eda_data


def load_hr(folder_path):
    file_path = folder_path + 'HR.csv'

    # Load data from .csv
    raw = genfromtxt(file_path, delimiter=',')

    # Create container object
    hr_data = HrContainer(raw[0], raw[1], raw[2:])

    return hr_data


def load_ibi(folder_path):
    file_path = folder_path + 'IBI.csv'

    # Load data from .csv
    raw = genfromtxt(file_path, delimiter=',')

    # Create container object
    ibi_data = IbiContainer(raw[0, 0], raw[1:, 0], raw[1:, 1])

    return ibi_data


def load_tags(folder_path):
    file_path = folder_path + 'tags.csv'

    # Load data from .csv
    raw = genfromtxt(file_path, delimiter=',')

    # Create container object
    tags_data = TagsContainer(raw)

    return tags_data


def load_temp(folder_path):
    file_path = folder_path + 'TEMP.csv'

    # Load data from .csv
    raw = genfromtxt(file_path, delimiter=',')

    # Create container object
    temp_data = TempContainer(raw[0], raw[1], raw[2:])

    return temp_data


class DataContainer:
    def __init__(self, acc_data, bvp_data, eda_data, hr_data, ibi_data, tags_data, temp_data):
        self.acc = acc_data
        self.bvp = bvp_data
        self.eda = eda_data
        self.hr = hr_data
        self.ibi = ibi_data
        self.tags = tags_data
        self.temp = temp_data
        self.game_id = None
        self.player_id = None

    def plot(self, show=False, file_name=None):
        self.acc.plot(subplot=611)
        self.bvp.plot(subplot=612)
        self.eda.plot(subplot=613)
        self.hr.plot(subplot=614)
        self.ibi.plot(subplot=615)
        self.temp.plot(subplot=616)

        if show:
            plt.show()

        if file_name is not None:
            plt.savefig(file_name)
    
    def crop(self, start_timestamp, length):
        self.acc.crop(start_timestamp, length)
        self.bvp.crop(start_timestamp, length)
        self.eda.crop(start_timestamp, length)
        self.hr.crop(start_timestamp, length)
        #self.ibi.crop(start_timestamp, length)
        self.temp.crop(start_timestamp, length)

    def resample(self, sample_rate):
        self.acc.resample(sample_rate)
        self.bvp.resample(sample_rate)
        self.eda.resample(sample_rate)
        self.hr.resample(sample_rate)
        #self.ibi.resample(sample_rate)
        self.temp.resample(sample_rate)




class AccContainer:
    def __init__(self, init_time, sample_rate, x_acc, y_acc, z_acc):
        self.units = '1/64g'
        self.init_time = init_time
        self.sample_rate = sample_rate
        self.x_acc = x_acc
        self.y_acc = y_acc
        self.z_acc = z_acc
        self.time = np.arange(0, len(self.x_acc)) / self.sample_rate + self.init_time

    def plot(self, show=False, file_name=None, subplot=None):
        if subplot is not None:
            plt.subplot(subplot)

        plt.plot(self.time, self.x_acc, label='X Acceleration')
        plt.plot(self.time, self.y_acc, label='Y Acceleration')
        plt.plot(self.time, self.z_acc, label='Z Acceleration')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (1/64g)')
        plt.legend()

        if show:
            plt.show(block=False)

        if file_name is not None:
            plt.savefig(file_name)
    
    def crop(self, start_timestamp, length):
        start_timestamp -= self.init_time
        start_timestamp *= self.sample_rate
        start_timestamp = int(start_timestamp)
        self.init_time = start_timestamp
        self.x_acc = self.x_acc[start_timestamp: start_timestamp + int(length * self.sample_rate)]
        self.y_acc = self.y_acc[start_timestamp: start_timestamp + int(length * self.sample_rate)]
        self.z_acc = self.z_acc[start_timestamp: start_timestamp + int(length * self.sample_rate)]
        self.time = np.arange(0, len(self.y_acc)) / self.sample_rate + self.init_time


    def resample(self, timesteps):
        self.sample_rate = 0.1
        interpolate = interp1d(self.time - self.init_time, self.x_acc, bounds_error=False, fill_value='extrapolate')
        self.x_acc = interpolate(np.linspace(0, self.time[-1] - self.init_time, timesteps))
        interpolate = interp1d(self.time - self.init_time, self.y_acc, bounds_error=False, fill_value='extrapolate')
        self.y_acc = interpolate(np.linspace(0, self.time[-1] - self.init_time, timesteps))
        interpolate = interp1d(self.time - self.init_time, self.z_acc, bounds_error=False, fill_value='extrapolate')
        self.z_acc = interpolate(np.linspace(0, self.time[-1] - self.init_time, timesteps))

        self.sample_rate = len(self.time) / self.sample_rate / timesteps
        self.time = np.arange(0, timesteps) / self.sample_rate + self.init_time


class BvpContainer:
    def __init__(self, init_time, sample_rate, bvp):
        self.units = 's'
        self.init_time = init_time
        self.sample_rate = sample_rate
        self.bvp = bvp
        self.time = np.arange(0, len(self.bvp)) / self.sample_rate + self.init_time

    def plot(self, show=False, file_name=None, subplot=None):
        if subplot is not None:
            plt.subplot(subplot)

        plt.plot(self.time, self.bvp)
        plt.xlabel('Time (s)')
        plt.ylabel('Blood Volume Pulse (s)')

        if show:
            plt.show(block=False)

        if file_name is not None:
            plt.savefig(file_name)

    def crop(self, start_timestamp, length):
        start_timestamp -= self.init_time
        start_timestamp *= self.sample_rate
        start_timestamp = int(start_timestamp)
        self.init_time = start_timestamp
        self.time = self.time[start_timestamp: start_timestamp + int(length * self.sample_rate)]
        self.bvp = self.bvp[start_timestamp: start_timestamp + int(length * self.sample_rate)]
    
    def resample(self, timesteps):
        self.sample_rate = 0.1
        interpolate = interp1d(self.time - self.init_time, self.bvp, bounds_error=False, fill_value='extrapolate')
        self.bvp = interpolate(np.linspace(0, self.time[-1] - self.init_time, timesteps))
        self.time = np.arange(0, len(self.bvp)) / self.sample_rate + self.init_time

class EdaContainer:
    def __init__(self, init_time, sample_rate, eda):
        self.units = 'microsiemens'
        self.init_time = init_time
        self.sample_rate = sample_rate
        self.eda = eda
        self.time = np.arange(0, len(self.eda)) / self.sample_rate + self.init_time

    def plot(self, show=False, file_name=None, subplot=None):
        if subplot is not None:
            plt.subplot(subplot)

        plt.plot(self.time, self.eda)
        plt.xlabel('Time (s)')
        plt.ylabel('Electrodermal Activity ($\mu$ S)')

        if show:
            plt.show(block=False)

        if file_name is not None:
            plt.savefig(file_name)

    def crop(self, start_timestamp, length):
        start_timestamp -= self.init_time
        start_timestamp *= self.sample_rate
        start_timestamp = int(start_timestamp)
        self.init_time = start_timestamp
        self.time = self.time[start_timestamp: start_timestamp + int(length * self.sample_rate)]
        self.eda = self.eda[start_timestamp: start_timestamp + int(length * self.sample_rate)]
    
    def resample(self, timesteps):
        self.sample_rate = 0.1
        interpolate = interp1d(self.time - self.init_time, self.eda, bounds_error=False, fill_value='extrapolate')
        self.eda = interpolate(np.linspace(0, self.time[-1] - self.init_time, timesteps))
        self.time = np.arange(0, len(self.eda)) / self.sample_rate + self.init_time

class HrContainer:
    def __init__(self, init_time, sample_rate, hr):
        self.units = ' '
        self.init_time = init_time
        self.sample_rate = sample_rate
        self.hr = hr
        self.time = np.arange(0, len(self.hr)) / self.sample_rate + self.init_time

    def plot(self, show=False, file_name=None, subplot=None):
        if subplot is not None:
            plt.subplot(subplot)

        plt.plot(self.time, self.hr)
        plt.xlabel('Time (s)')
        plt.ylabel('Heart Rate')

        if show:
            plt.show(block=False)

        if file_name is not None:
            plt.savefig(file_name)
    
    def crop(self, start_timestamp, length):
        start_timestamp -= self.init_time
        start_timestamp *= self.sample_rate
        start_timestamp = int(start_timestamp)
        self.init_time = start_timestamp
        self.time = self.time[start_timestamp: start_timestamp + int(length * self.sample_rate)]
        self.hr = self.hr[start_timestamp: start_timestamp + int(length * self.sample_rate)]

    def resample(self, timesteps):
        self.sample_rate = 0.1
        interpolate = interp1d(self.time - self.init_time, self.hr, bounds_error=False, fill_value='extrapolate')
        self.hr = interpolate(np.linspace(0, self.time[-1] - self.init_time, timesteps))
        self.time = np.arange(0, len(self.hr)) / self.sample_rate + self.init_time


class IbiContainer:
    def __init__(self, init_time, ibi_time, ibi_interval):
        self.units = 's'
        self.init_time = init_time
        self.time = ibi_time
        self.interval = ibi_interval

    def plot(self, show=False, file_name=None, subplot=None):
        if subplot is not None:
            plt.subplot(subplot)

        plt.plot(self.time, self.interval)
        plt.xlabel('Time (s)')
        plt.ylabel('Inter-beat Interval')

        if show:
            plt.show(block=False)

        if file_name is not None:
            plt.savefig(file_name)
    
    def crop(self, start_timestamp, length):
        start_timestamp -= self.init_time
        start_timestamp *= self.sample_rate
        start_timestamp = int(start_timestamp)
        self.init_time = start_timestamp
        self.time = self.time[start_timestamp: start_timestamp + int(length * self.sample_rate)]
        self.interval = self.interval[start_timestamp: start_timestamp + int(length * self.sample_rate)]

    def resample(self, timesteps):
        self.sample_rate = 0.1
        interpolate = interp1d(self.time - self.init_time, self.hintervalr, bounds_error=False, fill_value='extrapolate')
        self.hr = interpolate(np.linspace(0, self.time[-1] - self.init_time, timesteps))
        self.time = np.arange(0, len(self.hr)) / self.sample_rate + self.init_time



class TagsContainer:
    def __init__(self, tags):
        self.units = 'Unix timestamp in UTC'
        self.tags = tags


class TempContainer:
    def __init__(self, init_time, sample_rate, temp):
        self.units = 'degrees C'
        self.init_time = init_time
        self.sample_rate = sample_rate
        self.temp = temp
        self.time = np.arange(0, len(self.temp)) / self.sample_rate + self.init_time

    def plot(self, show=False, file_name=None, subplot=None):
        if subplot is not None:
            plt.subplot(subplot)

        plt.plot(self.time, self.temp)
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature ($^\circ$ C)')

        if show:
            plt.show(block=False)

        if file_name is not None:
            plt.savefig(file_name)

    def crop(self, start_timestamp, length):
        start_timestamp -= self.init_time
        start_timestamp *= self.sample_rate
        start_timestamp = int(start_timestamp)
        self.init_time = start_timestamp
        self.time = self.time[start_timestamp: start_timestamp + int(length * self.sample_rate)]
        self.temp = self.temp[start_timestamp: start_timestamp + int(length * self.sample_rate)]

    def resample(self, timesteps):
        self.sample_rate = 0.1
        interpolate = interp1d(self.time - self.init_time, self.temp, bounds_error=False, fill_value='extrapolate')
        self.temp = interpolate(np.linspace(0, self.time[-1] - self.init_time, timesteps))
        self.time = np.arange(0, len(self.temp)) / self.sample_rate + self.init_time

