# Functions for ceilometer data
# Started Elliott Wed 09 Sep 2015

# General info and setup

# module imports needed to run the function

import numpy as np
import sys

if sys.platform == 'linux2':
    sys.path.append('/net/home/mm0100/ewarren/Documents/AerosolBackMod/scripts/ellUtils') # general utils
    import ellUtils as eu
else:
    from ellUtils import ellUtils as eu



# --------------------------
# Constants to do with ceilometer
# --------------------------

# Sites with height pairs
# height above sea level (LUMA metadata)- height above ground (DTM) = height above surface
# surface height taken from DTM (Grimmond and ...) - Digital Terrain Model (surface only, no buildings)

# (original setup up) above ground level
site_bsc = {'CL31-A_KSS45W': 64.3 - 31.4, 'CL31-A_IMU': 92.1 - 21.6, 'CL31-B_RGS': 28.1 - 19.4, 'CL31-C_MR': 32.0 - 27.5,
            'CL31-D_NK': 27.0 - 23.2, 'CL31-E_NK': 27.0 - 23.2, 'CL31-D_SWT': 44.5 - 3.1}
# above ground level
site_bsc_agl = {'CL31-A_KSS45W': 64.3 - 32.4, 'CL31-A_IMU': 92.1 - 21.6, 'CL31-B_RGS': 28.1 - 19.4, 'CL31-C_MR': 32.0 - 27.5,
            'CL31-D_NK': 27.0 - 23.2, 'CL31-E_NK': 27.0 - 23.2, 'CL31-D_SWT': 44.5 - 3.1}
# above sea level
site_bsc_asl = {'CL31-A_KSS45W': 64.3, 'CL31-A_IMU': 92.1, 'CL31-B_RGS': 28.1, 'CL31-C_MR': 32.0,
            'CL31-D_NK': 27.0, 'CL31-E_NK': 27.0, 'CL31-D_SWT': 44.5}

# IMU: blue; RGS: orange; MR: red; NK: purple; SWT: green; KSSW: cyan; KSS45W: brown
site_bsc_colours = {'IMU': '#1f77b4', 'RGS': '#ff7f0e', 'MR': '#d62728', 'NK': '#9467bd', 'SWT': '#2ca02c', 'KSSW': 'c', 'KSS45W': '#8c564b'}

def extract_sites(site_list, height_type='agl'):

    """
    Extract out just the sites given in 'site_list
    :param site_list (list of strings): list of sites to extract, with each entry in the
        form of [ins]-[id]_[site], e.g. CL31-A_KSS45W
    :return: (dict): dictionary with site and height, extracted from site_bsc
    """

    # if height above ground level is required
    if height_type == 'agl':
        dict = {site_i: site_bsc[site_i] for site_i in site_list}
    elif height_type == 'asl':
        dict = {site_i: site_bsc_asl[site_i] for site_i in site_list}
    else:
        raise ValueError('height not defined as being asl or agl. Change height_type keyword variable!')

    return dict

# --------------------------
# Helper read in functions
# --------------------------

def createID(siteList):

    """
    will return an array with the sensor, site and type split up

    :param siteList:
    :return: IDs
    """

    import numpy as np

    sensors = np.vstack([i.split('_') for i in siteList])
    IDs = [i[0] + '-' + i[2] for i in sensors]
    return IDs

# create filename for ceil data
def create_filename(ceilDatadir, site, day, fType):

    """ Creates filename from site stinrg and day"""

    # site id (short) and site str in filename
    split = site.split('_')
    site_id = split[-1]
    bsc_site_name = split[0] + '_' + fType + '_' + split[-1]

    # date for the main day
    doyStr = day.strftime('%Y%j')

    # time resolution of data in filename
    if fType == 'MLH':
        timestr = '15min'
    elif fType == 'BSC':
        timestr = '15sec'
    elif fType == 'CLD':
        timestr = '15sec'
    elif fType == '':
        raise ValueError('fType variable not given!')
    else:
        raise ValueError('fType argument is not recognised. Please choose MLH, BSC, CLD or add new fType')

    # get filename
    bsc_fname = ceilDatadir + bsc_site_name + '_' + doyStr + '_' + timestr + '.nc'

    return bsc_fname, site_id

def get_ceil_gate_heights(datafile, ceil_range, ins_height):

    # read in height and time if not already first
    ceil_file_height = np.squeeze(datafile.variables['height'][:])
    # sometimes the height isn't corrected. So check here that
    # values might just be a scaler to represent height of instrument and not the actual height gates
    if ceil_file_height.size == 1:
        ceil_height = ceil_range + ins_height
        # if height of gates == range... then it needs correcting
    elif (ins_height != 0.0) & (ceil_file_height[0] == ceil_range[0]):
        ceil_height = ceil_file_height + ins_height

    return ceil_height

# time match variables to an array of datetimes
def time_match_data(data_obs, **kwargs):
    """
    Time match the variables to 'timeMatch'. None time based variables are simply copied

    :param data_obs:

    kwargs
    :param: timeMatch (array of datetimes): times to match variables to
    :return: data_obs:
    """

    from ellUtils.ellUtils import binary_search

    # find nearest time in ceil time
    # pull out ALL the nearest time idxs and differences
    t_idx = np.array([binary_search(data_obs['time'], t) for t in kwargs['timeMatch']])
    t_diff = np.array([data_obs['time'][t_idx_obs_i] - kwargs['timeMatch'][t_idx_match_i]
                       for t_idx_match_i, t_idx_obs_i in enumerate(t_idx)])

    # extract data
    data_obs['orig_time'] = data_obs['time']  # original time, before time matching
    data_obs['time'] = kwargs['timeMatch']  # aligned time

    # overwrite t_idx locations where t_diff is too high with nans
    # only keep t_idx values where the difference is below 5 minutes
    bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])

    # get keys, trim data for all but height and time
    for var, var_data in data_obs.iteritems():
        if var_data.shape != ():  # equivalent to 0 shape / scaler value
            if (var_data.shape[0] == len(data_obs['orig_time'])) & \
                    (var not in ['time', 'orig_time', 'height',
                                 'range']):  # if the dimension length is equal to time's length

                data_obs[var] = var_data[t_idx, ...]
            # else:
            #     raise AttributeError(
            #         'Tried to slice variable using t_idx but time was not the first dimension'
            #         'of the array! Need to reconsider this slicing approach!: ' + var + ' ' +
            #         'var shape: ' + str(var_data.shape))
            else:
                data_obs[var] = var_data

    return data_obs

# find what the ceilometer type is e.g. CT25K or CL31
def identify_sensorType(datafile):

    """ Identify the ceilometer sensor type, e.g. CT25K or CL31"""

    # get sensor type from metadata
    sensorType = str(datafile.Title).split('_')[0]

    # Test that sensorType is either CT25K or CL31, if not raise exception
    if (sensorType == 'CL31' or sensorType == 'CT25K') == False:
        # try to see if the sensor type is actually part of a longer string...
        #   this occurs in some of the older data files...
        if 'CL31' in sensorType:
            sensorType = 'CL31'
        elif 'CT25K' in sensorType:
            sensorType = 'CT25K'
        # if the sensor still can't be identified as a CL31 or CT25K, throw a ValueError
        if (sensorType == 'CL31' or sensorType == 'CT25K') == False:
            # if the sensor type can't be properly identified...
            raise ValueError('sensorType given is not CT25K or CL31')

    return sensorType

# calibrate BSC data
def calibrate_BSC_data(bsc_obs, site, day):

    """
    calibrate the bsc observations. One calibration coefficient to multiply with the backscatter, per day. Calibration
    values are stored in yearly files.

    :param bsc_obs (dict):
    :param site:
    :param day (datetime):
    :return:
    """

    from ellUtils.ellUtils import netCDF_read

    # site id (short) and site str in filename
    split = site.split('_')
    site_id = split[-1]
    cal_site_name = split[0] + '_CAL_' + split[-1]

    # L2 is the interpolated calibration data (vs time, window transmission or block avg.)
    calibdir = 'C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/data/L2/'
    filepath = calibdir + cal_site_name + '_' + day.strftime('%Y') + '.nc'

    calib_data = netCDF_read(filepath)

    # get correct calibration idx, given the bsc date
    time_idx = np.where(calib_data['time'] == day)[0][0]

    # apply calibration to bsc data
    bsc_obs['backscatter'] *= calib_data['c_pro'][time_idx]

    # set timezone as UTC
    #bsc_obs[site]['time'] = np.array([i.replace(tzinfo=tz.gettz('UTC')) for i in bsc_obs[site]['time']])

    return bsc_obs

# --------------------------
# Main read in functions
# --------------------------

# Metadata
def read_ceil_metadata(datadir, loc_filename='CeilsCSV.csv'):

    """
    Read in ceil metadata (lon, lat) into a dictionary
    :param datadir:
    :return:
    """

    # read in the ceil locations
    loc_fname = datadir + loc_filename

    ceil_rawmeta = eu.csv_read(loc_fname)

    # convert into dictionary - [lon, lat]
    ceil_metadata = {i[1]: [float(i[3]), float(i[4])] for i in ceil_rawmeta}


    return ceil_metadata

# read in one BSC ceil obs file
def netCDF_read_BSC(datapath, site, height, day, var_type='beta_tR', SNRcorrect=True, **kwargs):

    """
    Read in backscatter data

    Gets data level and instrument model from metadata. Converts dates into list of datetimes, again using metadata
    to find the relative epoch. keyword argument are not necessary for L0 data

    :param datapath: full path to file
    :param height: instrument height [m]
    :param var_type:
    :param day (datetime):
    :param SNRcorrect (bool): turn data witha low signal-to-noise ratio into NaN?

    kwargs
    :param calib (bool): calibrate BSC data?

    :return: data: (dictionary)
    :return ceilLevel: level of the ceilometer
    """

    from netCDF4 import Dataset
    import datetime as dt
    from dateutil import tz


    ## Read data in
    # ------------------------

    # Create variable that is linked to filename
    # opens the file, but doesn't do the reading in just yet.
    datafile = Dataset(datapath, 'r')

    # get ceil data and site level from metadata
    ceilLevel = str(datafile.Data_level)

    # identify the sensor type e.g. CT25K or CL31
    sensorType = identify_sensorType(datafile)

    # Extract data and remove single dimension entries at the same time
    if ceilLevel == 'L0':
        data = {'backscatter': np.squeeze(datafile.variables['BSC'][:])}

        if sensorType == 'CL31':
            data['backscatter'] = data['backscatter'] * 1e-8
        elif sensorType == 'CT25K':
            data['backscatter'] = data['backscatter'] * 1e-7

    elif ceilLevel == 'L1':
        data = {'backscatter': np.squeeze(datafile.variables[var_type][:]),
                'SNR': np.squeeze(datafile.variables['SNR'][:])}

        if sensorType == 'CL31':
            data['backscatter'] = data['backscatter'] * 1e-12
        elif sensorType == 'CT25K':
            raise ValueError('L1 CT25K data read in, edit script to assign multiplication value (e.g. 10-12)')

        if SNRcorrect == True:
            # Signal-to-noise ratio filter (higher threshold
            # 0.05 - relaxed (kept ~50% data of LUMA data)
            # 0.2 - stricter (kept ~10% data of LUMA data)
            # 0.3 - self set to further reduce noise above BL - checked against profile plots of high PM10 day (19/01/16)
            data['backscatter'][data['SNR'] < 0.3] = np.nan

    # create range [m]
    if sensorType == 'CL31':
        step = 10.0 # range gate resolution
    elif sensorType == 'CT25K':
        step = 30.0
    data['range'] = np.arange(step, 7700.0 + step, step)

    # get ceilometer gate height profile
    data['height']  = get_ceil_gate_heights(datafile, data['range'], height)

    # Time
    # -------------
    # Read in time and convert to list of datetimes
    # get time units for time conversion
    tstr = datafile.variables['time'].units
    rawtime = np.squeeze(datafile.variables['time'][:])
    data['time'] = np.array(eu.time_to_datetime(tstr, rawtime))
    #data['time'] = np.array([i.replace(tzinfo=tz.gettz('UTC')) for i in data['time']])

    # calibrate data
    if kwargs['calib'] == True:
        data = calibrate_BSC_data(data, site, day)
    else:
        print 'calib = False: BSC data has not been calibrated!'

    return data, ceilLevel

# read in one ceilometer file
def netCDF_read_CLD(datapath, ins_height, ftype, **kwargs):

    """
    Read in one day of ceilometer data specifically

    Gets data level and instrument model from metadata. Converts dates into list of datetimes, again using metadata
    to find the relative epoch.

    :param datapath: full path to file
    :param ins_height: instrument height [m]

    kwargs
    :param var_type:

    :return: data: (dictionary)
    :return ceilLevel: level of the ceilometer
    """

    from netCDF4 import Dataset
    import datetime as dt
    from dateutil import tz
    from ellUtils import time_to_datetime, get_all_varaible_names

    def masked_to_nan_array(datafile, var_i):

        """ read in a single variable and convert masked array to nan if required """

        array = np.squeeze(datafile.variables[var_i][:])

        if isinstance(array, np.ma.MaskedArray):
            try:
                mask = array.mask  # bool (True = bad/masked data)
                array = np.asarray(array, dtype='float32')
                array[mask] = np.nan
            except:
                warnings.warn('cannot convert ' + var_i + 'from masked to numpy!')

        return array

    ## Read data in
    # ------------------------

    # Create variable that is linked to filename
    # opens the file, but doesn't do the reading in just yet.
    datafile = Dataset(datapath, 'r')

    # get ceil data level from metadata
    ceilLevel = str(datafile.Data_level)

    # identify the sensor type e.g. CT25K or CL31
    sensorType = identify_sensorType(datafile)

    # get list of variables to read in
    # if no specific variables were given, find all the variable names within the first netCDF file
    if 'vars' not in kwargs:
        vars = get_all_varaible_names(datapath)
    else:
        vars = kwargs['vars']

    # Extract data and convert masked arrays to normal arrays with nans
    data = {str(var_i): masked_to_nan_array(datafile, var_i) for var_i in vars}

    # create range [m]
    if sensorType == 'CL31':
        step = 10.0 # range gate resolution
    elif sensorType == 'CT25K':
        step = 30.0
    data['range'] = np.arange(step, 7700.0 + step, step)

    # get ceilometer gate height profile
    data['height'] = get_ceil_gate_heights(datafile, data['range'], ins_height)

    # adjust CBH data to be height agl and not just the instrument range value
    for var, var_data in data.iteritems():
        if var_data.shape != ():  # equivalent to 0 shape / scaler value
            if 'CLD' in var: # if the dimension length is equal to time's length
                if var not in ['CLD_status']:
                    data[var] = data[var].astype('float32') + ins_height

    # Time
    # -------------
    # get time units for time conversion
    tstr = datafile.variables['time'].units
    # Read in time and convert to list of datetimes
    rawtime = np.squeeze(datafile.variables['time'][:])
    data['time'] = np.array(time_to_datetime(tstr, rawtime))
    #data['time'] = np.array([i.replace(tzinfo=tz.gettz('UTC')) for i in data['time']])

    return data, ceilLevel

def netCDF_read_ceil(datapath, ins_height, ftype, **kwargs):

    """
    Read in one day of ceilometer data specifically

    Gets data level and instrument model from metadata. Converts dates into list of datetimes, again using metadata
    to find the relative epoch.

    :param datapath: full path to file
    :param ins_height: instrument height [m]
    :param ftype: what file type (and variable) it is, e.g. 'CLD' or 'MLH'

    kwargs
    :param var_type:

    :return: data: (dictionary)
    :return ceilLevel: level of the ceilometer
    """

    from netCDF4 import Dataset
    import datetime as dt
    from dateutil import tz
    from ellUtils.ellUtils import time_to_datetime, get_all_varaible_names

    def masked_to_nan_array(datafile, var_i):

        """ read in a single variable and convert masked array to nan if required """

        array = np.squeeze(datafile.variables[var_i][:])

        if isinstance(array, np.ma.MaskedArray):
            try:
                mask = array.mask  # bool (True = bad/masked data)
                array = np.asarray(array, dtype='float32')
                array[mask] = np.nan
            except:
                warnings.warn('cannot convert ' + var_i + 'from masked to numpy!')

        return array

    ## Read data in
    # ------------------------

    # Create variable that is linked to filename
    # opens the file, but doesn't do the reading in just yet.
    datafile = Dataset(datapath, 'r')

    # get ceil data level from metadata
    ceilLevel = str(datafile.Data_level)

    # identify the sensor type e.g. CT25K or CL31
    sensorType = identify_sensorType(datafile)

    # get list of variables to read in
    # if no specific variables were given, find all the variable names within the first netCDF file
    if 'vars' not in kwargs:
        vars = get_all_varaible_names(datapath)
    else:
        vars = kwargs['vars']

    # Extract data and convert masked arrays to normal arrays with nans
    data = {str(var_i): masked_to_nan_array(datafile, var_i) for var_i in vars}

    # create range [m]
    if sensorType == 'CL31':
        step = 10.0 # range gate resolution
    elif sensorType == 'CT25K':
        step = 30.0
    data['range'] = np.arange(step, 7700.0 + step, step)

    # get ceilometer gate height profile
    data['height'] = get_ceil_gate_heights(datafile, data['range'], ins_height)

    # adjust CBH data to be height agl and not just the instrument range value
    if ftype == 'CLD':
        for var, var_data in data.iteritems():
            if var_data.shape != ():  # equivalent to 0 shape / scaler value
                if 'CLD' in var: # if the dimension length is equal to time's length
                    if var not in ['CLD_status']:
                        data[var] = data[var].astype('float32') + ins_height

    # adjust MLH variable heights
    elif ftype == 'MLH':
        for var, var_data in data.iteritems():
            if var_data.shape != ():  # equivalent to 0 shape / scaler value
                if (('LH' in var) | ('MH' in var)): # it IS mixing or other layer height
                    if 'sd' not in var: # but not std var - don't adjust the std dev vars
                        data[var] = data[var].astype('float32') + ins_height


    # Time
    # -------------
    # get time units for time conversion
    tstr = datafile.variables['time'].units
    # Read in time and convert to list of datetimes
    rawtime = np.squeeze(datafile.variables['time'][:])
    data['time'] = np.array(time_to_datetime(tstr, rawtime))
    #data['time'] = np.array([i.replace(tzinfo=tz.gettz('UTC')) for i in data['time']])

    return data, ceilLevel

# read in all BSC ceil files
def read_all_ceils_BSC(day, site_bsc, ceilDatadir, calib=True, var_type='beta_tR', **kwargs):

    """
    Read in ceilometer backscatter, time, height and SNR data and strip the hours out of it.
    Calibrate data if requested

    :param day:
    :param site_bsc:
    :param ceilDatadir:
    :param mod_data:
    :param calib:

    kwargs
    :param timeMatch: array of datetimes for the observed backscatter to be matched to. timeMatch becomes the processed
        obs new time array, with the obs original time array being renamed as 'orig_nearest_time'

    :return: bsc_obs
    """

    from os.path import exists
    from ellUtils.ellUtils import nearest, binary_search

    # contains all the sites time-upscaled data
    bsc_obs = {}

    for site, height in site_bsc.iteritems():

        # create filename
        bsc_fname, site_id = create_filename(ceilDatadir, site, day, fType='BSC')

        # check if data is there, else skip it
        if exists(bsc_fname):

            # this sites time-upscaled data
            bsc_obs[site] = {}

            # read backscatter data
            data_obs, ceilLevel = netCDF_read_BSC(bsc_fname, site, height, day, calib=calib, var_type=var_type, **kwargs)

            # time match model data?
            # if timeMatch has data, then time match against it
            if 'timeMatch' in kwargs:

                # find nearest time in ceil time
                # pull out ALL the nearest time idxs and differences
                t_idx = np.array([binary_search(data_obs['time'], t) for t in kwargs['timeMatch']])
                t_diff = np.array([data_obs['time'][t_idx_obs_i] - kwargs['timeMatch'][t_idx_match_i]
                                   for t_idx_match_i, t_idx_obs_i in enumerate(t_idx)])

                # extract data
                data_obs['SNR'] = data_obs['SNR'][t_idx, :]
                data_obs['backscatter'] = data_obs['backscatter'][t_idx, :]
                data_obs['orig_nearest_time'] = data_obs['time'][t_idx]
                data_obs['time'] = kwargs['timeMatch']

                # overwrite t_idx locations where t_diff is too high with nans
                # only keep t_idx values where the difference is below 5 minutes
                bad = np.array([abs(i.days * 86400 + i.seconds) > 10 * 60 for i in t_diff])

                data_obs['SNR'][bad, :] = np.nan
                data_obs['backscatter'][bad, :] = np.nan

            # put data into the bsc_obs dictionary for export
            bsc_obs[site] = data_obs


    return bsc_obs

# read in all BSC ceil files: tier2
def read_all_ceils(day, site_bsc, ceilDatadir, ftype, calib=True, **kwargs):

    """
    Read in ceilometer data, time, height and SNR data and strip the hours out of it.
    Calibrate data if requested

    :param day:
    :param site_bsc:
    :param ceilDatadir:
    :param mod_data:
    :param calib (bool): calibrate BSC data?

    kwargs
    :param timeMatch: array of datetimes for the observed backscatter to be matched to. timeMatch becomes the processed
        obs new time array, with the obs original time array being renamed as 'orig_nearest_time'

    :return: bsc_obs
    """

    from os.path import exists
    from ellUtils.ellUtils import nearest, binary_search, netCDF_read

    # contains all the sites time-upscaled data
    all_data_obs = {}

    for site, ins_height in site_bsc.iteritems():

        # create filename
        data_fname, site_id = create_filename(ceilDatadir, site, day, ftype)

        # check if data is there, else skip it
        if exists(data_fname):

            # this sites time-upscaled data
            all_data_obs[site] = {}

            # read backscatter data
            if ftype == 'BSC':
                data_obs, ceilLevel = netCDF_read_BSC(data_fname, ins_height)
            elif (ftype == 'CLD') | (ftype == 'MLH'):
                data_obs, ceilLevel = netCDF_read_ceil(data_fname, ins_height, ftype)
                # netCDF_read_CLD - old method for CLD

            # time match model data?
            # if timeMatch has data, then time match against it
            if 'timeMatch' in kwargs:

                # time match the variables to 'timeMatch'
                data_obs = time_match_data(data_obs, **kwargs)

            # put data into the all_data_obs dictionary for export
            all_data_obs[site] = data_obs

    return all_data_obs

# Processing

# --------------------------------------------------------------------------------

def site_info(site_str):

    """
    Split the sensor-site string into its constituent parts

    :param site_str:
    :return: sensorType, sensor, datasetType, site
    """

    # get dataset description variables from site_str
    # string split
    site_split = site_str.split('_')

    # extract vars
    # sensorType = site_split[0]
    sensorType = site_split[0].split('-')[0]  # works with or without sensor ID
    sensor = site_split[0]
    datasetType = site_split[1]
    site = site_split[2]

    return sensorType, sensor, datasetType, site

# ----------------------------------------------------------------------------------

# Plotting

def ceil_file_quick_plot(datapath, savedir ='', hmin = '', hmax = '', vmin = 1e-7, vmax = 1e-04):

    """
    Read in and plot one day of raw ceilometer backscatter FROM FILE


    :param datadir:
    :param filename:
    :param savedir:
    :param savename:
    :return:

    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.dates import DateFormatter
    from matplotlib import rcParams
    from datetime import datetime
    from ellUtils import nearest
    import numpy as np


    #! If the date extract doesn't end up working, get the date out of the data_obs['time'] variable

    # extract date our of datapath
    split = datapath.split('_')
    dateraw = split[-2]

    # get site string for saving
    s = datapath.split('/')
    s = s[-1].split('_')
    site_str = '_'.join(s[0:3])

    # convert to date string with day/month/year
    date = datetime.strptime(dateraw, '%Y%j')
    datestr = date.strftime('%d/%m/%Y')

    # ------------------------------------------------
    # READ

    data_obs = netCDF_read_BSC(datapath)

    # ------------------------------------------------
    # PROCESS

    # Height trimming
    if hmin is not '':
        _, idxs, _ = nearest(data_obs['height'], hmin)
    else:
        idxs = 0
    # set height range

    if hmax is not '':
        _, idxe, _ = nearest(data_obs['height'],hmax)
    else:
        idxe = len(data_obs['height'])

    # trim data to match
    data_obs['height'] = data_obs['height'][idxs:idxe + 1]
    data_obs['backscatter'] = data_obs['backscatter'][idxs:idxe + 1, :]

    # ------------------------------------------------
    # PLOT

    ## 4.1 Plot the data
    fig, ax = plt.subplots(1, figsize=(8, 4))

    # sort dates out
    # date_list = [i.strftime("%Y %m %d") for i in data_obs['time']]
    date_list = [i.strftime('%H:%M') for i in data_obs['time']]

    # Plot data
    # will create white space where x <= 0 due to logging
    mesh = ax.pcolormesh(data_obs['time'], data_obs['height'], np.transpose(data_obs['backscatter']),
                         norm=LogNorm(vmin=vmin, vmax=vmax))
    ## 4.2 Prettify the plot
    plt.title('Raw backscatter profile: ' + site_str + ' - ' + datestr)

    ax.set_xlabel('Time [HH:MM]', labelpad=2)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    ax.set_ylabel('height [m]')
    ax.get_yaxis().set_tick_params(right='off', direction='out')
    # ax.yaxis.set_ticks([2000, 4000, 6000, 8000])
    ax.axis('tight')

    # colourbar
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.ax.set_aspect(10)

    rcParams.update({'font.size': 9})
    plt.tight_layout()

    # save the plot
    if savedir == '':
        pass
    else:
        plt.savefig(savedir + site_str + '_' + dateraw + '.png')

    plt.close(fig)

    return fig

def ceil_plot_to_ax(x, y, c , ax, hmin='', hmax='', tmin='', tmax='', vmin=1e-7, vmax=1e-04):

    """
    Read in and plot one day of raw ceilometer backscatter FROM DATA already read in

    :param data_obs: dictionary with [site]['backscatter'] and list of datetimes
    :param hmin: min height
    :param hmax: max height
    :param vmin: min value on colourmap
    :param vmax: max value on colourmap
    :return: mesh: to get colourbar information from
    :return: ax:
    """

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.dates import DateFormatter
    from datetime import datetime
    from ellUtils import nearest
    import numpy as np


    # ------------------------------------------------
    # PROCESS

    # ToDo trim the data a little first to reduce plotting time(safely beyond the threshold)
    # Height trimming (setting idx range)
    if hmin is not '':
        ymin = hmin
    else:
        ymin = 0

    if hmax is not '':
        ymax = hmax
    else:
        ymax = data_obs['height'][-1]

    #
    # idx_range = np.arange(idxs, idxe)
    #
    # # trim data to match
    # data_obs['height'] = data_obs['height'][idx_range]
    # data_obs['backscatter'] = data_obs['backscatter'][:, idx_range]

    # ------------------------------------------------
    # PLOT

    # Plot data
    # will create white space where x <= 0 due to logging
    mesh = ax.pcolormesh(x, y, c, norm=LogNorm(vmin=vmin, vmax=vmax))

    ## 4.2 Prettify the plot
    # plt.title('Raw backscatter profile: ' + site_str + ' - ' + datestr)

    # ax.set_xlabel('Time [HH:MM]', labelpad=2)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))

    # plot limits
    ax.set_ylim([ymin, ymax])
    if (tmin is not '') & (tmax is not ''):
        ax.set_xlim([tmin, tmax])

    # ax.set_ylabel('Height [m]')
    ax.get_yaxis().set_tick_params(right='off', direction='out')
    # ax.yaxis.set_ticks([2000, 4000, 6000, 8000])
    # ax.axis('tight')

    return mesh, ax