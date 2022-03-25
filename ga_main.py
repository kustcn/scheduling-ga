# %%
#1. import
import os

import time
import random
import copy

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astroplan import FixedTarget
from pytz import timezone
from astroplan import Observer
from astropy.utils import iers

from astropy.utils import iers

iers.conf.auto_download = False
iers.IERS_A_URL = '/Users/wsl/data/astro/finals2000A.all'#iers_data
iers_a = iers.IERS_A.open(iers.IERS_A_URL)
iers.IERS.iers_table = iers.IERS_A.open(iers.IERS_A_URL)

# catalog_info
targets_file_path = os.getcwd() + '/targets.txt' # targets list for scheduling
length_columns_file = 12  # skd file infomations of column

# %%
# 2 basic configuration

el_lim = 10.
el_max = 65.
lat = 25.028  # unit: degree
lon = 102.796  # unit: degree
height = 1974  # unit: meter
# telecope
YNAO = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=height * u.m)

observer = Observer(name='40m Telescope',
                    location=YNAO,
                    pressure=0.615 * u.bar,
                    relative_humidity=0.11,
                    temperature=0 * u.deg_C,
                    timezone=timezone('Asia/Shanghai'),
                    description="40m Telescope on Kunming, Yunnan")

utcoffset = 8 * u.hour
slew_time = 6.4  # estimate slew time(unit: minute)


# %%

# 修正方位角和高度角
def PointModel(az, el):
    vpar = np.loadtxt('./vpar_all_0318.txt')
    az0 = np.deg2rad(az)  # 角度转弧度
    el0 = np.deg2rad(el)
    len_vpar = len(vpar)

    if len_vpar == 8:
        delt_az0 = vpar[0] + vpar[3] * np.tan(el0) + vpar[4] / np.cos(el0) \
                   + vpar[5] * np.cos(az0) * np.tan(el0) \
                   + vpar[6] * np.sin(az0) * np.tan(el0)
        tanel = np.tan(np.array(el0))
        if type(tanel) == np.float64:
            if tanel < 0.1:
                tanel = 0.1
        else:
            tanel[tanel < 0.1] = 0.1
        delt_el0 = vpar[1] + vpar[2] * np.cos(el0) - vpar[5] * np.sin(az0) \
                   + vpar[6] * np.cos(az0) + vpar[7] / tanel

    if len_vpar == 22:
        delt_az0 = vpar[0] + np.tan(el0) * np.cos(az0) * vpar[2] \
                   + np.tan(el0) * np.sin(az0) * vpar[3] \
                   + np.tan(el0) * vpar[4] - 1. / np.cos(el0) * vpar[5] \
                   + az0 * vpar[11] + np.cos(az0) * vpar[12] \
                   + np.sin(az0) * vpar[13] + np.cos(2 * az0) * vpar[16] \
                   + np.sin(2 * az0) * vpar[17]
        tanel = np.tan(np.array(el0))
        if type(tanel) == np.float64:
            if tanel < 0.1:
                tanel = 0.1
        else:
            tanel[tanel < 0.1] = 0.1
        delt_el0 = vpar[1] - np.sin(az0) * vpar[2] + np.cos(az0) * vpar[3] \
                   + np.cos(el0) * vpar[6] + vpar[7] / np.tan(el0) \
                   + el0 * vpar[8] + np.cos(el0) * vpar[9] \
                   + np.sin(el0) * vpar[10] + np.cos(2 * az0) * vpar[14] \
                   + np.sin(2 * az0) * vpar[15] + np.cos(8 * el0) * vpar[18] \
                   + np.sin(8 * el0) * vpar[19] + np.cos(az0) * vpar[20] \
                   + np.sin(az0) * vpar[21]
    az1 = az + delt_az0
    el1 = el + delt_el0

    return az1, el1


# 计算源的俯仰和方位，并加载指向模型，给出修正后的结果
def src_az_el(srcname, timeinfo):
    srcaltaz = srcname.transform_to(AltAz(obstime=timeinfo, location=YNAO))
    az = srcaltaz.az.degree
    el = srcaltaz.alt.degree
    az, el = PointModel(az, el)
    return az, el


# 计算源的俯仰，并加载指向模型，给出修正后的结果
def src_el(srcname, timeinfo):
    _, el = src_az_el(srcname, timeinfo)
    return el


# 函数定义：从src1到src2的换源时间
def transition1(src1, src2, t0, obs_sit):
    """
    To calculate the time needed for telescope transition from 'src1' to 'src2' at 't0'

    Parameters
    ----------
    src1: the SkyCoord of source 1;
        e.g. src1 = SkyCoord('04 37 16 -47 15 09', frame='icrs', unit=(u.hourangle, u.deg))
    src2: see 'src1'
    t0: the UT time to start the transition
    obs_sit: the telescope's location, e.g. km=EarthLocation(-1281152.8*u.m, 5640864.4*u.m, 2682653.5*u.m)

    Notes
    -----
    astropy modules needed:
        * astropy.coordinates.SkyCoord
        * astropy.coordinates.EarthLocation
        * astropy.coordinates.AltAz
    """
    v_az, v_el = 1.0, 0.6
    az_speed_up_degree = 2.449
    az_speed_down_degree = 3.531
    el_speed_up_degree = 1.460
    el_speed_down_degree = 0.879
    az_speed_up_time = 5.93
    az_speed_down_time = 14.95
    el_speed_up_time = 6.35
    el_speed_down_time = 10.87
    az_stable_degree = az_speed_up_degree + az_speed_down_degree
    az_stable_time = az_speed_up_time + az_speed_down_time
    el_stable_degree = el_speed_up_degree + el_speed_down_degree
    el_stable_time = el_speed_up_time + el_speed_down_time
    # t_stable = el_speed_down_time # for stablising at the beginning and the end
    az1, el1 = src_az_el(src1, t0)
    az2, el2 = src_az_el(src2, t0)  # 计算俯仰和方位，加载指向模型
    azdif, eldif = az2 - az1, el2 - el1
    # 进行判断 方位角还是俯仰角
    if abs(az2 - az1) / v_az > abs(el2 - el1) / v_el + el_stable_time - az_stable_time:
        t_stable = az_stable_time - az_stable_degree / v_az
    else:
        t_stable = el_stable_time - el_stable_degree / v_el

    if np.abs(azdif) <= az_speed_up_degree and np.abs(eldif) <= el_speed_up_degree:
        dtaz = np.abs(azdif) / az_speed_down_degree * az_speed_down_time
        dtel = np.abs(eldif) / el_speed_down_degree * el_speed_down_time
        return max(dtaz, dtel)
    else:
        niter = 5
        dt = 0
        # count = 0
        for k in range(niter):
            obstime = t0 + dt * u.second
            az1, el1 = src_az_el(src1, obstime)
            az2, el2 = src_az_el(src2, obstime)
            dt_buf = max(abs(az2 - az1) / v_az, abs(el2 - el1) / v_el)
            if abs(dt_buf - dt) < 0.002:
                break
            else:
                dt = dt_buf
            # count += 1
        return float(format(dt, '.5f')) + t_stable


# 函数定义：从初始位置到第一颗源src的切换时间
def transition2(az0, el0, src, t0, obs_sit):
    """
    To calculate the time needed for telescope transition from 'src1' to 'src2' at 't0'

    Parameters
    ----------
    src1: the SkyCoord of source 1;
        e.g. src1 = SkyCoord('04 37 16 -47 15 09', frame='icrs', unit=(u.hourangle, u.deg))
    src2: see 'src1'
    t0: the UT time to start the transition
    obs_sit: the telescope's location, e.g. km=EarthLocation(-1281152.8*u.m, 5640864.4*u.m, 2682653.5*u.m)

    Notes
    -----
    astropy modules needed:
        * astropy.coordinates.SkyCoord
        * astropy.coordinates.EarthLocation
        * astropy.coordinates.AltAz
    """
    v_az, v_el = 1.0, 0.6
    az_speed_up_degree = 2.449
    az_speed_down_degree = 3.531
    el_speed_up_degree = 1.460
    el_speed_down_degree = 0.879
    az_speed_up_time = 5.93
    az_speed_down_time = 14.95
    el_speed_up_time = 6.35
    el_speed_down_time = 10.87
    az_stable_degree = az_speed_up_degree + az_speed_down_degree
    az_stable_time = az_speed_up_time + az_speed_down_time
    el_stable_degree = el_speed_up_degree + el_speed_down_degree
    el_stable_time = el_speed_up_time + el_speed_down_time
    # t_stable = el_speed_down_time # for stablising at the beginning and the end
    az, el = src_az_el(src, t0)
    azdif, eldif = az - az0, el - el0
    if abs(az - az0) / v_az > abs(el - el0) / v_el + el_stable_time - az_stable_time:
        t_stable = az_stable_time - az_stable_degree / v_az
    else:
        t_stable = el_stable_time - el_stable_degree / v_el

    if np.abs(azdif) <= az_speed_up_degree and np.abs(eldif) <= el_speed_up_degree:
        dtaz = np.abs(azdif) / az_speed_down_degree * az_speed_down_time
        dtel = np.abs(eldif) / el_speed_down_degree * el_speed_down_time
        return max(dtaz, dtel)
    else:
        niter = 5
        dt = 0
        # count = 0
        for k in range(niter):
            obstime = t0 + dt * u.second
            az, el = src_az_el(src, obstime)
            dt_buf = max(abs(az - az0) / v_az, abs(el - el0) / v_el)
            if abs(dt_buf - dt) < 0.002:
                break
            else:
                dt = dt_buf
            # count += 1
        return float(format(dt, '.5f')) + t_stable


# 函数定义：检查积分时间段内的俯仰情况
def check_el_lim(src, inte_time, t_start, loc):
    time_range = np.linspace(0, inte_time, num=inte_time)
    m = 1
    for t in time_range:
        obstime = t_start + t * u.minute
        az, el = src_az_el(src, obstime)
        if el < el_lim or el > el_max:
            m = 0
            break
    return m


# %%
# 设置初始方位和俯仰
az0 = 240
el0 = 88
# 设置开始观测时间
time_range = Time(['2021-05-21 00:00:00', "2021-05-22 00:00:00"])
time_ut = time_range[0] + np.linspace(0, 24, 24 * 60) * u.hour


# 定义源
class Target(object):
    def __init__(self, id, ra, dec, int_time, config_file) -> None:
        super(Target).__init__()
        self.id = id
        self.ra = ra  # 赤经
        self.dec = dec  # 赤纬
        self.int_time = int_time  # 积分时间
        self.config_file = config_file
        # self.value = value
        self.sc = SkyCoord(ra, dec)  # astropy中的固定源
        self.t = FixedTarget(name=self.id, coord=self.sc)  # astroplan中的固定源

    # 根据时间参数获得修正后源的方向和高度角
    def src_az_el(self, current_time):
        srcaltaz = self.sc.transform_to(AltAz(obstime=current_time, location=YNAO))
        az = srcaltaz.az.degree
        el = srcaltaz.alt.degree
        az, el = PointModel(az, el)
        return az, el


# %%
# 加载源
targets_dict = {}
inte_time_list = []
target_id_list = []

with open(targets_file_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split()
        src_per_info = []
        if len(line) == length_columns_file:
            src = line[1]
            ra = line[2] + 'h' + line[3] + 'm' + line[4] + 's'
            dec = line[5] + 'd' + line[6] + 'm' + line[7] + 's'
            inte_time = float(line[9])
            config_file = line[10]
            target = Target(src, ra, dec, inte_time, config_file)
            target_id_list.append(src)
            targets_dict[src] = target
        else:
            print("**********Successfully read all the src info. from the originally SKD file!**********\n")

# print(f"共有{len(target_id_list)}源")


# %%
import time


def time_seconds(t):
    return time.mktime(time.strptime(t.value.split(".")[0], "%Y-%m-%d %H:%M:%S"))

time_seconds(time_ut[382])

# %%
# 计算每个源在这3天内的，可以观测的秒数(target_available_seconds)和时间段(target_available_times)
min_airmass, max_airmass = 1.0, 10.0

target_available_times = {}  # 可用的观测时间的开始时间和结束时间
target_available_seconds = {}  # 可用的秒数


def cal_avail_times(target):
    available_times = []
    available_seconds = []

    # print(f"computing {target.id}")
    airmass = observer.altaz(time_ut, target.t).secz

    idxs = np.argwhere((airmass >= min_airmass) & (airmass < max_airmass))

    idxs = idxs.flatten()

    # idxs = [i for i in idxs if src_el(target.sc, time_ut[i]) > el_lim and src_el(target.sc, time_ut[i]) < el_max]

    # 限位计算
    limit_idxs = []
    for i in idxs:
        el = src_el(target.sc, time_ut[i])
        if el > el_lim and el < el_max:
            limit_idxs.append(i)

    last_idx = limit_idxs[0]
    start_idx = limit_idxs[0]
    end_idx = -1

    for i, idx in enumerate(limit_idxs):
        if idx - last_idx > 1:
            end_idx = last_idx

        if end_idx != -1 or i == len(limit_idxs) - 1:
            if i == len(limit_idxs) - 1:
                end_idx = limit_idxs[-1]
            available_times.append((time_ut[start_idx], time_ut[end_idx]))
            available_seconds.append(time_seconds(time_ut[end_idx]) - time_seconds(time_ut[start_idx]))

            end_idx = -1
            start_idx = idx

        last_idx = idx
    return target.id, available_times, available_seconds


# %%
# #并行计算方法
import dask.bag as db


def group_by_id(id1):
    return id1[0]


def reduce_id(v):
    return v


b = db.from_sequence([target for _,target in targets_dict.items()], npartitions=1)
c = b.map(cal_avail_times).foldby(key = group_by_id, binop=reduce_id)
d = c.compute()


# 结果验证
for k in d:
    target_available_times[k[0]] = k[1][1]
    target_available_seconds[k[0]] = k[1][2]

    # print(k[0],'可用时间段：',[f"{i[0].value}到{i[1].value}" for i in k[1][1]])
    # print(k[0],'可用时长（分钟）：',[i//60 for i in k[1][2]])

# 目标函数1
def gain(el):
    gain = 0.089094 * (0.94942833 + 0.0025333343 * el - 3.1726174e-05 * el * el)
    return np.array(gain)

# 目标函数2
def get_time(gene_list):
    time_all = 0
    time_all_out = 0
    for i in range(len(gene_list)):
        src_name_time = gene_list[i]

        inte_time = targets_dict[src_name_time[0]].int_time

        if i == 0:
            src0 = targets_dict[src_name_time[0]].ra + '' + targets_dict[src_name_time[0]].dec
            src = SkyCoord(src0, frame='icrs', unit=(u.hourangle, u.deg))
            utc_obs_start = time_range[0]
            slewtime = float(transition2(float(az0), float(el0), src, utc_obs_start, YNAO)) / 60.
            record_scan_start_time = utc_obs_start + slewtime * u.minute
            record_scan_end_time = record_scan_start_time + inte_time * u.minute
            # 为下一个循环开始设置时间
            utc_obs_start = Time(record_scan_end_time)
            timeinfo1 = slewtime * u.minute + inte_time * u.minute
            timeinfo1_out = slewtime + inte_time
        else:
            utc_obs_start_y = utc_obs_start
            src_1 = targets_dict[src_name_time[0]].ra + ' ' + targets_dict[src_name_time[0]].dec
            src1 = SkyCoord(src_1, frame='icrs', unit=(u.hourangle, u.deg))
            src_2_name = gene_list[i - 1]
            src_2 = targets_dict[src_2_name[0]].ra + ' ' + targets_dict[src_2_name[0]].dec
            src2 = SkyCoord(src_2, frame='icrs', unit=(u.hourangle, u.deg))

            slewtime = float(transition1(src1, src2, utc_obs_start_y, YNAO)) / 60.
            record_scan_start_time = utc_obs_start_y + slewtime * u.minute
            record_scan_end_time = record_scan_start_time + inte_time * u.minute
            utc_obs_start = Time(record_scan_end_time)

            timeinfo = slewtime * u.minute + inte_time * u.minute
            timeinfo_out = slewtime + inte_time
            time_all += timeinfo
            time_all_out += timeinfo_out
    time_sum = timeinfo1 + time_all

    # return time_sum
    return timeinfo_out + time_all_out

#########################################################################################################

from operator import itemgetter

class Gene:
    def __init__(self, **data):
        self.__dict__.update(data)
        self.size = len(data['data'])

class GA:
    def __init__(self, parameter):
        self.parameter = parameter
        pop = []
        for i in range(self.parameter['NP']):
            geneinfo = self.get_init_solution()
            # fitness = self.get_init_solution()[0]
            fitness = self.fitness_func(geneinfo)
            pop.append({'Gene': Gene(data=geneinfo), 'fitness': fitness})
        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)

        # 评估每个个体的函数值，总耗时

    def get_init_solution(self, num_source=28):
        '''
        :todo:生成1个解
        '''

        geneinfo = []
        target_plan_list = []

        # 记录每个源被观测的次数
        # target_observed = copy.deepcopy(target_available_times)
        target_observed = dict()
        for i in range(0, 53):
            target_observed.update({target_id_list[i]: 0})

        for i in range(0, 53):
            if target_available_times[target_id_list[i]][0][0] == time_range[0]:
                target_plan_list.append(target_id_list[i])
        # 目前遍历所有可以作为种群第一个元素的源id
        src_name = random.sample(target_plan_list, 1)
        inte_time = targets_dict[src_name[0]].int_time  # 在src_name源上观测需要消耗的时间
        src0 = targets_dict[src_name[0]].ra + '' + targets_dict[src_name[0]].dec
        src = SkyCoord(src0, frame='icrs', unit=(u.hourangle, u.deg))
        utc_obs_start = time_range[0]
        slewtime = float(transition2(float(az0), float(el0), src, utc_obs_start, YNAO)) / 60.
        # 计算第一个源的开始时间和结束时间
        record_scan_start_time = utc_obs_start + slewtime * u.minute
        record_scan_end_time = record_scan_start_time + inte_time * u.minute

        target_observed[src_name[0]] += 1

        # 为下一个循环开始设置时间
        target_calibration_list = copy.deepcopy(target_id_list)
        # 将源名称和源开始时间和结束时间放在一起
        geneinfo.append([src_name[0], record_scan_start_time, record_scan_end_time])

        for x in range(0, num_source - 1):

            target_variable_list = []
            utc_obs_start = Time(record_scan_end_time)
            # 找出所有可用的源，存储到target_variable_list中
            for y in range(0, 52):
                try:
                    target_calibration_list.remove(src_name[0])
                except ValueError:
                    pass

                # 遍历剩余源
                # src_name_y = target_calibration_list[y]
                # utc_obs_start_y = utc_obs_start
                # inte_time2 = targets_dict[src_name_y].int_time
                # if len(target_available_times[src_name_y]) == 1:
                #     if (target_available_times[src_name_y][0][0] > utc_obs_start_y and \
                #             target_available_times[src_name_y][0][1] < utc_obs_start_y + inte_time2 + slew_time):
                #         target_variable_list.append(target_calibration_list[y])
                # if len(target_available_times[src_name_y]) == 2:
                #     if (target_available_times[src_name_y][0][0] > utc_obs_start_y and \
                #         target_available_times[src_name_y][0][1] < utc_obs_start_y + inte_time2 + slew_time) or \
                #             (target_available_times[src_name_y][1][0] > utc_obs_start_y and \
                #              target_available_times[src_name_y][1][1] < utc_obs_start_y + inte_time + slew_time):
                #         target_variable_list.append(target_calibration_list[y])
                # if len(target_available_times[src_name_y]) == 3:
                #     if (target_available_times[src_name_y][0][0] > utc_obs_start_y and \
                #         target_available_times[src_name_y][0][1] < utc_obs_start_y + inte_time2 + slew_time) or \
                #             (target_available_times[src_name_y][1][0] > utc_obs_start_y and \
                #              target_available_times[src_name_y][1][1] < utc_obs_start_y + inte_time + slew_time) or \
                #             (target_available_times[src_name_y][2][0] > utc_obs_start_y and \
                #              target_available_times[src_name_y][2][1] < utc_obs_start_y + inte_time + slew_time):
                #         target_variable_list.append(target_calibration_list[y])
                src_name_y = target_calibration_list[y]
                utc_obs_start_y = utc_obs_start
                inte_time = targets_dict[src_name_y].int_time
                src_1 = targets_dict[src_name_y].ra + ' ' + targets_dict[src_name_y].dec
                src1 = SkyCoord(src_1, frame='icrs', unit=(u.hourangle, u.deg))
                src_2 = targets_dict[geneinfo[x][0]].ra + ' ' + targets_dict[geneinfo[x][0]].dec
                src2 = SkyCoord(src_2, frame='icrs', unit=(u.hourangle, u.deg))

                slewtime1 = float(transition1(src1, src2, utc_obs_start_y, YNAO)) / 60.

                # 计算第二颗源到最后一颗源的开始时间和结束时间
                record_scan_start_time = utc_obs_start_y + slewtime1 * u.minute
                record_scan_end_time = record_scan_start_time + inte_time * u.minute
                #判断这个时间段是否位于源的可用时间段内
                flag = False
                if len(target_available_times[src_name_y]) == 1:
                    if (target_available_times[src_name_y][0][0] <= record_scan_start_time and \
                            target_available_times[src_name_y][0][1] >= record_scan_end_time):
                        flag = True
                elif len(target_available_times[src_name_y]) == 2:
                    if (target_available_times[src_name_y][0][0] <= record_scan_start_time and \
                            target_available_times[src_name_y][0][1] >= record_scan_end_time) or \
                            (target_available_times[src_name_y][1][0] <= record_scan_start_time and \
                             target_available_times[src_name_y][1][1] >= record_scan_end_time):
                        flag = True
                elif len(target_available_times[src_name_y]) == 3:
                    if (target_available_times[src_name_y][0][0] <= record_scan_start_time and \
                        target_available_times[src_name_y][0][1] >= record_scan_end_time) or \
                            (target_available_times[src_name_y][1][0] <= record_scan_start_time and \
                             target_available_times[src_name_y][1][1] >= record_scan_end_time) or \
                            (target_available_times[src_name_y][2][0] <= record_scan_start_time and \
                             target_available_times[src_name_y][2][1] >= record_scan_end_time):
                        flag = True
                if flag:
                    target_variable_list.append(target_calibration_list[y])


            # 选择target_variable_list中被观测次数最少的源作为下一颗源
            Value_min = 1E30
            target_variable_list_mini_observed = []
            for iSource in target_variable_list:
                if target_observed[iSource] < Value_min:
                    target_variable_list_mini_observed = []
                    target_variable_list_mini_observed.append(iSource)
                    Value_min = target_observed[iSource]
                elif target_observed[iSource] == Value_min:
                    target_variable_list_mini_observed.append(iSource)

            # 从target_variable_list_mini_observed随机选取一个作为下一颗源
            src_name_random = random.sample(target_variable_list_mini_observed, 1)
            target_observed[src_name_random[0]] += 1

            src_1 = targets_dict[src_name_random[0]].ra + ' ' + targets_dict[src_name_random[0]].dec
            src1 = SkyCoord(src_1, frame='icrs', unit=(u.hourangle, u.deg))
            src_2 = targets_dict[geneinfo[x][0]].ra + ' ' + targets_dict[geneinfo[x][0]].dec
            src2 = SkyCoord(src_2, frame='icrs', unit=(u.hourangle, u.deg))

            slewtime1 = float(transition1(src1, src2, utc_obs_start_y, YNAO)) / 60.

            # 计算第二颗源到最后一颗源的开始时间和结束时间
            inte_time = targets_dict[src_name_random[0]].int_time
            record_scan_start_time = utc_obs_start_y + slewtime1 * u.minute
            record_scan_end_time = record_scan_start_time + inte_time * u.minute
            geneinfo.append([src_name_random[0], record_scan_start_time, record_scan_end_time])

        return geneinfo

    def find_next_point(self, cur_solution, num_nextPoint=1):
        '''
        :todo: 根据现有的解cur_solution的情况，计算下num_nextPoint个源
        output:
            newPoint：计算得到的下一个源
        '''

        geneinfo = copy.deepcopy(cur_solution)

        # 记录每个源被观测的次数
        target_observed = dict()
        for i in range(0, 53):
            target_observed.update({target_id_list[i]: 0})
        for igeneinfo in geneinfo:
            target_observed[igeneinfo[0]] += 1

        target_plan_list = []
        record_scan_start_time = geneinfo[-1][1]
        record_scan_end_time = geneinfo[-1][2]

        # 为下一个循环开始设置时间
        target_calibration_list = copy.deepcopy(target_id_list)
        # geneinfo.append(src_name)
        # 将源名称和源开始时间和结束时间放在一起

        for x in range(0, num_nextPoint):

            target_variable_list = []
            utc_obs_start = Time(record_scan_end_time)
            # 找出所有可用的源，存储到target_variable_list中
            for y in range(0, 52):
                try:
                    # target_calibration_list.remove(src_name[0])
                    target_calibration_list.remove(geneinfo[0][0])
                except ValueError:
                    pass

                # 遍历剩余源
                src_name_y = target_calibration_list[y]
                utc_obs_start_y = utc_obs_start
                inte_time = targets_dict[src_name_y].int_time
                src_1 = targets_dict[src_name_y].ra + ' ' + targets_dict[src_name_y].dec
                src1 = SkyCoord(src_1, frame='icrs', unit=(u.hourangle, u.deg))
                src_2 = targets_dict[geneinfo[x][0]].ra + ' ' + targets_dict[geneinfo[x][0]].dec
                src2 = SkyCoord(src_2, frame='icrs', unit=(u.hourangle, u.deg))

                slewtime1 = float(transition1(src1, src2, utc_obs_start_y, YNAO)) / 60.

                # 计算第二颗源到最后一颗源的开始时间和结束时间
                record_scan_start_time = utc_obs_start_y + slewtime1 * u.minute
                record_scan_end_time = record_scan_start_time + inte_time * u.minute
                # 判断这个时间段是否位于源的可用时间段内
                flag = False
                if len(target_available_times[src_name_y]) == 1:
                    if (target_available_times[src_name_y][0][0] <= record_scan_start_time and \
                            target_available_times[src_name_y][0][1] >= record_scan_end_time):
                        flag = True
                elif len(target_available_times[src_name_y]) == 2:
                    if (target_available_times[src_name_y][0][0] <= record_scan_start_time and \
                        target_available_times[src_name_y][0][1] >= record_scan_end_time) or \
                            (target_available_times[src_name_y][1][0] <= record_scan_start_time and \
                             target_available_times[src_name_y][1][1] >= record_scan_end_time):
                        flag = True
                elif len(target_available_times[src_name_y]) == 3:
                    if (target_available_times[src_name_y][0][0] <= record_scan_start_time and \
                        target_available_times[src_name_y][0][1] >= record_scan_end_time) or \
                            (target_available_times[src_name_y][1][0] <= record_scan_start_time and \
                             target_available_times[src_name_y][1][1] >= record_scan_end_time) or \
                            (target_available_times[src_name_y][2][0] <= record_scan_start_time and \
                             target_available_times[src_name_y][2][1] >= record_scan_end_time):
                        flag = True
                if flag:
                    target_variable_list.append(target_calibration_list[y])

            # 选择target_variable_list中被观测次数最少的源作为下一颗源
            Value_min = 1E30
            target_variable_list_mini_observed = []
            for iSource in target_variable_list:
                if target_observed[iSource] < Value_min:
                    target_variable_list_mini_observed = []
                    target_variable_list_mini_observed.append(iSource)
                    Value_min = target_observed[iSource]
                elif target_observed[iSource] == Value_min:
                    target_variable_list_mini_observed.append(iSource)

            # 从target_variable_list_mini_observed随机选取一个作为下一颗源
            src_name_random = random.sample(target_variable_list_mini_observed, 1)
            target_observed[src_name_random[0]] += 1

            src_1 = targets_dict[src_name_random[0]].ra + ' ' + targets_dict[src_name_random[0]].dec
            src1 = SkyCoord(src_1, frame='icrs', unit=(u.hourangle, u.deg))
            src_2 = targets_dict[geneinfo[x][0]].ra + ' ' + targets_dict[geneinfo[x][0]].dec
            src2 = SkyCoord(src_2, frame='icrs', unit=(u.hourangle, u.deg))

            slewtime1 = float(transition1(src1, src2, utc_obs_start_y, YNAO)) / 60.

            # 计算第二颗源到最后一颗源的开始时间和结束时间
            inte_time = targets_dict[src_name_random[0]].int_time
            record_scan_start_time = utc_obs_start_y + slewtime1 * u.minute
            record_scan_end_time = record_scan_start_time + inte_time * u.minute
            geneinfo.append([src_name_random[0], record_scan_start_time, record_scan_end_time])

        return geneinfo

    # 目标函数3
    def get_angle(self, gene_list):
        '''
        遍历源，并计算角度偏移情况
        '''
        el_list = []
        for source in gene_list:
            start_time = source[1]
            while start_time < source[2]:
                el_list.append(src_el(targets_dict[source[0]].sc, start_time) - 45)
                start_time += self.parameter['TIME_INTERVAL'] * u.minute

        return sum(np.power(el_list, 2))


    def fitness_func(self, gene):

        get_time_total = float(get_time(gene))
        get_angle_total = self.get_angle(gene)

        return self.parameter['w1'] * get_angle_total + self.parameter['w2'] * get_time_total

    def islegal(self, gene):
        '''
        判断解gene是否合法
        若合法则不做任何处理，否则则进行处理
        '''
        for igene in range(1, len(gene.data)):
            flag = False
            if gene.data[igene][1] > gene.data[igene - 1][2] and gene.data[igene][1] < gene.data[igene - 1][2] + slew_time * u.minute:
                flag = True


            if not flag:
                # 若不满足约束，则重新选择一个点
                temp = self.find_next_point(gene.data[0 : igene])
                gene.data[0 : igene + 1] = temp

        return gene

    def selectBest(self, pop):
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=False)
        return s_inds[0]

    # 按照一定的概率选择个体

    def selection(self, individuals, k):

        s_inds = copy.deepcopy(sorted(individuals, key=itemgetter("fitness"), reverse=False))
        sum_fits = sum(ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop
        chosen = []
        for i in range(k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits], as threshold
            sum_ = 0
            for ind in s_inds:
                sum_ += ind['fitness']  # sum up the fitness
                if sum_ >= u:
                    chosen.append(copy.deepcopy(ind))
                    break
        chosen = sorted(chosen, key=itemgetter("fitness"), reverse=False)
        return chosen

    # 基因交叉
    def crossoperate(self, offspring):
        dim = min(len(offspring[0]['Gene'].data), len(offspring[1]['Gene'].data))

        geninfo1 = copy.deepcopy(offspring[0]['Gene'].data)  # Gene's data of first offspring chosen from the selected pop
        geninfo2 = copy.deepcopy(offspring[1]['Gene'].data)  # Gene's data of second offspring chosen from the selected pop
        if dim == 1:
            pos1 = 1
            pos2 = 1
        else:
            pos1 = random.randrange(1, dim)  # select a position in the range from 0 to dim-1,
            pos2 = random.randrange(1, dim)

        newoff1 = Gene(data=[])  # offspring1 produced by cross operation
        newoff2 = Gene(data=[])  # offspring2 produced by cross operation
        temp1 = []
        temp2 = []
        for i in range(dim):
            if min(pos1, pos2) <= i <= max(pos1, pos2):
                temp2.append(copy.deepcopy(geninfo2[i]))
                temp1.append(copy.deepcopy(geninfo1[i]))
            else:
                temp2.append(copy.deepcopy(geninfo1[i]))
                temp1.append(copy.deepcopy(geninfo2[i]))

        newoff1.data = temp1
        newoff2.data = temp2

        return newoff1, newoff2

    def get_time(self, gene_list):
        time_all = 0
        for i in range(len(gene_list)):
            src_name_time = gene_list[i]

            inte_time = targets_dict[src_name_time[0]].int_time

            if i == 0:
                src0 = targets_dict[src_name_time[0]].ra + '' + targets_dict[src_name_time[0]].dec
                src = SkyCoord(src0, frame='icrs', unit=(u.hourangle, u.deg))
                utc_obs_start = time_range[0]
                slewtime = float(transition2(float(az0), float(el0), src, utc_obs_start, YNAO)) / 60.
                record_scan_start_time = utc_obs_start + slewtime * u.minute
                record_scan_end_time = record_scan_start_time + inte_time * u.minute
                # 为下一个循环开始设置时间
                utc_obs_start = Time(record_scan_end_time)
                timeinfo1 = slewtime * u.minute + inte_time * u.minute
            else:
                utc_obs_start_y = utc_obs_start
                src_1 = targets_dict[src_name_time[0]].ra + ' ' + targets_dict[src_name_time[0]].dec
                src1 = SkyCoord(src_1, frame='icrs', unit=(u.hourangle, u.deg))
                src_2_name = gene_list[i - 1]
                src_2 = targets_dict[src_2_name[0]].ra + ' ' + targets_dict[src_2_name[0]].dec
                src2 = SkyCoord(src_2, frame='icrs', unit=(u.hourangle, u.deg))

                slewtime = float(transition1(src1, src2, utc_obs_start_y, YNAO)) / 60.
                record_scan_start_time = utc_obs_start_y + slewtime * u.minute
                record_scan_end_time = record_scan_start_time + inte_time * u.minute
                utc_obs_start = Time(record_scan_end_time)

                timeinfo = slewtime * u.minute + inte_time * u.minute
                time_all += timeinfo
        time_sum = timeinfo1 + time_all

        return time_sum

    def mutation(self, offspring):
        dim = len(offspring.data)
        pos = random.randrange(1, dim)
        temp = self.find_next_point(offspring.data[0 : pos])
        offspring.data[0: pos + 1] = temp

        return offspring

    def GA_main(self):
        popsize = self.parameter['NP']

        print("Start of evolution")

        # Begin the evolution

        aaa = time.time()
        trace = list()
        count = 0       #记录已经多少次没能优化了
        for g in range(self.parameter['maxGen']):

            print("############### Generation {} ###############".format(g))

            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, popsize)

            nextoff = []
            while len(nextoff) != popsize:
                # Apply crossover and mutation on the offspring

                # Select two individuals
                offspring = copy.deepcopy([selectpop.pop() for _ in range(2)])

                if random.random() < self.parameter['CXPB']:  # cross two individuals with probability CXPB
                    crossoff1, crossoff2 = self.crossoperate(offspring)
                    if random.random() < self.parameter['MUTPB']:

                        muteoff1 = self.mutation(crossoff1)
                        muteoff2 = self.mutation(crossoff2)

                        muteoff1 = self.islegal(muteoff1)
                        muteoff2 = self.islegal(muteoff2)
                        fit_muteoff1 = self.fitness_func(muteoff1.data)
                        fit_muteoff2 = self.fitness_func(muteoff2.data)

                        nextoff.append({'Gene': muteoff1, 'fitness': fit_muteoff1})
                        nextoff.append({'Gene': muteoff2, 'fitness': fit_muteoff2})
                    else:

                        # fit_crossoff1 = self.get_time(crossoff1.data)  # Evaluate the individuals
                        # fit_crossoff2 = self.get_time(crossoff2.data)
                        crossoff1 = self.islegal(crossoff1)
                        crossoff2 = self.islegal(crossoff2)
                        fit_crossoff1 = self.fitness_func(crossoff1.data)
                        fit_crossoff2  = self.fitness_func(crossoff2.data)
                        nextoff.append({'Gene': crossoff1, 'fitness': fit_crossoff1})
                        nextoff.append({'Gene': crossoff2, 'fitness': fit_crossoff2})
                else:
                    nextoff.extend(offspring)

            # The population is entirely replaced by the offspring
            self.pop = nextoff

            bbb = time.time()
            print('迭代一次时长：' + str(bbb - aaa))

            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
            best_ind = self.selectBest(self.pop)

            if best_ind['fitness'] < self.bestindividual['fitness']:
                self.bestindividual = best_ind
                count = 0
            else:
                count += 1
            trace.append(self.bestindividual['fitness'])

            for ibestIndiv in range(len(self.bestindividual['Gene'].data)):
                print('第' + str(ibestIndiv) + '个源：' + self.bestindividual['Gene'].data[ibestIndiv][0] + '\n')
                print('开始时间：' + str(self.bestindividual['Gene'].data[ibestIndiv][1]) + '\n')
                print('结束时间：' + str(self.bestindividual['Gene'].data[ibestIndiv][2]) + '\n')

                print("Best individual found is  {}".format(self.bestindividual['fitness']))

            if count < self.parameter['COUNT_STOP']:
                break
            
        print("------ End of (successful) evolution ------")
        if count == self.parameter['COUNT_STOP']:
            print('算法因{}次迭代不优化而停止'.format(self.parameter['COUNT_STOP']))
        else:
            print('算法因达到最大迭代次数而停止')

# %%
# begin 
params = {'NP' : 10, \
            'num_source': 28, \
            'w1' : 0.5, \
            'w2' : 0.5, \
            'maxGen' : 10, \
            'CXPB' : 0.8, \
            'MUTPB' : 0.1}
ga = GA(params)

ga.GA_main()
# %%
# import pickle
# a_f = open("./avail_time_slots.pickle","wb")
# pickle.dump(d,"avail_time_slots.pickle")
# a_f.close()
# %%
