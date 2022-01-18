# %%
#1,import packages and configuration
import os
import sys
import time
import random
import copy
import datetime
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

program_dir = os.getcwd() 
original_skd_file = 'targets.txt'
target_catalog = program_dir + '/' + original_skd_file#original skd file name
scheduled_targets = program_dir + '/final_' + original_skd_file
f4debug = program_dir + '/time_source.txt'
f4check = program_dir + '/time_check.txt'
length_src_info = 12# skd file infomations of colum

# %% [markdown]
# # 2. YRT40 site configuration
el_lim = 10.
el_max = 65.
lat = 25.028  # unit: degree
lon = 102.796 # unit: degree
height = 1974 # unit: meter
# 设置台站信息
YNAO = EarthLocation(lat = lat * u.deg, lon = lon * u.deg, height = height * u.m)

observer = Observer(name='40m Telescope',
               location=YNAO,
               pressure=0.615 * u.bar,
               relative_humidity=0.11,
               temperature=0 * u.deg_C,
               timezone=timezone('Asia/Shanghai'),
               description="40m Telescope on Kunming, Yunnan")

utcoffset = 8 * u.hour
slew_time = 6.4 # estimate slew time(unit: minute)

observer

# %% [markdown]
# # 3.pointing model

# %%
#antenna gain
def full_gain(el):
    gain = np.zeros(len(el))
    for i in range(len(el)):
        if (el[i] > el_max) or (el[i] < el_lim):
            gain[i] = 0
        else:
            gain[i] = 0.089094 * (0.94942833 + 0.0025333343 * el[i] - 3.1726174e-05 * el[i] * el[i])
    return np.array(gain)

def gain(el):
    gain = 0.089094 * (0.94942833 + 0.0025333343 * el - 3.1726174e-05 * el * el)
    return np.array(gain)

    
#修正方位角和高度角
def PointModel(az, el):
    vpar = np.loadtxt('./vpar_all_0318.txt')
    az0 = np.deg2rad(az)     #角度转弧度
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
    srcaltaz = srcname.transform_to(AltAz(obstime = timeinfo, location = YNAO))
    az = srcaltaz.az.degree
    el = srcaltaz.alt.degree
    az, el = PointModel(az, el)
    return az, el
    
# 计算源的俯仰，并加载指向模型，给出修正后的结果
def src_el(srcname, timeinfo):
    _, el = src_az_el(srcname, timeinfo)
    return el


#函数定义：从src1到src2的换源时间
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
    az2, el2 = src_az_el(src2, t0) #计算俯仰和方位，加载指向模型
    azdif, eldif = az2 - az1, el2 - el1
    #进行判断 方位角还是俯仰角
    if abs(az2 - az1) / v_az >  abs(el2 - el1) / v_el + el_stable_time - az_stable_time:
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
    if abs(az - az0) / v_az >  abs(el - el0) / v_el + el_stable_time - az_stable_time:
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
    time_range = np.linspace(0, inte_time, num = inte_time)
    m = 1
    for t in time_range:
        obstime = t_start + t * u.minute
        az, el = src_az_el(src, obstime)
        if el < el_lim or el > el_max:
            m = 0
            break
    return m



# %% [markdown]
# # 4.观测时间与定义源

# %%
#设置初始方位和俯仰
az0 = 240
el0 = 88
#设置开始观测时间
time_range = Time(['2021-05-21 00:00:00', "2021-05-22 00:00:00"])
time_ut = time_range[0] + np.linspace(0, 24, 24*60)*u.hour

#定义源
class Target(object):
    def __init__(self, id, ra, dec, int_time, config_file) -> None:
        super(Target).__init__()
        self.id = id
        self.ra = ra                                      #赤经
        self.dec = dec                                    #赤纬
        self.int_time = int_time                          #积分时间
        self.config_file = config_file
        # self.value = value      
        self.sc = SkyCoord(ra, dec)                       # astropy中的固定源
        self.t = FixedTarget(name=self.id, coord=self.sc) # astroplan中的固定源
    #根据时间参数获得修正后源的方向和高度角
    def src_az_el(self, current_time):
        srcaltaz = self.sc.transform_to(AltAz(obstime = current_time, location = YNAO))
        az = srcaltaz.az.degree
        el = srcaltaz.alt.degree
        az, el = PointModel(az, el)
        return az, el


# %% [markdown]
# # 5.加载源

# %%
#加载源
targets_dict = {}
inte_time_list = []
target_id_list = []
src_info_list = []


with open (target_catalog, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.split()
        src_per_info = []
        if len(line) == length_src_info:
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

print(f"共有{len(target_id_list)}源")
print(target_id_list)

# %% [markdown]
# # 6.可观测时间计算

# %%
from astropy.time import Time
from astroplan.utils import time_grid_from_range
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

from astroplan.plots import plot_airmass
from matplotlib import dates

i = 0

for k,target in targets_dict.items():
    # print(len(time_ut[dd]))
    print(f"{target.id}，ra:{target.ra}，dec:{target.dec}")

    plot_airmass(target.t, observer, time_ut, altitude_yaxis=True, max_airmass=10.0) 
    i = i + 1
    if i >= 3:  #只画3个
        break
    
# %%
import time
def time_seconds(t):
     return time.mktime(time.strptime(t.value.split(".")[0], "%Y-%m-%d %H:%M:%S"))
time_seconds(time_ut[382])

# %%
#计算每个源在这3天内的，可以观测的秒数(target_available_seconds)和时间段(target_available_times)
min_airmass,max_airmass = 1.0,10.0

target_available_times = {}#可用的观测时间的开始时间和结束时间
target_available_seconds = {}#可用的秒数

def cal_avail_times(target):
    available_times = []
    available_seconds = []
    
    print(f"computing {target.id}")
    airmass = observer.altaz(time_ut, target.t).secz

    idxs = np.argwhere((airmass >= min_airmass) & (airmass < max_airmass))

    idxs = idxs.flatten()

    #idxs = [i for i in idxs if src_el(target.sc, time_ut[i]) > el_lim and src_el(target.sc, time_ut[i]) < el_max]

    #限位计算
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

        if end_idx != -1 or i == len(limit_idxs)-1:
            if i == len(limit_idxs)-1:
                end_idx = limit_idxs[-1]
            available_times.append((time_ut[start_idx],time_ut[end_idx]))
            available_seconds.append(time_seconds(time_ut[end_idx])-time_seconds(time_ut[start_idx]))

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

b = db.from_sequence([target for _,target in targets_dict.items()], npartitions=53)
c = b.map(cal_avail_times).foldby(key = group_by_id, binop=reduce_id)
d = c.compute()

# %%
from datetime import datetime
#结果验证
for k in d:
    target_available_times[k[0]] = k[1][1]
    target_available_seconds[k[0]] = k[1][2]
    
    print(k[0],'可用时间段：',[f"{i[0].value}到{i[1].value}" for i in k[1][1]])
    print(k[0],'可用时长（分钟）：',[i//60 for i in k[1][2]])
target_gain_dict = {}

for i in range (len(target_id_list)):
    src_gain = targets_dict[target_id_list[i]].ra + ' ' + targets_dict[target_id_list[i]].dec
    srcgain = SkyCoord(src_gain, frame = 'icrs', unit = (u.hourangle, u.deg))
    goal_list = []
    for j in range(len(target_available_times[target_id_list[i]])):
        gain_src_info = 0
        time_begin = target_available_times[target_id_list[i]][j][0]
        
        max_slew_time = 6.4

        time_end = target_available_times[target_id_list[i]][j][1]
        
        time_difference_value = time_end-time_begin

        move_time = int(time_difference_value.sec/60)

        time_con = np.linspace(0, move_time, num = int(move_time) * 1 + 1)

        time_info = time_begin+max_slew_time*u.minute + time_con * u.minute
        
        az,el = src_az_el(srcgain,time_info)
        goal_gain = sum(gain(el))
        gain_src_info += goal_gain

    target_gain_dict[target_id_list[i]] = gain_src_info

    
# %%
print(target_available_times['2330-2005'][0])
print(target_available_times['2330-2005'][1])
print(target_gain_dict['0034-0721'])


# %% [markdown]
# # 7.定义遗传算法

# %%
#定义遗传算法
#Gene类 只有一个初始化方法，用于获取基因的内容和大小 内容为53个源排列顺序的列表
from operator import itemgetter
class Gene:
    def __init__(self, **data):
        self.__dict__.update(data)
        self.size = len(data['data'])

class GA:
    def __init__(self,parameter):
        self.parameter = parameter
        pop = []
        for i in range(self.parameter[2]):
            geneinfo, fitness = self.get_init_solution()
            pop.append({'Gene': Gene(data=geneinfo), 'fitness': fitness})
        self.pop = pop
        self.bestindividual = self.selectBest(self.pop)
        
        #评估每个个体的函数值，总耗时


    def get_init_solution(self):
        geneinfo = []
        target_plan_list = []
        
        for i in range (0,53):
            if target_available_times[target_id_list[i]][0][0] == time_range[0]:
                target_plan_list.append(target_id_list[i])        
        #目前遍历所有可以作为种群第一个元素的源id
        score = 0

        src_name = random.sample(target_plan_list,1)
        inte_time = targets_dict[src_name[0]].int_time    
        src0 = targets_dict[src_name[0]].ra + '' + targets_dict[src_name[0]].dec
        src = SkyCoord(src0, frame = 'icrs', unit = (u.hourangle, u.deg))#使用targets_dict[src_name[0]]代替
        utc_obs_start = time_range[0]
        slewtime = float(transition2(float(az0), float(el0), src, utc_obs_start, YNAO)) / 60.
        record_scan_start_time = utc_obs_start + slewtime * u.minute
        record_scan_end_time = record_scan_start_time + inte_time * u.minute

        #为下一个循环开始设置时间
        target_calibration_list = copy.deepcopy(target_id_list)
        timeinfo1 = slewtime * u.minute + inte_time * u.minute
        geneinfo.append(src_name)
        move_time = inte_time
        time_range_src0 = np.linspace(0, move_time, num = int(move_time) * 1 + 1)
        src0_gain_time = utc_obs_start + slewtime * u.minute + time_range_src0 * u.minute
        az,el = src_az_el(src,src0_gain_time)
        src0_gain = sum(gain(el))
        src0_target_gain = target_gain_dict[src_name[0]]

        src0_score = src0_target_gain/src0_gain*timeinfo1

        utc_obs_start = Time(record_scan_end_time)

        for x in range (0,52):
            
            target_variable_list = []
            utc_obs_start = Time(record_scan_end_time)
            for y in range (0,52):
                try:
                    target_calibration_list.remove(src_name[0])

                except ValueError:
                    pass
                
                #遍历剩余源
                src_name_y = target_calibration_list[y]
                utc_obs_start_y = utc_obs_start
                inte_time2 = targets_dict[src_name_y].int_time
                if len(target_available_times[src_name_y]) == 1:
                    if (target_available_times[src_name_y][0][0] > utc_obs_start_y and target_available_times[src_name_y][0][1] < utc_obs_start_y+inte_time2+slew_time):
                        target_variable_list.append(target_calibration_list[y])
                if len(target_available_times[src_name_y]) == 2:
                    if  (target_available_times[src_name_y][0][0] > utc_obs_start_y and target_available_times[src_name_y][0][1] < utc_obs_start_y+inte_time2+slew_time) or (target_available_times[src_name_y][1][0] > utc_obs_start_y and target_available_times[src_name_y][1][1] < utc_obs_start_y+inte_time+slew_time):
                        target_variable_list.append(target_calibration_list[y])
                if len(target_available_times[src_name_y]) == 3:
                    if (target_available_times[src_name_y][0][0] > utc_obs_start_y and target_available_times[src_name_y][0][1] < utc_obs_start_y+inte_time2+slew_time) or (target_available_times[src_name_y][1][0] > utc_obs_start_y and target_available_times[src_name_y][1][1] < utc_obs_start_y+inte_time+slew_time) or (target_available_times[src_name_y][2][0] > utc_obs_start_y and target_available_times[src_name_y][2][1] < utc_obs_start_y+inte_time+slew_time):
                        target_variable_list.append(target_calibration_list[y])

            try:                
                
                src_name_random = random.sample(target_variable_list,1)
                geneinfo.append(src_name_random)
                src_1 = targets_dict[src_name_random[0]].ra + ' ' + targets_dict[src_name_random[0]].dec
                src1 = SkyCoord(src_1, frame = 'icrs', unit = (u.hourangle, u.deg))
                src_2 = targets_dict[geneinfo[x][0]].ra + ' ' + targets_dict[geneinfo[x][0]].dec
                src2 = SkyCoord(src_2, frame = 'icrs', unit = (u.hourangle, u.deg))
           
                slewtime1 = float(transition1(src1, src2, utc_obs_start_y, YNAO)) / 60.
                timeinfo = slewtime1 * u.minute + inte_time2 * u.minute        
                move_time1 = inte_time2 
                time_range_src1 = np.linspace(0, move_time1, num = int(move_time1) * 1 + 1)
                src1_gain_time = utc_obs_start + slewtime1 * u.minute + time_range_src1 * u.minute
                az,el = src_az_el(src1,src1_gain_time)
                src_i_gain = sum(gain(el))
                src_i_target_gain = target_gain_dict[src_name_random[0]]

                src_i_score = src_i_target_gain/src_i_gain*timeinfo
                record_scan_start_time = utc_obs_start_y + slewtime1 * u.minute
                record_scan_end_time = record_scan_start_time + inte_time * u.minute

                utc_obs_start = Time(record_scan_end_time)
                    
                score += src_i_score
            except ValueError:
                pass

        score = score+src0_score

        return geneinfo,score

	
    def selectBest(self, pop):
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=False)          
        return s_inds[0]

    
    #按照一定的概率选择个体
    def selection(self, individuals, k):
        s_inds = sorted(individuals, key=itemgetter("fitness"),reverse=False)  
        sum_fits = sum(ind['fitness'] for ind in individuals)  # sum up the fitness of the whole pop
        chosen = []
        for i in range(k):
            u = random.random() * sum_fits  # randomly produce a num in the range of [0, sum_fits], as threshold
            sum_ = 0
            for ind in s_inds:
                sum_ += ind['fitness']  # sum up the fitness
                if sum_ >= u:
                    chosen.append(ind)
                    break
        chosen = sorted(chosen, key=itemgetter("fitness"), reverse=False)
        return chosen
    
    #基因交叉 
    def crossoperate(self,offspring):
        dim = min(len(offspring[0]['Gene'].data),len(offspring[1]['Gene'].data))
        geninfo1 = offspring[0]['Gene'].data  # Gene's data of first offspring chosen from the selected pop
        geninfo2 = offspring[1]['Gene'].data  # Gene's data of second offspring chosen from the selected pop
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
            if min(pos1, pos2) <= i < max(pos1, pos2):
                temp2.append(geninfo2[i])
                temp1.append(geninfo1[i])
            else:
                temp2.append(geninfo1[i])
                temp1.append(geninfo2[i])
        newoff1.data = temp1
        newoff2.data = temp2
    
        return newoff1, newoff2
        
    def get_time(self,gene_list):
        time_all = 0
        for i in range(len(gene_list)):
            src_name_time = gene_list[i]

            inte_time = targets_dict[src_name_time[0]].int_time
            
            if i == 0:
                src0 = targets_dict[src_name_time[0]].ra + '' + targets_dict[src_name_time[0]].dec
                src = SkyCoord(src0, frame = 'icrs', unit = (u.hourangle, u.deg))
                utc_obs_start = time_range[0]
                slewtime = float(transition2(float(az0), float(el0), src, utc_obs_start, YNAO)) / 60.
                record_scan_start_time = utc_obs_start + slewtime * u.minute
                record_scan_end_time = record_scan_start_time + inte_time * u.minute
                #为下一个循环开始设置时间
                utc_obs_start = Time(record_scan_end_time)
                timeinfo1 = slewtime * u.minute + inte_time * u.minute
            else:
                utc_obs_start_y = utc_obs_start
                src_1 = targets_dict[src_name_time[0]].ra + ' ' + targets_dict[src_name_time[0]].dec
                src1 = SkyCoord(src_1, frame = 'icrs', unit = (u.hourangle, u.deg))
                src_2_name = gene_list[i-1]
                src_2 = targets_dict[src_2_name[0]].ra + ' ' + targets_dict[src_2_name[0]].dec
                src2 = SkyCoord(src_2, frame = 'icrs', unit = (u.hourangle, u.deg))
                                
                                
                slewtime = float(transition1(src1, src2, utc_obs_start_y, YNAO)) / 60.
                record_scan_start_time = utc_obs_start_y + slewtime * u.minute
                record_scan_end_time = record_scan_start_time + inte_time * u.minute
                utc_obs_start = Time(record_scan_end_time)
    
                timeinfo = slewtime * u.minute + inte_time * u.minute
                time_all += timeinfo
        time_sum = timeinfo1+time_all

        return time_sum

    def mutation(self,offspring):
        dim = len(offspring.data)
        pos = random.randrange(1, dim)
        targetuse = offspring.data[:pos]
        for i in range(len(targetuse)):
            if i == 0 :
                targetuse1 = targetuse[i]
                src0 = targets_dict[targetuse1[0]].ra + '' + targets_dict[targetuse1[0]].dec
                src = SkyCoord(src0, frame = 'icrs', unit = (u.hourangle, u.deg))
                utc_obs_start = time_range[0]
                slewtime = float(transition2(float(az0), float(el0), src, utc_obs_start, YNAO)) / 60.
                record_scan_start_time = utc_obs_start + slewtime * u.minute
                record_scan_end_time = record_scan_start_time + inte_time * u.minute
                #为下一个循环开始设置时间
                utc_obs_start = Time(record_scan_end_time)
            else:
                targetuse1 = targetuse[i]
                utc_obs_start_y = utc_obs_start
                src_1 = targets_dict[targetuse1[0]].ra + ' ' + targets_dict[targetuse1[0]].dec
                src1 = SkyCoord(src_1, frame = 'icrs', unit = (u.hourangle, u.deg))
                targetuse2 = targetuse[i-1]
                src_2 = targets_dict[targetuse2[0]].ra + ' ' + targets_dict[targetuse2[0]].dec
                src2 = SkyCoord(src_2, frame = 'icrs', unit = (u.hourangle, u.deg))
                                
                                
                slewtime = float(transition1(src1, src2, utc_obs_start_y, YNAO)) / 60.
                record_scan_start_time = utc_obs_start_y + slewtime * u.minute
                record_scan_end_time = record_scan_start_time + inte_time * u.minute
                utc_obs_start = Time(record_scan_end_time)

        
        targetcopy = copy.deepcopy(target_id_list)
        targetcopy.remove(offspring.data[pos][0])
        for i in range (0,52):
            targetlist = []
            src_name_i = targetcopy[i]
            inte_time2 = targets_dict[src_name_i].int_time
            utc_obs_start_1 = utc_obs_start
            if len(target_available_times[src_name_i]) == 1:
                    if (target_available_times[src_name_i][0][0] > utc_obs_start_1 and target_available_times[src_name_i][0][1] < utc_obs_start_1+inte_time2+slew_time):
                        targetlist.append(targetcopy[i])
            if len(target_available_times[src_name_i]) == 2:
                    if  (target_available_times[src_name_i][0][0] > utc_obs_start_1 and target_available_times[src_name_i][0][1] < utc_obs_start_1+inte_time2+slew_time) or (target_available_times[src_name_i][1][0] > utc_obs_start_1 and target_available_times[src_name_i][1][1] < utc_obs_start_1+inte_time+slew_time):
                        targetlist.append(targetcopy[i])
            if len(target_available_times[src_name_i]) == 3:
                    if (target_available_times[src_name_i][0][0] > utc_obs_start_1 and target_available_times[src_name_i][0][1] < utc_obs_start_1+inte_time2+slew_time) or (target_available_times[src_name_i][1][0] > utc_obs_start_1 and target_available_times[src_name_i][1][1] < utc_obs_start_1+inte_time+slew_time) or (target_available_times[src_name_i][2][0] > utc_obs_start_1 and target_available_times[src_name_i][2][1] < utc_obs_start_1+inte_time+slew_time):
                        targetlist.append(targetcopy[i])
                        
        
        try:
            offspring.data[pos] = random.sample(targetlist,1)
        except ValueError:
            pass
        
            
        return offspring
                
          
    def GA_main(self):
        popsize = self.parameter[2]
 
        print("Start of evolution")
 
        # Begin the evolution
        for g in range(NGEN):
 
            print("############### Generation {} ###############".format(g))
 
            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, popsize)
            
            nextoff = []
            while len(nextoff) != popsize:
                # Apply crossover and mutation on the offspring
 
                # Select two individuals
                
                offspring = [selectpop.pop() for _ in range(2)]
                
                if random.random() < CXPB:  # cross two individuals with probability CXPB
                    crossoff1,crossoff2 = self.crossoperate(offspring)
                    if random.random()< MUTPB:
                    
                        muteoff1 = self.mutation(crossoff1)
                        muteoff2 = self.mutation(crossoff2)
                        fit_muteoff1 = self.get_time(muteoff1.data)  # Evaluate the individual
                        fit_muteoff2 = self.get_time(muteoff2.data)
                     
                        nextoff.append({'Gene': muteoff1, 'fitness': fit_muteoff1})
                        nextoff.append({'Gene': muteoff2, 'fitness': fit_muteoff2})
                    else:
                        
                        fit_crossoff1 = self.get_time(crossoff1.data)  # Evaluate the individuals
                        fit_crossoff2 = self.get_time(crossoff2.data)
                        nextoff.append({'Gene': crossoff1, 'fitness': fit_crossoff1})
                        nextoff.append({'Gene': crossoff2, 'fitness': fit_crossoff2})
                else:
                    
                    nextoff.extend(offspring)
                

            # The population is entirely replaced by the offspring
            self.pop = nextoff
 
            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]
            best_ind = self.selectBest(self.pop)
 
            if best_ind['fitness'] < self.bestindividual['fitness']:
                self.bestindividual = best_ind
 
            print("Best individual found is {}, {}".format(self.bestindividual['Gene'].data,
                                                           self.bestindividual['fitness']))
            print("  Max fitness of current pop: {}".format(min(fits)))
 
        print("------ End of (successful) evolution ------")


# %%
if __name__ == "__main__":
    CXPB,NGEN,popsize,MUTPB = 0.7,100,40,0.2
    parameter = [CXPB, NGEN, popsize, MUTPB]
    run = GA(parameter)
    run.GA_main()

# %%
geneinfo = []
target_plan_list = []
score = 0      
for i in range (0,53):
    if target_available_times[target_id_list[i]][0][0] == time_range[0]:
        target_plan_list.append(target_id_list[i])        
    #目前遍历所有可以作为种群第一个元素的源id
    

src_name = random.sample(target_plan_list,1)
inte_time = targets_dict[src_name[0]].int_time    
src0 = targets_dict[src_name[0]].ra + '' + targets_dict[src_name[0]].dec
src = SkyCoord(src0, frame = 'icrs', unit = (u.hourangle, u.deg))
utc_obs_start = time_range[0]
slewtime = float(transition2(float(az0), float(el0), src, utc_obs_start, YNAO)) / 60.
record_scan_start_time = utc_obs_start + slewtime * u.minute
record_scan_end_time = record_scan_start_time + inte_time * u.minute

#为下一个循环开始设置时间
target_calibration_list = copy.deepcopy(target_id_list)
timeinfo1 = slewtime * u.minute + inte_time * u.minute
geneinfo.append(src_name)
move_time = inte_time +slew_time
time_range_src0 = np.linspace(0, move_time, num = int(move_time) * 1 + 1)
src0_gain_time = utc_obs_start + slewtime * u.minute + time_range_src0 * u.minute
az,el = src_az_el(src,src0_gain_time)
src0_gain = sum(gain(el))
src0_target_gain = target_gain_dict[src_name[0]]

src0_score = src0_target_gain/src0_gain*timeinfo1

utc_obs_start = Time(record_scan_end_time)
print (geneinfo,src0_score)

for x in range (0,52):
            
    target_variable_list = []
    utc_obs_start = Time(record_scan_end_time)
    for y in range (0,52):

        try:
            target_calibration_list.remove(src_name[0])

        except ValueError:
            pass
                    
        #遍历剩余源
        src_name_y = target_calibration_list[y]
        utc_obs_start_y = utc_obs_start
        inte_time2 = targets_dict[src_name_y].int_time
        if len(target_available_times[src_name_y]) == 1:
            if (target_available_times[src_name_y][0][0] > utc_obs_start_y and target_available_times[src_name_y][0][1] < utc_obs_start_y+inte_time2+slew_time):
                target_variable_list.append(target_calibration_list[y])
        if len(target_available_times[src_name_y]) == 2:
            if  (target_available_times[src_name_y][0][0] > utc_obs_start_y and target_available_times[src_name_y][0][1] < utc_obs_start_y+inte_time2+slew_time) or (target_available_times[src_name_y][1][0] > utc_obs_start_y and target_available_times[src_name_y][1][1] < utc_obs_start_y+inte_time+slew_time):
                target_variable_list.append(target_calibration_list[y])
        if len(target_available_times[src_name_y]) == 3:
            if (target_available_times[src_name_y][0][0] > utc_obs_start_y and target_available_times[src_name_y][0][1] < utc_obs_start_y+inte_time2+slew_time) or (target_available_times[src_name_y][1][0] > utc_obs_start_y and target_available_times[src_name_y][1][1] < utc_obs_start_y+inte_time+slew_time) or (target_available_times[src_name_y][2][0] > utc_obs_start_y and target_available_times[src_name_y][2][1] < utc_obs_start_y+inte_time+slew_time):
                target_variable_list.append(target_calibration_list[y])

        try:                
                
                src_name_random = random.sample(target_variable_list,1)
                geneinfo.append(src_name_random)
                src_1 = targets_dict[src_name_random[0]].ra + ' ' + targets_dict[src_name_random[0]].dec
                src1 = SkyCoord(src_1, frame = 'icrs', unit = (u.hourangle, u.deg))
                src_2 = targets_dict[geneinfo[x][0]].ra + ' ' + targets_dict[geneinfo[x][0]].dec
                src2 = SkyCoord(src_2, frame = 'icrs', unit = (u.hourangle, u.deg))
                timeinfo = slewtime * u.minute + inte_time2 * u.minute
                slewtime = float(transition1(src1, src2, utc_obs_start_y, YNAO)) / 60.
                move_time1 = inte_time2
                time_range_src1 = np.linspace(0, move_time1, num = int(move_time1) * 1 + 1)
                src1_gain_time = utc_obs_start + slewtime * u.minute + time_range_src1 * u.minute
                az,el = src_az_el(src1,src1_gain_time)
                src_i_gain = sum(gain(el))
                src_i_target_gain = target_gain_dict[src_name_random[0]]
                src_i_score = src_i_target_gain/src_i_gain*timeinfo
                record_scan_start_time = utc_obs_start_y + slewtime * u.minute
                record_scan_end_time = record_scan_start_time + inte_time * u.minute
                utc_obs_start = Time(record_scan_end_time)                       
                score += src_i_score
        except ValueError:
            pass

score = score+src0_score
print (geneinfo,score)

# %%



