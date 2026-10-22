from datetime import datetime

import pandas as pd


def parse_coordinate(coordinate, type):
    """
    解析坐标字符串
    Args:
        coordinate: 字符串格式，同原子文件的格式
        type: 类型 Point/LineString

    Returns:

    """
    if type == 'LineString':
        coordinate = coordinate.replace('[', '')
        coordinate = coordinate.replace(']', '').split(',')
        lon1 = float(coordinate[0])
        lat1 = float(coordinate[1])
        lon2 = float(coordinate[2])
        lat2 = float(coordinate[3])
        return lon1, lat1, lon2, lat2
    elif type == 'Point':
        coordinate = coordinate.replace('[', '')
        coordinate = coordinate.replace(']', '').split(',')
        lon = float(coordinate[0])
        lat = float(coordinate[1])
        return lon, lat

def encode_time(timestamp: str) -> int:
    """
    Encode a timestamp into the model's minute-level time slot.

    Supports ISO timestamps with or without milliseconds/timezone suffix.
    Weekdays use [0, 1439], weekends use [1440, 2879].

    Parameters
    ----------
    timestamp : str

    Returns
    -------
    int
        Encoded time slot.
    """
    time = pd.Timestamp(timestamp).to_pydatetime()

    if time.weekday() in (5, 6):
        return time.hour * 60 + time.minute + 1440

    return time.hour * 60 + time.minute

# def encode_time(timestamp):
#     """
#     将字符串格式的时间戳编码
#     """
#     # 按一分钟编码，周末与工作日区分开来
#     time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
#     if time.weekday() == 5 or time.weekday() == 6:
#         return time.hour * 60 + time.minute + 1440
#     else:
#         return time.hour * 60 + time.minute