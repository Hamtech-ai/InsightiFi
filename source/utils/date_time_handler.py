import datetime as datetime
import jdatetime as jdatetime
from datetime import timedelta
import math as math

def timedelta_length_round(delta_t, divisor='day', round_type='ceil'):
    if divisor == 'second':
        divisor_period = datetime.timedelta(seconds=1)
    elif divisor == 'minute':
        divisor_period = datetime.timedelta(minutes=1)
    elif divisor == 'hour':
        divisor_period = datetime.timedelta(hours=1)
    elif divisor == 'week':
        divisor_period = datetime.timedelta(weeks=1)
    else: # day is default
        divisor_period = datetime.timedelta(days=1)

    time_proportion = delta_t.total_seconds() / divisor_period.total_seconds()

    # output return
    if round_type == 'floor':
        return math.ceil(time_proportion)
    else: # ceil is default
        return math.ceil(time_proportion)

def now_datetime():
    return datetime.datetime.now()

def today_date():
    return datetime.date.today()

def zero_date():
    return datetime.date(1,1,1)
    
def days_until_now(eval_date):
    now = datetime.date.today()
    delta_datetime = now - eval_date
    
    return timedelta_length_round(delta_datetime, divisor='day', round_type='ceil')

def string_to_datetime(eval_datetime_string, format='%y-%m-%d %h:%M:%s'):
    return datetime.datetime.strptime(eval_datetime_string, format)

def jdatetime_creator(year=1380, month=1, day=1, hour=0, minute=0, second=0):
    return jdatetime.datetime(year=year, month=month, day=day, hour=hour, minute=minute, second=second, locale='fa_IR')

def jdatetime_to_jdate(eval_jdatetime):
    return eval_jdatetime.date()

def jdate_to_georgian(eval_jdate):
    return eval_jdate.togregorian()

def datetime_to_date(eval_datetime):
    return eval_datetime.date()

def day_delta_time(num):
    return timedelta(days=num)