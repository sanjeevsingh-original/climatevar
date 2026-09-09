import numpy as np
import pandas as pd
import xarray as xr
from climatevar.time import calendar_name, drop_leap_day, expected_days_in_year, infer_frequency, season_year, time_step_seconds, year_complete_mask
from climatevar.precipitation import daily_amount

def test_calendar_year_lengths():
    assert expected_days_in_year(2000,"standard")==366; assert expected_days_in_year(2001,"standard")==365; assert expected_days_in_year(2000,"noleap")==365; assert expected_days_in_year(2000,"360_day")==360; assert expected_days_in_year(2000,"all_leap")==366

def test_gregorian_leap_year_completeness():
    time=pd.date_range("2000-01-01","2000-12-31",freq="D"); data=xr.DataArray(np.ones(time.size),coords={"time":time},dims="time"); assert bool(year_complete_mask(data).item()); assert not bool(year_complete_mask(data.isel(time=slice(0,300))).item())

def test_noleap_completeness_uses_365_days():
    time=pd.date_range("2001-01-01","2001-12-31",freq="D"); data=xr.DataArray(np.ones(time.size),coords={"time":time},dims="time"); data["time"].attrs["calendar"]="noleap"; assert calendar_name(data)=="noleap"; assert bool(year_complete_mask(data).item())

def test_djf_assigns_december_to_following_year():
    time=pd.date_range("2000-12-30","2001-01-02",freq="D"); data=xr.DataArray(np.ones(time.size),coords={"time":time},dims="time"); assert season_year(data).values.tolist()==[2001,2001,2001,2001]

def test_drop_leap_day():
    time=pd.date_range("2000-02-27","2000-03-01",freq="D"); data=xr.DataArray(np.arange(time.size),coords={"time":time},dims="time"); assert drop_leap_day(data).time.dt.day.values.tolist()==[27,28,1]

def test_time_bounds_are_preferred():
    time=pd.date_range("2000-01-01",periods=3,freq="3h"); bounds=xr.DataArray(np.array([[np.datetime64("1999-12-31T22:30"),np.datetime64("2000-01-01T01:30")],[np.datetime64("2000-01-01T01:30"),np.datetime64("2000-01-01T04:30")],[np.datetime64("2000-01-01T04:30"),np.datetime64("2000-01-01T07:30")]]),coords={"time":time,"bnds":[0,1]},dims=("time","bnds"),name="time_bnds"); ds=xr.DataArray(np.ones(3),coords={"time":time},dims="time").to_dataset(name="pr").assign(time_bnds=bounds); assert np.allclose(time_step_seconds(ds).values,10800.0); assert infer_frequency(ds)=="3-hourly"

def test_cf_time_bounds_drive_rate_integration():
    time=pd.to_datetime(["2000-01-01T00:00","2000-01-01T01:00"]); bounds=xr.DataArray(np.array([[np.datetime64("1999-12-31T23:00"),np.datetime64("2000-01-01T01:00")],[np.datetime64("2000-01-01T01:00"),np.datetime64("2000-01-01T02:00")]]),coords={"time":time,"bnds":[0,1]},dims=("time","bnds"),name="time_bnds"); data=xr.DataArray([1.,1.],coords={"time":time},dims="time",attrs={"units":"mm/hr"}); data=data.assign_coords(time_bnds=("time",np.array([np.datetime64("2000-01-01T00:00"),np.datetime64("2000-01-01T01:00")]))); ds=data.to_dataset(name="pr").assign(time_bnds=bounds); result=daily_amount(ds["pr"]); assert np.isclose(result.isel(time=0).item(),2.0)
