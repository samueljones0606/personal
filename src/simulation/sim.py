import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pricing.black_scholes as bs
from pandas.tseries.offsets import BDay


DAYS_PER_YEAR = 365

def simulate_spot(underlyings, start_date, end_date, vol, drift, simulate_divs = True):
    sim_dates = pd.date_range(start_date, end_date, freq = "B")
    spot = 1 + pd.Series(index=underlyings, data=-np.log(np.random.uniform(size=len(underlyings))))*0.5
    time_axis = pd.Series(index=sim_dates, data=(sim_dates - sim_dates[0]).days /DAYS_PER_YEAR).rename("t")
    rnd = pd.DataFrame(data = np.random.normal(size = (len(sim_dates),len(underlyings))), index = sim_dates, columns=underlyings)
    paths = rnd.mul(np.sqrt(pd.Series(rnd.index, index=rnd.index).diff().dt.days / DAYS_PER_YEAR), axis=0).fillna(0.0).cumsum()
    paths = np.exp((paths * vol).add(time_axis*(drift-vol**2), axis=0)).mul(spot, axis=1)
    
    # TODO a better way of simulating div dates....
    divs = pd.concat((
        pd.Series(index=pd.date_range(sim_dates[0],sim_dates[-1], freq="6M")+pd.DateOffset(days=5)+0*BDay(),data=0.02, name=underlyings[0]),
        pd.Series(index=pd.date_range(sim_dates[0],sim_dates[-1], freq="12M")+pd.DateOffset(days=27)+0*BDay(),data=0.04, name = underlyings[1]),
        pd.Series(index=pd.date_range(sim_dates[0],sim_dates[-1], freq="18M")+pd.DateOffset(days=64)+0*BDay(),data=0.05, name = underlyings[2])),
        axis=1).reindex(sim_dates).fillna(0.0)

    paths = (paths - divs.cumsum())
    return paths, divs

def simulate_simple_spot(underlyings, start_date, end_date, vol, drift, fix_spot = True):
    sim_dates = pd.date_range(start_date, end_date, freq = "B")
    if fix_spot:
        spot = 1 + pd.Series(index=underlyings, data=list(-np.log(np.random.uniform(size=1)))*len(underlyings))*0.5
    else:
        spot = 1 + pd.Series(index=underlyings, data=-np.log(np.random.uniform(size=len(underlyings))))*0.5
    time_axis = pd.Series(index=sim_dates, data=(sim_dates - sim_dates[0]).days /DAYS_PER_YEAR).rename("t")
    rnd = pd.DataFrame(data = np.random.normal(size = (len(sim_dates),len(underlyings))), index = sim_dates, columns=underlyings)
    paths = rnd.mul(np.sqrt(pd.Series(rnd.index, index=rnd.index).diff().dt.days / DAYS_PER_YEAR), axis=0).fillna(0.0).cumsum()
    paths = np.exp((paths * vol).add(time_axis*(drift-vol**2), axis=0)).mul(spot, axis=1)
    return paths

def generate_option_data(sim_dates, underlyings):
    
    # Assume optinos are issued at 10c intervals at 6m, 10c intervals at 3m, and 5c intervals at 1m.
    strikes_6m = np.arange(0,3.01,0.2)[1:]
    strikes_3m = np.arange(0,3.01,0.1)[1:]
    strikes_1m = np.arange(0,3.01,0.05)[1:]
    
    # then at any point in time the strike ranges for the next 6 expiry points are:
    strikes_all = (strikes_1m, strikes_3m, strikes_3m, strikes_6m, strikes_6m, strikes_6m)

    denorm = pd.DataFrame([
        (t, T, K, und, CP)
        for t in sim_dates
        for T, K_range in zip(pd.date_range(t, freq="M",periods=6), strikes_all)
        for K in K_range   
        for und in underlyings
        for CP in "CP"
    ],
    columns = ["valuation_date","expiry_date","strike","underlying","payoff"])
    
    #augment with an id and a time to expiry
    denorm["opt_id"] = (
        denorm["underlying"] + "_" +
        denorm["expiry_date"].apply(lambda x: x.strftime("%Y%m%d")) + "_" +
        (denorm["strike"]*100).astype(int).apply(lambda x: "{:04d}".format(x)) + "_" +
        denorm["payoff"]
    )

    denorm["expiry_time"]=(denorm["expiry_date"]-denorm["valuation_date"]).dt.days.astype(float)/DAYS_PER_YEAR
    
    return denorm

def price_options(denorm, paths, implied_vol):
    # join on spot price
    stacked_paths = paths.stack().rename("spot").to_frame()
    stacked_paths.index.names = ["valuation_date","underlying"]
    denorm = denorm.join(stacked_paths, on=["valuation_date","underlying"])
    denorm["implied_vol"] = implied_vol
    # pricing and greek for all options
    denorm["price"] = bs.option_value(denorm["strike"], denorm["spot"], denorm["implied_vol"], denorm["expiry_time"], denorm["payoff"]=="C")
    denorm["delta"] = bs.delta(denorm["strike"], denorm["spot"], denorm["implied_vol"], denorm["expiry_time"], denorm["payoff"]=="C")
    denorm["gamma"] = bs.gamma(denorm["strike"], denorm["spot"], denorm["implied_vol"], denorm["expiry_time"])
    denorm["vega"] = bs.vega(denorm["strike"], denorm["spot"], denorm["implied_vol"], denorm["expiry_time"])
    denorm["theta"] = bs.theta(denorm["strike"], denorm["spot"], denorm["implied_vol"], denorm["expiry_time"])
    return denorm

def vehicle_selection(denorm, target_expiry):
    
    def _vehicle_selection(df):
        # what's the nearest expiry to our target?
        times = df.groupby(["expiry_date", "expiry_time"]).size().reset_index()
        filtered = df[df["expiry_date"]==times.loc[(times["expiry_time"] - target_expiry).abs().idxmin(),"expiry_date"]]
        # and the nearest strikes to atm
        filtered = filtered[filtered["strike"]==filtered.loc[(filtered["strike"]-filtered["spot"]).abs().idxmin(),"strike"]]
        assert len(filtered)==2
        filtered = filtered["opt_id"].to_frame()
        filtered["weight"]=1.
        return filtered

    traded_options = denorm.groupby(["underlying","valuation_date"]).apply(_vehicle_selection)
    traded_options.index = traded_options.index.droplevel(-1)
    
    traded_options = traded_options.reset_index()
    traded_options = traded_options.set_index(traded_options["valuation_date"].dt.strftime("%Y%m%d")+"_"+traded_options["underlying"]+"_1m_ATM_straddle")
    traded_options.index.name = "vehicle_id"
    
    return traded_options

def filter_denorm(vehicles, denorm):

        # we care only about the option price from the date that it is first traded
        first_trade_date = vehicles.reset_index().groupby("opt_id")["valuation_date"].min().to_dict()
        
        # filter the denorm table to only the options that we need, only on the dates that we need them
        def filterer(df):
            df = df[df["valuation_date"]>=first_trade_date[df["opt_id"].values[0]]] # we only care about the option from the first date that it is traded
            df = df.sort_values("valuation_date")
            df["opt_pnl"] = df["price"].diff() # option pnl *to* the valuation date (zero on the first listing day, nonzero on expiry day)
            df["hedge_pnl"] = df["spot"].diff()*df["delta"].shift() # hedge pnl *to* the valuation date (assuming we hedged to the previous day's delta)
            return df
        filtered_denorm = denorm.groupby("opt_id").filter(lambda df :  df["opt_id"].values[0] in first_trade_date).groupby("opt_id").apply(filterer)
        return filtered_denorm
        
def build_vehicle_frame(sim_dates, vehicles, denorm, field):
    '''denorm contains only the relevant option prices on the relevant days for this vehicle set'''
    '''field is any linearly additive field that is present in the denorm, i.e.  price or any of its derivatives (greeks) that can therefore be used as risk measure'''
        
    # reshape from a (date, asset)-indexed denorm to a date x asset prices table of individual options
    opt_price_paths = denorm[["valuation_date","opt_id",field]].set_index(["valuation_date","opt_id"]).unstack(level="opt_id").reindex(sim_dates)
    opt_price_paths.columns = opt_price_paths.columns.droplevel()
        
    # group the individual options into the vehicles that we can trade (#TODO we will need an "ungrouped" version of this in due course, an "investable" is an
    # atomic subportfolio of asses with all associated flows & events, e.g. early exercise, coupons & divs, corp acts etc. Here we have the total "investible" path
    # we will eventually want it by asset)
    vehicle_paths = (opt_price_paths[vehicles["opt_id"]].mul(vehicles["weight"].values, axis=1))
    vehicle_paths = vehicle_paths.where(pd.Series(vehicle_paths.index).values.reshape(-1,1) >= vehicles["valuation_date"].values.reshape(1,-1))
    vehicle_paths = vehicle_paths.groupby(vehicles.index, axis=1).sum().where(~vehicle_paths.isnull().groupby(vehicles.index, axis=1).all())
        
    return vehicle_paths