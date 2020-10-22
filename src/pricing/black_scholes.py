import functools
import itertools
import numpy
from scipy.stats import norm

STR_NEGATIVE_STD_ERR = "Standard deviation must be non-negative"
STR_IGNORE = 'ignore'

import numpy
from scipy.stats import norm

def normalised_implied_volatility(x, v):
     
    shape = (x*v).shape
    
    converged = numpy.full(shape, False, dtype='bool')
    results = numpy.full(shape, numpy.nan)
    
    i = 0
    
    mask = 0.0 == v
    converged[mask] = True
    results[mask] = 0.0
    
    if converged.all():
        return results
        
    mask = numpy.logical_and(numpy.logical_not(converged), (1.0 == v))
    converged[mask] = True
    results[mask] = numpy.inf

    if converged.all():
        return results
    
    # only defined for +ve x (i.e. we should only be looking at OTM options)
    mask = x < 0.0
    converged[mask] = True
    results[mask] = numpy.nan

    # should have nothing below zero or above 1
    # options worth less than intrinsic or more that the underlying or strike
    mask = numpy.logical_or(v < 0.0, v > 1.0)
    converged[mask] = True
    results[mask] = numpy.nan
        
    y = numpy.where(converged, numpy.nan, 2.0 * norm.ppf((1.0 + v) / 2.0))
    
    mask = numpy.logical_and(numpy.logical_not(converged), y == numpy.inf)
    converged[mask] = True
    results[mask] = y[mask]
    if converged.all():
        return results
        
    mask = numpy.logical_and(numpy.logical_not(converged), 0 == x)
    converged[mask] = True
    results[mask] = y[mask]
    if converged.all():
        return results
         
    fOld = numpy.full(shape, numpy.inf)
    yOld = numpy.full(shape, numpy.inf)
    ex = numpy.where(converged, numpy.nan, numpy.exp(x))
    
    # testing over a range of values (different orders of magnitude over x and v)
    # a best worst case scenario seems to be two iterations of this fixed point approach
    # before reverting to newton/halley
    # (also, the "best worst-case" doesnt' much impact the 'iterations of the usual cases)
    for k in range(2):
        vEst = numpy.where(converged, numpy.nan, v + ex * norm.pdf(-x / y - y / 2.0))
        vEst[vEst >= 1.0] = numpy.broadcast_to(numpy.maximum(v,0.99), vEst.shape)[vEst >= 1.0]
        tmp = numpy.where(converged, numpy.nan, norm.ppf(vEst))
        tmp += numpy.sqrt(tmp * tmp + 2 * x)
        y[tmp != 0] = tmp[tmp != 0]
        ++i
    
    converged[0 == y] = True
    results[0 == y] = 0
    if converged.all():
        return results
         
    while i!=1000:
 
        if (y < 0.0).any():
            raise RuntimeError("oh dear, y should never go negative...")
        d1 = numpy.where(converged, numpy.nan, -x / y)
        y2 = numpy.where(converged, numpy.nan, y / 2.0)
        d2 = numpy.where(converged, numpy.nan, d1 - y2)
        d1 += y2
 
        p1 = numpy.where(converged, numpy.nan, norm.pdf(d1))
        p2 = numpy.where(converged, numpy.nan, norm.pdf(d2))
        r1 = numpy.where(converged, numpy.nan, norm.cdf(d1)/norm.pdf(d1)) # TODO maybe redundant: cdf is often implemented in terms of this ratio...
        r2 = numpy.where(converged, numpy.nan, norm.cdf(d2)/norm.pdf(d2))
        
        # if we're too far out & cdfToPdfRatio fails, rely on the fact that our guess is asymptotically correct
        mask = numpy.logical_and(numpy.logical_and(numpy.logical_not(converged), d2 < -30) ,numpy.isnan(r2))
        converged[mask] = True
        results[mask] = y[mask]
        if converged.all():
            return results 
 
        f = numpy.where(converged, numpy.nan, r1 * p1 - ex * r2 * p2 - v)
        mask = 0==f
        converged[mask] = True
        results[mask] = y[mask]
        if converged.all():
            return results
            
        mask = abs(fOld) <= abs(f)
        converged[mask] = True
        results[mask] = yOld[mask]
        if converged.all():
            return results
            
        yOld = numpy.where(converged, numpy.nan, y)
        fOld = numpy.where(converged, numpy.nan, f)
 
        dy = numpy.where(converged, numpy.nan, r1 - r2 - v / p1) # this would be a basic newton step
        dy = numpy.where(converged, numpy.nan, dy/(1.0 - dy * d1 * d2 / y)) # but we adjust, halley's method
        y = numpy.where(converged, numpy.nan, y -dy)
        i = i+1
 
    raise RuntimeError("Implied volatility failed to converge for inputs x = {}, v = {}".format(x,v))
    
def normalised_price(x, y):
    
    return numpy.where(numpy.logical_and(x <= 0.0, numpy.isinf(x)),1.0, # zero-strike call is worth 1 forward
                numpy.where(numpy.logical_and(x > 0.0, numpy.isinf(x)),0.0, # infinity-strike call is worthless
                                   numpy.where(y==0.0,numpy.maximum(1-numpy.exp(x), 0.0), # zero time call is intrinsic
                                               norm.cdf(-x/y+y/2)-numpy.exp(x)*norm.cdf(-x/y-y/2))))

def _d_normalised_price_dx(x, y):
    ''' First derivative of normalised_price formula wrt logstrike argument x '''
    x = numpy.asarray(x)
    y = numpy.asarray(y)

    if numpy.any(y < 0.0):
        raise ValueError(STR_NEGATIVE_STD_ERR)
    d2_value = -x / y - y / 2.0
    k = numpy.exp(x)

    # special cases of huge strikes or expiring ATM, the formula fails but the
    # limit is 0. deal with these edge cases...
    ret = -numpy.exp(x) * norm.cdf(d2_value)
    ret = numpy.where(numpy.logical_or(numpy.isinf(k), numpy.logical_and(x == 0, y == 0)), 0.0, ret)

    # annoyingly numpy.where always returns arrays even with scalar inputs
    if type(x) is numpy.float64 and type(y) is numpy.float64 and type(ret) is numpy.ndarray:
        assert len(ret.flatten() == 1)
        ret = ret.flatten()[0]

    assert not numpy.any(numpy.isnan(ret))
    return ret

def _d_normalised_price_dy(x, y):
    ''' First derivative of normalised_price formula wrt stdev argument y '''
    x = numpy.asarray(x)
    y = numpy.asarray(y)

    if numpy.any(y < 0.0):
        raise ValueError(STR_NEGATIVE_STD_ERR)
    d1_value = -x / y + y / 2.0
    ret = norm.pdf(d1_value)

    ret = numpy.where(numpy.logical_and(x == 0, y == 0), numpy.inf, ret)
    ret = numpy.where(numpy.logical_and(numpy.isinf(x), numpy.isinf(y)), 0.0, ret)

    # annoyingly numpy.where always returns arrays even with scalar inputs
    if type(x) is numpy.float64 and type(y) is numpy.float64 and type(ret) is numpy.ndarray:
        assert len(ret.flatten() == 1)
        ret = ret.flatten()[0]

    assert not numpy.any(numpy.isnan(ret))
    return ret

def option_value(strike, forward, volatility, time_to_expiry, is_call):
    ''' Undiscounted option price in forward-space Black-Scholes model '''
    f_or_s = numpy.where(is_call, forward, strike)
    s_or_f = numpy.where(is_call, strike, forward)
    return f_or_s * normalised_price(numpy.log(s_or_f / f_or_s), volatility * numpy.sqrt(time_to_expiry))

def delta(strike, forward, volatility, time_to_expiry, is_call):
    ''' Forward-delta of an undiscounted option in forward-space Black-Scholes model '''
    forward = numpy.asarray(forward)
    x = numpy.log(strike / forward)
    y = volatility * numpy.sqrt(time_to_expiry)
    if numpy.any(y < 0.0):
        raise ValueError(STR_NEGATIVE_STD_ERR)
    sign = numpy.where(is_call, 1.0, -1.0)
    return sign * norm.cdf(-sign * (x / y - y / 2.0))

def cash_delta(strike, forward, volatility, time_to_expiry, is_call):
    ''' Forward cash delta (forward x delta) of an undiscounted option in forward-space Black-Scholes model
        Cash delta is the dollar value of the amount of the underlying
        required to be held as a delta hedge for the option
    '''
    return forward * delta(strike, forward, volatility, time_to_expiry, is_call)

def dual_delta(strike, forward, volatility, time_to_expiry, is_call):
    ''' Dual delta, i.e. derivative wrt strike, of an undiscounted option in the forward-space Black-Scholes model '''
    forward = numpy.array(forward)
    x = numpy.log(strike / forward)
    y = volatility * numpy.sqrt(time_to_expiry)
    if numpy.any(y < 0.0):
        raise ValueError(STR_NEGATIVE_STD_ERR)
    sign = numpy.where(is_call, 1.0, -1.0)
    return -sign * norm.cdf(-sign * (x / y + y / 2.0))

def cash_dual_delta(strike, forward, volatility, time_to_expiry, is_call):
    ''' Dual (forward) Cash delta of a call option in the forward-space Black-Scholes model
        Dual Cash Delta is strike x delta, ie.. the analogue of cash delta in the strike variable
    '''
    return strike * dual_delta(strike, forward, volatility, time_to_expiry, is_call)

def gamma(strike, forward, volatility, time_to_expiry):
    ''' Gamma of an undiscounted option wrt the forward, in the forward-space Black Scholes model
        Note that this is the same for calls and puts
    '''
    forward = numpy.asarray(forward)
    x = numpy.log(strike / forward)
    y = volatility * numpy.sqrt(time_to_expiry)

    if numpy.any(y < 0.0):
        raise ValueError(STR_NEGATIVE_STD_ERR)
    d1_value = -x / y + y / 2.0
    ret = norm.pdf(d1_value) / (forward * y)

    ret = numpy.where(y == 0, 0.0, ret)
    ret = numpy.where(numpy.logical_and(x == 0, y == 0), numpy.inf, ret)
    ret = numpy.where(numpy.logical_or(numpy.isinf(x), numpy.isinf(y)), 0.0, ret)

    # annoyingly numpy.where always returns arrays even with scalar inputs
    if type(x) is numpy.float64 and type(y) is numpy.float64 and type(ret) is numpy.ndarray:
        assert len(ret.flatten() == 1)
        ret = ret.flatten()[0]

    assert not numpy.any(numpy.isnan(ret))
    return ret

def cash_gamma(strike, forward, volatility, time_to_expiry):
    ''' Cash Gamma of an option in the forward-space Black Scholes model
        Note that this is the same for calls and puts
        CashGamma / 100 is the change in CashDelta for a 1% move in the underlying
    '''
    return forward ** 2 * gamma(strike, forward, volatility, time_to_expiry)

def vega(strike, forward, volatility, time_to_expiry):
    ''' Vega of an undiscounted option in the forward-space Black Scholes model
        Identical for calls and puts
        Note we return mathematical derivative dprice/dvolatility, we don't scale by 1/100
        to return vega per vol point
    '''
    forward = numpy.asarray(forward)
    x = numpy.log(strike / forward)
    rtt = numpy.sqrt(time_to_expiry)
    y = volatility * rtt
    return forward * _d_normalised_price_dy(x, y) * rtt

def theta(strike, forward, volatility, time_to_expiry):
    ''' Theta of an undiscounted option in the forward-space Black Scholes model
        Note that while we don't have rates, dividend yields as input, i.e. they are assumed zero,
        the theta of calls and puts is the same. With r!=0, they differ, but in that case both calls and puts
        can be implemented in terms of this r==0 version
        We adopt the convention that theta is negative, the deriv wrt to time to expiry. that is, a negative amount,
        rather that a *positive* amount that is *lost* by being long options...
    '''
    forward = numpy.asarray(forward)
    x = numpy.log(strike / forward)
    rtt = numpy.sqrt(time_to_expiry)
    y = volatility * rtt

    # note in the divisor, we divide by 1 not 0
    # then we will always get 0 except at x=0 where we get inf, the correct limit behaviour
    return -0.5 * forward * volatility * _d_normalised_price_dy(x, y) / numpy.where(0.0 == rtt, 1.0, rtt)

def vanna(strike, forward, volatility, time_to_expiry):
    ''' Vanna, aka Dvegadspot or ddeltadvol, is the second order cross derivative of price wrt forward and volatility.
        It is the same for both calls and puts.
    '''
    forward = numpy.asarray(forward)
    x = numpy.log(strike / forward)
    y = volatility * numpy.sqrt(time_to_expiry)
    d1 = -x / y + y / 2.0
    d2 = d1 - y
    return -norm.pdf(d1) * d2 / volatility

def volga(strike, forward, volatility, time_to_expiry):
    ''' Volga is the second derivative wrt to volatility, i.e. dvegadvol.
        This one goes by a few names... i prefer "volga" being short for "volgamma" or "volatility-gamma", i.e. the
        second order greek (gamma) with respect to volatility rather than spot. Call it "Vomma" if you must, that just
        always made me think of "vomit".
    '''
    forward = numpy.asarray(forward)
    x = numpy.log(strike / forward)
    y = volatility * numpy.sqrt(time_to_expiry)
    v = vega(strike, forward, volatility, time_to_expiry)
    d1 = -x / y + y / 2.0
    d2 = d1 - y
    return v * d1 * d2 / volatility

def implied_volatility(strike, forward, price, time_to_expiry, is_call):
    ''' Compute the implied volatility for an undiscounted option price, using the specific (simpler)
        function for an out-of-the-money normalised call
    '''
    f_or_s = numpy.where(is_call, forward, strike)
    s_or_f = numpy.where(is_call, strike, forward)
    modified_price = numpy.where(f_or_s > s_or_f, 1.0 + (price - f_or_s) / s_or_f, price / f_or_s)
    return normalised_implied_volatility(
        numpy.abs(numpy.log(strike / forward)), modified_price) / numpy.sqrt(time_to_expiry)

def call_implied_volatility(strike, forward, price, time_to_expiry):
    ''' Black Scholes implied volatility for an undiscounted call option '''
    return implied_volatility(strike, forward, price, time_to_expiry, True)

def put_implied_volatility(strike, forward, price, time_to_expiry):
    ''' Black Scholes implied volatility for an undiscounted put option '''
    return implied_volatility(strike, forward, price, time_to_expiry, False)


def _call_value(strike, forward, volatility, time_to_expiry):
    ''' Undiscounted call price in forward-space Black-Scholes model '''
    return option_value(strike, forward, volatility, time_to_expiry, True)

def _put_value(strike, forward, volatility, time_to_expiry):
    ''' Undiscounted put price in forward-space Black-Scholes model '''
    return option_value(strike, forward, volatility, time_to_expiry, False)

def _call_delta(strike, forward, volatility, time_to_expiry):
    ''' Fowrard-delta of a call option in Black-Scholes model '''
    return delta(strike, forward, volatility, time_to_expiry, True)

def _call_cash_delta(strike, forward, volatility, time_to_expiry):
    ''' Cash delta (forward x delta)  of a call option in forward-space Black-Scholes model
        Cash delta is the dollar value of the amount of the underlying
        required to be held as a delta hedge for the option
    '''
    return  forward * _call_delta(strike, forward, volatility, time_to_expiry)

def _put_delta(strike, forward, volatility, time_to_expiry):
    ''' Delta of a put option in Black-Scholes model '''
    return delta(strike, forward, volatility, time_to_expiry, False)

def _put_cash_delta(strike, forward, volatility, time_to_expiry):
    ''' Cash delta (forward x delta) of a put option in forward-space Black-Scholes model
        Cash delta is the dollar value of the amount of the underlying
        required to be held as a delta hedge for the option
    '''
    return  forward * _put_delta(strike, forward, volatility, time_to_expiry)

def _call_dual_delta(strike, forward, volatility, time_to_expiry):
    ''' Dual delta, i.e. derivative wrt strike, of a call option in the forward-space Black-Scholes model '''
    return dual_delta(strike, forward, volatility, time_to_expiry, True)

def _call_cash_dual_delta(strike, forward, volatility, time_to_expiry):
    ''' Dual Cash delta of a call option in the forward-space Black-Scholes model
        Dual Cash Delta is strike x delta, ie.. the analogue of cash delta in the strike variable
    '''
    return  cash_dual_delta(strike, forward, volatility, time_to_expiry, True)

def _put_dual_delta(strike, forward, volatility, time_to_expiry):
    ''' Dual delta, i.e. derivative wrt strike, of a put option in the forward-space Black-Scholes model '''
    return dual_delta(strike, forward, volatility, time_to_expiry, False)

def _put_cash_dual_delta_(strike, forward, volatility, time_to_expiry):
    ''' Dual Cash delta of a put option in the forward-space Black-Scholes model
        Dual Cash Delta is strike x delta, ie.. the analogue of cash delta in the strike variable
    '''
    return  cash_dual_delta(strike, forward, volatility, time_to_expiry, False)

class TestForwardBlackScholes():
    ''' Class for testing Black Scholes functions '''

    def test_normalised_price(self):
        ''' Test the main black-scholes pricing function '''

        # the main bs function can be interpreted as being equal to
        # C/forward as a function of x = lnK/forward and y = volatility *sqrt(time_to_expiry)
        # when forward=1 and r=q=0
        # here we test the limiting behaviour etc of this function for extreme values

        # check that we get the right value for large y / infinte y
        x = numpy.linspace(-100, 100, 101)
        assert numpy.linalg.norm(normalised_price(x.flatten(), 100) - 1, numpy.inf) == 0.0
        assert numpy.linalg.norm(normalised_price(x.flatten(), numpy.inf) - 1) == 0.0

        # check that we get the right value for y=0 (including ATM at expiry, where the formula collapses)
        x = numpy.linspace(-10, 10, 101)
        assert numpy.linalg.norm(normalised_price(x.flatten(), 0) - numpy.maximum(1.0 - numpy.exp(x), 0.0), numpy.inf) == 0.0
        assert numpy.linalg.norm(normalised_price(x.flatten(), 0) - numpy.maximum(1.0 - numpy.exp(x), 1e-8), numpy.inf) < 1e-6

        y = numpy.linspace(0, 10, 101)

        # call tends to forward for very low/zero strikes (including at expiry)
        assert numpy.linalg.norm(normalised_price(-100, y) - 1.0, numpy.inf) == 0.0
        assert numpy.linalg.norm(normalised_price(-numpy.inf, y) - 1.0, numpy.inf) == 0.0

        # and to zero for very high/infinte strikes (including at expiry)
        assert numpy.linalg.norm(normalised_price(100, y), numpy.inf) < 1e-6
        assert numpy.linalg.norm(normalised_price(numpy.inf, y), numpy.inf) == 0.0

        # check that we have a valid and seemingly continuous function
        x = numpy.linspace(-10, 80, 101)
        y = numpy.linspace(0, 10, 101)
        assert not numpy.any(numpy.isnan(normalised_price(x, y)))
        assert numpy.linalg.norm((normalised_price(x + 1e-6, y) - normalised_price(x - 1e-6, y)).flatten(), numpy.inf) < 1e-4
        assert numpy.linalg.norm((normalised_price(x, y + 1e-6) - normalised_price(x, y)).flatten(), numpy.inf) < 1e-4

    def test_d_normalised_price_dx(self):
        ''' Test the first derivative of pricing function with respect to logstrike argument x '''

        x = numpy.linspace(-10, 80, 101).reshape(1, 101)
        y = numpy.linspace(0, 10, 101).reshape(101, 1)

        # check the finite difference approximation
        dx_fd = 1e-8

        # it's ok for y==0, we don't want to warn on 1/y
        finite_diff = (normalised_price(x + dx_fd, y) - normalised_price(x - dx_fd, y)) / (2 * dx_fd)
        analytic = _d_normalised_price_dx(x, y)
        assert numpy.linalg.norm((finite_diff - analytic).flatten(), numpy.inf) < 1e-6

        # check that we have a valid, negative, and seemingly continuous function
        assert not numpy.any(numpy.isnan(analytic))
        assert numpy.all(analytic <= 0.0)
        assert numpy.linalg.norm((_d_normalised_price_dx(x + 1e-6, y) - _d_normalised_price_dx(x - 1e-6, y)).flatten(), numpy.inf) < 1e-4
        assert numpy.linalg.norm((_d_normalised_price_dx(x, y + 1e-6) - _d_normalised_price_dx(x, y)).flatten(), numpy.inf) < 1e-4

        # tends to zero with y for nonnegative x...
        assert numpy.linalg.norm(_d_normalised_price_dx(numpy.array([xval for xval in x.flatten() if xval >= 0.0]), 0.0), numpy.inf) == 0.0
        assert numpy.linalg.norm(_d_normalised_price_dx(numpy.array([xval for xval in x.flatten() if xval >= 0.0]).flatten(), 1e-8), numpy.inf) == 0.0
        assert _d_normalised_price_dx(0.0, 0.0) == 0.0

        # ...but not for negative x
        assert numpy.linalg.norm(_d_normalised_price_dx(x.flatten(), 0.0), numpy.inf) > 0.1

        # left and right extrema (for infinite and large-but-finite x) are zero...
        assert numpy.linalg.norm(_d_normalised_price_dx(-1000, y.flatten()), numpy.inf) == 0.0
        assert numpy.linalg.norm(_d_normalised_price_dx(1000, y.flatten()), numpy.inf) == 0.0
        assert numpy.linalg.norm(_d_normalised_price_dx(-numpy.inf, y.flatten()), numpy.inf) == 0.0
        assert numpy.linalg.norm(_d_normalised_price_dx(-numpy.inf, y.flatten()), numpy.inf) == 0.0

        # ...including (already checked) the specific case for y01
        assert _d_normalised_price_dx(-numpy.inf, 0.0) == 0.0
        assert _d_normalised_price_dx(numpy.inf, 0.0) == 0.0

        # tends to zero for large x
        assert numpy.linalg.norm(_d_normalised_price_dx(x.flatten(), 1000), numpy.inf) == 0.0
        assert numpy.linalg.norm(_d_normalised_price_dx(x.flatten(), numpy.inf), numpy.inf) == 0.0

        # potentially a nice thing to check would be that the function seems continuous, i.e. that neighbouring values
        # are nearby but this would be redundant as we specifically validate the derivatives against finite difference
        # in other tests

    def test_d_normalised_price_dy(self):
        ''' Test the first derivative of pricing function with respect to stdev argument y '''

        x = numpy.linspace(-10, 80, 101).reshape(1, 101)
        y = numpy.linspace(0, 10, 101).reshape(101, 1)

        # check the finite difference approximation
        dy_fd = 1e-8

        # it's ok for y==0, we don't want to warn on 1/y
        finite_diff = (normalised_price(x, y + dy_fd) - normalised_price(x, y)) / dy_fd
        analytic = _d_normalised_price_dy(x, y)
        assert numpy.linalg.norm((finite_diff - analytic).flatten(), numpy.inf) < 1e-6

        # at y=0, we collapse to a delta function
        assert _d_normalised_price_dy(0.0, 0.0) == numpy.inf
        assert numpy.linalg.norm(_d_normalised_price_dy(x.flatten(), 0.0), numpy.inf) == 0.0

        # when x is large/small, or infinte +/- we go to zero
        assert numpy.linalg.norm(_d_normalised_price_dy(-numpy.inf, y.flatten()), numpy.inf) == 0.0
        assert numpy.linalg.norm(_d_normalised_price_dy(-200, y.flatten()), numpy.inf) < 1e-8
        assert numpy.linalg.norm(_d_normalised_price_dy(numpy.inf, y.flatten()), numpy.inf) == 0.0
        assert numpy.linalg.norm(_d_normalised_price_dy(200, y.flatten()), numpy.inf) < 1e-8

        # we also go to zero for large/infinte y
        assert numpy.linalg.norm(_d_normalised_price_dy(x.flatten(), numpy.inf), numpy.inf) == 0.0
        assert numpy.linalg.norm(_d_normalised_price_dy(x.flatten(), 1000), numpy.inf) == 0.0

        # we are positive everywhere
        assert numpy.all(_d_normalised_price_dy(x, y) >= 0.0)

        # right limit behaviour in the very far corners...
        assert _d_normalised_price_dy(-numpy.inf, numpy.inf) == 0.0
        assert _d_normalised_price_dy(numpy.inf, numpy.inf) == 0.0

    def test_calls_and_puts(self):
        ''' Test pricing of puts and calls '''

        value_point = 1.0  # avoid using 1 to week out any "wlog assume x=1" problems
        value_range = numpy.linspace(0, 2 * value_point, 101)

        times = numpy.linspace(0.0001, 10.0, 101)
        vol = 0.1

        # check symmetry: put as a function of strike is call as a function of forward and vice versa
        assert numpy.linalg.norm(_call_value(value_point, value_range, vol, times) - _put_value(value_range, value_point, vol, times), numpy.inf) < 1e-8
        assert numpy.linalg.norm(_call_value(value_range, value_point, vol, times) - _put_value(value_point, value_range, vol, times), numpy.inf) < 1e-8

        # we are a bit lazy and only check as time_to_expiry gets large. We know under the covers this is equivalent to
        # volatility getting large also, so don't check those cases...
        # a fixed strike call becomes equivalent to the forward, it will always be exercised
        # (in the case of infinite vol, we need to remove the forward=0 case. it's not well defined)
        assert numpy.linalg.norm(_call_value(value_point, value_range, 1.0, 1000.0) - value_range, numpy.inf) == 0.0

        assert numpy.linalg.norm(_call_value(value_point, value_range[1:], 1.0, numpy.inf) - value_range[1:], numpy.inf) == 0.0
        # given the forward level, upts are worth their strike level; the forward will go to zero given enough time
        # (in the case of infinite vol, we need to remove the strike=0 case. it's not well defined)
        assert numpy.linalg.norm(_put_value(value_range, value_point, 1.0, 1000.0) - value_range, numpy.inf) == 0.0
        assert numpy.linalg.norm(_put_value(value_range[1:], value_point, 1.0, numpy.inf) - value_range[1:], numpy.inf) == 0.0

        # also check as time_to_expiry gets small, we recover the intrinsic value, for put varying over strike with a
        # fixed forward, and calls varying over forward with fixed strike
        assert numpy.linalg.norm(_call_value(value_point, value_range, 1.0, 0.0) - numpy.maximum(value_range - value_point, 0.0), numpy.inf) < 1e-12
        assert numpy.linalg.norm(_put_value(value_range, value_point, 1.0, 0.0) - numpy.maximum(value_range - value_point, 0.0), numpy.inf) < 1e-12
        assert numpy.linalg.norm(_call_value(value_point, value_range, 1.0, 1e-12) - numpy.maximum(value_range - value_point, 0.0), numpy.inf) < 1e-6
        assert numpy.linalg.norm(_put_value(value_range, value_point, 1.0, 1e-12) - numpy.maximum(value_range - value_point, 0.0), numpy.inf) < 1e-6

        # zero strike put is worthless, as is a call when the forward has gone to zero
        assert numpy.linalg.norm(_put_value(0.0, value_point, 1.0, times), numpy.inf) == 0.0
        assert numpy.linalg.norm(_put_value(1e-12, value_point, 1.0, times), numpy.inf) < 1e-6
        assert numpy.linalg.norm(_call_value(value_point, 0.0, 1.0, times), numpy.inf) == 0.0
        assert numpy.linalg.norm(_call_value(value_point, 1e-12, 1.0, times), numpy.inf) < 1e-6

        # high strike put is worth the intrinsic, as is a call when forward gets very large
        big_num = 100
        assert numpy.linalg.norm(_put_value(big_num, value_point, 0.1, times) - (big_num - value_point)) == 0.0
        assert numpy.linalg.norm(_call_value(value_point, big_num, 0.1, times) - (big_num - value_point)) == 0.0

        # a fixed strike put becomes worth the strike as time gets large
        assert numpy.linalg.norm(_put_value(value_point, value_range, 1.0, 1000.0) - value_point, numpy.inf) == 0.0
        assert numpy.linalg.norm(_put_value(value_point, value_range, 1.0, numpy.inf) - value_point, numpy.inf) == 0.0
        # and given the forward level, calls of any strike become worth the same as the forward
        assert numpy.linalg.norm(_call_value(value_range, value_point, 1.0, 1000.0) - value_point, numpy.inf) == 0.0
        assert numpy.linalg.norm(_call_value(value_range, value_point, 1.0, numpy.inf) - value_point, numpy.inf) == 0.0

        # also check as time_to_expiry gets small, we recover the intrinsic value, for call varying over strike with a
        # fixed forward,and put varying over forward with fixed strike
        assert numpy.linalg.norm(_put_value(value_point, value_range, 1.0, 0.0) - numpy.maximum(value_point - value_range, 0.0), numpy.inf) < 1e-12
        assert numpy.linalg.norm(_call_value(value_range, value_point, 1.0, 0.0) - numpy.maximum(value_point - value_range, 0.0), numpy.inf) < 1e-12
        assert numpy.linalg.norm(_put_value(value_point, value_range, 1.0, 1e-12) - numpy.maximum(value_point - value_range, 0.0), numpy.inf) < 1e-6
        assert numpy.linalg.norm(_call_value(value_range, value_point, 1.0, 1e-12) - numpy.maximum(value_point - value_range, 0.0), numpy.inf) < 1e-6

        # zero strike call is the same as a forward, put with forward at zero is worth the strike.
        assert numpy.linalg.norm(_call_value(0.0, value_point, 1.0, times) - value_point, numpy.inf) == 0
        assert numpy.linalg.norm(_call_value(1e-12, value_point, 1.0, times) - value_point, numpy.inf) < 1e-6
        assert numpy.linalg.norm(_put_value(value_point, 0.0, 1.0, times) - value_point, numpy.inf) == 0.0
        assert numpy.linalg.norm(_put_value(value_point, 1e-12, 1.0, times) - value_point, numpy.inf) < 1e-6

        # high strike call is worth nothing, as is a put when forward gets very large
        big_num = 100
        assert numpy.linalg.norm(_call_value(big_num, value_point, 0.1, times)) < 1e-16
        assert numpy.linalg.norm(_call_value(numpy.inf, value_point, 0.1, times)) == 0.0
        assert numpy.linalg.norm(_put_value(value_point, big_num, 0.1, times)) < 1e-16
        assert numpy.linalg.norm(_put_value(value_point, numpy.inf, 0.1, times)) == 0.0

    def test_delta(self):
        ''' Test Black Scholes delta '''

        dx_fd = 1e-8

        numpy.random.seed(19791021)
        strike = numpy.random.uniform(dx_fd, 2.0 - dx_fd, 100).reshape(1, 100)
        forward = numpy.random.uniform(dx_fd, 2.0 - dx_fd, 100).reshape(1, 100)
        volatility = 0.2
        time_to_expiry = numpy.linspace(0, 10, 100).reshape(100, 1)

        finite_diff_value = (_call_value(strike, forward + dx_fd, volatility, time_to_expiry)
            - _call_value(strike, forward - dx_fd, volatility, time_to_expiry)) / (2 * dx_fd)
        analytic_value = _call_delta(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

        finite_diff_value = (_call_value(strike + dx_fd, forward, volatility, time_to_expiry)
            - _call_value(strike - dx_fd, forward, volatility, time_to_expiry)) / (2 * dx_fd)
        analytic_value = _call_dual_delta(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

        finite_diff_value = (_put_value(strike, forward + dx_fd, volatility, time_to_expiry)
            - _put_value(strike, forward - dx_fd, volatility, time_to_expiry)) / (2 * dx_fd)
        analytic_value = _put_delta(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

        finite_diff_value = (_put_value(strike + dx_fd, forward, volatility, time_to_expiry)
            - _put_value(strike - dx_fd, forward, volatility, time_to_expiry)) / (2 * dx_fd)
        analytic_value = _put_dual_delta(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

        # slightly disingenuous having analytic_value even no. of points, means we skip over the discontinuity at 1...
        x = numpy.linspace(0, 2, 100)

        # check that delta functions behave as as step func at time_to_expiry=0
        assert numpy.linalg.norm(_call_delta(1, x, 0, 0).flatten() - numpy.where(x < 1.0, 0.0, 1.0).flatten(), numpy.inf) == 0.0
        assert numpy.linalg.norm(_call_delta(x, 1.0, 0, 0).flatten() - numpy.where(x < 1.0, 1.0, 0.0).flatten(), numpy.inf) == 0.0
        assert numpy.linalg.norm(_put_delta(1, x, 0, 0).flatten() + numpy.where(x < 1.0, 1.0, 0.0).flatten(), numpy.inf) == 0.0
        assert numpy.linalg.norm(_put_delta(x, 1.0, 0, 0).flatten() + numpy.where(x < 1.0, 0.0, 1.0).flatten(), numpy.inf) == 0.0

        assert numpy.linalg.norm(_call_dual_delta(1, x, 0, 0).flatten() + numpy.where(x < 1.0, 0.0, 1.0).flatten(), numpy.inf) == 0.0
        assert numpy.linalg.norm(_call_dual_delta(x, 1.0, 0, 0).flatten() + numpy.where(x < 1.0, 1.0, 0.0).flatten(), numpy.inf) == 0.0
        assert numpy.linalg.norm(_put_dual_delta(1, x, 0, 0).flatten() - numpy.where(x < 1.0, 1.0, 0.0).flatten(), numpy.inf) == 0.0
        assert numpy.linalg.norm(_put_dual_delta(x, 1.0, 0, 0).flatten() - numpy.where(x < 1.0, 0.0, 1.0).flatten(), numpy.inf) == 0.0

        # TODO check all boundary conditions at ATM behaviour at expiry

    def test_gamma(self):
        ''' Test Black Scholes gamma '''

        dx_fd = 1e-8

        numpy.random.seed(19791021)
        strike = numpy.random.uniform(dx_fd, 2.0 - dx_fd, 100)
        forward = numpy.random.uniform(dx_fd, 2.0 - dx_fd, 100)
        volatility = 0.2
        time_to_expiry = numpy.linspace(0, 10, 100)

        finite_diff_value = (_call_delta(strike, forward + dx_fd, volatility, time_to_expiry)
            - _call_delta(strike, forward - dx_fd, volatility, time_to_expiry)) / (2 * dx_fd)
        analytic_value = gamma(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(analytic_value, numpy.inf) > 0.1  # we have numbers of reasonable magnitude
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

        finite_diff_value = (_put_delta(strike, forward + dx_fd, volatility, time_to_expiry)
            - _put_delta(strike, forward - dx_fd, volatility, time_to_expiry)) / (2 * dx_fd)
        analytic_value = gamma(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(analytic_value, numpy.inf) > 0.1  # we have numbers of reasonable magnitude
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

    def test_vega(self):
        ''' Test Black Scholes vega '''

        dx_fd = 1e-8

        numpy.random.seed(19791021)
        n_points =  1000
        strike = numpy.random.uniform(0.0, 2.0, n_points).reshape(n_points, 1)
        forward = numpy.random.uniform(0.0, 2.0, n_points).reshape(n_points, 1)
        volatility = numpy.random.uniform(dx_fd, 0.5 - dx_fd, n_points).reshape(n_points, 1)
        time_to_expiry = numpy.random.uniform(0.0, 5, n_points).reshape(n_points, 1)

        finite_diff_value = (_call_value(strike, forward, volatility + dx_fd, time_to_expiry)
            - _call_value(strike, forward, volatility - dx_fd, time_to_expiry)) / (2 * dx_fd)
        analytic_value = vega(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(analytic_value, numpy.inf) > 0.1  # we have numbers of reasonable magnitude
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

        finite_diff_value = (_put_value(strike, forward, volatility + dx_fd, time_to_expiry)
            - _put_value(strike, forward, volatility - dx_fd, time_to_expiry)) / (2 * dx_fd)
        analytic_value = vega(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(analytic_value, numpy.inf) > 0.1  # we have numbers of reasonable magnitude
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

    def test_theta(self):
        ''' Test Black Scholes theta '''

        dx_fd = 1e-8

        numpy.random.seed(19791021)
        n_points =  1000
        strike = numpy.random.uniform(0.0, 2.0, n_points).reshape(n_points, 1)
        forward = numpy.random.uniform(0.0, 2.0, n_points).reshape(n_points, 1)
        volatility = numpy.random.uniform(0.0, 0.5, n_points).reshape(n_points, 1)
        time_to_expiry = numpy.random.uniform(dx_fd, 5 - dx_fd, n_points).reshape(n_points, 1)

        finite_diff_value = -(_call_value(strike, forward, volatility, time_to_expiry + dx_fd)
            - _call_value(strike, forward, volatility, time_to_expiry - dx_fd)) / (2 * dx_fd)
        analytic_value = theta(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(analytic_value, numpy.inf) > 0.1  # we have numbers of reasonable magnitude
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

        finite_diff_value = -(_put_value(strike, forward, volatility, time_to_expiry + dx_fd)
            - _put_value(strike, forward, volatility, time_to_expiry - dx_fd)) / (2 * dx_fd)
        analytic_value = theta(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(analytic_value, numpy.inf) > 0.1  # we have numbers of reasonable magnitude
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

        assert numpy.all(analytic_value <= 0.0)

        # boundary conditions
        assert numpy.linalg.norm(theta(numpy.inf, forward, volatility, time_to_expiry).flatten(), numpy.inf) == 0.0
        assert numpy.linalg.norm(theta(10000, forward, volatility, time_to_expiry).flatten(), numpy.inf) < 1e-6

        assert numpy.linalg.norm(theta(0.0, forward, volatility, time_to_expiry).flatten(), numpy.inf) == 0.0
        assert numpy.linalg.norm(theta(1e-12, forward, volatility, time_to_expiry).flatten(), numpy.inf) < 1e-6

        assert numpy.linalg.norm(theta(strike, forward, volatility, numpy.inf).flatten(), numpy.inf) == 0.0
        assert numpy.linalg.norm(theta(strike, forward, volatility, 1e8).flatten(), numpy.inf) < 1e-6

        assert numpy.linalg.norm(theta(strike, forward, volatility, 0.0).flatten(), numpy.inf) == 0.0
        assert numpy.linalg.norm(theta(strike, forward, volatility, 1e-12).flatten(), numpy.inf) < 1e-6

        assert numpy.all(theta(strike, strike, volatility, 0.0) == -numpy.inf)

    def test_vanna(self):
        ''' Test Black Scholes vanna '''

        dx_fd = 1e-8

        numpy.random.seed(19791021)
        n_points =  1000
        strike = numpy.random.uniform(0.0, 2.0, n_points).reshape(n_points, 1)
        forward = numpy.random.uniform(0.0, 2.0, n_points).reshape(n_points, 1)
        volatility = numpy.random.uniform(0.0, 0.5, n_points).reshape(n_points, 1)
        time_to_expiry = numpy.random.uniform(dx_fd, 5 - dx_fd, n_points).reshape(n_points, 1)

        finite_diff_value = (vega(strike, forward + dx_fd, volatility, time_to_expiry)
            - vega(strike, forward - dx_fd, volatility, time_to_expiry)) / (2 * dx_fd)
        analytic_value = vanna(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(analytic_value, numpy.inf) > 0.1  # we have numbers of reasonable magnitude
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

        # TODO check all boundary conditions etc

    def test_volga(self):
        ''' Test Black Scholes volga '''

        dx_fd = 1e-8

        numpy.random.seed(19791021)
        n_points =  1000
        strike = numpy.random.uniform(0.0, 2.0, n_points).reshape(n_points, 1)
        forward = numpy.random.uniform(0.0, 2.0, n_points).reshape(n_points, 1)
        volatility = numpy.random.uniform(0.0, 0.5, n_points).reshape(n_points, 1)
        time_to_expiry = numpy.random.uniform(dx_fd, 5 - dx_fd, n_points).reshape(n_points, 1)

        finite_diff_value = (vega(strike, forward, volatility + dx_fd, time_to_expiry)
            - vega(strike, forward, volatility - dx_fd, time_to_expiry)) / (2 * dx_fd)
        analytic_value = volga(strike, forward, volatility, time_to_expiry)
        assert numpy.linalg.norm(analytic_value, numpy.inf) > 0.1  # we have numbers of reasonable magnitude
        assert numpy.linalg.norm(finite_diff_value - analytic_value, numpy.inf) < 1e-6

    def test_dimensions(self):
        ''' Test we can call all the Black Scholes functions with vector or scalar input and that
            the results come back as expected
        '''

        functions = {
            _call_value,
            _put_value,
            _call_delta,
            _put_delta,
            _call_dual_delta,
            _put_dual_delta,
            _call_cash_delta,
            _put_cash_delta,
            _call_cash_dual_delta,
            _put_cash_dual_delta_,
            gamma,
            cash_gamma,
            vega,
            theta,
            vanna,
            volga
            }

        n_strikes = 7
        n_expiries = 5  # different to n_strikes to weed out erroneous vector ops on data from different spaces

        strike_value = 1.0
        forward_value = 1.0
        volatility_value = 1.0
        time_to_exp_value = 1.0

        # vector inuts
        strike_vec = numpy.random.uniform(0, 2, n_strikes)
        forward_vec = numpy.random.uniform(0, 2, n_strikes)
        volatility_vec = numpy.random.uniform(0.1, 0.5, n_expiries)
        time_to_exp_vec = numpy.random.uniform(0, 2, n_expiries)

        # grids of inputs. Most useful for grid of volatility, where we have varying strike and time_to_expiry and
        # different implied vols at each point in KxT
        forward_grid = numpy.random.uniform(0, 2, n_strikes * n_expiries).reshape((n_expiries, n_strikes))
        strike_grid = numpy.random.uniform(0, 2, n_strikes * n_expiries).reshape((n_expiries, n_strikes))
        volatility_grid = numpy.random.uniform(0.1, 0.5, n_strikes * n_expiries).reshape((n_expiries, n_strikes))
        time_to_exp_grid = numpy.random.uniform(0.1, 0.5, n_strikes * n_expiries).reshape((n_expiries, n_strikes))

        # nb In the case where we don't want analytic_value mxn grid as return, we are just pricing a list of unrelated
        # strikes & expiries. in these cases we must .reshape the inputs to be kx1 or 1xk depending on the orientation
        # we want the outputs, and then flatten the results (we can't just input 1d vectors, else if we had num strikes
        # and num expiries, it woudl be unclear if we want num outputs or an nxn grid. we choose the grid output as the
        # default interpretation for 1d inputs, so we must explicitly size our inuts if we want 1d outputs. Hope this
        # makes sense! see examples below.)

        n_list = 11
        strike_vec = numpy.random.uniform(0, 2, n_list)
        forward_vec = numpy.random.uniform(0, 2, n_list)
        volatility_vec = numpy.random.uniform(0.1, 0.5, n_list)
        time_to_exp_vec = numpy.random.uniform(0, 2, n_list)

        for func in functions:

            # scalar output
            result = func(strike_value, forward_value, volatility_value, time_to_exp_value)
            assert type(result) is numpy.float64

            # any combination of validly sized vector inputs with scalars produces the right sized vector output
            # note we have to discard the first case where *all* inputs are scalars....
            input_combos = [
                [strike_value, strike_vec],
                [forward_value, forward_vec],
                [volatility_value, volatility_vec],
                [time_to_exp_value, time_to_exp_vec]]

            input_combos = list(itertools.product(*input_combos))[1:]

            for input_combo in input_combos:
                result = func(*input_combo)
                assert len(result.shape) == 1
                assert len(result == n_list)

            # any combination of validly sized matrix inputs and scalars products the right sized matrix output
            # note if we want a 2d array output, we allow for 1d inputs to be broadcast along rows/columns
            # these are expected to be in the form of nx1 and 1xm arrays depending on whether a row/col input
            # or we allow the full mxn grid, or scalars to be broadcast over the whole grid.
            input_combos = itertools.product(*[
                [strike_value,
                     strike_grid[0, :].reshape(1, strike_grid.shape[1]),
                     strike_grid[:, 0].reshape(strike_grid.shape[0], 1),
                     strike_grid],
                [forward_value,
                     forward_grid[0, :].reshape(1, forward_grid.shape[1]),
                     forward_grid[:, 0].reshape(forward_grid.shape[0], 1),
                     forward_grid],
                [volatility_value,
                    volatility_grid[0, :].reshape(1, volatility_grid.shape[1]),
                    volatility_grid[:, 0].reshape(volatility_grid.shape[0], 1),
                    volatility_grid],
                [time_to_exp_value,
                    time_to_exp_grid[0, :].reshape(1, time_to_exp_grid.shape[1]),
                    time_to_exp_grid[:, 0].reshape(time_to_exp_grid.shape[0], 1),
                    time_to_exp_grid]
            ])

            # filter our any combos that don't have full row/col setup
            # we must have at least one input that has all expiries (rows)
            # and at least one that has all strikes (columns)
            input_combos = [comb for comb in input_combos if
                (max([inp.shape[0] for inp in comb if type(inp) is numpy.ndarray] or [0]) == n_expiries
                and max([inp.shape[1] for inp in comb if type(inp) is numpy.ndarray] or [0]) == n_strikes)]

            for input_combo in input_combos:
                result = func(*input_combo)
                assert result.shape == (n_expiries, n_strikes)


    def test_basic_inversion_1(self):
        ''' Price options given the volatiltiy, convert back to vols and check the error is small
            for a variety of calls and puts of various maturities, both in & out of the money. Here we can't test
            very extreme values of the inputs, because numerical floating point accuracy puts unavoidable bounds
            on the accuracy of inversion of in-the-money options. See other more rigourous tests for OTM options
        '''

        numpy.random.seed(12345)
        num = 101
        easy_x = numpy.linspace(-0.5, 0.5, num).reshape(1, num)
        easy_y = numpy.linspace(0.1, 0.3, num).reshape(num, 1)

        forwards = numpy.exp(numpy.random.uniform(-10, 10, num)).reshape(1, num)
        strikes = forwards * numpy.exp(easy_x)
        expiries = numpy.random.uniform(0, 1, num).reshape(num, 1)
        vols = easy_y / numpy.sqrt(expiries)
        is_call = (numpy.random.uniform(0, 1, num) > 0.5).reshape(1, num)

        prices = option_value(strikes, forwards, vols, expiries, is_call)
        ivols = implied_volatility(strikes, forwards, prices, expiries, is_call)
        error = numpy.max(numpy.abs(ivols - vols))

        assert error < 1e-10

    def test_basic_inversion_2(self):
        ''' Imply option volatilities from prices, recompute the prices from thos vols and check the error is small
            for a variety of calls and puts of various maturities, both in & out of the money. Here we can't test
            very extreme values of the inputs, because numerical floating point accuracy puts unavoidable bounds
            on the accuracy of inversion of in-the-money options. See other more rigourous tests for OTM options
        '''

        numpy.random.seed(12345)
        num = 101
        easy_x = numpy.linspace(-0.5, 0.5, num).reshape(1, num)
        easy_v = numpy.linspace(0.1, 0.3, num).reshape(num, 1)

        forwards = numpy.exp(numpy.random.uniform(-10, 10, num)).reshape(1, num)
        strikes = forwards * numpy.exp(easy_x)

        is_call = (numpy.random.uniform(0, 1, num) > 0.5).reshape(1, num)
        prices = easy_v * numpy.where(is_call, forwards, strikes) + numpy.maximum(
            numpy.where(is_call, forwards - strikes, strikes - forwards), 0.0)
        expiries = numpy.random.uniform(0, 1, num).reshape(num, 1)

        ivols = implied_volatility(strikes, forwards, prices, expiries, is_call)
        reprices = option_value(strikes, forwards, ivols, expiries, is_call)

        error = numpy.max(numpy.abs((prices - reprices) / forwards))

        assert error < 1e-10


    def test_out_the_money_inversion(self):
        ''' More stringent tests for quality of inversion accuracy, specifically for out of the money options
            which can be computed more accurately because the have values approaching zero (as opposed to
            in-the-money which in the limit tend to 1, and so the important time-value of the option is dwarfed by
            comparison to the intrinsic and therefore computation of implied vol becomes difficult)
        '''

        numpy.random.seed(12345)
        num = 101
        x = numpy.linspace(0, 20, num)  # puts down to ~1/10000 of a bp
        y = numpy.linspace(0, 15, num)  # out to ~225 years at 100% vol or 1y at 1500%

        forwards = numpy.exp(numpy.random.uniform(-10, 10, num)).reshape(1, num)
        strikes = forwards * numpy.exp(x)
        expiries = numpy.random.uniform(0, 1, num).reshape(num, 1)
        vols = y / numpy.sqrt(expiries)
        is_call = strikes > forwards

        prices = option_value(strikes, forwards, vols, expiries, is_call)
        ivols = implied_volatility(strikes, forwards, prices, expiries, is_call)

        time_value = prices / numpy.where(is_call, forwards, strikes)

        # if the option is worth nothing or intrinsic, not a lot we can do. vol is good as inf/zero.
        # impled vol mathematically goes to zero as time_value goes to 1: in fact we hit np.inf slightly before
        # so allow a little bit of leniency (honestly, not that important...)
        mask = time_value > 0.99999999
        assert numpy.count_nonzero(mask) / (len(mask.flatten())) < 0.1
        vol_errors = numpy.where(mask, 0.0, ivols - vols)
        error = numpy.max(numpy.abs(vol_errors))
        assert error < 1e-6

        # for option below 99% of their max value, let's be more strict
        mask = time_value > 0.99
        assert numpy.count_nonzero(mask) / (num * num) < 0.5
        vol_errors = numpy.where(mask, 0.0, ivols - vols)
        error = numpy.max(numpy.abs(vol_errors))
        assert error < 1e-10

    def test_implied_vol_dimensions(self):
        ''' Test we can call implied volatility functions with vector, scalar or matrix input and that
            the results come back as expected
        '''

        strike = 1.1234
        forward = 1.2345
        price = 0.1234
        expiry = 0.567
        is_call = False

        result = implied_volatility(strike, forward, price, expiry, is_call)
        assert type(result) is numpy.float64

        scalar_flag = 0
        array_flag = 1

        class BsVar:
            ''' enum of the variables over which we are testing vectorisation '''
            strike = 0
            forward = 1
            price = 2
            expiry = 3
            is_call = 4

        input_combos = [[scalar_flag, array_flag]] * 5
        input_combos = list(itertools.product(*input_combos))
        input_combos = [dict(zip((BsVar.strike, BsVar.forward, BsVar.price, BsVar.expiry, BsVar.is_call),
                                           inputs))
                        for inputs in input_combos]
        input_combos = input_combos[1:]  # discard the first case corresponding to all scalar inputs

        array_length = 7
        for combo in input_combos:
            strikes = strike if combo[BsVar.strike] == scalar_flag else numpy.full(array_length, strike)
            forwards = strike if combo[BsVar.forward] == scalar_flag else numpy.full(array_length, forward)
            prices = strike if combo[BsVar.price] == scalar_flag else numpy.full(array_length, price)
            expiries = strike if combo[BsVar.expiry] == scalar_flag else numpy.full(array_length, expiry)
            is_call_flags = strike if combo[BsVar.is_call] == scalar_flag else numpy.full(array_length, is_call)
            implied_vol = implied_volatility(strikes, forwards, prices, expiries, is_call_flags)
            assert implied_vol.shape == (array_length,)

        scalar_flag = 0
        column_vector_flag = 1
        row_vector_flag = 2
        matrix_flag = 3

        input_combos = [[scalar_flag, column_vector_flag, row_vector_flag, matrix_flag]] * 5
        input_combos = list(itertools.product(*input_combos))
        input_combos = [dict(zip((BsVar.strike, BsVar.forward, BsVar.price, BsVar.expiry, BsVar.is_call),
                                           inputs))
                        for inputs in input_combos]
        input_combos = input_combos[1:]  # discard the first case corresponding to all scalar inputs

        height = 3
        width = 11

        for combo in input_combos:

            strikes = strike
            if combo[BsVar.strike] == column_vector_flag: strikes = numpy.full(height, strike).reshape(height, 1)
            if combo[BsVar.strike] == row_vector_flag: strikes = numpy.full(width, strike).reshape(1, width)
            if combo[BsVar.strike] == matrix_flag: strikes = numpy.full(height * width, strike).reshape(height, width)

            forwards = forward
            if combo[BsVar.forward] == column_vector_flag: forwards = numpy.full(height, forward).reshape(height, 1)
            if combo[BsVar.forward] == row_vector_flag: forwards = numpy.full(width, forward).reshape(1, width)
            if combo[BsVar.forward] == matrix_flag: forwards = numpy.full(height * width, forward).reshape(height, width)

            expiries = expiry
            if combo[BsVar.expiry] == column_vector_flag: expiries = numpy.full(height, expiry).reshape(height, 1)
            if combo[BsVar.expiry] == row_vector_flag: expiries = numpy.full(width, expiry).reshape(1, width)
            if combo[BsVar.expiry] == matrix_flag: expiries = numpy.full(height * width, expiry).reshape(height, width)

            prices = price
            if combo[BsVar.price] == column_vector_flag: prices = numpy.full(height, price).reshape(height, 1)
            if combo[BsVar.price] == row_vector_flag: prices = numpy.full(width, price).reshape(1, width)
            if combo[BsVar.price] == matrix_flag: prices = numpy.full(height * width, price).reshape(height, width)

            is_call_flags = is_call
            if combo[BsVar.is_call] == column_vector_flag: is_call_flags = numpy.full(height, is_call).reshape(height, 1)
            if combo[BsVar.is_call] == row_vector_flag: is_call_flags = numpy.full(width, is_call).reshape(1, width)
            if combo[BsVar.is_call] == matrix_flag: is_call_flags = numpy.full(height * width, is_call).reshape(height, width)

            results = implied_volatility(strikes, forwards, prices, expiries, is_call_flags)

            has_cols = column_vector_flag in combo.values()
            has_rows = row_vector_flag in combo.values()
            has_mats = matrix_flag in combo.values()

            assert has_cols or has_rows or has_mats

            expected_height = height if (has_cols or has_mats) else 1
            expected_width = width if (has_rows or has_mats) else 1

            assert (results.shape == (expected_height, expected_width))