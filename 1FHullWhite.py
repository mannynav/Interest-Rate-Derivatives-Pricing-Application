
import numpy as np
import enum
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate


class OneFactorHullWhite:

    mean_reversion = 0.0
    volatility = 0.0
    ZCB = 0
    FDStep = 0

    #Constructor
    def __init__(self,mean_reversion,volatility,ZCB,FDStep):
        self.mean_reversion = mean_reversion
        self.volatility = volatility
        self.ZCB = ZCB
        self.FDStep = FDStep

    def ForwardRate(self,t):
        return -(np.log(self.ZCB(t+0.0001)) - np.log(self.ZCB(t-0.0001)))/(2.0*0.0001)


    def Theta(self,t):
        temp = (1.0/self.mean_reversion)*(self.ForwardRate(t+self.FDStep) - self.ForwardRate(t-self.FDStep))/(2*self.FDStep) + self.ForwardRate(t) + self.volatility*self.volatility/(2.0*self.mean_reversion*self.mean_reversion)*(1.0-np.exp(-2.0*self.mean_reversion*t))
        return temp


    def A(self,Ti,TL):
        time_step = TL-Ti
        grid = np.linspace(0.0,time_step,250) #grid for integration
        B = lambda t: 1.0/self.mean_reversion*(np.exp(-self.mean_reversion*t)-1.0)
        temp = self.mean_reversion * integrate.trapz(self.Theta(TL-grid)*B(grid),grid)
        temp2 = self.volatility*self.volatility/(4.0*np.power(self.mean_reversion,3.0))*(np.exp(-2.0*self.mean_reversion*time_step)*(4*np.exp(self.mean_reversion*time_step)-1.0) -3.0) + self.volatility*self.volatility*time_step/(2.0*self.mean_reversion*self.mean_reversion)
        return temp + temp2


    def B(self,Ti,TL):
        return (1/self.mean_reversion)*(np.exp(-self.mean_reversion*(TL-Ti))-1.0)

    def ZeroCouponBond(self,Ti,TL,r0):
        A = self.A(Ti,TL)
        B = self.B(Ti,TL)
        #r0 = self.ForwardRate(0.00001)
        return np.exp(A + B*r0)

    def GeneratePaths(self,number_of_paths,number_of_steps,T):
        standard_normals = np.random.normal(0.0,1.0,[number_of_paths,number_of_paths])
        short_rate = np.zeros([number_of_paths,number_of_steps+1])

        short_rate[:,0] = self.ForwardRate(0.00001)

        time = np.zeros([number_of_steps+1])
        Numeraire = np.zeros([number_of_paths,number_of_steps])
        dt = T/number_of_steps
        for i in range(0,number_of_steps):
            short_rate[:,i+1] = short_rate[:,i] + self.mean_reversion*(self.Theta(time[i]) - short_rate[:,i])*dt + self.volatility*(np.power(dt,0.5))*standard_normals[:,i]
            time[i+1] = time[i] + dt

        for i in range(0,number_of_paths):
            Numeraire[i,:] = np.exp(np.cumsum(short_rate[i,:-1])*dt)
        return {"time":time,"short_rate":short_rate,"Numeraire":Numeraire}

    def TransformedTheta(self,t,T):

        temp = self.Theta(t) + ((self.volatility*self.volatility)/self.mean_reversion)*self.B(t,T)
        return temp


    def MeanForwardMeasure(self,T):

        grid = np.linspace(0.0,T,500)
        integrand = lambda i: self.TransformedTheta(i,T)*np.exp(-self.mean_reversion*(T-i))
        new_mean = self.ForwardRate(0.00001)*np.exp(-self.mean_reversion*T) + self.mean_reversion*integrate.trapz(integrand(grid),grid)
        return new_mean

    def ZCBFwdMeasurePrice(self,strike,Ti,TL):
        new_mean = self.MeanForwardMeasure(Ti)
        transformed_v = np.sqrt(self.volatility*self.volatility/(2.0*self.mean_reversion)*(1.0-np.exp(-2.0*self.mean_reversion*Ti)))

        adjusted_k = strike*np.exp(-self.A(Ti,TL))

        a = (np.log(adjusted_k)-self.B(Ti,TL)*new_mean)/(self.B(Ti,TL)*transformed_v)

        d1 = a - self.B(Ti,TL)*transformed_v
        d2 = d1+self.B(Ti,TL)*transformed_v

        term1 = np.exp(0.5*self.B(Ti,TL)*self.B(Ti,TL)*transformed_v*transformed_v + self.B(Ti,TL)*new_mean)*st.norm.cdf(d1) - adjusted_k*np.exp(-self.A(Ti,TL))*st.norm.cdf(d2)

        return self.ZCB(Ti)*np.exp(self.A(Ti,TL))*term1  - self.ZCB(TL) + strike*self.ZCB(Ti)


    def Caplet(self,notional,strike,Ti,TL):
        transformed_notional = notional*(1.0 + (TL-Ti)*strike)

        transformed_strike = ((TL-Ti)*strike) + 1.0

        price = transformed_notional*self.ZCBFwdMeasurePrice(1/transformed_strike,Ti,TL)

        return price


#Zero coupon bond. Will eventually get this from the market
Market_ZCB = lambda T: np.exp(-0.1*T)

#Expiries
T1 = 4.0
T2 = 8.0


#Hull White parameters
lamb = 0.02
eta = 0.02


steps = 25
end_time = 50
grid= np.linspace(0,end_time,steps)

#Create Hull-White object
HW = OneFactorHullWhite(lamb,eta,Market_ZCB,0.0001)

#Initial rate. This is input to the HW ZCB
r0 = HW.ForwardRate(0.00001)

values= np.zeros([steps,1])
for i,time_i in enumerate(grid):
    values[i] = HW.ZeroCouponBond(0.0,time_i,r0)

plt.figure(1)
plt.grid()
plt.plot(grid,values)

paths = HW.GeneratePaths(2000,1000,4)
short_rate = paths["short_rate"]
numeraire = paths["Numeraire"]

strikes = np.linspace(0.01,1.7,50)
ZCB_Price = HW.ZeroCouponBond(4,8,short_rate[:,-1]) #Closed form price to be used from T = 4 to T = 8.

call_prices = np.zeros([len(strikes),1])
put_prices = np.zeros([len(strikes),1])

for i,strike in enumerate(strikes):
    call_prices[i] = np.mean( 1.0/numeraire[:,-1] * np.maximum(ZCB_Price-strike,0.0))

for i,strike in enumerate(strikes):
    put_prices[i] = np.mean( 1.0/numeraire[:,-1] * np.maximum(strike-ZCB_Price,0.0))

plt.figure(2)
plt.grid()
plt.plot(strikes,call_prices)
plt.title('Call option on ZCB')


plt.figure(3)
plt.grid()
plt.plot(strikes,put_prices)
plt.title('Put option on ZCB')



P0T = lambda T: np.exp(-0.1*T)#np.exp(-0.03*T*T-0.1*T)
frwd = 1.0/(T2-T1) *(P0T(T1)/P0T(T2)-1.0)
K = np.linspace(frwd/2.0,3.0*frwd,26)
Notional = 1.0


capletPrice = np.zeros(len(K))
for i in range(0,len(K)):
    capletPrice[i] = HW.Caplet(Notional,K[i],T1,T2)



plt.figure(4)
plt.title('Caplet Price')
plt.plot(K,capletPrice)
plt.xlabel('strike')
plt.ylabel('Caplet Price')
plt.grid()
