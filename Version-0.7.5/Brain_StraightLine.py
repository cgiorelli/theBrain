'Set up'
##--------------##

from __future__ import division
from math import sin , atan, cos, pi
from numpy import arange
from numpy.testing import assert_almost_equal as aae
import matplotlib.pyplot as plt
#%matplotlib inline
##--------------##

debug = 'false'; #If true, returns linear velocity + accerelation, and angular velcocity + accerelation

'Vehicle Parameters'
##--------------##

mass = 272 #kg
area = 1.50 #m^2 surface area (cross-sectional)
rrad = 0.25 #meters rear wheel radius
frad = 0.25 #meters front wheel radius
effT = 0.80 #powertrain efficiency
pmax = 60.0 #kW cont. power (EmDrive 500)
gear = 4.5 #final drive gear ratio
wdis = 0.65 #Weight distrubtion, % of weight on rear wheels
base = 1.2 #Axle-to-Axle distance in meters
high = 0.25 #Center of Gravity's height in meters

mxRM = 6000#5500 #max motor Rev/min
mxPT = 140.01 #240.02
mxCT = 60.087 #96

#fric = 0.19 #friction coefficient
drag = 0.50 #coefficient of drag

'Constants'
##--------------##
grav = 9.81 #m/s^2
dens = 1.23 #kg/m^3

'Initial Values'
##--------------##
hertz = 25

# angl = 0.00 #initial hill angle in radians
accl = 0.00 #initial accerelaion in m/s^2
mRPM  = 0.00 #initial Motor RPM value
wRPM = 0.00 #initial Wheel RPM value
velo = 0.00 #initial velocity in m/s
xpos = 0.00 #initial X-position (total distance)
lpos = 0.00 #initial X-position relative to the current lap
time = 0.00 #initial time in seconds (Start time)
wAcc = 0.00 #initial angular accerelation

thrt = 75
##--------------##

'Motor Torque from Throttle Map'
#Equations were made by inputting points of the torque curves into Excel
##--------------##

'''Emrax 208 Equations'''
def Ctor(rpm):
    return (3e-13*rpm**3)-(1e-6*rpm**2)+(0.0059*rpm)+(60.087)
    
def Ptor(rpm):
    return (-1e-14*rpm**4)-(7e-11*rpm**3)+(2e-7*rpm**2)-(0.0002*rpm)+(140.01) 


'''Emrax 228 Equations
def Ctor(rpm):
    return (-6e-13*rpm**4)+(9e-9*rpm**3)-(5e-5*rpm**2)+(0.1082*rpm)+(96)
    
def Ptor(rpm):
    return (3e-13*rpm**4)-(3e-9*rpm**3)+(6e-6*rpm**2)-(0.0039*rpm)+(240.02) 
'''
   
def motorTorque(rpm, thrt): #Normally all the Torque functions are kept below, but are also above for plotting purposes

    def Ctor(rpm):
        return (3e-13*rpm**3)-(1e-6*rpm**2)+(0.0059*rpm)+(60.087)
    
    def Ptor(rpm):
        return (-1e-14*rpm**4)-(7e-11*rpm**3)+(2e-7*rpm**2)-(0.0002*rpm)+(140.01) 
    
    def interpolate(x1,x2,y1,y2,xvalue):
        #Linear interpolation between two points x1 and x2 with y values y1 and y2. Finds yvalue corresponding to xvalue
        slope = (y2-y1)/(x2-x1)
        yvalue = y1 + slope*(xvalue-x1)
        return yvalue

    if thrt >= 100:
        motorTorque = Ptor(rpm)
    elif thrt == 50:
        motorTorque = Ctor(rpm)
    elif thrt <= 0:
        motorTorque = 0
    elif (thrt > 50) and (thrt < 100):
        torque1 = Ctor(rpm)
        torque2 = Ptor(rpm)
        throttle1 = 50
        throttle2 = 100
        motorTorque = interpolate(throttle1,throttle2,torque1,torque2,thrt)
    elif (thrt > 0) and (thrt < 50):
        torque2 = Ctor(rpm)
        throttle2 = 50
        motorTorque = interpolate(0,throttle2,0,torque2,thrt)
    
    return motorTorque
##--------------##



'Drive Train Torque'
##--------------##
def driveTorque(gear,mRPM,mxRM,thrt):
    if mRPM > mxRM:
        mRPM = mxRM
    return (gear * motorTorque(mRPM, thrt))
##--------------##

'Augular Acceleration in Drive Train'
##--------------##
def AngularAcceleration(DriveTrainTorque):
    return DriveTrainTorque / 1 #assumes the Moment of Interia is 1 -- T = I*AngAccereleration
##--------------##
def AngularVelocity(AngAccel,AngVelo,hertz):
        return AngVelo + AngAccel*(1/hertz)


'Motor RPM from Forward Velocity'
##--------------##
def motorRPM(gear,rrad,velo):
    return ((velo*60)/(2*pi*rrad))*gear
##--------------##

'Slope Calculator'
##--------------##
def slope(base,frad,rrad,xpos):
        
    elevDataFile='elevationdata.txt' #filename (and path if located in a different folder) for elevation data profile
        
    def interpolate(x1,x2,y1,y2,xvalue):
        "Linear interpolation between two points x1 and x2 with y values y1 and y2. Finds yvalue corresponding to xvalue"
        slope = (y2-y1)/(x2-x1)
        yvalue = y1 + slope*(xvalue-x1)
        return yvalue
    
    def elevation(xpos):
        "Finds the elevation (y coordinate) for each x coordinate. Throws an exception if x-coordinate is out of range for the input file."
        import bisect
        xylist = []
        
        with open(elevDataFile) as elev:
            for line in elev.readlines():
                xylist.append([float(line.split(' ')[0]),float(line.split(' ')[1].rstrip())])

        xvalues=zip(*xylist)[0]
        yvalues=zip(*xylist)[1]
    
        #check that the xPosition value falls within the dataset provided, else return elevation of 0
        if xpos >= min(xvalues) and xpos <= max(xvalues):
            if xpos in xvalues:
                #if xPosition is literally in the list, then just lookup the y value associated with it
                elevation=float(yvalues[xvalues.index(xpos)])
                ##print 'found it in the list'
            else:
                #use a simple linear interpolation to calculate the elevation between two points.
                x1=xvalues[bisect.bisect(xvalues,xpos)-1]
                x2=xvalues[bisect.bisect(xvalues,xpos)]
                y1=yvalues[bisect.bisect(xvalues,xpos)-1]
                y2=yvalues[bisect.bisect(xvalues,xpos)]
                elevation = float(interpolate(x1,x2,y1,y2,xpos))
                ##print 'had to interpolate'
        else:
            elevation = float(0)
            print 'outta bounds'
        return(elevation)
    
    
    def update_angle(xpos,base,fTireDiam,rTireDiam):
        ##from math import pi, cos
        ##from numpy.testing import assert_almost_equal as aae
        
        'Update vehicle angle based on current x-position. Returns angle between wheelbase and y=0 line.'
        r_elevation = elevation(xpos)
        
        interval1 = (0.002*pi) #Results in 500 loops, 0.01 is 100 loops, 0.001 is 1000 loops
            
        for i in arange(-pi/2, pi/2, interval1): 
            r_axle = r_elevation + (rrad)
            
            f_axleY = r_axle + base*sin(i)
            front_X = xpos + base*cos(i)
        
            Ycon_Pt = f_axleY - (frad)
            f_elevation = elevation(front_X)
                
            try:
                aae(f_elevation, Ycon_Pt, decimal=2)
            except AssertionError:
                continue                
            else:
                return i
                
    return update_angle(base,frad,rrad,xpos)
##--------------##

'Normal Forces'
##--------------##
def Norm_f(accl,base,grade,grav,high,mass,wdis):
    return mass*grav*((1-wdis)*cos(grade))+(high/base)*sin(grade)-mass*accl*(high/base)

def Norm_r(accl,base,grade,grav,high,mass,wdis):
    return mass*grav*((wdis)*cos(grade))-(high/base)*sin(grade)+mass*accl*(high/base)
##--------------##
'Resistance Forces'
##--------------##
def aero(dens,area,velo,drag):
    # currently assumes that the wind's velocity = 0, therefore relative velocity = car velocity
    return (dens*area*drag*velo**2)/2
                      
def hill(grade,mass,grav):
    return mass*grav*sin(grade)

def rres(Fn,velo):
    rolling_coeff = (1e-6*velo**3)-(2e-5*velo**2)+(0.0004*velo)+0.005
    return Fn*rolling_coeff
##--------------##

'Magic Formula Class'
##--------------##
class Tires:
    
    def __init__(self, rotational_inertia, effective_radius, condition='dry'):
        self.J = rotational_inertia
        self.r = effective_radius
        self.condition = condition
    def calculate_slip_ratio(self, vehicle_velocity, hub_angular_velocity, velocity_threshold = 0.1):
        'Calcualate the Slip Ratio of the tire. Incorporates a minimum velocity_threshold to avoid singular evolution at zero.'
        wheel_slip_velocity = self.r * hub_angular_velocity - vehicle_velocity        
        if abs(vehicle_velocity) < velocity_threshold:
            wheel_slip_ratio = 2 * wheel_slip_velocity / ( velocity_threshold + (vehicle_velocity**2)/velocity_threshold )
        else:
            wheel_slip_ratio = wheel_slip_velocity / vehicle_velocity
        return wheel_slip_ratio
    def magic_formula(self, slip_ratio):
        'Calculate frictional coefficient mu given condition. Options are dry, wet, snow, or ice.'
        if slip_ratio >= 1.0:
            slip_ratio = 1.0
        if slip_ratio <= -1.0:
            slip_ratio = -1.0
        if self.condition.lower() == 'wet':
            B = 12.0
            #C = 2.0
            D = 0.82
            E = 1.00
            ##----------##
            # From "Simulation of Vehicle Longitudinal Dynamics"
            C = 2.3
        elif self.condition.lower() == 'snow':
            B = 5.00
            C = 2.00
            D = 0.30
            E = 1.00
        elif self.condition.lower() == 'ice':
            B = 4.00
            C = 2.00
            D = 0.10
            E = 1.00
        elif self.condition.lower() == 'wiki':
            B = 0.714
            C = 1.40
            D = 1.00
            E = -0.20
        else:   #dry conditions are assumed if user input is anything other than 'wet' 'snow' or 'ice'
            #B = 10.0
            #C = 1.5
            D = 1.0
            #E = 1.0
            ##----------##
            # From "Simulation of Vehicle Longitudinal Dynamics"
            B = 10
            C = 1.9
            E = 0.97

        # Calculation (see pacejka '94)
        ## adhesion_coefficient = D * sin(C* atan(B*(1-E)*slip_ratio+E) * slip_ratio + E * atan(B*slip_ratio))
        
        adhesion_coefficient = D*sin(C*atan(B*slip_ratio-E*(B*slip_ratio-atan(B*slip_ratio)))) 
        ## Above equation taken from "Simulation of Vehicle Longitudinal Dynamics" paper.
        return adhesion_coefficient
##--------------##

'Slip Ratio'
##--------------##

def slipRatio(mxRM,velo,rrad,wAcc,gear,mxPT):
    return ((wAcc*rrad)-velo)/max((mxPT*gear*rrad),(((mxRM/gear)*2*pi*rrad)/60))
##--------------##

'Acceleration Run -- While Loop'
##--------------##

# Resetting Parameters
accl = 0.00 #initial acceleration in m/s^2
mRPM = 0.00 #initial Motor RPM value
wRPM = 0.00 #initial Wheel RPM value
velo = 0.00 #initial velocity in m/s
xpos = 0.00 #initial X-position (total distance)
lpos = 0.00 #initial X-position relative to the current lap
time = 0.00 #initial time in seconds (Start time)
wAcc = 0.00 #initial angular accerelation
wVel = 0.00 #initial angular velocity
n = 0

# Setting up the graphing
tim = []
xvel = []
xpot = []
xacc = []
rpms = []
slip = []
fx = []
fx1 = []
fx2 = []

# Specific values for the acceleration run
thrt = 100
grade = 0

drytire = Tires(rotational_inertia=1.5, effective_radius=rrad)

while (xpos < (75)): # Will run until 75 m is reached
    n += 1
    t1 = time
    t2 = time + (1/hertz)
    
    # Initial position set up
    Fn_r = Norm_r(accl,base,grade,grav,high,mass,wdis)
    Fn_f = Norm_f(accl,base,grade,grav,high,mass,wdis)
    
    # Resistive forces
    AR = aero(dens,area,velo,drag)
    HR = hill(grade,mass,grav)
    RR_f = rres(Fn_f,velo)
    RR_r = rres(Fn_r,velo)
    
    # Drive Forces
    DT_t = driveTorque(gear,mRPM,mxRM,thrt)
    
    #Tire Dynamics
    wAcc = AngularAcceleration(DT_t)
    wVel = AngularVelocity(wAcc,wVel,hertz) 
    slip_ratio = slipRatio(mxRM,velo,rrad,wAcc,gear,mxPT)     
    mu = drytire.magic_formula(slip_ratio)
    
    Fx_1 = (DT_t/rrad)-AR-HR-RR_f-RR_r
    Fx_2 = mu*Fn_r
    Fx = min(Fx_1,Fx_2)
    
    # Kinematics // Update xpos
    accler = Fx / mass
    velo += accler*(t2-t1)
    xpos += velo*(t2-t1)+accler*(t2-t1)**2
    # end of loop functions
    mRPM = motorRPM(gear,rrad,velo)
    time = t2
    
    tim.append(t2)
    xpot.append(xpos)
    xvel.append(velo)
    xacc.append(accler)
    slip.append(slip_ratio)
    fx.append(Fx)
    fx1.append(Fx_1)
    fx2.append(Fx_2)        
    if (debug == 'true'):    
        print "Velocity: %5.3f, Max Velocity: %5.3f" % (velo,(((mxRM*2*pi*rrad)/(gear*60))))
        print "Ang Accerelation: %5.5f, Max Ang: %5.5f" % (wAcc,(mxPT*gear*rrad))    
        print "Ang Velocity: %5.5f, Max Ang: %5.5f \n" % (wVel,mxPT*gear*(1/hertz))
    
print 'Acceleration Run\nAt',thrt,'% throttle,',xpos,' m were completed in',t2,' seconds.'

# Plotting the acceleration run info
plt.figure(figsize=(13,16))

plt.subplot(521)
plt.plot(tim,xpot)
plt.xlabel('time (sec)')
plt.ylabel('x-position (meters)')

plt.subplot(522)
plt.plot(tim,xvel)
plt.xlabel('time (sec)')
plt.ylabel('velocity (meters/s)')

plt.subplot(523)
plt.plot(tim,xacc)
plt.xlabel('time (sec)')
plt.ylabel('acceleration (meters/sec^2)')

plt.subplot(524)
plt.plot(tim,[j / 9.81 for j in xacc])
plt.xlabel('time (sec)')
plt.ylabel('Longitudinal G\'s')

plt.subplot(525)
plt.plot(tim,slip)
plt.xlabel('time (sec)')
plt.ylabel('abs - Slip Ratio')

plt.subplot(526)
plt.plot(tim,fx)
plt.xlabel('time (sec)')
plt.ylabel('Fx')

plt.subplot(527)
plt.plot(tim,fx2)
plt.xlabel('time (sec)')
plt.ylabel('Fx2 = mu*Fn_r')
   
plt.subplot(528)
plt.plot(tim,fx1)
plt.xlabel('time (sec)')
plt.ylabel('Fx1 (Sum of forces)')

wettire = Tires(rotational_inertia=1.5, effective_radius=rrad, condition = 'wet')
snowtire = Tires(rotational_inertia=1.5, effective_radius=rrad, condition = 'snow')
icetire = Tires(rotational_inertia=1.5, effective_radius=rrad, condition = 'ice')

slip=[]
drymulist=[]
wetmulist=[]
snowmulist=[]
icemulist=[]
for i in arange(-1,1,0.01):
    drymu = drytire.magic_formula(i)
    wetmu = wettire.magic_formula(i)
    snowmu = snowtire.magic_formula(i)
    icemu = icetire.magic_formula(i)
    
    slip.append(i)
    drymulist.append(drymu)
    wetmulist.append(wetmu)
    snowmulist.append(snowmu)
    icemulist.append(icemu)

plt.subplot(529)
plt.plot(slip,drymulist,label='Dry Condition')
plt.plot(slip,wetmulist,label='Wet Condition')
plt.plot(slip,snowmulist,label='Snow Condition')
plt.plot(slip,icemulist,label='Ice Condition')
plt.xlabel('Slip Ratio %')
plt.ylabel('Coefficient of Adhesion')
plt.legend(loc=4)
plt.title('Coef. of Adhesion vs Slip Ratio',size=15)

peak = []
cont = []

for i in range(0,6000):
    peak.append(Ptor(i))
    cont.append(Ctor(i))
    
plt.subplot(5,2,10)
plt.plot(peak,label='Peak')
plt.plot(cont,label='Cont')

plt.title('Torque Curves')
plt.xlabel('RPM')
plt.ylabel('Torque (Nm)')

plt.legend()

plt.tight_layout()
plt.draw();
