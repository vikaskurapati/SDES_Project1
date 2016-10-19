
# coding: utf-8

## SDES Project1: LC Tank with a resistance(constant voltage applied)

# In[2]:

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


####### C is capacitance of the capacitor, L is inducatance of inductor, R is resistor's resistance,V is the voltage applied, $V_0$ is initial voltage across capacitor and $I_0$ being initial current across the inductor, $t_f$ is the total time and n is the number of time steps

# In[3]:

def problem(C,L,R,V,V0,I0,tf,n):
    det = ((R*R)/(L*L)) - (4/(L*C)) + 0.0j
    t = np.linspace(0.,tf,n)
    if abs(det) == 0:
        D1 = (I0/C)*np.exp(1/(2.0*R*C))
        D2 = (V0 - V)*np.exp(1/(2.0*R*C))
        V_matrix = V + np.exp(-(1/2.0*R*C)*t)*(D1*t + D2) 
    else :
        s1 = -0.5*R/L + 0.5*np.sqrt(det)
        s2 = -0.5*R/L - 0.5*np.sqrt(det)
        A1 = ((I0/C) - (V0 - V)*s2)/(s1 - s2)
        A2 = ((I0/C) - (V0 - V)*s1)/(s2 - s1)
        V_matrix = V + A1*np.exp(s1*t) + A2*np.exp(s2*t)
    return V_matrix,t



# In[4]:

def plot(t,V):
    plt.plot(t,V)
    plt.show()


# In[ ]:



def animate(X,Y,name):
    fig = plt.figure()
    ax = plt.axes(xlim=(np.amin(X), np.amax(X)+0.1), ylim=(np.amin(Y)-0.1, np.amax(Y)+0.1))
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([],[])
        return line,
    def animate(i):
        x = X[:i]
        y = Y[:i]
        #print x,y
        line.set_data(x, y)
        return line,
    
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=200, interval=20, blit=True)
    anim.save(name+'.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    plt.show()

    


# In[6]:

if __name__ == '__main__':
    V,t = problem(1.0, 1.0, 1.0, 10.0, 10.0, 1.0, 20.0, 200)
    plot(t,V)
    animate(t,V,'Voltage')

