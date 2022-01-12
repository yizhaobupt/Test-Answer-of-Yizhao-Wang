import numpy as np
import math as m
# from sympy import *
import matplotlib.pyplot as plt

Earray=[]       # store each E value at each time slot
Sarray=[]
ESarray=[]
Parray=[]
array=[]        #store each time slot
k1 = 100
k2 = 600
k3 = 150


def St(E,S,ES,P):
    y_deri = - k1*E*S + k2 * ES
    return y_deri

def ESt(E,S,ES,P):
    y_deri = k1*E*S - (k2+k3)*ES
    return y_deri

def Et(E,S,ES,P):
    y_deri = - k1*E*S + (k2+k3)*ES
    return y_deri

def Pt(E,S,ES,P):
    y_deri = k3*ES
    return y_deri
    

def RK4():
    h=0.0005            # time piece
    a=0                 # time label
    time_all = 0.02     # time total
    
    E = 1               # original concentration
    S = 10
    ES = 0
    P = 0
    while a<=time_all:  
        array.append(a)
 
        Earray.append(E)
        Sarray.append(S)
        ESarray.append(ES)
        Parray.append(P)
        a+=h

        E1=Et(E,S,ES,P) # step 1 of Runge-Kutta
        x1=E + E1*h/2
        S1=St(E,S,ES,P)
        y1=S + S1*h/2
        ES1=ESt(E,S,ES,P)
        z1=ES + ES1*h/2
        P1=Pt(E,S,ES,P)
        w1=P + P1*h/2

        E2=Et(x1,y1,z1,w1)  # step 2 of Runge-Kutta
        x2=E + E2*h/2
        S2=St(x1,y1,z1,w1)
        y2=S + S2*h/2
        ES2=ESt(x1,y1,z1,w1)
        z2=ES + ES2*h/2
        P2=Pt(x1,y1,z1,w1)
        w2=P + P2*h/2

        E3=Et(x2,y2,z2,w2)  # step 3 of Runge-Kutta
        x3=E + E3*h
        S3=St(x2,y2,z2,w2)
        y3=S + S3*h
        ES3=ESt(x2,y2,z2,w2)
        z3=ES + ES3*h
        P3=Pt(x2,y2,z2,w2)
        w3=P + P3*h
        
        E4=Et(x3,y3,z3,w3)  # step 4 of Runge-Kutta
        S4=St(x3,y3,z3,w3)
        ES4=ESt(x3,y3,z3,w3)
        P4=Pt(x3,y3,z3,w3)

        E = E + (E1 + 2*E2 + 2*E3 +E4) * h/6
        S = S + (S1 + 2*S2 + 2*S3 +S4) * h/6
        ES = ES + (ES1 + 2*ES2 + 2*ES3 +ES4) * h/6
        P = P + (P1 + 2*P2 + 2*P3 +P4) * h/6


def main():
    RK4()
    plt.figure('1', figsize=(10, 8))

    plt.subplot(221)                        # the relation between time and E's concentration
    plt.xlabel("t",size=12)
    plt.ylabel("E",size=12)
    plt.scatter(array, Earray, alpha=0.6)

    plt.subplot(222)                        # the relation between time and S's concentration
    plt.xlabel("t",size=12)
    plt.ylabel("S",size=12)
    plt.scatter(array, Sarray, alpha=0.6)

    plt.subplot(223)                        # the relation between time and ES's concentration
    plt.xlabel("t",size=12)
    plt.ylabel("ES",size=12)
    plt.scatter(array, ESarray, alpha=0.6)

    plt.subplot(224)                        # the relation between time and P's concentration
    plt.xlabel("t",size=12)
    plt.ylabel("P",size=12)
    plt.scatter(array, Parray, alpha=0.6)
    plt.show()

    plt.figure('2', figsize=(10, 8))

    plt.subplot(221)                        # the relation between ES' and E's concentration
    plt.xlabel("E",size=12)
    plt.ylabel("ES",size=12)
    plt.scatter(Earray, ESarray, alpha=0.6)

    plt.subplot(222)                        # the relation between P's and E's concentration
    plt.xlabel("E",size=12) 
    plt.ylabel("P",size=12)
    plt.scatter(Earray, Parray, alpha=0.6)

    plt.subplot(223)                        # the relation between S' and P's concentration
    plt.xlabel("S",size=12)
    plt.ylabel("P",size=12)
    plt.scatter(Sarray, Parray, alpha=0.6)

    plt.subplot(224)                        # the relation between ES's and P's concentration
    plt.xlabel("ES",size=12)
    plt.ylabel("P",size=12)
    plt.scatter(ESarray, Parray, alpha=0.6)
    plt.show()


if __name__ == "__main__":
    main()