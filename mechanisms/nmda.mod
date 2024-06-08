: from Durstewitz & Gabriel (2006), Cerebral Cortex

TITLE NMDA synapse 

NEURON {
	POINT_PROCESS NMDA
	NONSPECIFIC_CURRENT i
        RANGE g,a,b,tauD,tauF,util,tcon,tcoff,e
}

UNITS {
        (uS) = (microsiemens)
        (nA) = (nanoamp)
        (mV) = (millivolt)
}

PARAMETER {
	tcon = 2.3 (ms)
	tcoff = 95.0 (ms)
	e = 0 	(mV)
        tauD = 100         (ms)
        tauF = 100         (ms)
        util= .3
}

ASSIGNED {
	v 	(mV)
	i	(nA)
	g       (uS)
	factor
}

INITIAL { 
   a=0  
   b=0 
   factor=tcon*tcoff/(tcoff-tcon)
}

STATE {
      a
      b
}

BREAKPOINT {
	LOCAL s
	SOLVE states METHOD derivimplicit
	s = 1.50265/(1+0.33*exp(-0.0625*v))
        g = b-a
	i = g*s*(v-e)
        :printf("at t = %g: s = %g i = %g\n", t, s, i)
}

DERIVATIVE states {
	a' = -a/tcon
	b' = -b/tcoff
}

NET_RECEIVE(wgt,g_unit,R,u,tlast (ms),nspike) { LOCAL x
   INITIAL {
     nspike = 0
     R = 1
     u = util
     tlast = 0
   }
        
   :printf("entry flag=%g t=%g\n", flag, tlast)
   if (tauF>0) { u=util+(1-util)*u*exp(-(t-tlast)/tauF) }
   if (tauD>0) { R=1+(R*(1-u)-1)*exp(-(t-tlast)/tauD) }

   x=wgt*g_unit*factor*R*u
   a = a+x
   b = b+x
   tlast = t
   nspike = nspike+1
}