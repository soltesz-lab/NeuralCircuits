
TITLE Morris-Lecar spiking dynamics

NEURON {
  SUFFIX ml
  USEION k READ ek WRITE ik
  USEION ca READ eca WRITE ica
  NONSPECIFIC_CURRENT il
  RANGE gcabar, gkbar, gleak, el, gl, ica, ik, il, w, winit
  RANGE phi, betam, gammam, betaw, gammaw
  RANGE minf, winf, tauw
}

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S)  = (siemens)
}

PARAMETER {

	gcabar = 0.004 (S/cm2)
	gkbar  = 0.008 (S/cm2)

  gl  = 2e-3  (S/cm2)

  el = -70 (mV)

  phi = 0.067

  betam  = -1.2 (mV)
  gammam = 18 (mV)

  betaw  = 12 (mV)
  gammaw = 17 (mV)
  winit = 0
}

ASSIGNED {
  ica (mA/cm2)
  ik (mA/cm2)
  il (mA/cm2)
  gk (S/cm2)
  gca (S/cm2)
  minf
  winf
  tauw (ms)
  ek (mV)
  eca (mV)
  m
  v (mV)
}

STATE {
  w
}

INITIAL {
	rates(v)
	m = minf
	w = winit
}

BREAKPOINT {

  SOLVE states METHOD cnexp
  gca = gcabar * m
  gk  = gkbar * w
  ik  = gk * (v - ek)
  ica = gca * (v - eca)
  il  = gl * (v - el)
}

DERIVATIVE states {
	rates(v)
	w' = phi*(winf-w)/tauw
}

PROCEDURE rates( v (mV) ) {

  minf = 0.5*(1 + tanh((v-betam)/gammam))
  winf = 0.5*(1 + tanh((v-betaw)/gammaw))
  tauw = 1/(cosh((v-betaw)/(2*gammaw)))
  m = minf
}
