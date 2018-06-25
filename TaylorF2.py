# A standalone implementation of the TaylorF2 approximant
# The code follows the implementation in LALSuite
#
# MP 11/2017

import sampyl as smp
from sampyl import np
import warnings


class TaylorF2:
    Mf_ISCO = 1.0/(6**(3.0/2.0) * np.pi)

    @property
    def PN_v(self):
        return self.v

    @property
    def PN_vlogv(self):
        return self.vlogv

    @property
    def PN_vlogsq(self):
        return self.vlogvsq

    def __init__(self, m1, m2, chi1, chi2, qm_def1=1.0, qm_def2=1.0):
        """
        m1, m2 mass of bodies in solar masses
        chi1, chi2: dimensionless aligned spins
        """
        mtot = m1 + m2
        d = (m1 - m2) / (m1 + m2)
        eta = m1*m2/mtot/mtot
        m1M = m1/mtot
        m2M = m2/mtot

        # Use the spin-orbit variables from arXiv:1303.7412, Eq. 3.9
        # We write dSigmaL for their (\delta m/m) * \Sigma_\ell
        # There's a division by mtotal^2 in both the energy and flux terms
        # We just absorb the division by mtotal^2 into SL and dSigmaL
        SL = m1M*m1M*chi1 + m2M*m2M*chi2
        dSigmaL = d*(m2M*chi2 - m1M*chi1)
        pfaN = 3.0/(128.0 * eta) # Newtonian coefficient

        # reserve space for coefficient arrays
        PN_PHASING_SERIES_MAX_ORDER = 12
        #self.v = np.zeros(1 + PN_PHASING_SERIES_MAX_ORDER)
        self.vlogv = np.zeros(1 + PN_PHASING_SERIES_MAX_ORDER)
        self.vlogvsq = np.zeros(1 + PN_PHASING_SERIES_MAX_ORDER)


        # Non-spin phasing terms - see arXiv:0907.0700, Eq. 3.18 */
        EULER_GAMMA = 0.5772156649015328606065120900824024
        self.v = [1.0
                  ,0.0
                  ,5.0*(743.0/84.0 + 11.0 * eta)/9.0
                  ,-16.0*np.pi
                  ,5.0*(3058.673/7.056 + 5429.0/7.0 * eta + 617.0 * eta*eta)/72.0
                  ,5.0/9.0 * (7729.0/84.0 - 13.0 * eta) * np.pi
                  ,(11583.231236531/4.694215680 - 640.0/3.0 * np.pi*np.pi - 6848.0/21.0 * EULER_GAMMA)
                              + eta * (-15737.765635/3.048192 + 2255.0/12.0 * np.pi*np.pi)
                               + eta*eta * 76055.0/1728.0
                               - eta*eta*eta * 127825.0/1296.0
                               + (-6848.0/21.0)*np.log(4.0)
                  , np.pi * ( 77096675.0/254016.0 + 378515.0/1512.0 * eta - 74045.0/756.0 * eta*eta) ]
        self.v = np.array(self.v)


        # Compute 2.0PN SS, QM, and self-spin
        # See Eq. (6.24) in arXiv:0810.5336
        # 9b,c,d in arXiv:astro-ph/0504538

        # aligned spin
        chi1sq = chi1*chi1
        chi2sq = chi2*chi2
        chi1dotchi2 = chi1*chi2

        pn_sigma = eta * (721.0/48.0*chi1*chi2 - 247.0/48.0*chi1dotchi2);
        pn_sigma = pn_sigma + (720.0*qm_def1 - 1.0)/96.0 * m1M * m1M * chi1 * chi1
        pn_sigma = pn_sigma + (720.0*qm_def2 - 1.0)/96.0 * m2M * m2M * chi2 * chi2
        pn_sigma = pn_sigma -  (240.0*qm_def1 - 7.0)/96.0 * m1M * m1M * chi1sq
        pn_sigma = pn_sigma - (240.0*qm_def2 - 7.0)/96.0 * m2M * m2M * chi2sq

        pn_ss3 =  (326.75/1.12 + 557.5/1.8*eta) * eta*chi1*chi2
        pn_ss3 = pn_ss3 + ((4703.5/8.4 + 2935.0/6.0*m1M - 120.0*m1M*m1M) * qm_def1 + (-4108.25/6.72 - 108.5/1.2*m1M + 125.5/3.6*m1M*m1M)) * m1M*m1M * chi1sq
        pn_ss3 = pn_ss3 + ((4703.5/8.4 + 2935.0/6.0*m2M - 120.0*m2M*m2M) * qm_def2 + (-4108.25/6.72 - 108.5/1.2*m2M + 125.5/3.6*m2M*m2M)) * m2M*m2M * chi2sq


        # Spin-orbit terms - can be derived from arXiv:1303.7412, Eq. 3.15-16 */
        pn_gamma = (554345.0/1134.0 + 110.0*eta/9.0)*SL + (13915.0/84.0 - 10.0*eta/3.0)*dSigmaL
        correction = [ 0
                      ,0
                      ,0
                      ,188.0*SL/3.0 + 25.0*dSigmaL
                      , -10.0 * pn_sigma
                      , -1.0 * pn_gamma
                      , np.pi * (3760.0*SL + 1490.0*dSigmaL)/3.0 + pn_ss3
                      , (-8980424995.0/762048.0 + 6586595.0*eta/756.0 - 305.0*eta*eta/36.0)*SL - (170978035.0/48384.0 - 2876425.0*eta/672.0 - 4735.0*eta*eta/144.0) * dSigmaL]
        correction = np.array(correction)
        self.v = self.v + correction

        # multiply all coefficients by pfaN

        self.v = self.v * pfaN
        self.vlogv = self.vlogv * pfaN
        self.vlogvsq =  self.vlogvsq * pfaN


    def compute_phasing(self, Mf_array, Mf0):
        """
        Compute PN phasing from stored PN coefficients
        Mf_array: input array of geometric frequencies
        Mf0: reference frequency
        """
        if max(Mf_array) > self.Mf_ISCO:
            warnings.warn("Maximum geometric frequency {} is above ISCO".format(max(Mf_array)))

        PN_phase = []
        for Mf in Mf_array:
            v = (np.pi * (Mf/Mf0))**(1.0/3.0)
            logv = np.log(v)

            v2 = v * v;
            v3 = v * v2;
            v4 = v * v3;
            v5 = v * v4;
            v6 = v * v5;
            v7 = v * v6;
            v8 = v * v7;
            v9 = v * v8;
            v10 = v * v9;
            v12 = v2 * v10;
            phasing = 0.0;

            phasing = phasing + self.v[7] * v7
            phasing = phasing + (self.v[6] + self.vlogv[6] * logv) * v6
            phasing = phasing + (self.v[5] + self.vlogv[5] * logv) * v5
            phasing = phasing + self.v[4] * v4
            phasing = phasing +  self.v[3] * v3
            phasing = phasing + self.v[2] * v2
            phasing = phasing + self.v[1] * v
            phasing = phasing + self.v[0]

            phasing = np.divide(phasing , v5)
            PN_phase.append(phasing)

        return np.array(PN_phase)

    def compute_strain(self, Mf, Mf0, A0):
        """
        Newtonian point-particle amplitude where h(f) = h_+(f) + i h_x(f).
        Mf: array of geometric frequencies
        Mf0: reference frequency
        A0: amplitude factor
        """
        phi = self.compute_phasing(Mf, Mf0)
        strain = A0 * (Mf/Mf0)**(-7.0/6.0) * np.exp(1j * phi)
        return strain
