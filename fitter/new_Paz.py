import numpy as np

from scipy.interpolate import interp1d, UnivariateSpline
from numpy import sqrt, log10
import matplotlib.pyplot as plt


def vec_Paz(model, az_data, R_data, jns, current_pulsar):
    """ 
        Computes probability distribution for a range of line of sight accelerations at projected R : P(az|R)
        Returns the an array containing the probability distribution.  
        """

    # Under construction !!!

    # Return array of P(az|R)
    Paz_dist = []

    az_data = abs(az_data)  # Consider only positive values

    # Assumes units of az [m/s^2] if model.G == 0.004302, else models units
    # Conversion factor from [pc (km/s)^2/Msun] -> [m/s^2]
    az_fac = 1.0 / 3.0857e10 if (model.G == 0.004302) else 1

    # This flag will be set to false in the case of an interpolation error,
    # we will then ignore that point and not append to the list.
    append = True

    if R_data < model.rt:
        # Bumping this up by even 100x does not fix the z2 interpolation error.
        nz = model.nstep
        zt = sqrt(model.rt ** 2 - R_data ** 2)  # maximum z value at R

        # Original Implementation
        z = np.logspace(log10(model.r[1]), log10(zt), nz)

        # This was an attempt to make the z array more finely sampled
        # in the hope that the z2 interpolation errors would be eliminated.
        # In practice it seems to reduce the frequency of the errors however this is
        # very difficult to quantify.

        # Merge with another array to ensure z is finely enough sampled
        # z_extra = np.linspace(0, 8, 10000)
        # print("z_extra")
        # print(z_extra)
        # z_merge = np.r_[z, z_extra]
        # print("merged z")
        # print(z_merge)
        # z = np.sort(z_merge)
        # print("sorted z")
        # print(z)
        # nz = len(z)

        spl_Mr = UnivariateSpline(
            model.r, model.mc, s=0, ext=1
        )  # Spline for enclosed mass

        r = sqrt(R_data ** 2 + z ** 2)  # Local r array
        az = model.G * spl_Mr(r) * z / r ** 3  # Acceleration along los
        az[-1] = (
            model.G * spl_Mr(model.rt) * zt / model.rt ** 3
        )  # Ensure non-zero final data point

        az *= az_fac  # convert to [m/s^2]
        az_spl = UnivariateSpline(
            z, az, k=4, s=0, ext=1
        )  # 4th order needed to find max (can be done easier?)

        zmax = (
            az_spl.derivative().roots()
        )  # z where az = max(az), can be done without 4th order spline?
        azt = az[-1]  # acceleration at the max(z) = sqrt(r_t**2 - R**2)

        # Setup spline for rho(z)
        if jns == 0 and model.nmbin == 1:
            rho = model.rho
        else:
            rho = model.rhoj[jns]

        rho_spl = UnivariateSpline(model.r, rho, ext=1, s=0)
        rhoz = rho_spl(sqrt(z ** 2 + R_data ** 2))
        rhoz_spl = UnivariateSpline(z, rhoz, ext=1, s=0)

        # Now compute P(a_z|R)
        # There are 2 possibilities depending on R:
        #  (1) the maximum acceleration occurs within the cluster boundary, or
        #  (2) max(a_z) = a_z,t (this happens when R ~ r_t)

        nr, k = nz, 3  # bit of experimenting

        # Option (1): zmax < max(z)
        if len(zmax) > 0:
            zmax = zmax[0]  # Take first entry for the rare cases with multiple peaks
            # Set up 2 splines for the inverse z(a_z) for z < zmax and z > zmax
            z1 = np.linspace(z[0], zmax, nr)

            # What we want is a continuation of the acceleration space past the zmax point
            # so that we come to the z2 point for which we can calculate a separate probability.
            # The z2 point needs to be calculated separately because the calculation depends on
            # the density, which is diffrent at each z point.
            # Unclear why original implementation doesn't always work, seems perfectly fine.
            # The reason for the reverse is that it needs to be strictly increasing for the spline

            z2 = (np.linspace(zmax, z[-1], nr))[::-1]

            z1_spl = UnivariateSpline(az_spl(z1), z1, k=k, s=0, ext=1)

            # Original implementation
            # changing the spline here doesn't fix the z2 interpolation error.
            z2_spl = UnivariateSpline(az_spl(z2), z2, k=k, s=0, ext=1)

        # Option 2: zmax = max(z)
        else:
            zmax = z[-1]
            z1 = np.linspace(z[0], zmax, nr)
            z1_spl = UnivariateSpline(az_spl(z1), z1, k=k, s=0, ext=1)

        # Maximum acceleration along this los
        azmax = az_spl(zmax)

        # Now determine P(az_data|R)
        for az_point in az_data:
            if az_point < azmax:
                z1 = max([z1_spl(az_point), z[0]])  # first radius where az = az_data

                Paz = rhoz_spl(z1) / abs(az_spl.derivatives(z1)[1])

                if az_point > azt:
                    # Find z where a_z = a_z,t

                    # Ideally there would be a check to see if the z2 spline is actually defined
                    # but in practice this has never actually occurred.
                    z2 = z2_spl(az_point)
                    append = True

                    # This is the check that detects the z2 interpolation error,
                    # if it fails we just discard the point.
                    if z2 < zt:
                        Paz += rhoz_spl(z2) / abs(az_spl.derivatives(z2)[1])
                        append = True

                    else:
                        # Discard the point
                        append = False

                        # Debug Plots

                        # print("z2 point outside of truncation radius")
                        # print(current_pulsar)

                        # plt.figure()
                        # # plt.loglog()
                        # plt.plot(az, z2_spl(az), ".--")
                        # plt.title("z2 spline of az vs az")
                        # plt.xlabel("az")
                        # plt.axvline(az_point, color="orange", label="az_point")
                        # plt.axhline(z1_spl(az_point), color="yellow", label="z1spl")
                        # # plt.axhline(z2_spl(az_point), color="green", label="z2spl")
                        # # plt.axhline(zt, color="red", label="zt")
                        # plt.ylabel("z2_spl(az)")
                        # plt.legend()
                        # plt.savefig("z2spl_az.png")

                        # plt.figure()
                        # # plt.loglog()
                        # plt.plot(z2_spl(az), az, ".--")
                        # plt.xlim(0, 10)
                        # plt.title("az vs z2 spline of az")
                        # plt.xlabel("z2_spl(az)")
                        # plt.axhline(az_point, color="orange", label="az_point")
                        # plt.axvline(z1_spl(az_point), color="yellow", label="z1spl")
                        # # plt.axvline(z2_spl(az_point), color="green", label="z2spl")
                        # # plt.axvline(zt, color="red", label="zt")
                        # plt.ylabel("az")
                        # plt.legend()
                        # plt.savefig("az_z2spl.png")

                        # plt.figure()
                        # # plt.loglog()
                        # plt.plot(z2_spl(az), az, ".--", label="az vs z2spl")
                        # plt.plot(z, az, ".--", label="az vs z", alpha=0.5)
                        # plt.xlim(0, 10)
                        # plt.title("az vs z2 spline of az, az vs z also plotted")
                        # plt.xlabel("z2_spl(az)")
                        # plt.axhline(az_point, color="orange", label="az_point")
                        # plt.axvline(z1_spl(az_point), color="yellow", label="z1spl")
                        # # plt.axvline(z2_spl(az_point), color="green", label="z2spl")
                        # # plt.axvline(zt, color="red", label="zt")
                        # plt.ylabel("az")
                        # plt.legend()
                        # plt.savefig("az_z2spl_az_z.png")

                        # plt.figure()
                        # # plt.loglog()
                        # plt.plot(z2_spl(az), az, ".--", label="az vs z2spl")
                        # plt.plot(z, az, ".--", label="az vs z", alpha=0.5)
                        # # plt.xlim(0, 10)
                        # plt.title("az vs z2 spline of az, az vs z also plotted nolim")
                        # plt.xlabel("z2_spl(az)")
                        # plt.axhline(az_point, color="orange", label="az_point")
                        # plt.axvline(z1_spl(az_point), color="yellow", label="z1spl")
                        # # plt.axvline(z2_spl(az_point), color="green", label="z2spl")
                        # # plt.axvline(zt, color="red", label="zt")
                        # plt.ylabel("az")
                        # plt.legend()
                        # plt.savefig("az_z2spl_az_z_nolim.png")

                        # plt.figure()
                        # plt.loglog()
                        # plt.plot(z2_spl(az), az, ".--", label="az vs z2spl")
                        # plt.plot(z, az, ".--", label="az vs z", alpha=0.5)
                        # plt.xlim(0, 10)
                        # plt.title("az vs z2 spline of az, az vs z also plotted - log")
                        # plt.xlabel("z2_spl(az)")
                        # plt.axhline(az_point, color="orange", label="az_point")
                        # plt.axvline(z1_spl(az_point), color="yellow", label="z1spl")
                        # # plt.axvline(z2_spl(az_point), color="green", label="z2spl")
                        # # plt.axvline(zt, color="red", label="zt")
                        # plt.ylabel("az")
                        # plt.legend()
                        # plt.savefig("az_z2spl_az_z_log.png")

                        # plt.figure()
                        # # plt.loglog()
                        # plt.xlim(0, 10)
                        # plt.plot(z1_spl(az), az, ".--", label="az vs z1spl")
                        # plt.plot(z, az, ".--", label="az vs z", alpha=0.5)
                        # plt.title("az vs z1 spline of az, az vs z also plotted")
                        # plt.xlabel("z1_spl(az)")
                        # plt.axhline(az_point, color="orange", label="az_point")
                        # plt.axvline(z1_spl(az_point), color="yellow", label="z1spl")
                        # # plt.axvline(z2_spl(az_point), color="green", label="z2spl")
                        # # plt.axvline(zt, color="red", label="zt")
                        # plt.ylabel("az")
                        # plt.legend()
                        # plt.savefig("az_z1spl_az_z.png")

                        # plt.figure()
                        # # plt.loglog()
                        # plt.plot(z1_spl(az), az, ".--")
                        # plt.xlim(0, 10)
                        # plt.title("az vs z1 spline of az")
                        # plt.xlabel("z1_spl(az)")
                        # plt.axhline(az_point, color="orange", label="az_point")
                        # plt.axvline(z1_spl(az_point), color="yellow", label="z1spl")
                        # # plt.axvline(z2_spl(az_point), color="green", label="z2spl")
                        # # plt.axvline(zt, color="red", label="zt")
                        # plt.ylabel("az")
                        # plt.legend()
                        # plt.savefig("az_z1spl.png")

                        # plt.figure()
                        # # plt.loglog()
                        # plt.plot(z, az, ".--")
                        # plt.xlim(0, 10)
                        # plt.title("az vs z")
                        # plt.xlabel("z")
                        # plt.ylabel("az")
                        # plt.axvline(z1_spl(az_point), color="yellow", label="z1spl")
                        # # plt.axvline(z2_spl(az_point), color="green", label="z2spl")
                        # # plt.axvline(zt, color="red", label="zt")
                        # plt.axhline(az_point, color="orange", label="az_point")
                        # plt.legend()
                        # plt.savefig("az_z.png")

                        # plt.figure()
                        # # plt.loglog()
                        # plt.plot(az, z1_spl(az), ".--")
                        # plt.title("z1 spline of az vs az")
                        # plt.xlabel("az")
                        # plt.axvline(az_point, color="orange", label="az_point")
                        # plt.axhline(z1_spl(az_point), color="yellow", label="z1spl")
                        # # plt.axhline(z2_spl(az_point), color="green", label="z2spl")
                        # # plt.axhline(zt, color="red", label="zt")
                        # plt.legend()
                        # plt.ylabel("z1_spl(az)")
                        # plt.savefig("z1spl_az.png")

                        # plt.figure()
                        # # plt.loglog()
                        # plt.xlim(0, 10)
                        # plt.plot(z, rhoz_spl(z), ".--")
                        # plt.title("rhoz spline of z vs z")
                        # plt.xlabel("z")
                        # plt.axvline(z1_spl(az_point), color="orange", label="z1spl")
                        # # plt.axvline(z2_spl(az_point), color="green", label="z2spl")
                        # # plt.axvline(zt, color="red", label="zt")
                        # plt.ylabel("rhoz_spl(z)")
                        # plt.legend()
                        # plt.savefig("rhoz_spl_z.png")

                        # raise ValueError("z2 interpolation error")

                # Normalize to 1
                # This allows us to use the artificially interted bins
                # without worrying about their very low densities.
                Paz /= rhoz_spl.integral(0, zt)

                model.z = z
                model.az = az
                model.Paz = Paz
                model.azmax = azmax
                model.zmax = zmax
            else:
                model.Paz = 0

            # If we get an interpolation error, don't keep the value.
            if append:
                Paz_dist.append(model.Paz)

        else:
            model.Paz = 0
            # Paz_dist.append(0) This doesn't seem to be needed

    return Paz_dist
