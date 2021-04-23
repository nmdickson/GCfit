import string
import random

import numpy as np
import astropy.units as u
import shapely.geometry as geom
import shapely.prepared as prepgeom


__all__ = [
    "initialize_fields",
    "Field",
]


def initialize_fields(fields_var, cen):
    '''
    return a list of Field objects, one for each PI
    each field, make the coords use dset's 'fields_unit'

    cen is a tuple, aka (obs.mdata['RA'], obs.mdata['DEC'])

    returns fields: dict of PI:Field pairs, for each PI in
    '''

    PI_list = np.unique(fields_var).astype(str)
    fields = dict()

    unit = fields_var.mdata['field_unit']

    for PI in PI_list:

        # Single polygon
        try:
            coords = fields_var.mdata[PI]
        except KeyError:
            # try gathering all the '_a, _b' style fields
            coords = []
            for ch in string.ascii_letters:
                try:
                    coords.append(fields_var.mdata[f'{PI}_{ch}'])
                except KeyError:
                    # once it stops working, we're done here
                    break

            if not coords:
                mssg = f"PI {PI} has no field bounds in `fields_var.mdata`"
                raise RuntimeError(mssg)

        fields[PI] = Field(coords, cen=cen, unit=unit)

    return fields


class Field:
    '''helper class to handle the polygons for mf fields

    preprep means the preped polygons will be made once at the start, which
    obviously saves on the time taken to prep them, but this geometry cannot be
    pickled, so if you want to pickle Field (i.e. for mpi) this will fail
    '''

    def __contains__(self, other):
        if self._multi:
            return any(p.contains(other) for p in self._prepped)
        else:
            return self._prepped.contains(other)

    def _correct(self, crd, cen=(0, 0), unit=None):
        '''centre, convert and RA correct the given array of crds'''
        RA, DEC = crd[:, 0], crd[:, 1]

        # Add and convert units to arcmin
        if not hasattr(crd, 'unit'):
            RA, DEC = RA << u.Unit(unit), DEC << u.Unit(unit)

        cen <<= RA.unit

        # Centre and correct RA
        RA, DEC = (RA - cen[0]) * np.cos(DEC), DEC - cen[1]

        RA, DEC = RA.to_value(u.arcmin), DEC.to_value(u.arcmin)

        return np.c_[RA, DEC]

    def prep(self):
        if self._multi:
            self._prepped = [prepgeom.prep(p) for p in self.polygon]
        else:
            self._prepped = prepgeom.prep(self.polygon)

        return self._prepped

    def __init__(self, coords, cen=(0, 0), unit=None, preprep=False):

        # ------------------------------------------------------------------
        # Parse the coords argument
        # ------------------------------------------------------------------

        # if already a polygon, assume its already been corrected
        if isinstance(coords, geom.Polygon):
            self._multi = False
        elif isinstance(coords, geom.MultiPolygon):
            self._multi = True

        # is a single polygon of coordinates
        elif isinstance(coords, np.ndarray) and coords.ndim == 2:
            self._multi = False
            coords = self._correct(coords, cen, unit)

        # assume it's iterable of coords or polygons
        else:
            self._multi = True
            coords = [self._correct(c, cen, unit) for c in coords]

        # ------------------------------------------------------------------
        # Set up the polygons
        # ------------------------------------------------------------------

        # TODO prepared geometries can't be pickled, shit

        if self._multi:
            polys = [geom.Polygon(c).buffer(0) for c in coords]

            self.polygon = geom.MultiPolygon(polys)

        else:
            self.polygon = geom.Polygon(coords).buffer(0)

        if preprep:
            self.prep()

        self.area = self.polygon.area << u.arcmin**2

    def slice_radially(self, r1, r2):
        '''Return a new field which is this field and a radial slice'''
        # make sure that r1,r2 are in arcmin
        r1, r2 = r1.to_value('arcmin'), r2.to_value('arcmin')

        origin = geom.Point((0, 0))

        shell = origin.buffer(r2) - origin.buffer(r1)

        return Field(self.polygon & shell)

    def MC_sample(self, M, return_points=False):
        '''Random sampling of `M` points from this field
        will create a new sample every time,
        (psst, will also store last_sample in self._prev_sample, fyi)

        if return_points is True, returns a MultiPoint, otherwise returns the
        r values for all the points, in arcmin
        '''

        def rejection_sample(poly, prep_poly, M_i):

            if poly.is_empty:
                return []

            minx, miny, maxx, maxy = poly.bounds

            points = []

            while len(points) < M_i:
                rand_x = random.uniform(minx, maxx)
                rand_y = random.uniform(miny, maxy)

                test_pnt = geom.Point(rand_x, rand_y)

                if prep_poly.contains(test_pnt):
                    points.append(test_pnt)

            return points

        # If this field hasn't been prepped yet, do that now, once
        if not hasattr(self, '_prepped'):
            self.prep()

        # Do each poly of a multipoly seperate so the bounds aren't huge
        if self._multi:
            points = []
            tot_area = self.polygon.area

            for poly, prep_poly in zip(self.polygon, self._prepped):
                M_i = round(M * poly.area / tot_area)
                points += rejection_sample(poly, prep_poly, M_i)

        else:
            points = rejection_sample(self.polygon, self._prepped, M)

        # return the sample of points
        if return_points:
            self._prev_sample = geom.MultiPoint(points)
        else:
            origin = geom.Point((0, 0))
            self._prev_sample = [p.distance(origin) for p in points] << u.arcmin

        return self._prev_sample

    def MC_integrate(self, func, sample=None, M=None):
        '''Monte carlo integrate func over this field, using sample points, or
        generating own sample if not given
        '''
        # TODO could optioanlly return the error as well

        if sample is None:

            if M is None:
                mssg = "must supply one of `sample` or `M`"
                raise TypeError(mssg)

            sample = self.MC_sample(M)

        M = sample.size

        res = np.sum(func(sample))

        # Only use area units if the integrand has units as well
        V = self.area if hasattr(res, 'unit') else self.area.value

        return (V / M) * res
