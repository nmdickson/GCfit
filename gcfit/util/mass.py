import random

import numpy as np
import astropy.units as u
from shapely import ops
import shapely.geometry as geom
import shapely.prepared as prepgeom


__all__ = ["Field"]


class Field:
    '''Representation of an observational photometric field

    Constructed based on imaged field coordinate boundaries, in RA and DEC,
    this class exists to facilitate the analysis of all mass function data.

    Polygons representing the observed fields are constructed, corrected,
    merged and prepped in order to best handle the extraction of radial slices
    and the monte carlo integration of any function within it's bounds.

    Parameters
    ----------
    coords :  numpy.ndarray or shapely.geometry.Polygon
        Input polygon descriptor, either in the form of a already created
        polygon (or multi-polygon) or as an array of ordered coordinates

    cen : 2-tuple of float, optional
        The central coordinate of the relevant globular cluster, in RA and
        DEC. If `cen` does not have any associated units, one must be supplied
        to the `unit` parameter. This is only used for initialization, all
        coordinates will be centred around (0, 0). Defaults to assuming this
        centring has already happened

    unit : str, optional
        Unit associated with all coordinates. Must be provided if `cen` has
        no units. Note that all coordinates will be converted to arcmin during
        processing, this parameter only defines the input values.

    preprep : bool, optional
        Whether to run `Field.prep` upon initialization. Defaults to False

    Attributes
    ----------
    polygon : shapely.geometry.Polygon
        The `shapely` `Polygon` or `MultiPolygon` object representing the field

    area : astropy.Quantity
        The total area contained by the fields, in arcmin^2

    Notes
    -----
    `preprep` will construct the prepared polygons immediately, however this
    geometry cannot be pickled, and thus the `Field` will not be passable using
    something like mpi. If that is desired, `prep` will have to be called
    manually at a later time, before any calls to `__contains__`

    See Also
    --------
    shapely : Source for all geometry operations
    '''

    def __contains__(self, other):
        # TODO might not be fair to *require* _prepped. Also will error badly
        if self._multi:
            return any(p.contains(other) for p in self._prepped)
        else:
            return self._prepped.contains(other)

    def _correct(self, crd, cen=(0, 0), unit=None):
        '''Centre, convert and RA correct the given array of `crd`s'''
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
        '''Prepare the polygons for quicker containment searches'''
        if self._multi:
            self._prepped = [prepgeom.prep(p) for p in self.polygon.geoms]
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
            coords = coords.geoms

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

        # Combine and smooth all polygons
        if self._multi:

            self.polygon = ops.unary_union([geom.Polygon(c).buffer(0)
                                            for c in coords])

        else:
            self.polygon = geom.Polygon(coords).buffer(0)

        # Explicitly check the polygons again, as they sometimes change above
        if isinstance(self.polygon, geom.Polygon):
            self._multi = False
        elif isinstance(self.polygon, geom.MultiPolygon):
            self._multi = True

        # If desired, prep the polygons
        if preprep:
            self.prep()

        # Compute polygon area, in correct units
        self.area = self.polygon.area << u.arcmin**2

    @classmethod
    def from_dataset(cls, dataset, cen):
        '''Create this field from a corresponding `gcfit.core.data.Dataset`'''
        import string

        unit = dataset.mdata['field_unit']

        coords = []
        for ch in string.ascii_letters:
            try:
                coords.append(dataset['fields'].mdata[f'{ch}'])
            except KeyError:
                break

        if len(coords) == 1:
            coords = coords[0]

        return cls(coords, cen=cen, unit=unit)

    def slice_radially(self, r1, r2):
        '''Return a new field representing a radial "slice" of this field'''

        # make sure that r1,r2 are in arcmin
        r1, r2 = r1.to_value('arcmin'), r2.to_value('arcmin')

        origin = geom.Point((0, 0))

        shell = origin.buffer(r2) - origin.buffer(r1)

        return Field(self.polygon & shell)

    def MC_sample(self, M, return_points=False):
        '''Randomly sample `M` points from this field

        Random points are generated (using `random.uniform`) between the
        minimum and maximum bounds of each polygon, and simple rejection
        sampling is used to determine the points within each polygon.
        The number of points per polygon (in fields with multiple seperate
        polygons) is in proportion to their respective areas.

        Parameters
        ----------
        M : int
            The number of points to sample

        return_points : bool, optional
            If `True`, returns a `shapely.geometry.MultiPoint` object containing
            all the points, otherwise simply returns the radial position of
            each (in arcmin). Defaults to `False`

        Returns
        -------
        sample : astropy.Quantity or shapely.geometry.MultiPoint
            Array of `M` sampled points, either in radial positions or
            coordinates, based on `return_points`

        '''
        # TODO we should accept a seed or something for this randomness

        def rejection_sample(poly, prep_poly, M_i):
            '''Rejection-sample `Mj` points from this given sub-`poly`'''

            if poly.is_empty:
                return []

            minx, miny, maxx, maxy = poly.bounds

            points = []

            # TODO this randomness could probably be done better?
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

            for poly, prep_poly in zip(self.polygon.geoms, self._prepped):
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
        '''Monte Carlo integration of `func` over this field

        Using a random sample of points within this field, either generated
        beforehand or using `Field.MC_sample`, computes the integral of the
        given `func` over the entire field by simply evaluating and summing
        the function over all samples, and normalizing to the field area.

        Will support units, if supported by the given function.

        Parameters
        ----------
        func : callable
            The (vectorized) function to integrate over. Must accept an array
            of points represented by their radial distances from the origin.

        sample : numpy.ndarray, optional
            The sample of points to be used in the integration, represented by
            their radial distances from the origin. If None (default), a new
            sample will be generated using `Field.MC_sample` and `M`

        M : int, optional
            The number of points to be sampled from `Field.MC_sample`, if
            `sample` is None.
        '''

        if sample is None:

            if M is None:
                mssg = "must supply one of `sample` or `M`"
                raise TypeError(mssg)

            sample = self.MC_sample(M)

        M = sample.size

        res = np.sum(func(sample))

        # Only use area units if the integrand has units as well
        V = self.area

        # TODO have to do this cause I cant get composite equivalencies to work
        if hasattr(func, '_xunit'):
            V = V.to(func._xunit**2)

        if not hasattr(res, 'unit'):
            V = V.value

        return (V / M) * res

    # ----------------------------------------------------------------------
    # Plotting functionality
    # ----------------------------------------------------------------------

    def _patch(self, *args, **kwargs):
        '''Create a `PathPatch` based on the polygon boundaries'''
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch

        if self.polygon.is_empty:
            raise ValueError("This Field is empty, cannot create patch")

        coords, codes = [], []

        for poly in (self.polygon.geoms if self._multi else [self.polygon]):

            for line in [poly.exterior, *poly.interiors]:
                coords += line.coords

                codes += [Path.MOVETO]
                codes += ([Path.LINETO] * (len(line.coords) - 2))
                codes += [Path.CLOSEPOLY]

        path = Path(coords, codes)
        return PathPatch(path, *args, **kwargs)

    def plot(self, ax, prev_sample=False, adjust_view=True, *,
             sample_kw=None, **kwargs):
        '''Plot this field onto a given ax as a polygonal patch

        Given an already-initialized matplotlib axes, add a patch representing
        all polygons in this field, using a `PathPatch` constructed from
        each polygons boundary lines.

        This method only adds a patch to existing plots, all other plotting
        logic must be handled seperately

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The matplotlib axes object to add this plot to

        prev_sample : bool, optional
            If True, also attempt to add a scatterplot of the last sampled
            points. `Field.MC_sample` must have been recently run, with
            `return_points=True` for this to function. Defaults to False

        adjust_view : bool, optional
            If True (default), also calls the `autoscale_view` method of `ax`.
            Simply adding a patch to a plot will not move the view to match,
            and patches could end up outside of the window limits. This
            parameter corrects that.

        sample_kw : dict, optional
            kwargs to be passed to the `scatter` plot of the sample plot, if
            `prev_sample=True`
        '''

        pt = ax.add_patch(self._patch(**kwargs))

        if prev_sample:
            try:
                smpl = self._prev_sample

                if sample_kw is None:
                    sample_kw = {}

                if hasattr(smpl, 'geom_type'):
                    smpl_xy = zip(*[p.xy for p in smpl.geoms])
                    sc = ax.scatter(*smpl_xy, **sample_kw)
                    sc.set_zorder(pt.zorder + 1)

                else:
                    mssg = ("Must sample with `return_points=True` in order to"
                            "plot sampled points")
                    raise RuntimeError(mssg)

            except AttributeError:
                mssg = ("No previous sample, call `Field.MC_sample` with "
                        "`return_points=True` first")
                raise RuntimeError(mssg)

        if adjust_view:
            ax.autoscale_view()

        if prev_sample:
            return pt, sc
        else:
            return pt
