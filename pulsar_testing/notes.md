# Just some notes to keep track of what needs to be done


With the isolated pulsars things generally seem OK. I think the issue was our target normalization
was se to 0.5 when it should have been 1.0, this cuts the domain off early and was making the
distributions narrow enough that some pulsars landed in the zero probability range. Upping the
target to 1.0 means all but one pulsar fall into the accepted range.

The spin period of pulsar S is inconsistent with basically any model. Use only the binary period for
now.

Look into whether fixing a big black hole population allows for the spin period. S is also badly fit
in Kiziltan and it might be driving the fit?

Giersz, Heggie 2011 (MC models of 47Tuc): "The critical object in this plot is the pulsar with the
    largest negative period derivative. It is known as 47 Tuc S (Freire et al. 2003), and these
    authors showed that it implies a projected mass/light ratio M/L > 1.4 in the region of the
    pulsar. In fact, the projected value of M/L at the location of this pulsar in our model is about
    1.1. Note, however, that our model is a little bright in the core (Section 4.1), which depresses
    the value. (Incidentally, the projected value of M/L increases from this central value to about
    2.3 at large radii; the global value is about 1.52.) Generally speaking, the tension between the
    requirements of the central surface brightness, on the one hand, and the acceleration of 47 Tuc
    S, on the other hand, was the single most important constraint in our attempts to find a
    satisfactory model."


This specific starting position can accommodate pulsar S without a crazy amount of BHs (~670 Msol)
`theta = [6.427, 0.959, 7.37, 1.293, 0.884, 0.454, 0.006, 0.262, 0.493, 0.563, 2.581, 1.1, 4.438]`

Generally the orbital periods have less assumptions but actually provide similar leverage so fine to
prefer the orbital data over the spin data when one is obviously inconsistent. 


# TODO:

* Clean up the normalization stuff, make sure it's all integrating to 1.0.

* We unfortunately are also seeing the bizarre bug where the normalization sometimes overflows the
  bounds of the array? Still have no idea what causes that, definitely not good. Seems to only
  happen under emcee, should test if it happens with only a singe thread.

