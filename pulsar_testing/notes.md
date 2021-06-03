# Just some notes to keep track of what needs to be done


With the isolated pulsars things generally seem OK. I think the issue was our target normalization was se to 0.5 when it should
have been 1.0, this cuts the domain off early and was making the distributions narrow enough that some pulsars landed in the
zero probability range. Upping the target to 1.0 means all but one pulsar fall into the accepted range.

The spin period of pulsar S is inconsistent with basically any model. Use only the binary period for now.

Look into whether fixing a big black hole population allows for the spin period. S is also badly fit in Kiziltan
and it might be driving the fit?

Generally the orbital periods have less assumptions but actually provife similar leverage so fine to prefer the orbital
data over the spin data when one is obviously inconsistent. 


# TODO:

* Clean up the normalization stuff, make sure it's all integrating to 1.0.

* We unfortunately are also seeing the bizarre bug where the normalization sometimes overflows the bounds of the array? Still
have no idea what causes that, definitely not good.

* Add in ~2300 Msol of BH to see if that accomodates pulsar S, could be an interesting discussion point re-Kiziltan's results. 
