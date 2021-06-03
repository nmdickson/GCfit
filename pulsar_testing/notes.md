# Just some notes to keep track of what needs to be done


With the isolated pulsars things generally seem OK. I *think* we might have an issue where our target normalization was set
to 0.5 when it should have been 1.0, this cuts the domain off early and was making the distributions narrow enough that
some pulsars landed in the zero probability range. Upping the target to 1.0 means all but one pulsar fall into the accepted
range.

We unfortunately are also seeing the bizarre bug where the normalization sometimes overflows the bounds of the array? Still
have no idea what causes that, definitely not good.

Looks like the orbital dists are still being over-sampled, look into this. Might be a problem with step size, could be something
else.

Update: with the pulsars I have locally (not from the ATNF catalogue) all pulsars fall into the non-zero probability range,
should have a look at the pulsar that doesn't and see if we can locate it in one of the two 47 Tuc pulsar papers. Otherwise see
what paper it's in and if there's any notes or anything. It could be a perfectly fine timing solution that just isn't compatible
with the current best fit models though I'd be pretty surprised if that was the case.

Further update: The pulsar in question is Pulsar S which is a pulsar that we have both orbital and spin periods for. The reason
hadn't come up in my own testing is that I was only using the spin period likelihoods for pulsars that didn't have binary periods
available. Going to test this with my version and see if it's broken there too.

Contd: Pulsar S (spin period) is broken in my version too, I think I have our version in sync now so that makes sense. It seems like
it's right on the border of being in the non-zero zone but I can't seem to generate a cluster that will allow for it. Even doubling
the mass and cutting the half-light radius in half doesn't allow for it. Maybe look into a softening constant added, just so we don't
have -inf constantly?
