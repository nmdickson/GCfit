# Just some notes to keep track of what needs to be done


I *think* the orbital pulsars are working properly now, remains to be seen if they will remain stable under the MCMC chain.

With the isolated pulsars things generally seem OK. I *think* we might have an issue where our target normalization was set
to 0.5 when it should have been 1.0, this cuts the domain off early and was making the distributions narrow enough that
some pulsars landed in the zero probability range. Upping the target to 1.0 means all but one pulsar fall into the accepted
range.

We unfortunately are also seeing the bizarre bug where the normalization sometimes overflows the bounds of the array? Still
have no idea what causes that, definitely not good.

Update: with the pulsars I have locally (eg not from the ATNF catalogue) all pulsars fall into the non-zero probability range,
should have a look at the pulsar that doesn't and see if we can locate it in one of the two 47 Tuc pulsar papers.
