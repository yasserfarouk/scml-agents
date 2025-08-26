Changelog
=========

0.4.12 (2025.08.26)
-------------------

* Adding SCML 2025 Agents

0.4.11 (2025.03.16)
-------------------

* Requiring SCML 0.7.6 and NegMAS 0.11.2. Making ArtisanKangaroo compatible with NegMAS 0.11.2
* Adding a missing report
* Ignore out of range contract comment
* Avoiding out-of-range agreements in S5s This is not a change in the strategy. It avoids failing due to a bug in SCML-Std implementation which allowed negotiation issues to exceed simulation time near the end of the simulation. This bug was fixed in the version of SCML used for the competition but this fix is necessary to make S5s work with earlier versions without throwing an exception.
* minor README update
* Fixing the default agent for 2024 std tests
