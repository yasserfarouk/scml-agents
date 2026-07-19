Changelog
=========

0.5.1 (2026.07.19)
-------------------

* Adding SCML 2026 Agents, qualification lists, and finalists (4 oneshot, 6 standard)
* Implementing lazy imports to fix slow import times (~95s -> ~3s)
* Removing an unused 8.8MB CSV file
* CI: uninstalling triton to fix a segfault (exit 139) in tests
* Various test corrections and README updates

0.5.0 (2026.03.15)
-------------------

* Migrating from setup.py to pyproject.toml
* Switching from black/isort to ruff for formatting and linting
* Replacing deprecated get_ami() with get_nmi() and adding None checks for its return value
* Fixing pandas MultiIndex indexing and a PyTorch macOS segfault
* Adding a GitHub Actions workflow to publish to PyPI on tags

0.4.14 (2026.03.14)
-------------------

* Fixing numpy imports for compatibility with newer numpy versions

0.4.13 (2025.08.27)
-------------------

* Pinning numpy version to avoid issues with numpy 1.26.

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
