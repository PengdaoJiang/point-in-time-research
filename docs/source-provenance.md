# Source provenance

The six modules under `qplatform/` and four original test modules were copied from the maintained research platform, with line endings normalized. The statistics test file retains its five standalone unit tests unchanged; two platform-wide integration checks and their imports are excluded because the backtest/IC/research executors are outside this release. Those original integration tests remain in the source project, and this repository does not claim to run them. The package initializer, public demo, public-entry test, dependency file and CI are release adaptations.

The public release is a complete dependency closure for the temporal-validation kernel, not a copy of the full platform, its account integrations or every strategy. A private source manifest records original file hashes and public file hashes. Public commit dates describe the release work; historical Git dates are not manufactured.

The synthetic public case is separate from original research evidence. Implemented checks, actually exercised behavior and unverified business conclusions remain distinct.
