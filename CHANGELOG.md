# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-04-16

### Added
- Initial professional release with provider-agnostic core logic.
- Support for shared database (Postgres/SQLite) usage tracking.
- Centralized pricing management and automatic database seeding.
- Native Google ADK integration via `CostTrackerPlugin`.
- Automated CI/CD with multi-OS and multi-Python version support.
- CLI reporting utility (`adk-cost-report`).

### Fixed
- Synchronized version strings across `pyproject.toml`, `_version.py`, and `uv.lock`.
- Purged all LLM-generated symbols and decorative Unicode.
