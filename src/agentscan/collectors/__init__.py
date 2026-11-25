"""Collectors read one source each and add raw facts to the AccountSnapshot.

Every collector is read-only and metadata-only. A collector that fails records a
SourceStatus(ok=False) instead of aborting the scan, so the report states its own coverage.
"""
