"""HTTP surface for PitchLens.

The FastAPI application lives in `api.main`. This package is deliberately inert
on import so that `import api` costs nothing; the ASGI target is `api.main:app`.
"""
