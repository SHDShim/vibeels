# Release Checklist

Use this checklist before making the repository public or publishing packages.

## Repository Audit

- Confirm no private data, unpublished datasets, credentials, local paths, or institution-sensitive content are tracked.
- Review generated assets in `asset/` and packaged assets in `vibeels/assets/`.
- Confirm the license choice in `LICENSE`, `pyproject.toml`, and `CITATION.cff`.
- Confirm the GitHub repository URL in `pyproject.toml` and `CITATION.cff`.

## Local Checks

```bash
python -m pytest
python -m build
python -m twine check dist/*
```

## TestPyPI

```bash
python -m twine upload --repository testpypi dist/*
python -m venv /tmp/vibeels-test
source /tmp/vibeels-test/bin/activate
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ vibeels
python -m vibeels
```

## Release Order

1. Finish repository audit.
2. Build and test locally.
3. Make the GitHub repository public.
4. Enable Zenodo GitHub integration.
5. Configure PyPI/TestPyPI Trusted Publishing.
6. Create the GitHub release tag.
7. Confirm Zenodo DOI.
8. Publish to PyPI.
9. Add DOI and PyPI badges to `README.md`.
