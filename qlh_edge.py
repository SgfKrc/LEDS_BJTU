"""Repository-root launcher for the minimal QLH Edge service."""

from src.qlh_edge import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
