"""Optional-dependency availability flags used across submodules."""

from __future__ import annotations

try:
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: F401
    ASE_ADAPTOR_AVAILABLE = True
except Exception:
    ASE_ADAPTOR_AVAILABLE = False

try:
    import ase.io  # noqa: F401
    ASE_IO_AVAILABLE = True
except Exception:
    ASE_IO_AVAILABLE = False

try:
    from dscribe.descriptors import SOAP  # noqa: F401
    DSCRIBE_AVAILABLE = True
except Exception:
    DSCRIBE_AVAILABLE = False
