"""Stage-independent infrastructure mechanisms."""

from .artifacts import atomic_bytes, atomic_csv, atomic_json, atomic_npz, atomic_text
from .config import ConfigError, LoadedConfig, load_dataoverview_config
from .content_chain import (
    ContentChainError,
    code_projection,
    derive_chain_closure,
    derive_chain_commitment,
    derive_chain_receipt,
    verify_chain,
    verify_chain_link,
)
from .hashing import canonical_json, canonical_json_bytes, sha256_file, sha256_json
from .paths import DIAGNOSTIC_ROOT, SMOKE_ROOT, PathBoundaryError, find_repo_root, scoped_path
from .schema import SchemaValidationError, validate_artifact
from .terms import CountryProfile, discover_country_profiles, load_country_profile

__all__ = [
    "ConfigError",
    "ContentChainError",
    "CountryProfile",
    "DIAGNOSTIC_ROOT",
    "LoadedConfig",
    "PathBoundaryError",
    "SMOKE_ROOT",
    "SchemaValidationError",
    "atomic_bytes",
    "atomic_csv",
    "atomic_json",
    "atomic_npz",
    "atomic_text",
    "canonical_json",
    "canonical_json_bytes",
    "code_projection",
    "discover_country_profiles",
    "derive_chain_closure",
    "derive_chain_commitment",
    "derive_chain_receipt",
    "find_repo_root",
    "load_country_profile",
    "load_dataoverview_config",
    "scoped_path",
    "sha256_file",
    "sha256_json",
    "validate_artifact",
    "verify_chain",
    "verify_chain_link",
]
