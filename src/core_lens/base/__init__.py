"""Public API surface for core_lens plugin authors.

Plugin authors should import only from this package::

    from core_lens.base import BaseEntity, View, Result

Internal modules (namespaces, schema helpers) are not re-exported here.
"""

__module__: str = "core_lens.base"

from core_lens.base.entity import BaseEntity, EntityValidationError
from core_lens.base.view import View
from core_lens.base.result import Result

__all__ = [
    "BaseEntity",
    "EntityValidationError",
    "Result",
    "View",
]
