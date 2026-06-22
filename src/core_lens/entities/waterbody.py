from core_lens.base.entity import BaseEntity


class WaterbodyEntity(BaseEntity):
    """Waterbody entity.

    Backed by:
    - Static: waterbodies/static
    - Annual: waterbodies/annual (partitioned by basin)

    All paths are relative to the AoI ``data_root``.
    """

    @property
    def key_cols(self) -> list[str]:
        return ["wb_id"]

    @property
    def geometry_col(self) -> str:
        return "geometry"

    @property
    def static_path(self) -> str:
        return "waterbodies/static"

    @property
    def annual_path(self) -> str | None:
        return "waterbodies/annual"

    @property
    def fortnightly_path(self) -> str | None:
        return "waterbodies/fortnightly"
