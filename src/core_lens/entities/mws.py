from core_lens.base.entity import BaseEntity


class MWSEntity(BaseEntity):
    """Microwatershed (MWS) entity.

    Backed by:
    - Static: mws/static (partitioned by year)
    - Annual: mws/annual (partitioned by year)
    - Fortnightly: mws/fortnightly (partitioned by basin then year)

    All paths are relative to the AoI ``data_root``.
    """

    @property
    def key_cols(self) -> list[str]:
        return ["mws_id"]

    @property
    def geometry_col(self) -> str:
        return "geometry"

    @property
    def static_path(self) -> str:
        return "mws/static"

    @property
    def annual_path(self) -> str | None:
        return "mws/annual"

    @property
    def fortnightly_path(self) -> str | None:
        return "mws/fortnightly"
