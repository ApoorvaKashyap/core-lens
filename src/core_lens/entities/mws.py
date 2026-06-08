from core_lens.base.entity import BaseEntity


class MWSEntity(BaseEntity):
    """Microwatershed (MWS) entity.

    Backed by:
    - Static: data/mws/static (partitioned by year)
    - Annual: data/mws/annual (partitioned by year)
    - Fortnightly: data/mws/fortnightly (partitioned by basin then year)
    """

    @property
    def key_cols(self) -> list[str]:
        return ["mws_id"]

    @property
    def geometry_col(self) -> str:
        return "geometry"

    @property
    def static_path(self) -> str:
        return "data/mws/static"

    @property
    def annual_path(self) -> str | None:
        return "data/mws/annual"

    @property
    def fortnightly_path(self) -> str | None:
        return "data/mws/fortnightly"
