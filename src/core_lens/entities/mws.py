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
        """Returns:
        list[str]: The key columns for the entity.
        """
        return ["mws_id"]

    @property
    def geometry_col(self) -> str:
        """Returns:
        str: The geometry column name.
        """
        return "geometry"

    @property
    def static_path(self) -> str:
        """Returns:
        str: The relative path to the static data.
        """
        return "mws/static"

    @property
    def annual_path(self) -> str | None:
        """Returns:
        str | None: The relative path to the annual data.
        """
        return "mws/annual"

    @property
    def fortnightly_path(self) -> str | None:
        """Returns:
        str | None: The relative path to the fortnightly data.
        """
        return "mws/fortnightly"
