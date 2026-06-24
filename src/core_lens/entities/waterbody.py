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
        """Returns:
        list[str]: The key columns for the entity.
        """
        return ["wb_id"]

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
        return "waterbodies/static"

    @property
    def annual_path(self) -> str | None:
        """Returns:
        str | None: The relative path to the annual data.
        """
        return "waterbodies/annual"
