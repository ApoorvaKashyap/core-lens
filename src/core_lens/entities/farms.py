from core_lens.base import BaseEntity


class FarmEntity(BaseEntity):
    """Farms entity.

    Backed by:
    - Static: farms/static
    - Annual: farms/annual
    - SubAnnual: farms/sub_annual

    All paths are relative to the AoI ``data_root``.
    """

    @property
    def key_cols(self) -> list[str]:
        """Get the property value.

        Returns:
            list[str]: The key columns for the entity.

        """
        return ["farm_id"]

    @property
    def geometry_col(self) -> str:
        """Get the property value.

        Returns:
            str: The geometry column name.

        """
        return "geometry"

    @property
    def static_path(self) -> str:
        """Get the property value.

        Returns:
            str: The relative path to the static data.

        """
        return "farms/static.parquet"

    @property
    def annual_path(self) -> str:
        """Get the property value.

        Returns:
            str: The relative path to the annual data.

        """
        return "farms/annual.parquet"

    @property
    def sub_annual_path(self) -> str:
        """Get the property value.

        Returns:
            str: The relative path to the monthly data.

        """
        return "farms/sub_annual.parquet"
