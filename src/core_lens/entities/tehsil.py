"""Module for Core Lens."""

from core_lens.base.entity import BaseEntity
from core_lens.schema.profile import SchemaProfile


class TehsilEntity(BaseEntity):
    """Tehsil administrative boundary entity.

    Static file columns (from parquet schema):
        tehsil_id     — unique tehsil key
        tehsil        — tehsil name
        district      — parent district name
        state         — parent state name
        shape_area    — polygon area
        shape_length  — polygon perimeter
        geometry      — WKB geometry
        bbox          — struct<xmin, ymin, xmax, ymax> pre-computed bounds
    """

    @property
    def key_cols(self) -> list[str]:
        """Get the property value.

        Returns:
            list[str]: The key columns for the entity.

        """
        return ["tehsil_id"]

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
        return "tehsil/static.parquet"

    @property
    def schema_profile(self) -> SchemaProfile:
        """Get the property value.

        Returns:
            SchemaProfile: The hardcoded schema profile.

        """
        if not hasattr(self, "_schema_profile"):
            self._schema_profile = SchemaProfile(
                key_cols=["tehsil_id"],
                geometry_col="geometry",
                geometry_type="wkb",
                annual_time_col=None,
                sub_annual_time_col=None,
                bbox_cols=(
                    "bbox.xmin",
                    "bbox.ymin",
                    "bbox.xmax",
                    "bbox.ymax",
                ),
                extra_static_cols=[
                    "tehsil",
                    "district",
                    "state",
                    "shape_area",
                    "shape_length",
                ],
            )
        return self._schema_profile
