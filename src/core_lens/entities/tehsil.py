from core_lens.base.entity import BaseEntity
from core_lens.schema.profile import SchemaProfile


class TehsilEntity(BaseEntity):
    """Tehsil administrative boundary entity.

    Static file columns:
        id       — unique tehsil key
        TEHSIL   — tehsil name
        District — parent district name
        STATE    — parent state name
        geom     — WKB geometry
    """

    @property
    def key_cols(self) -> list[str]:
        """Returns:
        list[str]: The key columns for the entity.
        """
        return ["id"]

    @property
    def geometry_col(self) -> str:
        """Returns:
        str: The geometry column name.
        """
        return "geom"

    @property
    def static_path(self) -> str:
        """Returns:
        str: The relative path to the static data.
        """
        return "tehsil/static/tehsil.parquet"

    @property
    def schema_profile(self) -> SchemaProfile:
        """Returns:
        SchemaProfile: The hardcoded schema profile.
        """
        if not hasattr(self, "_schema_profile"):
            self._schema_profile = SchemaProfile(
                key_cols=["id"],
                geometry_col="geom",
                geometry_type="wkb",
                annual_time_col=None,
                fortnightly_time_col=None,
                bbox_cols=None,
                extra_static_cols=[
                    "TEHSIL",
                    "District",
                    "STATE",
                    "Shape_Area",
                    "Shape_Leng",
                ],
            )
        return self._schema_profile
