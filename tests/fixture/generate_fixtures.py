import datetime
import pathlib
import random

import polars as pl
import shapely.geometry as sgeom
import shapely.wkb as swkb
from faker import Faker

fake = Faker()


def generate_static(
    output_dir: pathlib.Path,
    entity_name: str,
    num_items: int,
    id_col: str,
    schema: dict[str, str],
    id_prefix: str = "",
) -> list[str]:
    # Fixed columns: id_col and geometry
    entity_dir = output_dir / "data" / entity_name
    entity_dir.mkdir(parents=True, exist_ok=True)

    ids = [f"{id_prefix}{i:03d}" for i in range(1, num_items + 1)]
    geometries = [
        swkb.dumps(sgeom.box(73.0 + i * 0.01, 15.0, 73.01 + i * 0.01, 15.01))
        for i in range(num_items)
    ]

    data = {id_col: ids, "geometry": geometries}

    # Generate random data based on schema
    for col, dtype in schema.items():
        if dtype == "int":
            data[col] = [random.randint(10, 100000) for _ in range(num_items)]
        elif dtype == "float":
            data[col] = [random.random() * 100.0 for _ in range(num_items)]
        elif dtype == "string":
            data[col] = [fake.word() for _ in range(num_items)]
        elif dtype == "state":
            data[col] = [fake.state() for _ in range(num_items)]
        elif dtype == "district":
            data[col] = [fake.city() for _ in range(num_items)]
        else:
            data[col] = [fake.word() for _ in range(num_items)]

    df = pl.DataFrame(data)
    df.write_parquet(entity_dir / "static.parquet")
    print(f"Generated {entity_name}/static.parquet with {len(data)} columns")
    return ids


def generate_annual(
    output_dir: pathlib.Path,
    entity_name: str,
    ids: list[str],
    id_col: str,
    years: list[int],
    schema: dict[str, str],
) -> None:
    entity_dir = output_dir / "data" / entity_name
    entity_dir.mkdir(parents=True, exist_ok=True)

    num_items = len(ids) * len(years)
    id_list = []
    year_list = []
    for entity_id in ids:
        for year in years:
            id_list.append(entity_id)
            year_list.append(year)

    data = {id_col: id_list, "year": year_list}

    for col, dtype in schema.items():
        if dtype == "int":
            data[col] = [random.randint(10, 1000) for _ in range(num_items)]
        elif dtype == "float":
            data[col] = [random.random() * 100.0 for _ in range(num_items)]
        elif dtype == "string":
            data[col] = [fake.word() for _ in range(num_items)]

    df = pl.DataFrame(data)
    df.write_parquet(entity_dir / "annual.parquet")
    print(f"Generated {entity_name}/annual.parquet with {len(data)} columns")


def generate_fortnightly(
    output_dir: pathlib.Path,
    entity_name: str,
    ids: list[str],
    id_col: str,
    dates: list[datetime.date],
    schema: dict[str, str],
) -> None:
    entity_dir = output_dir / "data" / entity_name
    entity_dir.mkdir(parents=True, exist_ok=True)

    num_items = len(ids) * len(dates)
    id_list = []
    date_list = []
    for entity_id in ids:
        for d in dates:
            id_list.append(entity_id)
            date_list.append(d)

    data = {id_col: id_list, "fortnightly_date": date_list}

    for col, dtype in schema.items():
        if dtype == "int":
            data[col] = [random.randint(1, 100) for _ in range(num_items)]
        elif dtype == "float":
            data[col] = [random.random() * 10.0 for _ in range(num_items)]
        elif dtype == "string":
            data[col] = [fake.word() for _ in range(num_items)]

    df = pl.DataFrame(data)
    df.write_parquet(entity_dir / "fortnightly.parquet")
    print(f"Generated {entity_name}/fortnightly.parquet with {len(data)} columns")


def generate_fixtures(output_dir: pathlib.Path) -> None:
    # MWS
    mws_static_schema = {
        "district": "district",
        "area_ha": "float",
        "state": "state",
        "elevation": "float",
        "basin_name": "string",
    }
    mws_ids = generate_static(
        output_dir, "mws", 100, "mws_id", mws_static_schema, id_prefix="13_"
    )

    mws_annual_schema = {
        "ndvi_mean": "float",
        "precip_total": "float",
        "tmax_mean": "float",
        "tmin_mean": "float",
        "soil_moisture": "float",
        "crop_yield": "float",
        "evapotranspiration": "float",
        "solar_radiation": "float",
        "groundwater_level": "float",
        "surface_runoff": "float",
    }
    generate_annual(
        output_dir, "mws", mws_ids, "mws_id", [2021, 2022], mws_annual_schema
    )

    mws_fortnightly_schema = {
        "ndvi": "float",
        "precip": "float",
        "tmax": "float",
        "tmin": "float",
        "soil_moisture": "float",
        "lai": "float",
        "fapar": "float",
        "albedo": "float",
        "lst": "float",
        "ndwi": "float",
    }
    dates = [
        datetime.date(2022, 1, 1),
        datetime.date(2022, 1, 15),
        datetime.date(2022, 2, 1),
    ]
    generate_fortnightly(
        output_dir, "mws", mws_ids, "mws_id", dates, mws_fortnightly_schema
    )

    # Tehsil
    tehsil_static_schema = {
        "state": "state",
        "district": "district",
        "population": "int",
        "area_sqkm": "float",
        "num_villages": "int",
    }
    generate_static(
        output_dir, "tehsil", 10, "tehsil_id", tehsil_static_schema, id_prefix="T_"
    )


if __name__ == "__main__":
    fixture_dir = pathlib.Path(__file__).parent
    generate_fixtures(fixture_dir)
