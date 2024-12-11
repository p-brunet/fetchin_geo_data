import pathlib
from typing import Dict, List, Union

import json
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import earthaccess

from loguru import logger
from retrying import retry

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions


output_dir = pathlib.Path("data/TEMPO_raw")
execution_log = pathlib.Path("data/execution_log.json")

logger.add("execution.log", rotation="1 MB")


def load_execution_state() -> Dict[str, Union[str, List[str]]]:
    if execution_log.exists():
        with open(execution_log, "r") as f:
            return json.load(f)
    return {"last_execution": None, "processed_granules": []}


def save_execution_state(state: Dict[str, Union[str, List[str]]]) -> None:
    with open(execution_log, "w") as f:
        json.dump(state, f, indent=4)


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def safe_open(results: List[str]):
    return earthaccess.open(results)


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def crop_granule(granule, bbox: tuple) -> Union[xr.Dataset, None]:
    try:
        logger.info(f"Opening: {granule} with xrarry as dataset")
        ds = xr.open_dataset(granule)
        logger.info(f"{granule} opened with xrarry as dataset")
        west, south, east, north = bbox

        lon_idx = np.where((ds['longitude'] >= west) & (ds['longitude'] <= east))[0]
        lat_idx = np.where((ds['latitude'] >= south) & (ds['latitude'] <= north))[0]

        subset_ds = ds.isel(longitude=lon_idx, latitude=lat_idx)
        return subset_ds
    except Exception as e:
        logger.error(f"Error cropping granule {granule}: {e}")
        raise


def debug_granule(granule):
    print(granule)
    xr.open_dataset(granule)
    logger.info(f"Processing granule: {granule}")
    return granule


def process_granule(granule, bbox: tuple, state: Dict[str, Union[str, List[str]]]) -> None:
    if granule in state["processed_granules"]:
        logger.info(f"Skipping already processed granule: {granule}")
        return

    try:
        ds = crop_granule(granule, bbox)

        if ds is not None:
            granule_name = ds.attrs.get("local_granule_id")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{granule_name}"

            encoding = {var: {'zlib': True, 'complevel': 5, 'shuffle': True} for var in ds.data_vars}

            ds.to_netcdf(output_file, format='NETCDF4', encoding=encoding)
            logger.info(f"Subset dataset saved to {output_file}")
            state["processed_granules"].append(granule_name)
            save_execution_state(state)
    except Exception as e:
        logger.error(f"Error processing granule {granule}: {e}")


if __name__ == "__main__":

    # auth
    auth = earthaccess.login()

    doi = "10.5067/IS-40e/TEMPO/NO2_L3.003"
    start_date = datetime.strptime("2024-03-01T12:00:00", "%Y-%m-%dT%H:%M:%S")
    end_date = datetime.strptime("2024-04-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
    step = timedelta(days=1)
    bbox = (-120, 49, -110, 60)

    state = load_execution_state()

    # Reload from a stamp
    current_date = start_date if not state["last_execution"] else datetime.strptime(state["last_execution"], "%Y-%m-%dT%H:%M:%S")

    while current_date < end_date:
        next_date = current_date + step
        time_range = (current_date.strftime("%Y-%m-%dT%H:%M:%S"), next_date.strftime("%Y-%m-%dT%H:%M:%S"))

        logger.info(f"Searching data from {time_range[0]} to {time_range[1]}...")
        results = earthaccess.search_data(doi=doi, temporal=time_range, bounding_box=bbox, cloud_hosted=True)

        if not results:
            logger.warning(f"No granules found for time range {time_range}.")
            current_date = next_date
            continue

        logger.info(f"Found {len(results)} granules. Processing...")

        try:
            fs = safe_open(results)
        except Exception as e:
            logger.error(f"Failed to open granules: {e}")
            break

        options = PipelineOptions()
        with beam.Pipeline(options=options) as p:
            (p | "Create granule collection" >> beam.Create(fs)
               | "Process granules" >> beam.Map(lambda granule: process_granule(granule, bbox, state)))

        state["last_execution"] = current_date.strftime("%Y-%m-%dT%H:%M:%S")
        save_execution_state(state)

        current_date = next_date
