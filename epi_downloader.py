#!/usr/bin/env python3
"""EPI downloader is a tool for bulk downloading datasets from the IHME website.

The EPI visualisation website from which the datasets are downloaded can be found here:
        https://vizhub.healthdata.org/epi
"""

import argparse
import asyncio
import json
import sys
from collections.abc import Iterable, Mapping
from io import StringIO
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd
import platformdirs
from httpx import AsyncClient

APP_NAME = "epi_downloader"
EPI_BASE_URL = "https://vizhub.healthdata.org/epi"
REQUIRED_VARS = ("model", "measure", "year", "age", "sex")
EXAMPLE_CONFIG = {
    "model": ["Diabetes Mellitus - Total"],
    "measure": ["Prevalence"],
    "year": ["2015"],
    "age": ["20-24 years"],
    "sex": ["Male"],
}

Metadata = dict[str, dict[str, int]]
Versions = list[dict[str, Any]]
Config = dict[str, list[int]]


class CacheClient:
    """A wrapper around AsyncClient which caches downloaded data on disk."""

    def __init__(
        self, client: AsyncClient, cache_path: Path, ignore_cache: bool
    ) -> None:
        """Create a new cache client.

        Args:
            client: HTTP client
            cache_path: Where to save cached data
            ignore_cache: If true, redownload data even if it is in the cache
        """
        self._client = client
        self._cache_path = cache_path
        self._ignore_cache = ignore_cache

    async def get(self, url: str, file_name: str, *args: Any, **kwargs: Any) -> str:
        """Load data from the specified URL.

        Args:
            url: The URL from which to download data
            file_name: The unique name to use for the cached file
            args: Arguments to pass to AsyncClient.get
            kwargs: Keyword arguments to pass to AsyncClient.get
        """
        file_path = self._cache_path / file_name

        # If the data has already been cached (and the user hasn't opted out of using
        # the cache) then load it from disk
        if not self._ignore_cache and file_path.exists():
            with file_path.open(encoding="utf-8") as file:
                return file.read()

        # Make HTTP request
        response = await self._client.get(url, *args, **kwargs)
        response.raise_for_status()

        # Cache data on disk
        with file_path.open("w", encoding="utf-8") as file:
            file.write(response.text)

        return response.text


async def load_model_versions(client: CacheClient, model: int) -> Versions:
    """Load a list of possible data versions fo the given model."""
    # NB: I'm not sure what the step param means, but it seems to be empty for the
    # datasets I've been downloading -- AD
    params = {"model": model, "step": ""}
    text = await client.get(
        "/api/model/versions", f"versions_model{model}.json", params=params
    )
    return list(json.loads(text)["data"].values())


def get_latest_model_version(versions: Versions, measure: int | None) -> int:
    """Get the latest version ID matching the specified measure.

    This assumes that newer datasets have higher version IDs.
    """
    return max(v["version"] for v in versions if v["measure"] == measure)


def get_model_version(versions: Versions, measure: int) -> int:
    """Try to get the best-matching model version for the specified measure.

    The metadata is in a slightly confusing form here. Sometimes there are versions
    available which have measure IDs explicitly specified, in which case we can just
    check whether there is a version available with the desired measure ID. However, at
    other times, there is no measure ID specified (the field is null), but seemingly
    data for the desired measure can still be downloaded. So we first try to get an
    exact match for the measure ID and fall back on using a model version with no
    measure ID.
    """
    try:
        return get_latest_model_version(versions, measure)
    except ValueError:
        return get_latest_model_version(versions, None)


def id_to_str(id_dict: dict[str, int], id: int) -> str:
    """Convert an integer ID for a param to its string representation."""
    return next(k for k, v in id_dict.items() if v == id)


async def load_all_model_versions(
    client: CacheClient, model_ids: dict[str, int], models: list[int]
) -> dict[int, Versions]:
    """Load model versions for multiple models.

    Args:
        client: HTTP client
        model_ids: The string representations for each model
        models: Which models to get the dataset versions for
    """
    futures = (load_model_versions(client, model) for model in models)
    all_versions = await asyncio.gather(*futures, return_exceptions=True)

    # Check if versions could not be retrieved for any of the models
    failed = [m for m, v in zip(models, all_versions) if isinstance(v, BaseException)]
    if failed:
        raise RuntimeError(
            "Could not load dataset versions for the following models: "
            + ", ".join(id_to_str(model_ids, m) for m in failed)
        )

    return dict(zip(models, all_versions))


def load_config(config_path: str, metadata: Metadata) -> Config:
    """Load the config file from disk."""
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)

    out: Config = {}
    errors: dict[str, list[str]] = {}
    for key, values in config.items():
        out[key] = []
        for value in values:
            try:
                # Convert the names to corresponding integer IDs
                out[key].append(metadata[key][str(value)])
            except KeyError:
                if key not in errors:
                    errors[key] = []
                errors[key].append(str(value))

    if errors:
        raise RuntimeError(
            f"The following values in the config file are invalid: {errors!r}"
        )

    return out


async def load_dataset(
    client: CacheClient, params: dict[str, int], version: int
) -> pd.DataFrame:
    """Load a dataset for the given parameter set.

    Args:
        client: HTTP client
        params: Parameter set
        version: Which dataset version (i.e. ID number) to download
    """
    # Append version ID to params
    all_params: dict[str, Any] = params | {"version": version}

    # Keep params in same order so filename is consistent
    all_params = dict(sorted(all_params.items()))

    # A unique file name for this set of params
    param_str = "_".join(f"{key}{value}" for key, value in params.items())
    data_file_name = f"data_{param_str}.csv"

    # Extra parameters -- not sure what they mean
    all_params |= {
        "type": "final",
        "bundle": "",
        "step": "",
        "crosswalk": "",
        "clinical": False,
        "adjusted": False,
        "location": 1,
        "population": 1,
    }

    # Download CSV file
    text = await client.get(
        "/api/model/results/download", data_file_name, params=all_params
    )
    if not text:
        raise RuntimeError("No data available for the specified parameters")

    return pd.read_csv(StringIO(text))


def permute_parameter_grid(
    param_grid: Mapping[str, Iterable[int]],
) -> Iterable[dict[str, int]]:
    """Generate each combination of parameters for the given parameter grid.

    This function is a generator so the grid is computed lazily.

    >>> list(_permute_parameter_grid({'a': range(2), 'b': range(3)}))
    [{'a': 0, 'b': 0}, {'a': 0, 'b': 1}, {'a': 0, 'b': 2}, {'a': 1, 'b': 0}, {'a': 1, 'b': 1}, {'a': 1, 'b': 2}]
    """  # noqa: E501
    if not param_grid:
        return

    items = sorted(param_grid.items())
    keys, values = zip(*items)
    for v in product(*values):
        yield dict(zip(keys, v))


async def load_all_data(
    client: CacheClient,
    metadata: Metadata,
    config: Config,
    model_versions: dict[int, Versions],
) -> list[pd.DataFrame]:
    """Load datasets for all specified parameters."""
    data_futures = []
    all_params = list(permute_parameter_grid(config))
    for params in all_params:
        versions = model_versions[params["model"]]
        version = get_model_version(versions, params["measure"])
        data_futures.append(load_dataset(client, params, version))

    print(f"{len(data_futures)} data files to download")
    all_data = await asyncio.gather(*data_futures, return_exceptions=True)

    # Print out a message indicating which param sets failed to download
    failed: list[dict[str, str]] = []
    for params, data in zip(all_params, all_data):
        if isinstance(data, BaseException):
            # Convert params from integer IDs to string representation
            failed.append({k: id_to_str(metadata[k], v) for k, v in params.items()})
    if failed:
        print(
            f"WARNING: Failed to download data for {len(failed)}/{len(all_data)} "
            "parameter sets failed. The parameter sets were:"
        )
        for p in failed:
            print(f" - {p!r}")

    return [d for d in all_data if isinstance(d, pd.DataFrame)]


def save_datasets(datasets: list[pd.DataFrame], output_path: str) -> None:
    """Merge multiple datasets and save to a single output file."""
    # Merge DataFrames
    df = pd.concat(datasets, ignore_index=True)

    # Save combined dataset to file
    print(f"Saving data to {output_path}")
    df.to_csv(output_path, index=False)


def parse_metadata(metadata: dict[str, Any]) -> Metadata:
    """Parse the raw metadata retrieved from the EPI website.

    Returns:
        A dict where the key is a string representation and the value is the
        corresponding integer ID.
    """
    out = {}
    for var in REQUIRED_VARS:
        out[var] = {
            str(item["name"]): item[f"{var}_id"] for item in metadata[var].values()
        }

    return out


def write_json(file_name: str, data: Any) -> None:
    """Save the specified data to disk in JSON format."""
    print(f"Saving {file_name}")
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


async def load_metadata(client: CacheClient) -> Metadata:
    """Load the metadata from the EPI website.

    This metadata contains the names of various parameter values (e.g. disease type, age
    group) and corresponding ID numbers.
    """
    text = await client.get("/api/metadata", "metadata.json")
    return parse_metadata(json.loads(text)["data"])


def parse_cli_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog=APP_NAME, description="A tool to download datasets from IHME"
    )
    group = parser.add_argument_group()
    group.add_argument(
        "--dump-config",
        dest="dump_config",
        action="store_true",
        help="Dump metadata and example config files",
    )
    group.add_argument(
        "-c",
        "--config",
        dest="config_path",
        type=str,
        help="Path to config file",
    )
    group.add_argument(
        "-o",
        "--output",
        dest="output_path",
        type=str,
        help="Path to save CSV data to",
    )
    group.add_argument(
        "--no-cache",
        dest="no_cache",
        action="store_true",
        help=(
            "Redownload data files even if they are in the cache. This is useful for "
            "checking whether new data has become available since the last search."
        ),
    )

    # There might be a smarter way to validate arguments, but I can't figure it out
    try:
        if len(sys.argv) <= 1:
            raise RuntimeError()

        args = parser.parse_args()
        if args.dump_config:
            if args.config_path or args.output_path or args.no_cache:
                raise RuntimeError(
                    "--dump-config cannot be combined with other options"
                )
        elif not (args.config_path and args.output_path):
            raise RuntimeError("Both --config and --output options are required")
    except RuntimeError as ex:
        parser.print_usage()
        print(ex)
        raise SystemExit()

    return args


async def main() -> int:
    """Main entry point for the program."""
    args = parse_cli_args()

    async with AsyncClient(base_url=EPI_BASE_URL) as http_client:
        cache_path = platformdirs.user_cache_path(APP_NAME, ensure_exists=True)
        client = CacheClient(http_client, cache_path, args.no_cache)
        metadata = await load_metadata(client)

        if args.dump_config:
            write_json("metadata.json", metadata)
            write_json("example_config.json", EXAMPLE_CONFIG)

        if args.config_path:
            config = load_config(args.config_path, metadata)
            model_versions = await load_all_model_versions(
                client, metadata["model"], config["model"]
            )

            if datasets := await load_all_data(
                client, metadata, config, model_versions
            ):
                save_datasets(datasets, args.output_path)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
