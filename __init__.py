"""
Historical marine-animal telemetry: tagged animals and their recorded fixes.

The router calls only the three functions below. Each registered provider
serves one published dataset (see base.py for the rules they all follow and
movebank.py for the one dataset served today). No provider ever falls back to
generated data: with no verified copy of a dataset, the error says so.

Animal ids come from the source datasets. They are unique across the datasets
registered here; a provider added later must keep them so.
"""

from __future__ import annotations

from datetime import datetime

from app.services.animals import context
from app.services.animals.base import (
    DATA_STATUS_HISTORICAL,
    AmbiguousFixError,
    AnimalDataUnavailableError,
    AnimalNotFoundError,
    AnimalTrackProvider,
    FixNotFoundError,
)
from app.services.animals.context import HistoricalContextError
from app.services.animals.movebank import CHAGOS_GREEN_TURTLES, CHAGOS_MEGAFAUNA, MovebankDataPackage

__all__ = [
    "AmbiguousFixError",
    "AnimalDataUnavailableError",
    "AnimalNotFoundError",
    "FixNotFoundError",
    "HistoricalContextError",
    "PROVIDERS",
    "get_animal",
    "get_fix_environment",
    "get_track",
    "get_track_environment",
    "list_animals",
]

# One provider per published package. Animal ids are unique across them, so a
# lookup by id is unambiguous; `study_name` and `dataset_id` travel with every
# animal because one species can appear in more than one study.
PROVIDERS: list[AnimalTrackProvider] = [
    MovebankDataPackage(CHAGOS_GREEN_TURTLES),
    MovebankDataPackage(CHAGOS_MEGAFAUNA),
]


def _provider_for(animal_id: str) -> AnimalTrackProvider:
    for provider in PROVIDERS:
        if provider.has_animal(animal_id):
            return provider
    raise AnimalNotFoundError(animal_id)


def list_animals(species: str | None = None) -> dict:
    """Every tagged animal, optionally one species (scientific or common name)."""
    animals, datasets = [], []
    for provider in PROVIDERS:
        animals.extend(provider.list_animals(species=species))
        _, _, _, retrieved_at = provider.load()
        datasets.append(provider.dataset.as_dict(retrieved_at))
    return {
        "count": len(animals),
        "data_status": DATA_STATUS_HISTORICAL,
        "species_filter": species,
        "animals": animals,
        "datasets": datasets,
    }


def get_animal(animal_id: str) -> dict:
    return _provider_for(animal_id).get_animal(animal_id)


def get_track(animal_id: str, start_time: datetime | None = None, end_time: datetime | None = None) -> dict:
    """Recorded fixes in time order; start and end are inclusive, in UTC."""
    return _provider_for(animal_id).get_track(animal_id, start_time, end_time)


# ---------------------------------------------------------------------------
# Historical surface-ocean context (see context.py for what the values mean)
# ---------------------------------------------------------------------------


def _animal_block(track: dict, fix: dict) -> dict:
    """The observation, kept apart from the model values attached to it."""
    return {
        "animal_id": track["animal_id"],
        "timestamp": fix["timestamp"],
        "lat": fix["lat"],
        "lon": fix["lon"],
        "depth_m": fix["depth_m"],          # null: never the model's depth
        "source": track["provenance"]["provider"],
        "data_status": track["data_status"],
        "tracking_method": track["tracking_method"],
        "position_type": "Observed GPS telemetry",
        "source_event_id": fix["source_event_id"],
    }


def _context_for(provider: AnimalTrackProvider, animal_id: str):
    return context.find_context(provider.dataset.doi, animal_id)


def _products_for(provider: AnimalTrackProvider, animal_id: str):
    """The optional products: waves, sea level anomaly, satellite chlorophyll.

    Loaded once per request from their own manifests. Missing or unusable
    products are reported per fix, never raised: an optional product must not
    be able to take the core context down with it.
    """
    return context.find_products(provider.dataset.doi, animal_id)


def _with_products(sample: dict, products, fix: dict) -> dict:
    """Attach each optional product's own result to the core context.

    Every product carries its own availability, so a failing one appears as an
    unavailable entry beside working ones rather than emptying the response.
    """
    if not sample.get("environment_available") or not sample.get("environment"):
        return sample
    return {**sample, "environment": {**sample["environment"], **context.sample_products(products, fix)}}


def get_fix_environment(animal_id: str, timestamp: str | None = None, event_id: str | None = None) -> dict:
    """Surface-ocean context for one recorded fix, named by its timestamp or event id."""
    provider = _provider_for(animal_id)
    track = provider.get_track(animal_id)
    matches = [
        fix for fix in track["positions"]
        if (event_id is not None and fix["source_event_id"] == event_id)
        or (event_id is None and fix["timestamp"] == timestamp)
    ]
    if not matches:
        raise FixNotFoundError(event_id or timestamp)
    if len(matches) > 1:
        raise AmbiguousFixError(f"{len(matches)} fixes of {animal_id} share {timestamp}; name one by event_id")
    fix = matches[0]
    ctx = _context_for(provider, animal_id)
    products = _products_for(provider, animal_id)
    return {
        "animal": _animal_block(track, fix),
        "environment_scope": context.ENVIRONMENT_SCOPE,
        **_with_products(context.sample_fix(ctx, fix, animal_label=f"animal {animal_id}"), products, fix),
        "context_note": context.CONTEXT_NOTE,
        "context_provenance": context.context_provenance(ctx) if ctx else None,
        "product_provenance": context.products_provenance(products),
        "animal_provenance": track["provenance"],
    }


def get_track_environment(animal_id: str, start_time: datetime | None = None, end_time: datetime | None = None) -> dict:
    """Context for every recorded fix in the period, in time order. None are dropped."""
    provider = _provider_for(animal_id)
    track = provider.get_track(animal_id, start_time, end_time)
    ctx = _context_for(provider, animal_id)
    products = _products_for(provider, animal_id)
    records = [
        {"animal": _animal_block(track, fix),
         **_with_products(context.sample_fix(ctx, fix, animal_label=f"animal {animal_id}"), products, fix)}
        for fix in track["positions"]
    ]
    return {
        "animal_id": animal_id,
        "species_common": track["species_common"],
        "species_scientific": track["species_scientific"],
        "requested_start": track["requested_start"],
        "requested_end": track["requested_end"],
        "position_count": len(records),
        "context_available_count": sum(r["environment_available"] for r in records),
        **_product_counts(records),
        "environment_scope": context.ENVIRONMENT_SCOPE,
        "context_note": context.CONTEXT_NOTE,
        "context_provenance": context.context_provenance(ctx) if ctx else None,
        "product_provenance": context.products_provenance(products),
        "animal_provenance": track["provenance"],
        "records": records,
    }



def _product_counts(records: list[dict]) -> dict:
    """How many fixes each optional product answered for, counted from the
    records themselves -- never a figure written into the code."""
    counts = {spec.key: 0 for spec in context.PRODUCT_SPECS}
    interpolated = 0
    for record in records:
        environment = record.get("environment") or {}
        for key in counts:
            block = environment.get(key) or {}
            counts[key] += bool(block.get("available"))
        chlorophyll = environment.get("chlorophyll") or {}
        interpolated += bool(chlorophyll.get("available") and chlorophyll.get("interpolated"))
    return {"product_available_counts": counts, "chlorophyll_interpolated_count": interpolated}
