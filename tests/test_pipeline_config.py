from scripts.run_pipeline import load_config, resolve_city_config, resolve_runtime_location


def test_resolve_city_config_defaults_to_pune() -> None:
    config = load_config()
    city_config = resolve_city_config(config)

    assert city_config["name"] == "pune"
    assert city_config["latitude"] == 18.5204


def test_resolve_city_config_contains_multiple_search_points() -> None:
    config = load_config()
    city_config = resolve_city_config(config, city="pune")

    assert len(city_config["search_points"]) > 1


def test_resolve_city_config_accepts_search_point_alias_with_spaces() -> None:
    config = load_config()
    city_config = resolve_city_config(config, city="pimpri chinchwad")

    assert city_config["name"] == "pimpri_chinchwad"
    assert city_config["latitude"] == 18.6298
    assert len(city_config["search_points"]) == 1


def test_resolve_runtime_location_accepts_live_coordinates() -> None:
    config = load_config()
    city_config = resolve_runtime_location(
        config,
        latitude=18.675617,
        longitude=73.841995,
        label="current user",
    )

    assert city_config["name"] == "current_user"
    assert city_config["latitude"] == 18.675617
    assert city_config["longitude"] == 73.841995
    assert city_config["source"] == "coordinates"
