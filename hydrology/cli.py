import argparse
from hydrology.hydrology_cli import run_hydrology_cli


def run_cli():
    """
    Legacy entrypoint for hydrology EDA; now delegates to run_hydrology_cli.
    """
    parser = argparse.ArgumentParser(
        prog="HydroEDA",
        description="Hydropower Reservoir EDA Toolkit"
    )

    parser.add_argument("--download", action="store_true",
        help="Download and clean hydrology data.")
    parser.add_argument("--plots", action="store_true",
        help="Generate hydrology plots (outputs/hydro).")

    args = parser.parse_args()
    run_hydrology_cli(args)


if __name__ == "__main__":
    run_cli()
