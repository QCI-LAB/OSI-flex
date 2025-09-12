import time
from src.config_loader import load_config, print_config
from src.data_loader import load_data
from src.osiflex import OSIFlex
from src.results_saver import create_results_dir, save_results

CONFIG_FILE = "configs/sim_small_shear.yaml"
CONFIG_FILE = "configs/sim_medium_shear.yaml"
CONFIG_FILE = "configs/sim_large_shear.yaml"
CONFIG_FILE = "configs/sim_different_magnitudes.yaml"
#CONFIG_FILE = "configs/sim_nonorthogonal_shears.yaml"
#CONFIG_FILE = "configs/sim_single_shear.yaml"
# CONFIG_FILE = "configs/sim_shear_finetune.yaml"

# CONFIG_FILE = "configs/test_target.yaml"
# CONFIG_FILE = "configs/printed_cells.yaml"
# CONFIG_FILE = "configs/thyroid.yaml"

CONFIG_FILE = "configs/test_target_single_shear.yaml"
# CONFIG_FILE = "configs/printed_cells_single_shear.yaml"
# CONFIG_FILE = "configs/thyroid_single_shear.yaml"


def main(config_file: str = CONFIG_FILE) -> None:
    """
    Main script for OSI-Flex phase retrieval.
    Loads configuration and data, runs the OSIFlex algorithm, and saves results.
    """
    # Load configuration and data
    config = load_config(config_file)
    print_config(config_file)
    shearing_data = load_data(config.input_images)

    # Create result folder
    results_path = create_results_dir()

    # Retrieve object phase distribution
    start_time = time.time()
    osiflex = OSIFlex(shearing_data, config)
    phase, shears, costs = osiflex.run()
    elapsed = time.time() - start_time

    print(f"Main algorithm finished in {elapsed:.2f} seconds")
    print(f"Estimated shear vectors: {shears}")

    # Save results
    save_results(
        phase=phase,
        shearing_data=shearing_data,
        costs=costs,
        shears=shears,
        config=config,
        output_dir=results_path
    )


if __name__ == '__main__':
    main()
