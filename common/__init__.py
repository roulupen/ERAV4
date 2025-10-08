"""
Common reusable modules for ERAV4 assignments.

This package contains reusable training infrastructure that can be shared
across different assignments and datasets.
"""

from .utils import (
    set_random_seed,
    get_device,
    get_model_summary,
    save_checkpoint,
    load_checkpoint,
    plot_training_history,
    save_training_info,
    load_training_info,
    calculate_accuracy,
    format_time,
    print_training_summary,
    create_directory_structure,
    setup_optimizer_and_scheduler,
    test_architecture_only
)

from .trainer import (
    Trainer,
    create_trainer,
    train_model
)

from .config import (
    TrainingConfig,
    parse_arguments,
    get_default_config,
    create_config_from_dict,
    print_config,
    validate_config
)

__all__ = [
    # Utils
    'set_random_seed',
    'get_device',
    'get_model_summary',
    'save_checkpoint',
    'load_checkpoint',
    'plot_training_history',
    'save_training_info',
    'load_training_info',
    'calculate_accuracy',
    'format_time',
    'print_training_summary',
    'create_directory_structure',
    'setup_optimizer_and_scheduler',
    'test_architecture_only',
    
    # Trainer
    'Trainer',
    'create_trainer',
    'train_model',
    
    # Config
    'TrainingConfig',
    'parse_arguments',
    'get_default_config',
    'create_config_from_dict',
    'print_config',
    'validate_config',
]

__version__ = '1.0.0'

