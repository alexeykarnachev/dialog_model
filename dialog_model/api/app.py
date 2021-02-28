import logging

from fastapi import FastAPI

from dialog_model.api.endpoints import EndpointsRegister
from dialog_model.log_config import prepare_logging
from dialog_model.model import load_response_candidates_generator_from_experiment_dir

_LOGGER = logging.getLogger(__name__)


def prepare(experiment_dir, checkpoint_name, device, logs_dir) -> FastAPI:
    prepare_logging(logs_dir)
    generator = load_response_candidates_generator_from_experiment_dir(
        experiment_dir=experiment_dir, checkpoint_name=checkpoint_name, device=device)

    app = FastAPI()
    endpoints_register = EndpointsRegister(app=app, generator=generator)
    endpoints_register.register_all_views()

    _LOGGER.info('Application successfully prepared.')

    return app
