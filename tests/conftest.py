import pytest
from fastapi.testclient import TestClient

from shirabe.api import app

ABSTRACT = """We examined whether the timing of feedback affects performance on a repeated learning task. Participants completed a series of trials under three feedback schedules, with assignment randomized before the first session. We measured accuracy and response time at baseline and after each training block. Immediate feedback was associated with higher accuracy during training, but the difference was smaller at the delayed assessment. The estimates were similar after accounting for baseline performance and the number of completed trials. However, the sample was limited to volunteers from a single institution, and the study did not assess transfer to other tasks. These results suggest that feedback timing may influence short-term learning, while its effect on retention remains uncertain. Further work with a larger and more varied sample is needed to estimate the conditions under which the observed differences persist."""


@pytest.fixture
def abstract():
    return ABSTRACT


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client
