import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import api
from model_provider import GPTClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('HUMANIZER_ENABLE_REBUILD', raising=False)
    with TestClient(api.app) as client:
        yield client


def test_health_and_interface_need_no_model_download(client):
    info = client.get('/health').json()
    assert info['status'] == 'ok'
    assert not next(m for m in info['available_models'] if m['id'] == 'gpt-4.1-mini')['configured']
    assert client.get('/ui/').status_code == 200
    assert client.get('/ui/stream.mjs').status_code == 200
    assert client.post('/rebuild-index').status_code == 403


@pytest.mark.parametrize('payload,status', [
    ({'text': '  '}, 400), ({'text': 'x' * 2001}, 400),
    ({'text': 'سلام', 'style': 'unknown'}, 422),
    ({'text': 'سلام', 'model': 'unknown'}, 422),
    ({'text': 'سلام', 'model': 'gpt-4.1-mini'}, 400),
])
def test_invalid_requests_never_reach_generation(client, monkeypatch, payload, status):
    def forbidden(*args):
        pytest.fail('Invalid request reached generation')
    monkeypatch.setattr(api, 'run_humanizer', forbidden)
    assert client.post('/humanize', json=payload).status_code == status


def test_result_uses_selected_style_and_model(client, monkeypatch):
    def fake(text, style, model):
        assert (text, style, model) == ('سلام', 'formal', 'llama3.1:8b')
        return {'humanized_text': 'سلام به شما', 'retrieved_examples': [1], 'evaluation': {}}
    monkeypatch.setattr(api, 'run_humanizer', fake)
    response = client.post('/humanize', json={'text': ' سلام ', 'style': 'formal'})
    assert response.status_code == 200
    assert response.json()['examples_used'] == 1


def test_failed_generation_is_not_a_success_or_raw_error(client, monkeypatch):
    def fail(*args):
        raise RuntimeError('private-provider-details')
    monkeypatch.setattr(api, 'run_humanizer', fail)
    response = client.post('/humanize', json={'text': 'سلام'})
    assert response.status_code == 502
    assert 'private-provider-details' not in response.text
    monkeypatch.setattr(api, 'stream_humanizer', fail)
    response = client.post('/humanize/stream', json={'text': 'سلام'})
    assert '__ERROR__' in response.text
    assert 'private-provider-details' not in response.text
    assert response.headers['cache-control'] == 'no-store'


def test_sse_contract_preserves_json_tokens(client, monkeypatch):
    monkeypatch.setattr(api, 'stream_humanizer', lambda *args: iter([
        'data: __TOKEN__"سلام\\nدنیا"\n\n', 'data: __TEXT__"سلام دنیا"\n\n', 'data: __DONE__\n\n',
    ]))
    response = client.post('/humanize/stream', json={'text': 'سلام'})
    assert response.text.endswith('data: __DONE__\n\n')
    assert response.headers['content-type'].startswith('text/event-stream')


@pytest.mark.parametrize('status', [401, 403, 429, 500])
def test_provider_error_bodies_are_never_exposed(status):
    response = httpx.Response(status, text='private-provider-details')
    with pytest.raises(RuntimeError) as caught:
        GPTClient._check(response)
    assert 'private-provider-details' not in str(caught.value)
