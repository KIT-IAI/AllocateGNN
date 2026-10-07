"""Atomic artifact writers under contention and interrupted publication."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from threading import Barrier

import pytest

from sglib.core.infra import artifacts
from sglib.core.infra.hashing import sha256_file

pytestmark = pytest.mark.produce


@pytest.mark.parametrize('exclusive', [False, True])
def test_json_byte_format_and_file_hash(tmp_path, exclusive):
    document = {'z': [1, 2], 'a': '土地\n第二行\r\n末尾\\n不是换行',
                '嵌套': {'quote': '"quoted"', 'literal': '\\r\\n'}}
    target = tmp_path / 'value.json'
    artifacts.atomic_json(document, target, exclusive=exclusive)
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    reference = tmp_path / 'legacy_reference.json'
    if exclusive:
        # This is the exact old learned writer's native text-stream operation.
        reference.write_text(payload, encoding='utf-8')
    else:
        # The original core writer encoded directly, retaining literal LF.
        reference.write_text(payload, encoding='utf-8', newline='\n')
        assert reference.read_bytes() == payload.encode('utf-8')
    assert target.read_bytes() == reference.read_bytes()
    assert sha256_file(target) == sha256_file(reference)
    assert json.loads(target.read_text(encoding='utf-8')) == document
    ending = os.linesep.encode('utf-8') if exclusive else b'\n'
    assert target.read_bytes().endswith(ending)


def test_exclusive_json_has_exactly_one_winner(tmp_path):
    target = tmp_path / 'receipt.json'
    barrier = Barrier(8)
    def publish(number):
        barrier.wait()
        try:
            artifacts.atomic_json({'writer': number}, target, exclusive=True)
            return number
        except FileExistsError:
            return None
    with ThreadPoolExecutor(max_workers=8) as pool:
        winners = [x for x in pool.map(publish, range(8)) if x is not None]
    assert len(winners) == 1
    assert json.loads(target.read_text(encoding='utf-8')) == {'writer': winners[0]}
    assert not target.with_name('.receipt.json.part').exists()


def test_exclusive_json_keeps_interrupted_evidence_and_existing_target(tmp_path):
    target = tmp_path / 'receipt.json'
    partial = tmp_path / '.receipt.json.part'
    partial.write_bytes(b'interrupted evidence')
    with pytest.raises(FileExistsError):
        artifacts.atomic_json({}, target, exclusive=True)
    assert partial.read_bytes() == b'interrupted evidence'
    assert not target.exists()
    partial.unlink()
    target.write_bytes(b'original receipt')
    with pytest.raises(FileExistsError):
        artifacts.atomic_json({}, target, exclusive=True)
    assert target.read_bytes() == b'original receipt'
    assert not partial.exists()


def test_exclusive_json_cleans_owned_partial_on_publication_error(tmp_path, monkeypatch):
    target = tmp_path / 'receipt.json'
    def fail_link(*args):
        raise OSError('publication failed')
    monkeypatch.setattr(artifacts.os, 'link', fail_link)
    with pytest.raises(OSError, match='publication failed'):
        artifacts.atomic_json({}, target, exclusive=True)
    assert list(tmp_path.iterdir()) == []


def test_exclusive_json_cleans_owned_partial_on_flush_error(tmp_path, monkeypatch):
    def fail_flush(*args):
        raise OSError('flush failed')
    monkeypatch.setattr(artifacts.os, 'fsync', fail_flush)
    with pytest.raises(OSError, match='flush failed'):
        artifacts.atomic_json({}, tmp_path / 'receipt.json', exclusive=True)
    assert list(tmp_path.iterdir()) == []
