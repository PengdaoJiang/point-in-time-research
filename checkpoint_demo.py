"""Offline failure/recovery walkthrough using the actual checkpoint kernel."""
import hashlib,json,tempfile
from pathlib import Path
from qplatform.research_checkpoints import CheckpointStore
from qplatform.research_readiness import audit_label_support

def demonstrate(root):
    root=Path(root);inputs=root/'input';inputs.mkdir()
    (inputs/'unit.json').write_text('{"synthetic_total": 7}\n',encoding='utf-8')
    binding=hashlib.sha256(b'original synthetic experiment binding').hexdigest()
    store=CheckpointStore(root/'checkpoints',binding)
    first=store.seal('window-1',inputs,['unit.json']);again=store.load('window-1')
    assert first['receipt_sha256']==again['receipt_sha256']
    changed=CheckpointStore(root/'checkpoints',hashlib.sha256(b'changed input binding').hexdigest())
    assert changed.load('window-1') is None
    (again['directory']/'unit.json').write_text('tampered',encoding='utf-8')
    try:store.load('window-1')
    except ValueError as e:rejection=str(e)
    else:raise AssertionError('tampering must not be accepted')
    readiness=audit_label_support([
        dict(status='RESOLVED_MARKET',value=0.01,available_at='2026-01-02T00:00:00+00:00',training_role='action')],
        cutoff='2026-01-02T00:00:00+00:00',required_roles=['action'])
    assert not readiness['support_available'] and not readiness['actual_fit_verified']
    return dict(resume_verified=True,changed_binding_reused=False,tampering_rejected=rejection,cutoff_readiness=readiness,market_performance_claim=False)

if __name__=='__main__':
    with tempfile.TemporaryDirectory(prefix='checkpoint-demo-') as d:print(json.dumps(demonstrate(d),indent=2))
