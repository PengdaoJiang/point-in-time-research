import tempfile
from checkpoint_demo import demonstrate

def test_checkpoint_walkthrough():
    with tempfile.TemporaryDirectory() as root:
        result=demonstrate(root)
        assert result['resume_verified']
        assert not result['changed_binding_reused']
        assert not result['cutoff_readiness']['actual_fit_verified']
