from scripts.slow_seed_positioning import MAX_STEP_DEG, next_step

def test_next_step_is_bounded_and_converges_each_joint() -> None:
    current=(-46.0,2.5,65.5,63.0,-22.4);target=(-9.0,-14.0,40.0,68.0,-17.0)
    result=next_step(current,target)
    assert result==(-44.0,0.5,63.5,65.0,-20.4)
    assert all(abs(a-b)<=MAX_STEP_DEG for a,b in zip(result,current,strict=True))
