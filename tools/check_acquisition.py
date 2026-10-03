"""Acquisition regression tests using actual IQ, plus monotonic lifecycle checks."""
import sys,time,threading
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fpv_rf import acquisition,alerts,bands,dsp,sdr

class Replay:
    kind='file'
    retunable=True
    sample_rate=10_000_000
    frequency_hz=5_695_000_000
    def __init__(self, chunks): self.chunks=iter(chunks); self.calls=0
    def retune_settled(self,freq,n,timeout=2):
        self.calls+=1; self.frequency_hz=freq
        return next(self.chunks,np.zeros(0,dtype=np.complex64))

def main():
    src=sdr.SimSource(5_695_000_000)
    src.start()
    try:
        assert src.wait_for_bytes(1_200_000,timeout=3)
        iq=src.take_iq(420000)
    finally: src.stop()
    golden=sdr._u8_to_iq((Path(__file__).parent/'fixtures'/'real_5802_signed.u8').read_bytes())
    assert acquisition.confirm(Replay([golden[:420000],golden[420000:840000]]),5802000000)
    print('PASS: independent windows of the recorded hardware fixture confirm video')
    rng=np.random.default_rng(2)
    noise=(rng.normal(size=420000)+1j*rng.normal(size=420000)).astype(np.complex64)*.1
    tone=(.2*np.exp(2j*np.pi*.17*np.arange(420000))).astype(np.complex64)
    for data in [noise,tone]:
        assert acquisition.confirm(Replay([data]*3),5695000000) is None
    assert acquisition.confirm(Replay([iq,noise,noise]),5695000000) is None
    hit=acquisition.confirm(Replay([iq,iq]),5695000000)
    assert hit is not None and hit.decodable
    print('PASS: video requires persistent evidence; noise, CW and transient hits rejected')
    src=Replay([tone,iq,iq]); src.frequency_hz=5800000000
    res=acquisition.acquire(src,5650000000,5950000000,frequencies=[5695000000,5800000000])
    assert res.candidates and res.candidates[0].frequency_hz==5695000000
    assert res.hops==2 and res.lo_hz==res.hi_hz==5695000000
    print('PASS: carrier at current tuning cannot hide video on the next channel')
    cancel=threading.Event();cancel.set(); src=Replay([iq])
    res=acquisition.acquire(src,5650000000,5950000000,cancel=cancel)
    assert res.stopped_early and src.calls==0
    res=acquisition.acquire(Replay([]),5650000000,5950000000)
    assert res.error and res.failed_hops==3 and not res.candidates
    print('PASS: cancellation and missing receiver samples do not report acquisition')
    m=acquisition.Monitor(); m.completed(10)
    assert not m.due(11) and m.due(12)
    m.acquired(12); assert not m.due(14,False)
    assert not m.due(15,True); assert not m.due(17,False)
    assert m.due(18,False); assert not m.due(18,False,busy=True)
    m.enabled=False; assert not m.due(100)
    engine=alerts.AlertEngine(); engine.note(hit); engine.lost(hit.frequency_hz); engine.note(hit)
    assert [e.kind for e in engine.log]==[alerts.AlertKind.FOUND,alerts.AlertKind.LOST,alerts.AlertKind.FOUND]
    print('PASS: retry/loss deadlines, pause, busy exclusion and reacquisition alerts')
    print('RESULT: PASS')
    return 0
if __name__=='__main__': raise SystemExit(main())
