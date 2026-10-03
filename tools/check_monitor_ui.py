"""Exercise transmitter on/off/reacquire through the real Qt scan thread."""
import os,sys,time
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from fpv_rf import ui,sdr,bands,alerts
from tools.check_ui import Args
app=QApplication.instance() or QApplication([])
src=sdr.SimSource(5695000000,video=False,signal_hz=5695000000)
src.start(); win=ui.MainWindow(src,Args())
win._active_band=bands.Band('T','test',(5695000000,))
win.monitor.retry_s=.2;win.monitor.loss_s=.5
win.continuous.setChecked(True); win.show()
start=time.monotonic(); stage=0; error=[]
def poll():
 global stage
 try:
  age=time.monotonic()-start
  found=win.alerts.counts[alerts.AlertKind.FOUND]
  lost=win.alerts.counts[alerts.AlertKind.LOST]
  if age>15: raise AssertionError(f'timeout stage={stage} found={found} lost={lost}')
  if stage==0 and age>1:
   assert found==0,'noise caused an acquisition'
   src.video=True;stage=1
  elif stage==1 and found==1 and win.worker.locked:
   src.video=False;stage=2
  elif stage==2 and lost==1:
   src.video=True;stage=3
  elif stage==3 and found==2:
   win._on_cancel();assert not win.monitor.enabled
   print('PASS: quiet startup, transmitter appearance, loss, reacquisition and Stop')
   stage=4;timer.stop();win.close();QTimer.singleShot(600,app.quit)
 except Exception as exc:
  error.append(str(exc));timer.stop();win.close();QTimer.singleShot(600,app.quit)
timer=QTimer();timer.timeout.connect(poll);timer.start(50)
app.exec();src.stop()
print('RESULT: FAIL '+str(error) if error or stage!=4 else 'RESULT: PASS')
raise SystemExit(1 if error or stage!=4 else 0)
