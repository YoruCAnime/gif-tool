"""The watchdog with made-up readings: no video card work, no heavy memory. test_watchdog.py"""
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cgm import watchdog as w

OK_CARD = (60.0, 0.3)


def guard(ram=8.0, app=1.0, card=OK_CARD, **kw):
    state = {"ram": ram, "app": app, "card": card, "now": 0.0}
    said = []
    g = w.Guard(say=said.append, ram=lambda: state["ram"], app=lambda: state["app"], card=lambda: state["card"],
                clock=lambda: state["now"], look_every=0.02, card_every=0.0, **kw)
    return g, state, said


g, s, said = guard()
g.look()
assert g.waiting is None and g.ended is None
print("all fine: job runs")

g, s, said = guard(ram=1.2)
g.look()
assert g.waiting and "memory" in g.waiting and not g.ended, g.waiting
s["ram"] = 3.0
g.look()
assert g.waiting is None and said[-1] == "Carrying on"
print("low memory: waits, then carries on")

g, s, said = guard(ram=0.5)
g.look()
assert g.ended and "out of memory" in g.ended
print("nearly out of memory: ends:", g.ended)

g, s, said = guard(app=7.5)
g.look()
assert g.ended and "too big" in g.ended
g, s, said = guard(app=6.2)
g.look()
assert g.waiting and not g.ended
print("the app itself too big: waits at 6.2 GB, ends at 7.5 GB")

g, s, said = guard(card=(84.0, 0.3))
g.look()
assert g.waiting and "Cooling" in g.waiting
s["card"] = (78.0, 0.3)
g.look()
assert g.waiting, "must keep cooling until 75 C"
s["card"] = (75.0, 0.3)
g.look()
assert g.waiting
s["card"] = (74.0, 0.3)
g.look()
assert g.waiting is None
print("hot card: waits from 83 C down to under 75 C")

g, s, said = guard(card=(60.0, 0.95))
g.look()
assert g.waiting and "memory" in g.waiting
print("card memory full: waits")

g, s, said = guard(ram=1.0, patience=100.0)
g.look()
s["now"] = 99.0
g.look()
assert not g.ended
s["now"] = 101.0
g.look()
assert g.ended and "too long" in g.ended and "GB" in g.ended
print("no improvement: gives up:", g.ended)

g, s, said = guard()
g.look()
s["card"] = None
g.look()
assert not g.ended, "one missed answer is not a lost card"
g.look()
assert g.ended and "stopped responding" in g.ended
print("card stops answering: ends:", g.ended)

g, s, said = guard(card=None)
g.look()
g.look()
g.look()
assert not g.ended and not g.waiting
print("PC without an NVIDIA card: not treated as lost")

# the real path: a job waiting in check(), released, then ended; and Stop while waiting
g, s, said = guard(ram=1.0)
with g:
    done = []

    def job():
        w.check()
        done.append("passed")
        while True:
            try:
                w.check()
            except w.Stopped as e:
                done.append(str(e))
                return
            time.sleep(0.01)

    t = threading.Thread(target=job, daemon=True)
    t.start()
    time.sleep(0.2)
    assert not done, "job must wait while memory is low"
    s["ram"] = 8.0
    time.sleep(0.2)
    assert done == ["passed"], done
    s["ram"] = 0.4
    t.join(3)
    assert len(done) == 2 and "out of memory" in done[1], done
print("job in a thread: waits, carries on, then ends with the message")

g, s, said = guard(ram=1.0)
stop = threading.Event()
stop.set()
with g:
    try:
        w.check(stop)
        raise SystemExit("Stop was ignored")
    except InterruptedError:
        pass
print("Stop button works during a wait")
w.check()
print("no guard running: check does nothing")

print("real readings: %.1f GB free, app %.2f GB, card %s" % (w.read_ram(), w.read_app(), w.read_card()))
print("OK")
