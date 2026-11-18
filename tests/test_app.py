import sys, itertools, time, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest


def test_every_filter_combination_runs():
    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    sites = at.selectbox[0].options; weeks = at.selectbox[1].options
    print(len(sites), "sites", len(weeks), "weeks")
    n = 0; t = time.time()
    for s, w in itertools.product(sites, weeks):
        at.selectbox[0].set_value(s); at.selectbox[1].set_value(w); at.run()
        assert not at.exception, (s, w, at.exception); n += 1
    print("site x week ok", n, f"{time.time()-t:.0f}s")
    for u in [50, 95]:
        at.slider[0].set_value(u); at.run(); assert not at.exception
    at.radio[0].set_value("80% upper bound"); at.run(); assert not at.exception
    # sidebar extremes and closed store
    labels = [sl.label for sl in at.slider]
    def sl(label): return [x for x in at.slider if x.label.startswith(label)][0]
    sl("Capture rate").set_value(6.0); sl("Conversion").set_value(80); sl("Staff minutes").set_value(15.0); at.run(); assert not at.exception
    sl("Capture rate").set_value(0.5); sl("Conversion").set_value(5); sl("Staff minutes").set_value(1.0); at.run(); assert not at.exception
    sl("Opening hours Mon").set_value((9, 9)); sl("Opening hours Sunday").set_value((11, 11)); at.run()
    assert not at.exception, at.exception
    print("closed-store messages:", [m.value[:60] for m in at.warning] + [m.value[:60] for m in at.info])
    sl("Opening hours Mon").set_value((6, 23)); sl("Opening hours Sunday").set_value((6, 23)); at.run(); assert not at.exception
    print("all ok")

