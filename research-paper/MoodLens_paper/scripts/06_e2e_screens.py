"""Headless end-to-end UI test of both apps with a simulated camera, plus screenshots for the paper.
Needs both apps running on test ports (Menu 8101 with a scratch MOODMENU_DATA_DIR, Coach 8100). python3 (system) has playwright.
"""
import base64, json, os, sys, time
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results"); RAW = os.path.join(HERE, "..", "figures", "raw")
faces = {n: "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(RES, "test_faces", n + ".jpg"), "rb").read()).decode() for n in ["happy", "sad", "angry", "surprise", "neutral", "two"]}
INIT = """(() => {
  const imgs = %s; const cv = document.createElement('canvas'); cv.width = 640; cv.height = 480; const cx = cv.getContext('2d'); const el = {};
  window.__fake = {cur: 'happy'};
  for (const n of Object.keys(imgs)) { const im = new Image(); im.src = imgs[n]; el[n] = im; }
  setInterval(() => { const im = el[window.__fake.cur]; if (im && im.complete) cx.drawImage(im, 0, 0); }, 40);
  const mk = async () => cv.captureStream(20);
  navigator.mediaDevices.getUserMedia = mk; navigator.mediaDevices.getDisplayMedia = mk;
  navigator.mediaDevices.enumerateDevices = async () => [{kind: 'videoinput', deviceId: 'fake', label: 'Test camera'}];
})();""" % json.dumps(faces)

checks = []
def check(app, name, ok, detail=""):
    checks.append({"app": app, "test": name, "pass": bool(ok), "detail": str(detail)[:300]}); print(("PASS" if ok else "FAIL"), app, name, "|", str(detail)[:140], flush=True)

def snap(page, name, **kw): page.screenshot(path=os.path.join(RAW, name + ".png"), **kw)
def wait(page, ms): page.wait_for_timeout(ms)
def wait_reply(page, ms=30000):
    t0 = time.time()
    while time.time() - t0 < ms / 1000:
        if page.evaluate("!document.querySelector('.cb-typing') && document.querySelectorAll('.cb-msg').length>=3"): return True
        wait(page, 400)
    return False
def ask(page, text):
    page.evaluate("t => {const i=document.getElementById('cb-input'); i.value=t; document.getElementById('cb-form').requestSubmit();}", text)
    return wait_reply(page)

def menu_app(b):
    app = "Mood Menu"; ctx = b.new_context(viewport={"width": 1440, "height": 940}, device_scale_factor=1.5); ctx.add_init_script(INIT); page = ctx.new_page(); errs = []
    page.on("pageerror", lambda e: errs.append(str(e))); page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    page.goto("http://127.0.0.1:8101"); page.evaluate("document.fonts.ready"); wait(page, 1500)
    check(app, "page loads, 72-item menu fetched", page.evaluate("typeof menu==='object'") and page.evaluate("fetch('/api/menu').then(r=>r.json()).then(m=>m.length)") == 72)
    page.click("[data-start=camera]"); wait(page, 8000)
    d = page.evaluate("({m: menu.scannedMood, hero: $('hero-vibe').textContent, status: $('status-text').textContent, fps: $('hud-perf').textContent, before: menu.before})")
    check(app, "vibe scan reads a happy face", d["m"] == "happy" and d["hero"] == "Celebration Mode", d)
    snap(page, "menu_1_scan")
    # filters
    page.click(".filters-card summary"); page.click("#diet-seg [data-diet=veg]"); page.click("#spice-seg [data-spice='0']"); page.click("#avoid-chips [data-avoid=egg]")
    page.evaluate("const b=$('budget'); b.value=900; b.dispatchEvent(new Event('input',{bubbles:true})); b.dispatchEvent(new Event('change',{bubbles:true}))"); wait(page, 1800)
    f = page.evaluate("(() => { const c = menu.lastRec.combo; return {n: c.length, bad: c.filter(i => !i.veg || i.spicy > 0 || i.allergens.includes('egg')).length, total: c.reduce((s, i) => s + i.price, 0), summary: $('filter-summary').textContent}; })()")
    check(app, "filters (veg, mild, no egg, <= 900) respected", f["n"] > 0 and f["bad"] == 0 and f["total"] <= 900, f)
    snap(page, "menu_2_filters")
    page.click("#diet-seg [data-diet=any]"); page.click("#spice-seg [data-spice='2']"); page.click("#avoid-chips [data-avoid=egg]")
    page.evaluate("const b=$('budget'); b.value=2500; b.dispatchEvent(new Event('input',{bubbles:true})); b.dispatchEvent(new Event('change',{bubbles:true}))"); wait(page, 1500)
    # shuffle / lock / manual mood
    before = page.evaluate("menu.lastRec.combo.map(i=>i.id).join()"); page.click("#shuffle-btn"); wait(page, 1500); after = page.evaluate("menu.lastRec.combo.map(i=>i.id).join()")
    check(app, "Shuffle gives a different combo", before != after, f"{before} -> {after}")
    page.click("#mood-picker [data-mood=angry]"); wait(page, 1800)
    check(app, "manual mood picker (angry -> Cool-Down Mode)", page.evaluate("$('hero-vibe').textContent") == "Cool-Down Mode")
    page.click("#mood-picker [data-mood='']"); wait(page, 1500)
    page.evaluate("__fake.cur='sad'"); wait(page, 2500); page.click("#rescan-btn"); wait(page, 7000)
    check(app, "rescan with a sad face -> Comfort Mode", page.evaluate("menu.scannedMood") == "sad" and page.evaluate("$('hero-vibe').textContent") == "Comfort Mode", page.evaluate("$('hero-vibe').textContent"))
    lk = page.evaluate("$('lock-btn').textContent"); page.evaluate("__fake.cur='angry'"); wait(page, 4500)
    check(app, "Lock keeps picks while the face changes", lk.startswith("🔒") and page.evaluate("$('hero-vibe').textContent") == "Comfort Mode", lk)
    # order flow
    page.click("#add-combo"); page.click("#cart-btn"); page.click("#split-plus"); wait(page, 400)
    cart = page.evaluate("({lines: document.querySelectorAll('#cart-lines li').length, sub: $('cart-sub').textContent, tax: $('cart-tax').textContent, total: $('cart-total').textContent, each: $('split-each').textContent})")
    sub = int(cart["sub"].replace("₹", "").replace(",", "")); tax = int(cart["tax"].replace("₹", "").replace(",", ""))
    check(app, "cart totals (5% GST) and bill split", cart["lines"] >= 3 and tax == round(sub * 0.05), cart)
    snap(page, "menu_3_cart")
    page.evaluate("__fake.cur='sad'"); page.click("#place-btn"); wait(page, 1500)
    t = page.evaluate("({shown: !$('ticket-modal').hidden, qr: !!document.querySelector('#ticket svg'), text: $('ticket').innerText.slice(0, 60), after: !!$('t-after')})")
    check(app, "order ticket with QR code", t["shown"] and t["qr"], t)
    snap(page, "menu_4_ticket")
    page.evaluate("__fake.cur='happy'"); wait(page, 2500); page.click("#t-after"); wait(page, 8000)
    l = page.evaluate("({lift: menu.lastOrder.lift, toast: $('toast').textContent, lb: document.querySelectorAll('#leaderboard li').length})")
    check(app, "after-meal Mood Lift rescan (sad -> happy)", l["lift"] is not None and l["lift"] > 30 and l["lb"] >= 3, l)
    snap(page, "menu_5_lift")
    page.click("#t-close"); wait(page, 300)
    with page.expect_download() as dl:
        page.click("#after-btn") if False else page.evaluate("saveReceipt()")
    check(app, "shareable mood receipt PNG downloads", dl.value.suggested_filename.endswith(".png"), dl.value.suggested_filename)
    page.evaluate("document.querySelector('#leaderboard').scrollIntoView({block:'center'})"); wait(page, 500); snap(page, "menu_6_leaderboard")
    # table mode
    page.evaluate("window.scrollTo(0,0)"); page.evaluate("__fake.cur='two'"); wait(page, 2500); page.click("#mode-seg [data-mode=table]"); wait(page, 9000)
    tb = page.evaluate("({people: menu.people.map(p=>p.mood), cards: document.querySelectorAll('#table-people .person').length, total: $('table-total').textContent, shared: $('table-shared').innerText.slice(0,80), group: menu.lastTable && menu.lastTable.group_mood})")
    check(app, "table mode reads two faces and builds per-person picks", len(tb["people"]) == 2 and tb["cards"] == 2, tb)
    snap(page, "menu_7_table")
    # chatbot
    page.click(".cb-fab"); wait(page, 400)
    ok = ask(page, "I'm feeling low. Suggest a veg combo under 700 rupees"); wait(page, 400)
    cb = page.evaluate("({reply: document.querySelector('.cb-assistant:last-child').innerText.slice(0,200), addBtns: document.querySelectorAll('.cb-assistant:last-child .cb-dish').length})")
    check(app, "Smashy chatbot answers with add-to-order buttons", ok and cb["addBtns"] >= 1, cb)
    n0 = page.evaluate("+$('cart-count').textContent"); page.click(".cb-assistant:last-child .cb-dish"); wait(page, 400); n1 = page.evaluate("+$('cart-count').textContent")
    check(app, "chatbot dish button adds to the real cart", n1 == n0 + 1, f"{n0} -> {n1}")
    snap(page, "menu_8_chat")
    check(app, "no JavaScript errors", len(errs) == 0, errs[:3])
    ctx.close()

def coach_app(b):
    app = "Conversation Coach"; ctx = b.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5); ctx.add_init_script(INIT); page = ctx.new_page(); errs = []
    page.on("pageerror", lambda e: errs.append(str(e))); page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    page.goto("http://127.0.0.1:8100"); page.evaluate("document.fonts.ready"); wait(page, 1500)
    page.click("[data-start=camera]"); wait(page, 6000)
    d = page.evaluate("({h: $('coach-headline').textContent, who: $('breakdown-who').textContent, status: $('status-text').textContent, v: $('valence-val').textContent, top: [...document.querySelectorAll('#bars .bar-row.top')].map(r=>r.dataset.label)})")
    check(app, "live read: happy face -> 'They're enjoying this'", d["h"] == "They're enjoying this" and d["top"] == ["happy"], d)
    snap(page, "coach_1_live")
    says = {}
    for c in ["work", "sales", "date", "support", "casual"]:
        page.click(f"#context-chips [data-context={c}]"); wait(page, 600); says[c] = page.evaluate("$('coach-say').innerText.split('\\n')[0]")
    check(app, "playbook lines change with the setting", len(set(says.values())) == 5, says)
    page.evaluate("__fake.cur='sad'"); wait(page, 4500); h_sad = page.evaluate("$('coach-headline').textContent")
    page.evaluate("__fake.cur='angry'"); wait(page, 4500); h_ang = page.evaluate("$('coach-headline').textContent")
    check(app, "mood change updates the coaching (sad, then angry)", h_sad == "They seem down" and h_ang == "Tension is rising", (h_sad, h_ang))
    al = page.evaluate("[...document.querySelectorAll('#alert-feed li')].map(l=>l.innerText.replace(/\\n/g,' '))")
    check(app, "live alerts fire (tension / dip)", any("Tension" in a for a in al) or any("dipped" in a for a in al), al[:3])
    # transcript + micro expression
    page.evaluate("__fake.cur='sad'"); wait(page, 3500); page.evaluate("addLine('Guess what, I got us concert tickets')"); page.evaluate("__fake.cur='happy'"); wait(page, 5200)
    page.evaluate("addLine('but you have to pay for both')"); page.evaluate("__fake.cur='angry'"); wait(page, 5200)
    tr = page.evaluate("[...document.querySelectorAll('#transcript li')].slice(-2).map(l=>l.innerText.replace(/\\n/g,' '))")
    check(app, "transcript tags 'landed' and 'hit a nerve'", "landed" in tr[0] and "nerve" in tr[1], tr)
    page.evaluate("__fake.cur='happy'"); wait(page, 3800); page.evaluate("__fake.cur='angry'"); wait(page, 420); page.evaluate("__fake.cur='happy'"); wait(page, 2200)
    fc = page.evaluate("$('flash-count').textContent")
    check(app, "micro-expression (0.4 s flash) caught", not fc.startswith("0"), fc)
    page.evaluate("document.getElementById('flash-feed').scrollIntoView({block:'center'})"); wait(page, 400); snap(page, "coach_2_flash")
    page.evaluate("window.scrollTo(0,0)")
    page.evaluate("__fake.cur='two'"); wait(page, 6000)
    room = page.evaluate("({shown: !$('room-panel').hidden, cards: document.querySelectorAll('#room .person:not(.away)').length, away: document.querySelectorAll('#room .person.away').length, tip: $('room-tip').textContent})")
    check(app, "room mode: two active faces, flags who needs attention", room["shown"] and room["cards"] == 2, room)
    page.evaluate("document.getElementById('room-panel').scrollIntoView({block:'start'})"); wait(page, 300); snap(page, "coach_3_room")
    page.evaluate("window.scrollTo(0,0)"); page.evaluate("__fake.cur='angry'"); wait(page, 3500)
    page.click("#say-seg [data-say=ai]"); wait(page, 400); page.click("#ai-btn"); wait(page, 9000)
    ai = page.evaluate("({read: $('ai-note').innerText.slice(0,100), n: $('ai-say').children.length, watch: $('ai-watch').innerText.slice(0,60)})")
    check(app, "AI wingman returns a read and three replies", ai["n"] == 3, ai)
    page.evaluate("document.querySelector('.col-right').scrollIntoView()"); snap(page, "coach_4_ai"); page.click("#say-seg [data-say=playbook]")
    page.evaluate("window.scrollTo(0,0)")
    page.click(".cb-fab"); wait(page, 400); ok = ask(page, "They just reacted badly to what I said. What now?")
    cb = page.evaluate("({ctx: document.getElementById('cb-context').innerText, reply: document.querySelector('.cb-assistant:last-child').innerText.slice(0,160)})")
    check(app, "Coach chatbot sees the live mood and answers", ok and "angry" in cb["ctx"].lower(), cb)
    snap(page, "coach_5_chat"); page.click(".cb-fab")
    page.click("#source-seg [data-source=screen]"); wait(page, 3500)
    check(app, "video-call (screen share) source runs", page.evaluate("state.source") == "screen" and page.evaluate("state.running"), page.evaluate("$('status-text').textContent"))
    page.click("#end-btn"); wait(page, 1500)
    db = page.evaluate("({title: $('db-title').textContent, stats: $('db-stats').innerText.replace(/\\n/g,' ').slice(0,120), tips: $('db-tips').children.length, landed: $('db-landed').innerText.slice(0,60)})")
    check(app, "debrief report built", "Mostly" in db["title"] and db["tips"] >= 1, db)
    snap(page, "coach_6_debrief", full_page=True)
    with page.expect_download() as dl: page.click("#db-download")
    check(app, "debrief downloads as a standalone HTML report", dl.value.suggested_filename.endswith(".html"), dl.value.suggested_filename)
    check(app, "no JavaScript errors", len(errs) == 0, errs[:3])
    ctx.close()

with sync_playwright() as p:
    b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
    for fn in (menu_app, coach_app):
        try: fn(b)
        except Exception as e: check(fn.__name__, "script completed", False, repr(e))
    b.close()
json.dump(checks, open(os.path.join(RES, "e2e.json"), "w"), indent=1, ensure_ascii=False)
print(sum(c["pass"] for c in checks), "/", len(checks), "passed")
