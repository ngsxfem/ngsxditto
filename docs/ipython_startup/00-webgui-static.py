# IPython startup for the Sphinx/nbsphinx docs build, loaded via IPYTHONDIR
# (see docs/generate_docs.sh). nbsphinx executes every notebook in a fresh
# IPython kernel, and IPython kernels run every profile_default/startup/*.py
# before any notebook cell -- so this patches ngsolve.webgui.Draw without the
# example notebooks needing to know about it (they keep the plain
# `from ngsolve.webgui import Draw` / `import ngsolve.webgui as ngw`).
#
# Each scene is shown as a lightweight PREVIEW image; the interactive 3D
# (webgui) is loaded only when the reader clicks -- similar to the NGSolve
# documentation. This keeps the pages small and fast to load.
#
#   * a headless-chromium screenshot of the scene becomes the preview JPEG,
#   * the full self-contained scene is written to $WEBGUI_SCENE_DIR (copied
#     into the built site under <base>/webgui_scenes/ by generate_docs.sh),
#     and loaded into the iframe on click,
#   * if chromium / the scene dir is unavailable, we fall back to a plain
#     click-to-load box (or, outside a static build, to the eager scene).
import base64
import hashlib
import html as _htmlesc
import os
import subprocess
import tempfile
import warnings

_SCENE_DIR = os.environ.get("WEBGUI_SCENE_DIR")        # e.g. <docs>/webgui_scenes
_BASE = os.environ.get("WEBGUI_BASE", "")              # optional absolute prefix, e.g. /myproject


def _scene_url(name):
    # Relative by default: the built pages are flat (build/html/*.html) with the
    # scenes next to them in webgui_scenes/, and a relative URL keeps working
    # when the site is served under a subpath (ngsxfem.github.io/ngsxditto/,
    # GitLab Pages /<project>/) -- an absolute "/webgui_scenes/..." 404s there.
    # srcdoc iframes inherit the parent page's base URL, so the relative link
    # resolves against the page that embeds the preview.
    return (_BASE + "/" if _BASE else "") + "webgui_scenes/" + name


def _screenshot(html):
    """Headless-chromium screenshot of a webgui scene -> small JPEG data-uri (or None)."""
    d = tempfile.mkdtemp()
    hp = os.path.join(d, "s.html")
    png = os.path.join(d, "s.png")
    with open(hp, "w") as f:
        f.write(html)
    # Headless chromium needs no X display -- but a *stale* DISPLAY (the docs CI
    # exports one for VTK/pyvista) makes its GPU process try GLX on a display
    # that may not answer, and the WebGL canvas then silently stays blank
    # (white screenshots, no error). Scrub it from the subprocess environment.
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    for exe in ("chromium", "chromium-browser", "google-chrome"):
        try:
            if os.path.exists(png):                # don't mistake a previous
                os.remove(png)                     # attempt's file for ours
            subprocess.run(
                [exe, "--headless=new", "--disable-gpu", "--no-sandbox",
                 "--disable-dev-shm-usage",          # use /tmp, not a tiny Docker /dev/shm
                 "--use-gl=angle", "--use-angle=swiftshader-webgl",
                 "--enable-unsafe-swiftshader", "--hide-scrollbars",
                 "--window-size=1000,540", "--virtual-time-budget=12000",
                 "--screenshot=" + png, "file://" + hp],
                timeout=120, capture_output=True, check=False, env=env)
            if os.path.exists(png):
                try:                                   # PNG of a 3-D scene is huge; ship a small JPEG still
                    import io
                    from PIL import Image
                    im = Image.open(png).convert("RGB")
                    # A (nearly) uniform image means the WebGL canvas never
                    # rendered (e.g. no usable GL in the build environment).
                    # Better an honest click-to-load box than a white "preview"
                    # -- and the note makes the failure visible in the build log.
                    if all(hi - lo < 8 for lo, hi in im.getextrema()):
                        print("webgui preview screenshot is blank (" + exe
                              + ") -- falling back to click-to-load")
                        continue
                    im.thumbnail((760, 760))
                    buf = io.BytesIO(); im.save(buf, "JPEG", quality=70, optimize=True)
                    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
                except Exception:
                    return "data:image/png;base64," + base64.b64encode(open(png, "rb").read()).decode()
        except (FileNotFoundError, subprocess.SubprocessError):
            continue
    return None


def _iframe(srcdoc, extra=""):
    return ('<iframe srcdoc="' + _htmlesc.escape(srcdoc, quote=True) + '" '
            'style="width:100%;height:520px;border:0;border-radius:6px;" '
            + extra + '></iframe>')


class _DummyScene:
    """Stand-in for the WebGLScene normally returned by Draw(show=False).

    A few ngsxditto visualizations (e.g. UnfittedNGSWebguiScene) keep the
    returned scene around and call .Redraw() on it after every step to
    update a *live* widget. There is no live widget in a static preview, so
    Redraw() is a no-op here -- the reader still sees the first frame and can
    click through to the interactive scene.
    """
    def Redraw(self, *args, **kwargs):
        pass

    def GenerateHTML(self, *args, **kwargs):
        return ""

    def _repr_html_(self):
        # keep the Out[...] cell empty -- without this, notebooks whose last
        # expression is the Draw() result render "<__main__._DummyScene at 0x...>"
        return ""


try:
    import ngsolve.webgui as _ngw
    from IPython.display import display as _display, HTML as _HTML

    _orig_draw = _ngw.Draw

    def _static_draw(obj, *args, **kwargs):
        try:
            scene_html = _orig_draw(obj, *args, show=False, **kwargs).GenerateHTML()
        except Exception as _de:
            # A single un-drawable object must NOT break the whole page: show a
            # visible note instead (also surfaces in the build log).
            msg = "webgui Draw skipped: " + type(_de).__name__ + ": " + str(_de)
            print(msg)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _display(_HTML('<div style="padding:10px;border:1px dashed #c0392b;'
                               'border-radius:6px;color:#c0392b;font-family:system-ui">'
                               '⚠ ' + _htmlesc.escape(msg) + '</div>'))
            return _DummyScene()

        preview = _screenshot(scene_html) if _SCENE_DIR else None
        if preview and _SCENE_DIR:
            # lightweight preview that loads the external scene on click
            os.makedirs(_SCENE_DIR, exist_ok=True)
            name = hashlib.sha1(scene_html.encode()).hexdigest() + ".html"
            with open(os.path.join(_SCENE_DIR, name), "w") as f:
                f.write(scene_html)
            url = _scene_url(name)
            srcdoc = (
                '<!DOCTYPE html><html><head><style>'
                'html,body{margin:0;height:100%;overflow:hidden;'
                'font-family:system-ui,sans-serif}'
                '#p{width:100%;height:100%;object-fit:contain;background:#f3f4f6;'
                'display:block;filter:grayscale(35%)}'
                # greyish tint overlay -> clearly a still, not the live webgui
                '#o{position:absolute;inset:0;cursor:pointer;'
                'background:rgba(100,110,130,.32);'
                'display:flex;align-items:center;justify-content:center}'
                '#b{background:rgba(20,20,20,.62);color:#fff;padding:9px 18px;'
                'border-radius:24px;font-size:15px}'
                '</style></head><body>'
                '<img id="p" src="' + preview + '">'
                '<div id="o" onclick="location.href=\'' + url + '\'">'
                '<div id="b">▶ click to load interactive 3D</div></div>'
                '</body></html>')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # IPython "use IFrame" notice
                _display(_HTML(_iframe(srcdoc)))
        elif _SCENE_DIR:
            # No preview (chromium missing/failed) but we ARE the static build: write the scene
            # out and offer a plain click-to-load. NEVER inline the full scene here -- that
            # would bloat the page and defeat the point of the static preview.
            os.makedirs(_SCENE_DIR, exist_ok=True)
            name = hashlib.sha1(scene_html.encode()).hexdigest() + ".html"
            with open(os.path.join(_SCENE_DIR, name), "w") as f:
                f.write(scene_html)
            url = _scene_url(name)
            srcdoc = (
                '<!DOCTYPE html><html><head><style>'
                'html,body{margin:0;height:100%;font-family:system-ui,sans-serif}'
                '#o{position:absolute;inset:0;cursor:pointer;background:#eef0f4;'
                'display:flex;align-items:center;justify-content:center}'
                '#b{background:rgba(20,20,20,.62);color:#fff;padding:9px 18px;'
                'border-radius:24px;font-size:15px}'
                '</style></head><body>'
                '<div id="o" onclick="location.href=\'' + url + '\'">'
                '<div id="b">▶ click to load interactive 3D</div></div>'
                '</body></html>')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _display(_HTML(_iframe(srcdoc)))
        else:
            # live frontend, no scene dir: render the full scene eagerly (still isolated)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _display(_HTML(_iframe(scene_html, 'loading="lazy"')))

        return _DummyScene()

    _ngw.Draw = _static_draw
except Exception as _e:                            # pragma: no cover
    print("webgui static-startup patch failed:", _e)
