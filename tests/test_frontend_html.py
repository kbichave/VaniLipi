"""
Tests for the built Vite frontend HTML template (frontend/dist/index.html).
Static checks only — no browser required.
"""
from pathlib import Path

DIST_DIR = Path(__file__).parent.parent / "frontend" / "dist"
INDEX_HTML = DIST_DIR / "index.html"


def _html() -> str:
    assert INDEX_HTML.exists(), (
        "frontend/dist/index.html not found. "
        "Run: cd frontend && npm install && npm run build"
    )
    return INDEX_HTML.read_text(encoding="utf-8")


class TestHTMLStructure:
    def test_file_exists(self):
        assert INDEX_HTML.exists()

    def test_has_doctype(self):
        assert "<!doctype html>" in _html().lower()

    def test_has_root_div(self):
        assert 'id="root"' in _html()

    def test_has_module_script(self):
        # Vite injects a <script type="module"> pointing to the hashed bundle
        assert 'type="module"' in _html()

    def test_assets_dir_referenced(self):
        assert "/assets/" in _html()

    def test_no_cdn_react(self):
        # Should NOT have CDN react — it's bundled now
        html = _html()
        assert "unpkg.com/react" not in html
        assert "cdn.jsdelivr.net/npm/react" not in html

    def test_no_babel_standalone(self):
        assert "babel/standalone" not in _html()

    def test_no_tailwind_cdn(self):
        assert "cdn.tailwindcss.com" not in _html()


class TestHTMLMetadata:
    def test_has_lang_attribute(self):
        assert 'lang="en"' in _html()

    def test_has_viewport_meta(self):
        assert "viewport" in _html()

    def test_has_charset_meta(self):
        assert "UTF-8" in _html() or "utf-8" in _html().lower()

    def test_has_title(self):
        assert "<title>" in _html()
        assert "VaniLipi" in _html()

    def test_data_theme_on_html(self):
        assert "data-theme" in _html()


class TestPrivacy:
    def test_no_analytics_in_html(self):
        html = _html().lower()
        for tracker in ["google-analytics", "gtag", "segment.com", "mixpanel", "hotjar"]:
            assert tracker not in html, f"Found analytics tracker in HTML: {tracker}"

    def test_no_external_fonts_cdn(self):
        # Fonts are now self-hosted via @fontsource — not loaded from Google CDN
        assert "fonts.googleapis.com" not in _html()


class TestAssetsExist:
    def test_assets_directory_exists(self):
        assert (DIST_DIR / "assets").exists()

    def test_js_bundle_exists(self):
        js_files = list((DIST_DIR / "assets").glob("index-*.js"))
        assert len(js_files) >= 1, "No built JS bundle found in dist/assets/"

    def test_css_bundle_exists(self):
        css_files = list((DIST_DIR / "assets").glob("index-*.css"))
        assert len(css_files) >= 1, "No built CSS bundle found in dist/assets/"

    def test_font_files_bundled(self):
        # Inter and Noto Sans Devanagari woff2 files should be in dist/assets/
        font_files = list((DIST_DIR / "assets").glob("*.woff2"))
        assert len(font_files) > 0, "No woff2 font files found in dist/assets/"
