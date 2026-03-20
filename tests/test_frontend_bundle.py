"""
Tests for the built Vite JS/CSS bundles (frontend/dist/assets/index-*.js / *.css).
Vite does NOT mangle string literals in production builds, so API route strings,
key user-visible text, and keyboard shortcut strings survive in the bundle.
Static checks only — no browser required.
"""
from pathlib import Path

DIST_ASSETS = Path(__file__).parent.parent / "frontend" / "dist" / "assets"


def _bundle() -> str:
    """Read the main JS bundle. Fails clearly if not yet built."""
    files = list(DIST_ASSETS.glob("index-*.js"))
    assert files, (
        "No built JS bundle found in frontend/dist/assets/. "
        "Run: cd frontend && npm install && npm run build"
    )
    return files[0].read_text(encoding="utf-8")


def _css() -> str:
    files = list(DIST_ASSETS.glob("index-*.css"))
    assert files, "No built CSS found in frontend/dist/assets/"
    return files[0].read_text(encoding="utf-8")


class TestAPIRoutes:
    def test_upload_endpoint(self):
        assert "/api/upload" in _bundle()

    def test_transcribe_endpoint(self):
        assert "/api/transcribe" in _bundle()

    def test_retranslate_endpoint(self):
        assert "/api/retranslate" in _bundle()

    def test_export_endpoint(self):
        assert "/api/export/" in _bundle()

    def test_projects_endpoint(self):
        assert "/api/projects" in _bundle()

    def test_models_status_endpoint(self):
        assert "/api/models/status" in _bundle()

    def test_models_download_endpoint(self):
        assert "/api/models/download/" in _bundle()

    def test_websocket_stream_endpoint(self):
        assert "/api/stream/" in _bundle()


class TestUIStrings:
    def test_app_title(self):
        assert "VaniLipi" in _bundle()

    def test_transcribe_button(self):
        assert "Transcribe" in _bundle()

    def test_record_button(self):
        assert "Record" in _bundle()

    def test_export_button(self):
        assert "Export" in _bundle()

    def test_settings_string(self):
        assert "Settings" in _bundle()

    def test_offline_claim(self):
        assert "offline" in _bundle().lower()

    def test_apple_silicon_mention(self):
        assert "Apple Silicon" in _bundle()


class TestExportFormats:
    def test_srt_format(self):
        assert "SRT" in _bundle()

    def test_vtt_format(self):
        assert "VTT" in _bundle()

    def test_txt_format(self):
        assert "TXT" in _bundle()

    def test_docx_format(self):
        assert "DOCX" in _bundle()

    def test_pdf_format(self):
        assert "PDF" in _bundle()

    def test_json_format(self):
        assert "JSON" in _bundle()


class TestLanguages:
    def test_marathi(self):
        assert "Marathi" in _bundle()

    def test_hindi(self):
        assert "Hindi" in _bundle()

    def test_auto_detect(self):
        assert "Auto-detect" in _bundle() or "auto" in _bundle()


class TestLandingPage:
    def test_landing_page_hero_text(self):
        """Landing page product copy is in the bundle."""
        assert "Transcribe" in _bundle()
        assert "Translate" in _bundle()

    def test_landing_page_languages(self):
        """Landing page shows supported languages."""
        assert "Marathi" in _bundle()
        assert "Hindi" in _bundle()


class TestKeyboardShortcuts:
    def test_space_shortcut(self):
        assert "Space" in _bundle()

    def test_meta_key_shortcuts(self):
        assert "metaKey" in _bundle() or "ctrlKey" in _bundle()

    def test_export_shortcut(self):
        # ⌘E shortcut
        bundle = _bundle()
        assert '"e"' in bundle or "'e'" in bundle

    def test_search_shortcut(self):
        # ⌘F shortcut
        bundle = _bundle()
        assert '"f"' in bundle or "'f'" in bundle

    def test_tab_navigation(self):
        assert "Tab" in _bundle()


class TestCSSVariables:
    def test_light_mode_surface(self):
        assert "#FFFFFF" in _css() or "#ffffff" in _css()

    def test_accent_color(self):
        # Amber accent palette: amber-600 (#D97706) in light, amber-400 (#FBBF24) in dark
        assert "#D97706" in _css() or "#d97706" in _css() or "#FBBF24" in _css() or "#fbbf24" in _css()

    def test_dark_mode_surface(self):
        # Stone-950 (#0C0A09) is the dark mode base
        assert "#0C0A09" in _css() or "#0c0a09" in _css()

    def test_data_theme_dark_selector(self):
        assert 'data-theme="dark"' in _css() or "[data-theme=dark]" in _css()

    def test_css_variable_definitions(self):
        css = _css()
        assert "--accent" in css
        assert "--border" in css
        assert "--surface-0" in css or "--bg-primary" in css


class TestPrivacy:
    def test_no_analytics_in_bundle(self):
        bundle = _bundle().lower()
        for tracker in ["google-analytics", "gtag(", "segment.com", "mixpanel", "hotjar"]:
            assert tracker not in bundle, f"Found analytics tracker in bundle: {tracker}"
