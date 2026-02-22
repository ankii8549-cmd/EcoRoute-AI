"""
Basic frontend validation tests.

Tests verify:
- Static files exist and are accessible
- HTML structure is valid
- JavaScript file is present
- CSS file is present

Requirements: 5.1, 5.8
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Import the FastAPI app
from app.main import app

# Create test client
client = TestClient(app)


class TestStaticFiles:
    """Tests for static file existence and accessibility"""
    
    def test_static_directory_exists(self):
        """Test that static directory exists"""
        static_dir = Path("app/static")
        assert static_dir.exists()
        assert static_dir.is_dir()
    
    def test_index_html_exists(self):
        """Test that index.html exists"""
        index_file = Path("app/static/index.html")
        assert index_file.exists()
        assert index_file.is_file()
    
    def test_styles_css_exists(self):
        """Test that styles.css exists"""
        css_file = Path("app/static/styles.css")
        assert css_file.exists()
        assert css_file.is_file()
    
    def test_app_js_exists(self):
        """Test that app.js exists"""
        js_file = Path("app/static/app.js")
        assert js_file.exists()
        assert js_file.is_file()
    
    def test_index_html_accessible_via_api(self):
        """Test that index.html is accessible through the API"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_static_css_accessible(self):
        """Test that CSS file is accessible through static route"""
        response = client.get("/app/static/styles.css")
        assert response.status_code == 200
        assert "text/css" in response.headers["content-type"]
    
    def test_static_js_accessible(self):
        """Test that JavaScript file is accessible through static route"""
        response = client.get("/app/static/app.js")
        assert response.status_code == 200
        # JavaScript can be served as application/javascript or text/javascript
        assert any(ct in response.headers["content-type"] for ct in ["javascript", "text/plain"])


class TestHTMLStructure:
    """Tests for HTML structure validation"""
    
    def test_html_has_doctype(self):
        """Test that HTML file has DOCTYPE declaration"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "<!DOCTYPE html>" in content or "<!doctype html>" in content
    
    def test_html_has_head_section(self):
        """Test that HTML has head section"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "<head>" in content
        assert "</head>" in content
    
    def test_html_has_body_section(self):
        """Test that HTML has body section"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "<body>" in content
        assert "</body>" in content
    
    def test_html_has_title(self):
        """Test that HTML has title tag"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "<title>" in content
        assert "</title>" in content
    
    def test_html_links_to_css(self):
        """Test that HTML links to CSS file"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "styles.css" in content
        assert '<link' in content and 'stylesheet' in content
    
    def test_html_links_to_js(self):
        """Test that HTML links to JavaScript file"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        assert "app.js" in content
        assert "<script" in content
    
    def test_html_has_form_elements(self):
        """Test that HTML has form elements for user input"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have form or input elements
        assert "<form" in content or "<input" in content
    
    def test_html_has_vehicle_input(self):
        """Test that HTML has vehicle number input field"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have input for vehicle number
        assert "vehicle" in content.lower()
    
    def test_html_has_location_inputs(self):
        """Test that HTML has source and destination input fields"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have inputs for source and destination
        assert "source" in content.lower()
        assert "destination" in content.lower()
    
    def test_html_has_submit_button(self):
        """Test that HTML has submit button"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have button or submit input
        assert "<button" in content or 'type="submit"' in content
    
    def test_html_has_results_section(self):
        """Test that HTML has section for displaying results"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have some element for results (div, section, etc.)
        assert "result" in content.lower() or "route" in content.lower()


class TestCSSContent:
    """Tests for CSS file content"""
    
    def test_css_file_not_empty(self):
        """Test that CSS file is not empty"""
        with open("app/static/styles.css", "r", encoding="utf-8") as f:
            content = f.read()
        
        assert len(content.strip()) > 0
    
    def test_css_has_styling_rules(self):
        """Test that CSS has styling rules"""
        with open("app/static/styles.css", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have CSS selectors and rules
        assert "{" in content
        assert "}" in content
        assert ":" in content


class TestJavaScriptContent:
    """Tests for JavaScript file content"""
    
    def test_js_file_not_empty(self):
        """Test that JavaScript file is not empty"""
        with open("app/static/app.js", "r", encoding="utf-8") as f:
            content = f.read()
        
        assert len(content.strip()) > 0
    
    def test_js_has_api_endpoint_reference(self):
        """Test that JavaScript references API endpoints"""
        with open("app/static/app.js", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should reference the eco-route endpoint
        assert "eco-route" in content or "/eco-route" in content
    
    def test_js_has_fetch_or_ajax(self):
        """Test that JavaScript has API call functionality"""
        with open("app/static/app.js", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should use fetch or XMLHttpRequest for API calls
        assert "fetch" in content or "XMLHttpRequest" in content or "axios" in content
    
    def test_js_has_event_listeners(self):
        """Test that JavaScript has event listeners for user interaction"""
        with open("app/static/app.js", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have event listeners
        assert "addEventListener" in content or "onclick" in content or ".on(" in content
    
    def test_js_has_error_handling(self):
        """Test that JavaScript has error handling"""
        with open("app/static/app.js", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have try-catch or .catch for error handling
        assert "catch" in content or "error" in content.lower()


class TestResponsiveDesign:
    """Tests for responsive design elements"""
    
    def test_html_has_viewport_meta(self):
        """Test that HTML has viewport meta tag for mobile responsiveness"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have viewport meta tag
        assert "viewport" in content
        assert "width=device-width" in content
    
    def test_css_has_media_queries(self):
        """Test that CSS has media queries for responsive design"""
        with open("app/static/styles.css", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have media queries for different screen sizes
        # This is optional but recommended
        has_media_queries = "@media" in content
        
        # If no media queries, at least should have flexible units
        has_flexible_units = "%" in content or "rem" in content or "em" in content or "vw" in content
        
        assert has_media_queries or has_flexible_units


class TestAccessibility:
    """Basic accessibility tests"""
    
    def test_html_has_lang_attribute(self):
        """Test that HTML has lang attribute"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have lang attribute on html tag
        assert 'lang=' in content
    
    def test_form_inputs_have_labels(self):
        """Test that form inputs have associated labels"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have label elements
        if "<input" in content:
            assert "<label" in content or "aria-label" in content or "placeholder" in content


class TestUIComponents:
    """Tests for specific UI components"""
    
    def test_has_loading_indicator_element(self):
        """Test that HTML has element for loading indicator"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have some element for loading state
        assert "loading" in content.lower() or "spinner" in content.lower() or "loader" in content.lower()
    
    def test_has_error_display_element(self):
        """Test that HTML has element for error messages"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have element for displaying errors
        assert "error" in content.lower() or "message" in content.lower()
    
    def test_has_route_display_elements(self):
        """Test that HTML has elements for displaying route information"""
        with open("app/static/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Should have elements for route display
        assert "route" in content.lower() or "result" in content.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
