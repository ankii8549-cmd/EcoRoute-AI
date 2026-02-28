// API Configuration
const API_BASE_URL = window.location.origin;

// Google Maps Variables
let map;
let directionsService;
let directionsRenderer;
let markers = [];
let mapsLoaded = false;

// DOM Elements
const routeForm = document.getElementById('routeForm');
const errorSection = document.getElementById('errorSection');
const errorText = document.getElementById('errorText');
const resultsSection = document.getElementById('resultsSection');
const recommendedRoute = document.getElementById('recommendedRoute');
const alternativeRoutes = document.getElementById('alternativeRoutes');
const alternativeRoutesContainer = document.getElementById('alternativeRoutesContainer');
const savingsSummary = document.getElementById('savingsSummary');
const submitButton = routeForm.querySelector('button[type="submit"]');
const btnText = submitButton.querySelector('.btn-text');
const spinner = submitButton.querySelector('.spinner');

// Initialize Google Maps
function initMap() {
    try {
        // Check if map container exists
        const mapContainer = document.getElementById('map');
        if (!mapContainer) {
            console.error('Map container not found');
            return;
        }

        // Default center (India)
        const defaultCenter = { lat: 20.5937, lng: 78.9629 };
        
        map = new google.maps.Map(mapContainer, {
            zoom: 5,
            center: defaultCenter,
            mapTypeControl: true,
            streetViewControl: false,
            fullscreenControl: true
        });
        
        directionsService = new google.maps.DirectionsService();
        directionsRenderer = new google.maps.DirectionsRenderer({
            map: map,
            suppressMarkers: false,
            polylineOptions: {
                strokeColor: '#10b981',
                strokeWeight: 5,
                strokeOpacity: 0.8
            }
        });
        
        mapsLoaded = true;
        console.log('Google Maps initialized successfully');
    } catch (error) {
        console.error('Error initializing Google Maps:', error);
    }
}

// Make initMap globally accessible
window.initMap = initMap;

// Display Route on Map
function displayRouteOnMap(source, destination) {
    if (!mapsLoaded || !directionsService || !directionsRenderer) {
        console.error('Google Maps not initialized');
        return;
    }
    
    const request = {
        origin: source,
        destination: destination,
        travelMode: google.maps.TravelMode.DRIVING,
        provideRouteAlternatives: true
    };
    
    directionsService.route(request, (result, status) => {
        if (status === 'OK') {
            directionsRenderer.setDirections(result);
            
            // Fit map to route bounds
            const bounds = new google.maps.LatLngBounds();
            result.routes[0].legs[0].steps.forEach(step => {
                bounds.extend(step.start_location);
                bounds.extend(step.end_location);
            });
            map.fitBounds(bounds);
        } else {
            console.error('Directions request failed:', status);
        }
    });
}

// Form Submission Handler
routeForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form values
    const formData = {
        vehicle_no: document.getElementById('vehicle_no').value.trim(),
        source: document.getElementById('source').value.trim(),
        destination: document.getElementById('destination').value.trim()
    };
    
    // Client-side validation
    if (!validateForm(formData)) {
        return;
    }
    
    // Clear previous results and errors
    hideError();
    hideResults();
    
    // Show loading state
    showLoading();
    
    try {
        // Call API
        const response = await fetch(`${API_BASE_URL}/eco-route`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            // Handle API errors
            throw new Error(data.detail || 'An error occurred while fetching routes');
        }
        
        // Display results
        displayResults(data);
        
        // Display route on map
        displayRouteOnMap(formData.source, formData.destination);
        
    } catch (error) {
        // Display error message
        showError(error.message);
    } finally {
        // Hide loading state
        hideLoading();
    }
});

// Client-side Validation
function validateForm(formData) {
    if (!formData.vehicle_no) {
        showError('Please enter a vehicle number');
        return false;
    }
    
    if (formData.vehicle_no.length > 50) {
        showError('Vehicle number is too long (max 50 characters)');
        return false;
    }
    
    if (!formData.source) {
        showError('Please enter a source location');
        return false;
    }
    
    if (formData.source.length > 500) {
        showError('Source location is too long (max 500 characters)');
        return false;
    }
    
    if (!formData.destination) {
        showError('Please enter a destination location');
        return false;
    }
    
    if (formData.destination.length > 500) {
        showError('Destination location is too long (max 500 characters)');
        return false;
    }
    
    return true;
}

// Display Results
function displayResults(data) {
    // Display recommended route
    recommendedRoute.innerHTML = createRouteCard(data.recommended_route, true);
    
    // Display emission savings
    if (data.emission_savings_kg > 0) {
        savingsSummary.innerHTML = `
            <h4>🎉 Emission Savings</h4>
            <div class="savings-value">${data.emission_savings_kg.toFixed(2)} kg CO₂</div>
            <div class="savings-description">
                By choosing the recommended route, you'll save ${data.emission_savings_percent.toFixed(1)}% emissions
                compared to the highest-emission alternative
            </div>
        `;
        savingsSummary.style.display = 'block';
    } else {
        savingsSummary.style.display = 'none';
    }
    
    // Display alternative routes (excluding the recommended one)
    const alternatives = data.all_routes.filter(
        route => route.route_number !== data.recommended_route.route_number
    );
    
    if (alternatives.length > 0) {
        alternativeRoutes.innerHTML = alternatives
            .map(route => createRouteCard(route, false))
            .join('');
        alternativeRoutesContainer.style.display = 'block';
    } else {
        alternativeRoutesContainer.style.display = 'none';
    }
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Create Route Card HTML
function createRouteCard(route, isRecommended) {
    const trafficClass = route.traffic_level.toLowerCase();
    const badgeHtml = isRecommended 
        ? '<span class="route-badge best">Best Choice</span>' 
        : '';
    
    return `
        <div class="route-header">
            <span class="route-number">Route ${route.route_number}</span>
            ${badgeHtml}
        </div>
        ${route.summary ? `<div class="route-summary">${route.summary}</div>` : ''}
        <div class="route-details">
            <div class="detail-item">
                <span class="detail-label">Distance</span>
                <span class="detail-value">${route.distance_km.toFixed(1)} km</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Duration</span>
                <span class="detail-value">${formatDuration(route.duration_minutes)}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">Traffic Level</span>
                <span class="detail-value">
                    <span class="traffic-indicator ${trafficClass}">${route.traffic_level}</span>
                </span>
            </div>
            <div class="detail-item">
                <span class="detail-label">CO₂ Emission</span>
                <span class="detail-value emission">${route.predicted_co2_kg.toFixed(2)} kg</span>
            </div>
        </div>
    `;
}

// Format Duration
function formatDuration(minutes) {
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    
    if (hours > 0) {
        return `${hours}h ${mins}m`;
    }
    return `${mins}m`;
}

// Show Error
function showError(message) {
    errorText.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Hide Error
function hideError() {
    errorSection.style.display = 'none';
    errorText.textContent = '';
}

// Show Results
function showResults() {
    resultsSection.style.display = 'block';
}

// Hide Results
function hideResults() {
    resultsSection.style.display = 'none';
}

// Show Loading State
function showLoading() {
    submitButton.disabled = true;
    btnText.textContent = 'Finding Routes...';
    spinner.style.display = 'inline-block';
}

// Hide Loading State
function hideLoading() {
    submitButton.disabled = false;
    btnText.textContent = 'Find Eco-Routes';
    spinner.style.display = 'none';
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('Vehicle Emission Eco-Route System initialized');
});

// Make initMap available globally for Google Maps callback
window.initMap = initMap;
