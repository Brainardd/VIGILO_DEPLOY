// scripts.js

async function fetchMetrics() {
    try {
        const response = await fetch('/metrics');
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const data = await response.json();
        const metricList = document.getElementById('metric-list');
        metricList.innerHTML = ''; // Clear existing metrics

        // Round and format values for better readability
        for (const [key, value] of Object.entries(data.metrics)) {
            const roundedValue = typeof value === 'number' ? value.toFixed(2) : value;
            const li = document.createElement('li');
            li.textContent = `${key}: ${roundedValue}`;
            metricList.appendChild(li);
        }
    } catch (error) {
        console.error('Error fetching metrics:', error);
    }

    setTimeout(fetchMetrics, 1000); // Refresh every second
}

function downloadLogs() {
    window.location.href = "/download_logs";
}

fetchMetrics();
