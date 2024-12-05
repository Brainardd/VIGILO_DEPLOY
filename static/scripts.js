const video = document.getElementById('video'); // Use the existing video element

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        video.addEventListener('loadedmetadata', () => {
            processFrames();
        });
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

async function processFrames() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame onto the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frameData = canvas.toDataURL('image/jpeg');

    try {
        const response = await fetch('/process_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame: frameData }),
        });

        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const data = await response.json();

        // Update metrics on the frontend
        updateMetrics(data.metrics);

        // Display the processed frame
        const processedFrame = document.getElementById('processed-frame');
        processedFrame.src = data.processed_frame;

    } catch (error) {
        console.error('Error processing frame:', error);
    }

    requestAnimationFrame(processFrames);
}

const csvFileElement = document.getElementById('csv-filename');

// Fetch and display the CSV filename
async function fetchCsvFilename() {
    try {
        const response = await fetch('/get_csv_filename');
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const data = await response.json();
        if (data.csv_filename) {
            csvFileElement.textContent = `CSV File: ${data.csv_filename}`;
            csvFileElement.dataset.filename = data.csv_filename; // Store the filename for the download
        } else {
            csvFileElement.textContent = "CSV File: Not Available";
        }
    } catch (error) {
        console.error('Error fetching CSV filename:', error);
        csvFileElement.textContent = "CSV File: Error fetching file";
    }
}

// Call this function on page load
fetchCsvFilename();

function updateMetrics(metrics, csvFilename) {
    const metricList = document.getElementById('metric-list');
    metricList.innerHTML = ''; // Clear existing metrics

    for (const [key, value] of Object.entries(metrics)) {
        const li = document.createElement('li');
        li.textContent = `${key}: ${value}`;
        metricList.appendChild(li);
    }

    // Check for fatigue
    checkForFatigue(metrics);

    // Update the CSV filename display
    if (csvFilename) {
        const csvFileElement = document.getElementById('csv-filename');
        csvFileElement.textContent = `CSV File: ${csvFilename}`;
    }
}


function downloadLogs() {
    window.location.href = "/download_logs"; // Trigger file download
}

// Fetch the filename on page load
fetchCsvFilename();

const csvUpdatesElement = document.getElementById('csv-updates');

async function fetchCsvUpdates() {
    try {
        const response = await fetch('/get_csv_updates');
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const data = await response.json();
        if (data.lines) {
            csvUpdatesElement.innerHTML = ''; // Clear the previous content

            data.lines.forEach((line) => {
                const columns = line.trim().split(','); // Split CSV row into columns
                const row = document.createElement('tr');

                // Create table cells for each column
                columns.forEach((col) => {
                    const cell = document.createElement('td');
                    cell.textContent = col.trim();
                    row.appendChild(cell);
                });

                csvUpdatesElement.appendChild(row);
            });
        } else {
            csvUpdatesElement.innerHTML = '<tr><td colspan="9">No data available</td></tr>';
        }
    } catch (error) {
        console.error('Error fetching CSV updates:', error);
        csvUpdatesElement.innerHTML = '<tr><td colspan="9">Error fetching updates</td></tr>';
    }
}

// Poll for updates every 5 seconds
setInterval(fetchCsvUpdates, 5000);

// Fetch updates initially on page load
fetchCsvUpdates();

// Modal Elements
const fatigueModal = document.getElementById('fatigue-modal');
const breakTimerModal = document.getElementById('break-timer-modal');
const timerDisplay = document.getElementById('timer-display');
const startBreakButtons = document.querySelectorAll('.start-break');
const closeTimerButton = document.getElementById('close-timer-btn');

// Audio Elements
const notificationSound = document.getElementById('notification-sound');
const breakEndSound = document.getElementById('break-end-sound');

// Fatigue Detection Counter
let fatigueCounter = 0;

// Variable to track the timer and looping sound
let breakInterval = null;
let breakEndInterval = null; // Interval for looping the break-end sound

// Show fatigue notification
function showFatigueNotification() {
    fatigueModal.style.display = 'flex'; // Show modal
    notificationSound.play(); // Play notification sound
}

// Hide fatigue notification
function hideFatigueNotification() {
    fatigueModal.style.display = 'none'; // Hide modal
}

// Show break timer modal
function showBreakTimerModal() {
    breakTimerModal.style.display = 'flex'; // Show modal
}

// Hide break timer modal and stop break-end sound
function hideBreakTimerModal() {
    breakTimerModal.style.display = 'none'; // Hide modal
    stopBreakEndSound(); // Stop looping sound
}

// Start looping the break-end sound
function startBreakEndSound() {
    breakEndSound.play();
    breakEndInterval = setInterval(() => {
        breakEndSound.currentTime = 0; // Restart the sound
        breakEndSound.play();
    }, breakEndSound.duration * 1000); // Loop after the sound duration
}

// Stop looping the break-end sound
function stopBreakEndSound() {
    clearInterval(breakEndInterval); // Stop the interval
    breakEndSound.pause(); // Pause the sound
    breakEndSound.currentTime = 0; // Reset the sound to the beginning
}

// Start break timer
function startBreakTimer(duration) {
    hideFatigueNotification();
    showBreakTimerModal();

    let remainingTime = duration * 60; // Convert minutes to seconds
    updateTimerDisplay(remainingTime);

    // Clear any existing timer
    clearInterval(breakInterval);

    breakInterval = setInterval(() => {
        remainingTime--;
        updateTimerDisplay(remainingTime);

        if (remainingTime <= 0) {
            clearInterval(breakInterval);
            startBreakEndSound(); // Start looping the break-end sound
        }
    }, 1000);
}

// Update timer display
function updateTimerDisplay(seconds) {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    timerDisplay.textContent = `${minutes}:${secs.toString().padStart(2, '0')}`;
}

// Attach event listeners to break buttons
startBreakButtons.forEach((button) => {
    button.addEventListener('click', () => {
        const duration = parseInt(button.dataset.duration, 10); // Get duration from button
        startBreakTimer(duration);
    });
});

// Close the break timer modal when the user clicks the Close Timer button
closeTimerButton.addEventListener('click', () => {
    clearInterval(breakInterval); // Stop the timer
    hideBreakTimerModal(); // Close the modal
    alert('You ended the break early!');
});

// Detect fatigue and show notification if it occurs 3 consecutive times
function checkForFatigue(metrics) {
    if (
        metrics.Fatigue === 'Fatigue Detected' ||
        metrics.Fatigue === 'Fatigue Detected While Yawning'
    ) {
        fatigueCounter++;
        if (fatigueCounter >= 3) {
            showFatigueNotification();
            fatigueCounter = 0; // Reset counter after showing notification
        }
    } else {
        fatigueCounter = 0; // Reset counter if no fatigue detected
    }
}

startCamera();
