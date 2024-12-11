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



let audioContext, mediaRecorder, audioChunks = [];

// Start capturing audio
navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
    audioContext = new AudioContext();
    mediaRecorder = new MediaRecorder(stream);

    const audioProcessor = audioContext.createScriptProcessor(2048, 1, 1);
    const source = audioContext.createMediaStreamSource(stream);

    source.connect(audioProcessor);
    audioProcessor.connect(audioContext.destination);

    // Display audio levels
    audioProcessor.onaudioprocess = (event) => {
        const inputBuffer = event.inputBuffer.getChannelData(0);
        const rms = Math.sqrt(inputBuffer.reduce((sum, val) => sum + val * val, 0) / inputBuffer.length);
        console.log("Sample Rate:", audioContext.sampleRate);
    };

    // Capture audio chunks for backend processing
    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);   
    };

    mediaRecorder.start(1000); // Record audio in 1-second intervals
}).catch(error => {
    console.error("Error accessing microphone:", error);
});

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

        // Display processed fatigue notifications
        checkForFatigue(data.metrics);

    } catch (error) {
        console.error('Error processing frame:', error);
    }

    requestAnimationFrame(processFrames);
}

let analyser;
let audioData;
let canvas;
let canvasCtx;

function startAudioVisualizer() {
    // Set up the canvas
    canvas = document.getElementById('audio-visualizer');
    if (!canvas) {
        console.error('Canvas element with id "audio-visualizer" not found.');
        return;
    }
    canvasCtx = canvas.getContext('2d');

    // Get microphone access
    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
        // Initialize AudioContext
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);

        // Set up analyser node
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 256; // Determines the frequency data resolution
        audioData = new Uint8Array(analyser.frequencyBinCount);

        // Connect the audio source to the analyser
        source.connect(analyser);

        // Start visualizing
        visualizeAudio();
    }).catch(error => {
        console.error('Error accessing microphone:', error);
    });
}

// Function to visualize microphone audio
function visualizeAudio() {
    // Schedule the next frame
    requestAnimationFrame(visualizeAudio);

    // Get frequency data from the analyser
    analyser.getByteFrequencyData(audioData);

    // Clear the canvas
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate the height of the visualizer bar
    const barWidth = canvas.width; // Single slim bar
    const heightMultiplier = canvas.height / 256; // Scale height to fit canvas
    const barHeight = Math.max(...audioData) * heightMultiplier; // Use maximum frequency value for the bar height

    // Draw the visualizer bar
    canvasCtx.fillStyle = '#74c365'; // Bar color
    canvasCtx.fillRect(0, canvas.height - barHeight, barWidth, barHeight);
}

// Start the audio visualizer
startAudioVisualizer();

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

        // Format numeric values to two decimal places
        const formattedValue = typeof value === 'number' ? value.toFixed(2) : value;
        li.textContent = `${key}: ${formattedValue}`;

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
const csvContainer = document.querySelector('.csv-live-updates');
let currentPage = 1; // Start with the first page
let isFetching = false; // Prevent multiple simultaneous fetches
let isUserScrolling = false; // Track if the user is actively scrolling
let scrollTimeout; // Timeout to detect when scrolling stops

async function fetchCsvUpdates(page) {
    if (isFetching) return; // Avoid multiple fetch calls
    isFetching = true;

    try {
        const response = await fetch(`/get_csv_updates?page=${page}`); // Include pagination
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

        const data = await response.json();
        if (data.lines && data.lines.length > 0) {
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

            // Only auto-scroll if the user is not scrolling and the scroll is already at the bottom
            if (isAtBottom() && !isUserScrolling) {
                scrollToBottom();
            }
        } else {
            console.log('No more data available');
        }
    } catch (error) {
        console.error('Error fetching CSV updates:', error);
    } finally {
        isFetching = false;
    }
}

// Function to scroll to the bottom of the container
function scrollToBottom() {
    csvContainer.scrollTop = csvContainer.scrollHeight;
}

// Check if the scroll is at the very bottom
function isAtBottom() {
    return csvContainer.scrollTop + csvContainer.clientHeight >= csvContainer.scrollHeight - 10;
}

// Detect user scrolling
csvContainer.addEventListener('scroll', () => {
    isUserScrolling = true;

    // Reset the scrolling status after 2 seconds of no user interaction
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
        isUserScrolling = false;
    }, 2000);
});

// Initial fetch and scroll to the bottom
fetchCsvUpdates(currentPage).then(() => scrollToBottom());

// Poll for updates every 5 seconds
setInterval(() => {
    fetchCsvUpdates(currentPage);
}, 5000);


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

    let remainingTime = duration * 2; // Convert minutes to seconds
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

// Select elements
const helpButton = document.getElementById('helpButton');
const manualPopup = document.getElementById('manualPopup');
const closePopup = document.getElementById('closePopup');

// Show popup
helpButton.addEventListener('click', () => {
    manualPopup.classList.remove('hidden');
});

// Hide popup
closePopup.addEventListener('click', () => {
    manualPopup.classList.add('hidden');
});

// Hide popup when clicking outside the content area
manualPopup.addEventListener('click', (event) => {
    if (event.target === manualPopup) {
        manualPopup.classList.add('hidden');
    }
});

// Show the download options modal
function showDownloadOptions() {
    const modal = document.getElementById('downloadOptionsModal');
    modal.style.display = 'flex'; // Show modal as a flexbox for centering
}

// Close the download options modal
function closeDownloadOptions() {
    const modal = document.getElementById('downloadOptionsModal');
    modal.style.display = 'none';
}


// Handle the download based on user choice
function downloadLogs(format) {
    if (format === 'pdf') {
        fetch('/download_logs') // Fetch the entire CSV file from the server
            .then((response) => {
                if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
                return response.text();
            })
            .then((csvData) => {
                convertCsvToPdf(csvData); // Convert CSV to PDF
            })
            .catch((error) => console.error('Error fetching CSV data:', error));
    } else if (format === 'csv') {
        window.location.href = "/download_logs"; // Trigger CSV download
    }
    closeDownloadOptions();
}

// Convert CSV data to PDF
function convertCsvToPdf(csvData) {
    const { jsPDF } = window.jspdf;
    const pdf = new jsPDF();

    // Split CSV into rows and columns
    const rows = csvData.split('\n').map((line) => line.split(','));

    // Use autoTable plugin to add the CSV data to the PDF
    pdf.autoTable({
        head: [rows[0]], // First row as table headers
        body: rows.slice(1), // Remaining rows as table data
    });

    // Save the PDF
    pdf.save('logs.pdf');
}

startCamera();

