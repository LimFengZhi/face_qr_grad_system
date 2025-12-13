let statusInterval = null;
let queueStarted = false;

function showTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    if (tab === 'scanner') {
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
        document.getElementById('scanner-tab').classList.add('active');
    } else {
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
        document.getElementById('queue-tab').classList.add('active');
        loadStudents();
        loadQueue();
        loadStats();
        loadAllStudentsForDelete();  // Add this line
    }
}

async function startQueue() {
    const res = await fetch('/api/start', {method: 'POST'});
    const data = await res.json();
    if (data.success) {
        queueStarted = true;
        document.getElementById('video-placeholder').style.display = 'none';
        document.getElementById('video-feed').style.display = 'block';
        document.getElementById('video-feed').src = '/video_feed?' + Date.now();
        document.getElementById('start-btn').style.display = 'none';
        document.getElementById('stop-btn').style.display = 'inline-block';
        document.getElementById('skip-btn').style.display = 'inline-block';
        document.getElementById('waiting-info').style.display = 'block';
        startStatusPolling();
    } else {
        alert(data.error || 'Failed to start');
    }
}

async function stopQueue() {
    await fetch('/api/stop', {method: 'POST'});
    queueStarted = false;
    document.getElementById('video-feed').style.display = 'none';
    document.getElementById('video-placeholder').style.display = 'flex';
    document.getElementById('start-btn').style.display = 'inline-block';
    document.getElementById('stop-btn').style.display = 'none';
    document.getElementById('skip-btn').style.display = 'none';
    document.getElementById('waiting-info').style.display = 'none';
    stopStatusPolling();
}

async function skipCurrent() {
    await fetch('/api/skip', {method: 'POST'});
}

function startStatusPolling() {
    statusInterval = setInterval(updateStatus, 200);
}

function stopStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);
}

async function updateStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        document.getElementById('algo-display').textContent = data.algorithm;
        
        // Update face status
        const faceStatus = document.getElementById('face-status');
        if (data.face_detected) {
            const expectedId = data.current_student ? data.current_student.student_id : null;
            if (expectedId && String(data.face_id) === String(expectedId)) {
                faceStatus.className = 'status-box success';
                faceStatus.textContent = '✅ Face: ' + data.face_id;
            } else if (data.face_id) {
                faceStatus.className = 'status-box warning';
                faceStatus.textContent = '⚠️ Face: ' + data.face_id + ' (Expected: ' + expectedId + ')';
            } else {
                faceStatus.className = 'status-box info';
                faceStatus.textContent = '👤 Face: Unknown';
            }
        } else {
            faceStatus.className = 'status-box error';
            faceStatus.textContent = '❌ No Face';
        }
        
        // Update QR status
        const qrStatus = document.getElementById('qr-status');
        if (data.qr_detected) {
            const expectedId = data.current_student ? data.current_student.student_id : null;
            if (expectedId && String(data.qr_id) === String(expectedId)) {
                qrStatus.className = 'status-box success';
                qrStatus.textContent = '✅ QR: ' + data.qr_id;
            } else {
                qrStatus.className = 'status-box warning';
                qrStatus.textContent = '⚠️ QR: ' + data.qr_id + ' (Expected: ' + expectedId + ')';
            }
        } else {
            qrStatus.className = 'status-box error';
            qrStatus.textContent = '❌ No QR';
        }
        
        // Update match status
        const matchStatus = document.getElementById('match-status');
        if (data.verification_state === 'displaying') {
            matchStatus.className = 'status-box success';
            matchStatus.textContent = '✅ VERIFIED';
        } else if (data.face_id && data.qr_id && data.current_student) {
            const expectedId = data.current_student.student_id;
            const faceMatch = String(data.face_id) === String(expectedId);
            const qrMatch = String(data.qr_id) === String(expectedId);
            const faceQrMatch = String(data.face_id) === String(data.qr_id);
            
            if (faceMatch && qrMatch) {
                matchStatus.className = 'status-box success';
                matchStatus.textContent = '✅ Match!';
            } else if (!faceMatch && !qrMatch) {
                // Both wrong
                matchStatus.className = 'status-box error';
                matchStatus.textContent = '❌ Wrong Face & QR';
            } else if (!faceMatch) {
                // Face wrong, QR correct
                matchStatus.className = 'status-box warning';
                matchStatus.textContent = '⚠️ Wrong Face (QR OK)';
            } else if (!qrMatch) {
                // Face correct, QR wrong
                matchStatus.className = 'status-box warning';
                matchStatus.textContent = '⚠️ Wrong QR (Face OK)';
            } else if (!faceQrMatch) {
                // Face and QR don't match each other
                matchStatus.className = 'status-box error';
                matchStatus.textContent = '❌ Face ≠ QR Mismatch';
            }
        } else if (data.face_id && !data.qr_id) {
            matchStatus.className = 'status-box info';
            matchStatus.textContent = '👤 Face detected, waiting QR...';
        } else if (!data.face_id && data.qr_id) {
            matchStatus.className = 'status-box info';
            matchStatus.textContent = '📱 QR detected, waiting Face...';
        } else {
            matchStatus.className = 'status-box info';
            matchStatus.textContent = '⏳ Scanning...';
        }

        
        // Update waiting info
        if (data.current_student) {
            document.getElementById('waiting-name').textContent = data.current_student.name;
            document.getElementById('waiting-pos').textContent = `Position: ${data.current_index + 1}/${data.total}`;
        }
        
        // Update verified student
        const verifiedCard = document.getElementById('verified-card');
        if (data.verification_state === 'displaying' && data.verified_student) {
            let html = '';
            if (data.verified_img) {
                html += `<img src="${data.verified_img}" alt="Student">`;
            }
            html += `<h3>🎓 ${data.verified_student.name}</h3>`;
            html += `<div class="graduation-level">${data.graduation_level || ''}</div>`;
            html += `<div class="student-details">`;
            html += `<p><strong>ID:</strong> ${data.verified_student.student_id}</p>`;
            html += `<p><strong>Faculty:</strong> ${data.verified_student.faculty || 'N/A'}</p>`;
            html += `<p><strong>Course:</strong> ${data.verified_student.course || 'N/A'}</p>`;
            if (data.verified_student.cgpa) {
                html += `<p><strong>CGPA:</strong> ${data.verified_student.cgpa}</p>`;
            }
            html += `</div>`;
            html += `<div class="countdown">⏱️ ${data.remaining}s</div>`;
            verifiedCard.innerHTML = html;
        } else if (!data.queue_started) {
            verifiedCard.innerHTML = '<p style="color:#666;">Waiting for verification...</p>';
        }
        
        // Check if done
        if (!data.queue_started && queueStarted) {
            stopQueue();
            if (data.current_index >= data.total) {
                alert('🎉 All students verified!');
            }
        }
    } catch (e) {
        console.error(e);
    }
}

async function updateSettings() {
    const algo = document.getElementById('algorithm-select').value;
    const tts = document.getElementById('tts-checkbox').checked;
    
    await fetch('/api/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({algorithm: algo, tts_enabled: tts})
    });
}

async function loadStudents() {
    try {
        // Only get students who haven't attended yet
        const res = await fetch('/api/students?show_attended=false');
        const students = await res.json();
        const queueRes = await fetch('/api/queue');
        const queueData = await queueRes.json();
        const queueIds = queueData.queue.map(s => s.student_id);
        
        const select = document.getElementById('student-select');
        if (!select) return;  // Exit if element doesn't exist
        
        select.innerHTML = '<option value="">Select student...</option>';
        students.forEach(s => {
            if (!queueIds.includes(s.student_id)) {
                select.innerHTML += `<option value="${s.student_id}">${s.name} (${s.student_id})</option>`;
            }
        });
        console.log(`Loaded ${students.length} students`);  // Debug log
    } catch (e) {
        console.error('Error loading students:', e);
    }
}

async function loadQueue() {
    const res = await fetch('/api/queue');
    const data = await res.json();
    const list = document.getElementById('queue-list');
    
    if (data.queue.length === 0) {
        list.innerHTML = '<p style="color:#666; text-align:center;">Queue empty</p>';
        return;
    }
    
    list.innerHTML = '';
    data.queue.forEach((s, i) => {
        const cls = i < data.current_index ? 'queue-item done' : (i === data.current_index ? 'queue-item current' : 'queue-item');
        const status = s.attended ? '✅' : '⏳';
        list.innerHTML += `
            <div class="${cls}">
                <div>
                    <span class="name">${i+1}. ${s.name}</span>
                    <span class="id">${s.student_id}</span>
                    <span>${status}</span>
                </div>
                <div class="queue-actions">
                    <button class="remove-btn" onclick="removeFromQueue(${i})" title="Remove from queue">✕</button>
                </div>
            </div>
        `;
    });
}

async function loadStats() {
    const res = await fetch('/api/stats');
    const stats = await res.json();
    document.getElementById('stat-attended').textContent = stats.attended;
    document.getElementById('stat-total').textContent = stats.total;
    document.getElementById('stat-rate').textContent = stats.attendance_rate + '%';
}

async function addStudent() {
    const id = document.getElementById('student-select').value;
    if (!id) return;
    await fetch('/api/queue/add', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({student_id: id})
    });
    loadStudents();
    loadQueue();
}

async function addAllStudents() {
    await fetch('/api/queue/add_all', {method: 'POST'});
    loadStudents();
    loadQueue();
}

async function clearQueue() {
    await fetch('/api/queue/clear', {method: 'POST'});
    loadStudents();
    loadQueue();
}

async function removeFromQueue(idx) {
    await fetch('/api/queue/remove', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({index: idx})
    });
    loadStudents();
    loadQueue();
}

async function resetAttendance() {
    if (!confirm('⚠️ Are you sure you want to reset ALL attendance records?')) {
        return;
    }
    
    await fetch('/api/reset_attendance', {method: 'POST'});
    // Wait a moment for database to update
    await new Promise(resolve => setTimeout(resolve, 100));
    // Reload all data
    await loadStats();
    await loadStudents();
    await loadQueue();
    await loadAllStudentsForDelete();  // Add this line to refresh delete list
    
    // Force refresh the select element
    const select = document.getElementById('student-select');
    if (select) {
        select.dispatchEvent(new Event('change'));
    }
    
    alert('✅ Attendance reset successfully!');
}

async function toggleEmail() {
    const enabled = document.getElementById('email-checkbox').checked;
    
    await fetch('/api/email/toggle', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({enabled: enabled})
    });
}

async function loadEmailStatus() {
    try {
        const res = await fetch('/api/email/status');
        const data = await res.json();
        document.getElementById('email-checkbox').checked = data.enabled;
    } catch (e) {
        console.error('Error loading email status:', e);
    }
}

// ==================== REGISTRATION FUNCTIONS ====================
let regFormValid = false;
let regFaceCaptured = false;

function initRegistrationPage() {
    // Check if we're on registration page
    const cameraFeed = document.getElementById('camera-feed');
    if (!cameraFeed) return;
    
    // Start camera feed - FIX: add /api prefix
    cameraFeed.src = '/api/register/video_feed?' + Date.now();
    
    // Setup validation listeners
    setupRegistrationValidation();
}

function setupRegistrationValidation() {
    const studentIdInput = document.getElementById('student_id');
    const nameInput = document.getElementById('name');
    const emailInput = document.getElementById('email');
    
    if (!studentIdInput) return;
    
    // Validate student ID (check if exists)
    studentIdInput.addEventListener('blur', async function() {
        const id = this.value.trim();
        const errorEl = document.getElementById('student_id_error');
        
        if (!id) {
            this.classList.remove('valid', 'error');
            errorEl.textContent = '';
            return;
        }
        
        try {
            const res = await fetch('/api/register/check_id', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({student_id: id})
            });
            const data = await res.json();
            
            if (data.valid) {
                this.classList.remove('error');
                this.classList.add('valid');
                errorEl.textContent = '';
            } else {
                this.classList.remove('valid');
                this.classList.add('error');
                errorEl.textContent = data.error;
            }
        } catch (e) {
            console.error('Error checking student ID:', e);
        }
        validateRegistrationForm();
    });
    
    // Validate email format
    emailInput.addEventListener('blur', function() {
        const email = this.value.trim();
        const errorEl = document.getElementById('email_error');
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        
        if (email && !emailRegex.test(email)) {
            this.classList.add('error');
            errorEl.textContent = 'Invalid email format';
        } else {
            this.classList.remove('error');
            errorEl.textContent = '';
        }
        validateRegistrationForm();
    });
    
    // Validate name
    nameInput.addEventListener('blur', function() {
        const name = this.value.trim();
        const errorEl = document.getElementById('name_error');
        
        if (name && name.length < 2) {
            this.classList.add('error');
            errorEl.textContent = 'Name must be at least 2 characters';
        } else {
            this.classList.remove('error');
            errorEl.textContent = '';
        }
        validateRegistrationForm();
    });
    
    // Validate form on any input change
    document.querySelectorAll('#register-form input, #register-form select').forEach(el => {
        el.addEventListener('input', validateRegistrationForm);
    });
}

function validateRegistrationForm() {
    const studentId = document.getElementById('student_id');
    const name = document.getElementById('name');
    const email = document.getElementById('email');
    const faculty = document.getElementById('faculty');
    const course = document.getElementById('course');
    const captureBtn = document.getElementById('capture-btn');
    const captureStatus = document.getElementById('capture-status');
    
    if (!studentId) return;
    
    const hasErrors = document.querySelectorAll('.form-group input.error').length > 0;
    const allFilled = studentId.value && name.value && email.value && 
                      faculty.value && course.value;
    
    regFormValid = allFilled && !hasErrors && studentId.classList.contains('valid');
    
    if (captureBtn) {
        captureBtn.disabled = !regFormValid;
    }
    
    if (regFormValid) {
        document.getElementById('step1').classList.add('completed');
        document.getElementById('step2').classList.add('active');
        if (captureStatus) {
            captureStatus.textContent = '📷 Position your face and click Capture';
        }
    }
}

async function captureFace() {
    const studentId = document.getElementById('student_id').value;
    const statusEl = document.getElementById('capture-status');
    const cameraFeed = document.getElementById('camera-feed');
    
    statusEl.className = 'capture-status pending';
    statusEl.textContent = '⏳ Capturing face...';
    
    try {
        const res = await fetch('/api/register/capture_face', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({student_id: studentId})
        });
        const data = await res.json();
        
        if (data.success) {
            regFaceCaptured = true;
            statusEl.className = 'capture-status success';
            statusEl.textContent = '✅ ' + data.message;
            document.getElementById('submit-btn').disabled = false;
            document.getElementById('step2').classList.add('completed');
            document.getElementById('step3').classList.add('active');
            
            // Show the captured preview image instead of live feed
            if (data.preview) {
                cameraFeed.src = data.preview;
            }
            
            // Disable capture button after successful capture
            document.getElementById('capture-btn').textContent = '✅ Face Captured';
            document.getElementById('capture-btn').disabled = true;
        } else {
            statusEl.className = 'capture-status error';
            statusEl.textContent = '❌ ' + data.error;
        }
    } catch (e) {
        statusEl.className = 'capture-status error';
        statusEl.textContent = '❌ Error capturing face';
        console.error(e);
    }
}

async function submitRegistration() {
    if (!regFormValid || !regFaceCaptured) return;
    
    const submitBtn = document.getElementById('submit-btn');
    submitBtn.disabled = true;
    submitBtn.textContent = '⏳ Registering...';
    
    const formData = {
        student_id: document.getElementById('student_id').value,
        name: document.getElementById('name').value,
        email: document.getElementById('email').value,
        faculty: document.getElementById('faculty').value,
        course: document.getElementById('course').value,
        cgpa: document.getElementById('cgpa').value
    };
    
    try {
        const res = await fetch('/api/register/submit', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(formData)
        });
        const data = await res.json();
        
        const resultDiv = document.getElementById('registration-result');
        resultDiv.style.display = 'block';
        
        if (data.success) {
            document.getElementById('step3').classList.add('completed');
            
            // Show algorithm registration status
            let statusHtml = '<strong>✅ Registration Complete!</strong><br><br>';
            statusHtml += '<strong>Registered in:</strong><br>';
            data.registered_algorithms.forEach(algo => {
                statusHtml += `<span class="success">✓ ${algo}</span><br>`;
            });
            
            if (data.failed_algorithms && data.failed_algorithms.length > 0) {
                statusHtml += '<br><strong>Warnings:</strong><br>';
                data.failed_algorithms.forEach(fail => {
                    statusHtml += `<span class="failed">✗ ${fail}</span><br>`;
                });
            }
            
            statusHtml += '<br><em>QR code generated in data/qr_codes/</em>';
            resultDiv.innerHTML = statusHtml;
            
            submitBtn.textContent = '✅ Registered!';
            submitBtn.style.background = '#2ecc71';
            
            // Redirect after 3 seconds
            setTimeout(() => {
                window.location.href = '/';
            }, 3000);
        } else {
            resultDiv.innerHTML = `<span class="failed">❌ ${data.error}</span>`;
            if (data.details) {
                data.details.forEach(d => {
                    resultDiv.innerHTML += `<br><span class="failed">• ${d}</span>`;
                });
            }
            submitBtn.disabled = false;
            submitBtn.textContent = '✅ Complete Registration';
        }
    } catch (e) {
        document.getElementById('registration-result').style.display = 'block';
        document.getElementById('registration-result').innerHTML = '<span class="failed">❌ Error submitting registration</span>';
        submitBtn.disabled = false;
        submitBtn.textContent = '✅ Complete Registration';
        console.error(e);
    }
}

// ==================== DELETE STUDENT FUNCTIONS ====================

async function loadAllStudentsForDelete() {
    try {
        const res = await fetch('/api/students?show_attended=true');
        const students = await res.json();
        
        const select = document.getElementById('delete-student-select');
        if (!select) return;
        
        select.innerHTML = '<option value="">Select student to delete...</option>';
        students.forEach(s => {
            // Show ✅ if attended, empty if not
            const attended = s.attended ? ' ✅' : '';
            select.innerHTML += `<option value="${s.student_id}" data-name="${s.name}">${s.name} (${s.student_id})${attended}</option>`;
        });
    } catch (e) {
        console.error('Error loading students for delete:', e);
    }
}

async function deleteSelectedStudent() {
    const select = document.getElementById('delete-student-select');
    const studentId = select.value;
    
    if (!studentId) {
        alert('Please select a student to delete');
        return;
    }
    
    const studentName = select.options[select.selectedIndex].dataset.name || studentId;
    await deleteStudentEntirely(studentId, studentName);
    
    // Reload the delete dropdown
    loadAllStudentsForDelete();
}

async function deleteStudentEntirely(studentId, studentName) {
    if (!confirm(`⚠️ Are you sure you want to PERMANENTLY delete ${studentName} (${studentId})?\n\nThis will remove:\n• Database record\n• Face encodings (all algorithms)\n• Student image\n• QR code\n\nThis action cannot be undone!`)) {
        return;
    }
    
    try {
        const res = await fetch('/api/student/delete', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({student_id: studentId})
        });
        const data = await res.json();
        
        if (data.success) {
            alert(`✅ ${data.message}`);
            if (data.warnings && data.warnings.length > 0) {
                console.warn('Deletion warnings:', data.warnings);
            }
            // Refresh all lists
            loadStudents();
            loadQueue();
            loadStats();
            loadAllStudentsForDelete();
        } else {
            alert(`❌ ${data.error}`);
        }
    } catch (e) {
        alert('❌ Error deleting student');
        console.error(e);
    }
}

// Update DOMContentLoaded to also init registration
document.addEventListener('DOMContentLoaded', function() {
    loadQueue();
    loadStats();
    loadEmailStatus();
    loadAllStudentsForDelete();  // Add this line
    initRegistrationPage();
});