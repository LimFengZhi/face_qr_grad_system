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
        const status = i < data.current_index ? '✅' : '⏳';
        list.innerHTML += `
            <div class="${cls}">
                <div>
                    <span class="name">${i+1}. ${s.name}</span>
                    <span class="id">${s.student_id}</span>
                    <span>${status}</span>
                </div>
                <button class="remove-btn" onclick="removeFromQueue(${i})">✕</button>
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
    await fetch('/api/reset_attendance', {method: 'POST'});
    // Wait a moment for database to update
    await new Promise(resolve => setTimeout(resolve, 100));
    // Reload all data
    await loadStats();
    await loadStudents();
    await loadQueue();
    // Force refresh the select element
    const select = document.getElementById('student-select');
    if (select) {
        select.dispatchEvent(new Event('change'));
    }
}
// Initial load
document.addEventListener('DOMContentLoaded', function() {
    loadQueue();
    loadStats();
});