function syncSlider(el, valId) {
    document.getElementById(valId).innerText = el.value;
    const min = parseFloat(el.min);
    const max = parseFloat(el.max);
    const val = parseFloat(el.value);
    
    let pct = 50; 
    if (max !== min) {
        pct = ((val - min) / (max - min)) * 100;
    }
    el.style.setProperty('--fill', pct + '%');
}

// Removed sliderAltitude to prevent crashes
['sliderSpeed','sliderXPos', 'sliderAngle'].forEach(id => {
    const el = document.getElementById(id);
    const valId = { 
        sliderSpeed:'valSpeed', 
        sliderXPos:'valXPos',
        sliderAngle:'valAngle'
    }[id];
    syncSlider(el, valId);
});

document.getElementById('launchBtn').addEventListener('click', function() {
    const btn = this;
    btn.classList.add('busy');
    btn.textContent = '▶ SIMULATING...';
    document.getElementById('statusText').innerText = 'ACTIVE';
    document.getElementById('statusDot').style.background = '#ff4d6a';
    document.getElementById('statusDot').style.boxShadow = '0 0 8px #ff4d6a';

    const rawSpeed = document.getElementById('sliderSpeed').value;

    fetch('/launch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            altitude: 500.0, // Hardcoded the locked altitude!
            speed: -Math.abs(rawSpeed), 
            x_pos: document.getElementById('sliderXPos').value,
            angle: document.getElementById('sliderAngle').value 
        })
   }).then(() => {
        document.getElementById('video-screen').src = '/video_feed?' + Date.now();
        document.getElementById('telemetryPanel').classList.add('active');
        
        // ---> THE NEW PIPELINE: Open a permanent connection to the server <---
        const telStream = new EventSource('/telemetry_stream');
        
        // This runs automatically exactly when the server pushes new data down the pipe
        telStream.onmessage = function(event) {
            // Unpack the data
            const data = JSON.parse(event.data);
            
            if (data.altitude !== undefined) {
                // Update the text
                document.getElementById('telAlt').innerText = data.altitude.toFixed(1) + ' m';
                document.getElementById('telVelX').innerText = data.vel_x.toFixed(1) + ' m/s';
                document.getElementById('telVelY').innerText = data.vel_y.toFixed(1) + ' m/s';
                document.getElementById('telAngle').innerText = data.angle.toFixed(1) + '°';
                document.getElementById('telMain').innerText = data.main_thrust + '%';
                document.getElementById('telNose').innerText = data.nose_thrust + '%';
                document.getElementById('telCenter').innerText = data.center_thrust + '%';
                document.getElementById('telDrag').innerText = data.drag.toFixed(1) + ' N';
                
                // Update the fuel bar
                document.getElementById('telFuelTxt').innerText = Math.round(data.fuel_pct) + '%';
                const fuelFill = document.getElementById('telFuelFill');
                fuelFill.style.width = data.fuel_pct + '%';
                
                // Change color based on fuel level!
                if (data.fuel_pct > 50) fuelFill.style.background = '#00e676';
                else if (data.fuel_pct > 20) fuelFill.style.background = '#ffd600';
                else fuelFill.style.background = '#ff4d6a';
            }

            // If the simulation is over, shut down the pipeline and reset the button
            if (data.active === false) {
                telStream.close(); // Hang up the phone!
                
                btn.classList.remove('busy');
                btn.textContent = 'INITIATE DROP';
                document.getElementById('statusText').innerText = 'STANDBY';
                document.getElementById('statusDot').style.background = '';
                document.getElementById('statusDot').style.boxShadow = '';
                document.getElementById('telemetryPanel').classList.remove('active');
            }
        };
    });
});