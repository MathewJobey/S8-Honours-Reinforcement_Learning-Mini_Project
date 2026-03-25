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
        setTimeout(() => {
            btn.classList.remove('busy');
            btn.textContent = 'INITIATE DROP';
            document.getElementById('statusText').innerText = 'STANDBY';
            const dot = document.getElementById('statusDot');
            dot.style.background = '';
            dot.style.boxShadow = '';
        }, 1200);
    });
});