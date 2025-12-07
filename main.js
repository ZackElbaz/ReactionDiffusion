// Karl Sims Reaction-Diffusion - Starting from scratch
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true });

if (!gl) {
    alert('WebGL not supported');
    throw new Error('WebGL not supported');
}

// Set canvas to fill window
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

// Simulation parameters
let feed = 0.037;
let kill = 0.06;
let currentColorMap = 'grayscale';
let gradientOrientation = 'right-left';

// Simple vertex shader
const vertexShaderSource = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;
    void main() {
        v_texCoord = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

// STEP 1: Just display what we have (for testing)
const displayShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;
    uniform int u_colorMap;

    void main() {
        vec2 state = texture2D(u_state, v_texCoord).rg;
        float A = state.r;
        float B = state.g;

        // Simple grayscale: show B concentration
        // B=0 (no pattern) = white
        // B=1 (full pattern) = black
        vec3 color = vec3(1.0 - B);

        gl_FragColor = vec4(color, 1.0);
    }
`;

// STEP 2: Initialize with random blobs
const initShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform float u_seed;

    float random(vec2 st) {
        return fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453 * u_seed);
    }

    void main() {
        // Start with A=1, B=0 everywhere
        float A = 1.0;
        float B = 0.0;

        // Create some random blobs of B
        float rand = random(v_texCoord * 10.0);
        if (rand > 0.95) {
            B = 1.0;
            A = 0.0;
        }

        gl_FragColor = vec4(A, B, 0.0, 1.0);
    }
`;

// STEP 3: Gray-Scott simulation
const rdShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;
    uniform vec2 u_resolution;
    uniform float u_feed;
    uniform float u_kill;

    void main() {
        vec2 pixel = 1.0 / u_resolution;

        // Sample current state
        vec4 center = texture2D(u_state, v_texCoord);
        float A = center.r;
        float B = center.g;

        // 5-point Laplacian
        vec2 lap = center.rg * -4.0;
        lap += texture2D(u_state, v_texCoord + vec2(0.0, pixel.y)).rg;
        lap += texture2D(u_state, v_texCoord + vec2(pixel.x, 0.0)).rg;
        lap += texture2D(u_state, v_texCoord - vec2(0.0, pixel.y)).rg;
        lap += texture2D(u_state, v_texCoord - vec2(pixel.x, 0.0)).rg;

        // Constants
        float Da = 1.0;
        float Db = 0.5;
        float dt = 1.0;
        float f = u_feed;
        float k = u_kill;

        // Reaction term
        float reaction = A * B * B;

        // Gray-Scott equations
        float A_new = A + (Da * lap.r - reaction + f * (1.0 - A)) * dt;
        float B_new = B + (Db * lap.g + reaction - (k + f) * B) * dt;

        gl_FragColor = vec4(A_new, B_new, 0.0, 1.0);
    }
`;

// Compile shader
function compileShader(source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
    }

    return shader;
}

// Create program
function createProgram(vertexSource, fragmentSource) {
    const vertexShader = compileShader(vertexSource, gl.VERTEX_SHADER);
    const fragmentShader = compileShader(fragmentSource, gl.FRAGMENT_SHADER);

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        console.error('Program link error:', gl.getProgramInfoLog(program));
        return null;
    }

    return program;
}

// Create texture
function createTexture(width, height) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    return texture;
}

// Create framebuffer
function createFramebuffer(texture) {
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    return fb;
}

// Setup fullscreen quad
function setupQuad() {
    const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);
    return buffer;
}

// Create programs
const rdProgram = createProgram(vertexShaderSource, rdShaderSource);
const displayProgram = createProgram(vertexShaderSource, displayShaderSource);
const initProgram = createProgram(vertexShaderSource, initShaderSource);

// Create textures for ping-pong rendering
const width = canvas.width;
const height = canvas.height;
const textures = {
    ping: createTexture(width, height),
    pong: createTexture(width, height)
};

const framebuffers = {
    ping: createFramebuffer(textures.ping),
    pong: createFramebuffer(textures.pong)
};

let current = 'ping';

// Create quad
const quadBuffer = setupQuad();

// Initialize simulation
function initialize() {
    console.log('Initializing...');
    gl.viewport(0, 0, width, height);
    gl.useProgram(initProgram);

    const posLoc = gl.getAttribLocation(initProgram, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Random seed for different initialization each time
    const randomSeed = Math.random() * 1000.0 + 1.0;
    gl.uniform1f(gl.getUniformLocation(initProgram, 'u_seed'), randomSeed);

    // Initialize both buffers
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.ping);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.pong);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    console.log('Initialization complete');
}

// Run one simulation step
function simulate() {
    const next = current === 'ping' ? 'pong' : 'ping';

    gl.viewport(0, 0, width, height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers[next]);
    gl.useProgram(rdProgram);

    const posLoc = gl.getAttribLocation(rdProgram, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    gl.uniform2f(gl.getUniformLocation(rdProgram, 'u_resolution'), width, height);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_feed'), feed);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_kill'), kill);

    // Bind current state texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[current]);
    gl.uniform1i(gl.getUniformLocation(rdProgram, 'u_state'), 0);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    current = next;
}

// Display to screen
function display() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.useProgram(displayProgram);

    const posLoc = gl.getAttribLocation(displayProgram, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Bind current state
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[current]);
    gl.uniform1i(gl.getUniformLocation(displayProgram, 'u_state'), 0);

    // Set color map
    gl.uniform1i(gl.getUniformLocation(displayProgram, 'u_colorMap'), 0);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// Main loop
function loop() {
    // Run 8 simulation steps per frame
    for (let i = 0; i < 8; i++) {
        simulate();
    }

    // Display result
    display();

    requestAnimationFrame(loop);
}

// Setup UI controls
function setupControls() {
    const resetBtn = document.getElementById('resetBtn');
    const paramSelector = document.getElementById('paramSelector');
    const paramCrosshair = document.getElementById('paramCrosshair');
    const feedValue = document.getElementById('feedValue');
    const killValue = document.getElementById('killValue');
    const menu = document.getElementById('menu');

    resetBtn.addEventListener('click', () => {
        console.log('Reset button clicked');
        initialize();
    });

    // Parameter selector (X/Y for kill/feed)
    let isDragging = false;

    function updateParameters(e) {
        const rect = paramSelector.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        const y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));

        // X axis: kill (0.01413 to 0.06534)
        kill = 0.01413 + x * (0.06534 - 0.01413);

        // Y axis: feed (0.002 to 0.12) - inverted (top = high)
        feed = 0.12 - y * (0.12 - 0.002);

        // Update display
        killValue.textContent = kill.toFixed(5);
        feedValue.textContent = feed.toFixed(5);

        // Update crosshair position
        paramCrosshair.style.left = (x * 100) + '%';
        paramCrosshair.style.top = (y * 100) + '%';
    }

    paramSelector.addEventListener('mousedown', (e) => {
        isDragging = true;
        updateParameters(e);
    });

    document.addEventListener('mousemove', (e) => {
        if (isDragging) {
            updateParameters(e);
        }
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
    });

    // Initialize crosshair position
    const initialX = (kill - 0.01413) / (0.06534 - 0.01413);
    const initialY = 1.0 - (feed - 0.002) / (0.12 - 0.002);
    paramCrosshair.style.left = (initialX * 100) + '%';
    paramCrosshair.style.top = (initialY * 100) + '%';

    // Keyboard control - 'm' key toggles menu
    document.addEventListener('keydown', (e) => {
        if (e.key === 'm' || e.key === 'M') {
            menu.classList.toggle('closed');
        }
    });

    // Start with menu closed
    menu.classList.add('closed');
}

// Start
console.log('Starting Gray-Scott simulation');
console.log('feed =', feed, ', kill =', kill);

setupControls();
initialize();
loop();
