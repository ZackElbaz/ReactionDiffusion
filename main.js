// Gray-Scott Reaction-Diffusion - Clean Implementation
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true });

if (!gl) {
    alert('WebGL not supported');
    throw new Error('WebGL not supported');
}

// Grid settings
const GRID_SIZE = 100;
canvas.width = GRID_SIZE;
canvas.height = GRID_SIZE;
canvas.style.width = '100vw';
canvas.style.height = '100vh';
canvas.style.imageRendering = 'pixelated'; // Make grid visible

// Simulation parameters - "worms" preset for classic pattern
let feed = 0.0367;
let kill = 0.0649;
let currentColorMap = 'custom';

// Custom gradient colors (RGB in 0-1 range)
let customColor1 = [1.0, 1.0, 1.0]; // White (low B)
let customColor2 = [0.0, 0.0, 0.0]; // Black (high B)

// Gray-Scott constants (standard values for pattern formation)
const Da = 1.0;     // Diffusion rate for A
const Db = 0.5;     // Diffusion rate for B (A diffuses 2x faster)
const dt = 0.1;     // Small time step for stability with normalized Laplacian

// Animation state
let isPlaying = false;
let animationId = null;

// Simple vertex shader
const vertexShaderSource = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;
    void main() {
        v_texCoord = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

// Display shader
const displayShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;
    uniform vec3 u_customColor1;
    uniform vec3 u_customColor2;

    void main() {
        vec2 state = texture2D(u_state, v_texCoord).rg;
        float B = state.g;

        // Linear gradient between custom colors based on B concentration
        vec3 color = mix(u_customColor1, u_customColor2, B);

        gl_FragColor = vec4(color, 1.0);
    }
`;

// Initialization shader - random cluster with small B seeding
const initShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform vec2 u_clusterPos;
    uniform float u_seed;

    float random(vec2 st) {
        return fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453 * u_seed);
    }

    void main() {
        // Default: A=1.0, B=0.0 everywhere
        float A = 1.0;
        float B = 0.0;

        // Create a central cluster at random position
        float dist = distance(v_texCoord, u_clusterPos);

        if (dist < 0.1) {
            // Central high concentration of B
            B = 1.0;
            A = 0.0;
        }

        // Add small random B seeding across the canvas (critical for pattern formation)
        float rand = random(v_texCoord * 100.0);
        if (rand > 0.98) {
            B = 0.5 + random(v_texCoord * 50.0) * 0.5;
            A = 1.0 - B;
        }

        gl_FragColor = vec4(A, B, 0.0, 1.0);
    }
`;

// Gray-Scott simulation shader - EXACT equations from user
const rdShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;
    uniform vec2 u_resolution;
    uniform float u_feed;
    uniform float u_kill;
    uniform float u_Da;
    uniform float u_Db;
    uniform float u_dt;

    void main() {
        vec2 pixel = 1.0 / u_resolution;

        // Sample current state
        vec4 center = texture2D(u_state, v_texCoord);
        float A = center.r;
        float B = center.g;

        // Normalized 5-point Laplacian: ∇²
        // center: -1.0, each cardinal neighbor: 0.2
        vec2 laplacian = -1.0 * center.rg;
        laplacian += 0.2 * texture2D(u_state, v_texCoord + vec2(pixel.x, 0.0)).rg;
        laplacian += 0.2 * texture2D(u_state, v_texCoord - vec2(pixel.x, 0.0)).rg;
        laplacian += 0.2 * texture2D(u_state, v_texCoord + vec2(0.0, pixel.y)).rg;
        laplacian += 0.2 * texture2D(u_state, v_texCoord - vec2(0.0, pixel.y)).rg;

        // Reaction term: A·B²
        float reaction = A * B * B;

        // Gray-Scott equations (EXACT from user specification):
        // A′ = A + (Dₐ∇²A − A·B² + f(1−A)) Δt
        // B′ = B + (Db∇²B + A·B² − (k+f)B) Δt

        float A_new = A + (u_Da * laplacian.r - reaction + u_feed * (1.0 - A)) * u_dt;
        float B_new = B + (u_Db * laplacian.g + reaction - (u_kill + u_feed) * B) * u_dt;

        // Clamp to [0,1] to prevent numerical issues
        A_new = clamp(A_new, 0.0, 1.0);
        B_new = clamp(B_new, 0.0, 1.0);

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
function createTexture() {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, GRID_SIZE, GRID_SIZE, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
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
const textures = {
    ping: createTexture(),
    pong: createTexture()
};

const framebuffers = {
    ping: createFramebuffer(textures.ping),
    pong: createFramebuffer(textures.pong)
};

let current = 'ping';

// Create quad
const quadBuffer = setupQuad();

// Random cluster position (will be set on initialization)
let clusterPos = [Math.random(), Math.random()];

// Initialize simulation
function initialize() {
    console.log('Initializing with cluster at:', clusterPos);

    gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);
    gl.useProgram(initProgram);

    const posLoc = gl.getAttribLocation(initProgram, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Set random cluster position and seed
    clusterPos = [Math.random(), Math.random()];
    gl.uniform2f(gl.getUniformLocation(initProgram, 'u_clusterPos'), clusterPos[0], clusterPos[1]);
    gl.uniform1f(gl.getUniformLocation(initProgram, 'u_seed'), Math.random() * 1000.0 + 1.0);

    // Initialize both buffers
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.ping);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers.pong);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    current = 'ping';

    // Display initial state
    display();

    console.log('Initialization complete');
}

// Run one simulation step
function step() {
    const next = current === 'ping' ? 'pong' : 'ping';

    gl.viewport(0, 0, GRID_SIZE, GRID_SIZE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffers[next]);
    gl.useProgram(rdProgram);

    const posLoc = gl.getAttribLocation(rdProgram, 'a_position');
    gl.enableVertexAttribArray(posLoc);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
    gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

    // Set uniforms
    gl.uniform2f(gl.getUniformLocation(rdProgram, 'u_resolution'), GRID_SIZE, GRID_SIZE);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_feed'), feed);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_kill'), kill);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_Da'), Da);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_Db'), Db);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_dt'), dt);

    // Bind current state texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[current]);
    gl.uniform1i(gl.getUniformLocation(rdProgram, 'u_state'), 0);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    current = next;
}

// Animation loop
function animate() {
    if (!isPlaying) return;

    // Run multiple iterations per frame (more needed with smaller dt)
    for (let i = 0; i < 50; i++) {
        step();
    }

    // Display result
    display();

    // Continue animation
    animationId = requestAnimationFrame(animate);
}

// Toggle play/pause
function togglePlayPause() {
    isPlaying = !isPlaying;

    const playPauseBtn = document.getElementById('playPauseBtn');

    if (isPlaying) {
        playPauseBtn.textContent = 'Pause';
        animate();
    } else {
        playPauseBtn.textContent = 'Play';
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    }
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

    // Set custom colors
    gl.uniform3f(gl.getUniformLocation(displayProgram, 'u_customColor1'), customColor1[0], customColor1[1], customColor1[2]);
    gl.uniform3f(gl.getUniformLocation(displayProgram, 'u_customColor2'), customColor2[0], customColor2[1], customColor2[2]);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// Helper function to convert hex color to RGB 0-1 range
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? [
        parseInt(result[1], 16) / 255,
        parseInt(result[2], 16) / 255,
        parseInt(result[3], 16) / 255
    ] : [1.0, 1.0, 1.0];
}

// Setup UI controls
function setupControls() {
    const resetBtn = document.getElementById('resetBtn');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const paramSelector = document.getElementById('paramSelector');
    const paramCrosshair = document.getElementById('paramCrosshair');
    const feedValue = document.getElementById('feedValue');
    const killValue = document.getElementById('killValue');
    const menu = document.getElementById('menu');
    const color1Picker = document.getElementById('color1');
    const color2Picker = document.getElementById('color2');

    // Play/Pause button
    playPauseBtn.addEventListener('click', () => {
        console.log('Play/Pause button clicked');
        togglePlayPause();
    });

    // Color pickers
    color1Picker.addEventListener('input', (e) => {
        customColor1 = hexToRgb(e.target.value);
        display(); // Update display immediately
    });

    color2Picker.addEventListener('input', (e) => {
        customColor2 = hexToRgb(e.target.value);
        display(); // Update display immediately
    });

    // Reset button
    resetBtn.addEventListener('click', () => {
        console.log('Reset button clicked');

        // Stop animation if playing
        if (isPlaying) {
            togglePlayPause();
        }

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

    // Initialize crosshair position for default values
    const initialX = (kill - 0.01413) / (0.06534 - 0.01413);
    const initialY = 1.0 - (feed - 0.002) / (0.12 - 0.002);
    paramCrosshair.style.left = (initialX * 100) + '%';
    paramCrosshair.style.top = (initialY * 100) + '%';
    feedValue.textContent = feed.toFixed(5);
    killValue.textContent = kill.toFixed(5);

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
console.log('Default parameters: feed =', feed, ', kill =', kill);
console.log('Constants: Da =', Da, ', Db =', Db, ', dt =', dt);

setupControls();
initialize();
