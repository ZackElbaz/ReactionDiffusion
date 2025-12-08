// Karl Sims Reaction-Diffusion - Starting from scratch
const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl', { preserveDrawingBuffer: true });

if (!gl) {
    alert('WebGL not supported');
    throw new Error('WebGL not supported');
}

// Set canvas display size to fill window
canvas.style.width = '100vw';
canvas.style.height = '100vh';

// Use small grid to see how the code is working
const GRID_SIZE = 100;
canvas.width = GRID_SIZE;
canvas.height = GRID_SIZE;

// Simulation parameters - using "Mazes" preset from Karl Sims
let feed = 0.029;
let kill = 0.057;
let currentColorMap = 'custom';
let gradientOrientation = 'right-left';

// Custom gradient colors (RGB in 0-1 range)
let customColor1 = [1.0, 1.0, 1.0]; // White
let customColor2 = [0.0, 0.0, 0.0]; // Black

// Time counter for perturbations
let timeCounter = 0.0;

// Simple vertex shader
const vertexShaderSource = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;
    void main() {
        v_texCoord = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

// Display shader with color maps
const displayShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;
    uniform int u_colorMap;
    uniform vec3 u_customColor1;
    uniform vec3 u_customColor2;

    void main() {
        vec2 state = texture2D(u_state, v_texCoord).rg;
        float A = state.r;
        float B = state.g;

        vec3 color;

        // Color maps - B concentration mapped to colors
        if (u_colorMap == -1) {
            // Custom gradient: mix between two custom colors based on B
            color = mix(u_customColor1, u_customColor2, B);
        } else if (u_colorMap == 0) {
            // Grayscale: B=0 white, B=1 black
            color = vec3(1.0 - B);
        } else if (u_colorMap == 1) {
            // Blue
            color = mix(vec3(1.0), vec3(0.0, 0.0, 1.0), B);
        } else if (u_colorMap == 2) {
            // Orange-Blue
            color = mix(vec3(1.0), vec3(1.0, 0.5, 0.0), B);
        } else if (u_colorMap == 3) {
            // Green-Purple
            color = mix(vec3(1.0), vec3(0.5, 0.0, 0.5), B);
        } else if (u_colorMap == 4) {
            // Cyan-Magenta
            color = mix(vec3(1.0), vec3(1.0, 0.0, 1.0), B);
        } else if (u_colorMap == 5) {
            // Rainbow
            float hue = B * 6.0;
            vec3 c = vec3(
                abs(hue - 3.0) - 1.0,
                2.0 - abs(hue - 2.0),
                2.0 - abs(hue - 4.0)
            );
            color = clamp(c, 0.0, 1.0);
        } else if (u_colorMap == 6) {
            // Yellow-Blue
            color = mix(vec3(1.0), vec3(1.0, 1.0, 0.0), B);
        } else if (u_colorMap == 7) {
            // Red-Yellow
            color = mix(vec3(1.0), vec3(1.0, 0.0, 0.0), B);
        } else if (u_colorMap == 8) {
            // Teal-Orange
            color = mix(vec3(1.0), vec3(0.0, 0.8, 0.8), B);
        } else if (u_colorMap == 9) {
            // Purple-Yellow
            color = mix(vec3(1.0), vec3(0.8, 0.0, 0.8), B);
        } else if (u_colorMap == 10) {
            // Fire
            if (B < 0.33) {
                color = mix(vec3(1.0), vec3(0.8, 0.0, 0.0), B * 3.0);
            } else if (B < 0.66) {
                color = mix(vec3(0.8, 0.0, 0.0), vec3(1.0, 0.5, 0.0), (B - 0.33) * 3.0);
            } else {
                color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 0.5), (B - 0.66) * 3.0);
            }
        } else {
            // Vibrant
            float t = B * 4.0;
            if (t < 1.0) {
                color = mix(vec3(1.0), vec3(0.0, 1.0, 1.0), t);
            } else if (t < 2.0) {
                color = mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), t - 1.0);
            } else if (t < 3.0) {
                color = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), t - 2.0);
            } else {
                color = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t - 3.0);
            }
        }

        gl_FragColor = vec4(color, 1.0);
    }
`;

// STEP 2: Initialize with tiny concentration variations
const initShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform float u_seed;

    float random(vec2 st) {
        return fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453 * u_seed);
    }

    void main() {
        // Start with A=1.0, B=0.0 everywhere (chemical A fills the space)
        float A = 1.0;
        float B = 0.0;

        // Create multiple random seed points scattered across the grid
        for (int i = 0; i < 8; i++) {
            vec2 seedPos = vec2(
                random(vec2(float(i) * 13.7, u_seed * 2.3)),
                random(vec2(float(i) * 7.1, u_seed * 5.9))
            );
            float dist = distance(v_texCoord, seedPos);

            // Small circular seed regions
            if (dist < 0.03) {
                float strength = 1.0 - (dist / 0.03);
                B = max(B, strength);
                A = min(A, 1.0 - strength);
            }
        }

        // Add random noise everywhere for variation
        float noise = (random(v_texCoord * 100.0) - 0.5) * 0.05;
        B = clamp(B + noise, 0.0, 1.0);

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
    uniform float u_time;

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

        // Constants - using much smaller values for numerical stability
        float Da = 1.0;  // Diffusion rate for A
        float Db = 0.5;  // Diffusion rate for B (slower than A)
        float dt = 0.2;  // Smaller time step prevents divergence
        float f = u_feed;
        float k = u_kill;

        // Reaction term
        float reaction = A * B * B;

        // Gray-Scott equations
        float A_new = A + (Da * lap.r - reaction + f * (1.0 - A)) * dt;
        float B_new = B + (Db * lap.g + reaction - (k + f) * B) * dt;

        // Add tiny random perturbations to prevent complete equilibrium
        float rand = fract(sin(dot(v_texCoord + u_time, vec2(12.9898, 78.233))) * 43758.5453);
        float perturbation = (rand - 0.5) * 0.001;
        B_new += perturbation;

        // Clamp values to prevent divergence outside [0,1]
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
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_time'), timeCounter);

    // Bind current state texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, textures[current]);
    gl.uniform1i(gl.getUniformLocation(rdProgram, 'u_state'), 0);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    current = next;
    timeCounter += 0.01;
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
    const colorMaps = {
        'custom': -1,
        'grayscale': 0, 'blue': 1, 'orange-blue': 2, 'green-purple': 3,
        'cyan-magenta': 4, 'rainbow': 5, 'yellow-blue': 6, 'red-yellow': 7,
        'teal-orange': 8, 'purple-yellow': 9, 'fire': 10, 'vibrant': 11
    };
    gl.uniform1i(gl.getUniformLocation(displayProgram, 'u_colorMap'), colorMaps[currentColorMap] || 0);

    // Set custom colors
    gl.uniform3f(gl.getUniformLocation(displayProgram, 'u_customColor1'), customColor1[0], customColor1[1], customColor1[2]);
    gl.uniform3f(gl.getUniformLocation(displayProgram, 'u_customColor2'), customColor2[0], customColor2[1], customColor2[2]);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// Main loop
function loop() {
    // Run many iterations per frame (more needed with smaller dt)
    for (let i = 0; i < 32; i++) {
        simulate();
    }

    // Display result
    display();

    requestAnimationFrame(loop);
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
    const colorMapSelect = document.getElementById('colorMap');
    const resetBtn = document.getElementById('resetBtn');
    const paramSelector = document.getElementById('paramSelector');
    const paramCrosshair = document.getElementById('paramCrosshair');
    const feedValue = document.getElementById('feedValue');
    const killValue = document.getElementById('killValue');
    const menu = document.getElementById('menu');
    const color1Picker = document.getElementById('color1');
    const color2Picker = document.getElementById('color2');

    // Color map selector
    colorMapSelect.addEventListener('change', (e) => {
        currentColorMap = e.target.value;
        console.log('Color map changed to:', currentColorMap);
    });

    // Color pickers
    color1Picker.addEventListener('input', (e) => {
        customColor1 = hexToRgb(e.target.value);
        console.log('Color 1 changed to:', customColor1);
    });

    color2Picker.addEventListener('input', (e) => {
        customColor2 = hexToRgb(e.target.value);
        console.log('Color 2 changed to:', customColor2);
    });

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
