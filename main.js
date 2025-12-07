// Karl Sims Reaction-Diffusion Implementation
// Following: https://www.karlsims.com/rd.html

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
// Good starting values for labyrinthine patterns
let feed = 0.055;  // f: 0.002 - 0.12 (Y axis)
let kill = 0.062;  // k: 0.01413 - 0.06534 (X axis)
let currentColorMap = 'grayscale';
let gradientOrientation = 'right-left'; // 'right-left', 'left-right', 'top-bottom', 'bottom-top'

// Vertex shader (simple passthrough)
const vertexShaderSource = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;
    void main() {
        v_texCoord = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

// Reaction-Diffusion shader (Gray-Scott model)
const rdShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;
    uniform vec2 u_resolution;
    uniform float u_feed;
    uniform float u_kill;
    uniform int u_gradientOrientation;

    void main() {
        // Grid spacing for Laplacian computation
        vec2 pixel = 1.0 / u_resolution;

        // Karl Sims standard parameters
        float dA = 1.0;
        float dB = 0.5;
        float dt = 1.0;

        // Sample current state
        vec2 state = texture2D(u_state, v_texCoord).rg;
        float a = state.r;
        float b = state.g;

        // Compute Laplacian using 3x3 convolution
        // Center: -1, Adjacent: 0.2, Diagonals: 0.05
        vec2 laplacian = vec2(0.0);

        // Center
        laplacian += texture2D(u_state, v_texCoord).rg * -1.0;

        // Adjacent (4-neighbors)
        laplacian += texture2D(u_state, v_texCoord + vec2(-pixel.x, 0.0)).rg * 0.2;
        laplacian += texture2D(u_state, v_texCoord + vec2(pixel.x, 0.0)).rg * 0.2;
        laplacian += texture2D(u_state, v_texCoord + vec2(0.0, -pixel.y)).rg * 0.2;
        laplacian += texture2D(u_state, v_texCoord + vec2(0.0, pixel.y)).rg * 0.2;

        // Diagonals
        laplacian += texture2D(u_state, v_texCoord + vec2(-pixel.x, -pixel.y)).rg * 0.05;
        laplacian += texture2D(u_state, v_texCoord + vec2(pixel.x, -pixel.y)).rg * 0.05;
        laplacian += texture2D(u_state, v_texCoord + vec2(-pixel.x, pixel.y)).rg * 0.05;
        laplacian += texture2D(u_state, v_texCoord + vec2(pixel.x, pixel.y)).rg * 0.05;

        // Style map: gradient controls feed AND kill rate variation
        // Creates pattern transitions: labyrinths → spots → holes
        float gradientValue = 0.0;
        if (u_gradientOrientation == 0) {
            // Right to Left
            gradientValue = v_texCoord.x;
        } else if (u_gradientOrientation == 1) {
            // Left to Right
            gradientValue = 1.0 - v_texCoord.x;
        } else if (u_gradientOrientation == 2) {
            // Top to Bottom
            gradientValue = 1.0 - v_texCoord.y;
        } else {
            // Bottom to Top
            gradientValue = v_texCoord.y;
        }

        // Vary both feed and kill rates based on gradient
        float feedVariation = 0.02;
        float killVariation = 0.005;
        float f = u_feed - feedVariation + gradientValue * feedVariation * 2.0;
        float k = u_kill - killVariation + gradientValue * killVariation * 2.0;

        // Gray-Scott equations
        float abb = a * b * b;
        float da = dA * laplacian.r - abb + f * (1.0 - a);
        float db = dB * laplacian.g + abb - (k + f) * b;

        // Update state
        a += da * dt;
        b += db * dt;

        // Clamp
        a = clamp(a, 0.0, 1.0);
        b = clamp(b, 0.0, 1.0);

        gl_FragColor = vec4(a, b, 0.0, 1.0);
    }
`;

// Display shader with color maps (matching Karl Sims' style)
const displayShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;
    uniform int u_colorMap;

    void main() {
        vec2 state = texture2D(u_state, v_texCoord).rg;
        float a = state.r;
        float b = state.g;

        vec3 color;

        if (u_colorMap == 0) {
            // Grayscale - show B chemical (bright where B is high)
            color = vec3(b);
        } else if (u_colorMap == 1) {
            // Blue gradient (dark blue to light blue)
            color = vec3(b * 0.3, b * 0.5, 0.5 + b * 0.5);
        } else if (u_colorMap == 2) {
            // Orange-Blue
            color = mix(vec3(0.0, 0.3, 0.6), vec3(1.0, 0.5, 0.0), b);
        } else if (u_colorMap == 3) {
            // Green-Purple
            color = mix(vec3(0.0, 0.5, 0.0), vec3(0.5, 0.0, 0.5), b);
        } else if (u_colorMap == 4) {
            // Cyan-Magenta
            color = mix(vec3(0.0, 0.8, 0.8), vec3(0.8, 0.0, 0.8), b);
        } else if (u_colorMap == 5) {
            // Rainbow spectrum
            float hue = b * 6.0;
            vec3 c = vec3(
                abs(hue - 3.0) - 1.0,
                2.0 - abs(hue - 2.0),
                2.0 - abs(hue - 4.0)
            );
            color = clamp(c, 0.0, 1.0);
        } else if (u_colorMap == 6) {
            // Yellow-Blue
            color = mix(vec3(0.0, 0.0, 0.6), vec3(1.0, 1.0, 0.0), b);
        } else if (u_colorMap == 7) {
            // Red-Yellow
            color = mix(vec3(0.5, 0.0, 0.0), vec3(1.0, 1.0, 0.0), b);
        } else if (u_colorMap == 8) {
            // Teal-Orange
            color = mix(vec3(0.0, 0.5, 0.5), vec3(1.0, 0.4, 0.0), b);
        } else if (u_colorMap == 9) {
            // Purple-Yellow
            color = mix(vec3(0.3, 0.0, 0.5), vec3(1.0, 1.0, 0.3), b);
        } else if (u_colorMap == 10) {
            // Fire (black-red-orange-yellow)
            if (b < 0.33) {
                color = mix(vec3(0.0, 0.0, 0.0), vec3(0.8, 0.0, 0.0), b * 3.0);
            } else if (b < 0.66) {
                color = mix(vec3(0.8, 0.0, 0.0), vec3(1.0, 0.5, 0.0), (b - 0.33) * 3.0);
            } else {
                color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 0.5), (b - 0.66) * 3.0);
            }
        } else {
            // Vibrant multi-color
            float t = b * 4.0;
            if (t < 1.0) {
                color = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t);
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

// Initialize shader (A=1, B=random seed)
const initShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform float u_seed;

    float random(vec2 st) {
        return fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453 * u_seed);
    }

    void main() {
        float a = 1.0;
        float b = 0.0;

        // Seed small random area with B=1
        float rand = random(v_texCoord * 10.0);
        if (rand > 0.97) {
            b = 1.0;
        }

        gl_FragColor = vec4(a, b, 0.0, 1.0);
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

    // Set gradient orientation for style map
    const orientations = { 'right-left': 0, 'left-right': 1, 'top-bottom': 2, 'bottom-top': 3 };
    gl.uniform1i(gl.getUniformLocation(rdProgram, 'u_gradientOrientation'), orientations[gradientOrientation] || 0);

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
    const colorMaps = {
        'grayscale': 0, 'blue': 1, 'orange-blue': 2, 'green-purple': 3,
        'cyan-magenta': 4, 'rainbow': 5, 'yellow-blue': 6, 'red-yellow': 7,
        'teal-orange': 8, 'purple-yellow': 9, 'fire': 10, 'vibrant': 11
    };
    gl.uniform1i(gl.getUniformLocation(displayProgram, 'u_colorMap'), colorMaps[currentColorMap] || 0);

    // Draw
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// Main loop
function loop() {
    // Run simulation step
    simulate();

    // Display result
    display();

    requestAnimationFrame(loop);
}

// Setup UI controls
function setupControls() {
    const colorMapSelect = document.getElementById('colorMap');
    const orientationSelect = document.getElementById('orientation');
    const resetBtn = document.getElementById('resetBtn');
    const menu = document.getElementById('menu');
    const paramSelector = document.getElementById('paramSelector');
    const paramCrosshair = document.getElementById('paramCrosshair');
    const feedValue = document.getElementById('feedValue');
    const killValue = document.getElementById('killValue');

    colorMapSelect.addEventListener('change', (e) => {
        currentColorMap = e.target.value;
    });

    orientationSelect.addEventListener('change', (e) => {
        gradientOrientation = e.target.value;
    });

    resetBtn.addEventListener('click', () => {
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

    // Initialize crosshair position based on current parameters
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
console.log('Karl Sims Reaction-Diffusion');
console.log('feed =', feed, ', kill =', kill);

setupControls();
initialize();
loop();
