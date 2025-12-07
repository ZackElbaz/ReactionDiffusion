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

// Karl Sims' standard parameters
const DA = 1.0;  // Diffusion rate for chemical A
const DB = 0.5;  // Diffusion rate for chemical B
const dt = 1.0;  // Time step

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
    uniform float u_dA;
    uniform float u_dB;
    uniform float u_dt;
    uniform float u_feed;
    uniform float u_kill;

    void main() {
        vec2 pixel = 1.0 / u_resolution;

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

        // Style map: gradient from dark (right) to light (left)
        // Right side (x=1): dark = low feed
        // Left side (x=0): light = high feed
        float feedVariation = 0.03;
        float f = u_feed + v_texCoord.x * feedVariation;
        float k = u_kill;

        // Gray-Scott equations
        float abb = a * b * b;
        float da = u_dA * laplacian.r - abb + f * (1.0 - a);
        float db = u_dB * laplacian.g + abb - (k + f) * b;

        // Update state
        a += da * u_dt;
        b += db * u_dt;

        // Clamp
        a = clamp(a, 0.0, 1.0);
        b = clamp(b, 0.0, 1.0);

        gl_FragColor = vec4(a, b, 0.0, 1.0);
    }
`;

// Display shader (A=white, B=black)
const displayShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;

    void main() {
        vec2 state = texture2D(u_state, v_texCoord).rg;
        float a = state.r;
        float b = state.g;

        // A is white, B is black
        vec3 color = vec3(1.0 - b);

        gl_FragColor = vec4(color, 1.0);
    }
`;

// Initialize shader (A=1, B=random seed)
const initShaderSource = `
    precision highp float;
    varying vec2 v_texCoord;

    float random(vec2 st) {
        return fract(sin(dot(st, vec2(12.9898, 78.233))) * 43758.5453);
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
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_dA'), DA);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_dB'), DB);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_dt'), dt);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_feed'), 0.055);
    gl.uniform1f(gl.getUniformLocation(rdProgram, 'u_kill'), 0.062);

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

// Start
console.log('Karl Sims Reaction-Diffusion');
console.log('DA =', DA, ', DB =', DB);
console.log('feed = 0.055, kill = 0.062');
console.log('Style map: gradient from dark (right) to light (left)');

initialize();
loop();
