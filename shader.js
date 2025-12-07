// Vertex shader - simple passthrough
const vertexShaderSource = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;

    void main() {
        v_texCoord = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

// Camera processing shader - converts to greyscale and inverts
const cameraProcessShader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_videoTexture;

    void main() {
        vec2 uv = vec2(v_texCoord.x, 1.0 - v_texCoord.y);
        vec4 color = texture2D(u_videoTexture, uv);

        // Convert to greyscale
        float grey = dot(color.rgb, vec3(0.299, 0.587, 0.114));

        // Invert: darker input = higher value
        float inverted = 1.0 - grey;

        gl_FragColor = vec4(inverted, inverted, inverted, 1.0);
    }
`;

// Reaction-Diffusion shader - Karl Sims' exact Gray-Scott implementation
const reactionDiffusionShader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;
    uniform sampler2D u_feedMap;
    uniform vec2 u_resolution;
    uniform float u_feed;
    uniform float u_kill;
    uniform float u_timestep;

    void main() {
        vec2 pixelSize = 1.0 / u_resolution;

        // Sample current state
        vec2 state = texture2D(u_state, v_texCoord).rg;
        float a = state.r;
        float b = state.g;

        // Compute Laplacian using Karl Sims' 3x3 convolution kernel
        // Center: -1, Adjacent: 0.2, Diagonals: 0.05
        vec2 laplacian = texture2D(u_state, v_texCoord).rg * -1.0;

        // Adjacent (4-neighbors)
        laplacian += texture2D(u_state, v_texCoord + vec2(-pixelSize.x, 0.0)).rg * 0.2;
        laplacian += texture2D(u_state, v_texCoord + vec2(pixelSize.x, 0.0)).rg * 0.2;
        laplacian += texture2D(u_state, v_texCoord + vec2(0.0, -pixelSize.y)).rg * 0.2;
        laplacian += texture2D(u_state, v_texCoord + vec2(0.0, pixelSize.y)).rg * 0.2;

        // Diagonals
        laplacian += texture2D(u_state, v_texCoord + vec2(-pixelSize.x, -pixelSize.y)).rg * 0.05;
        laplacian += texture2D(u_state, v_texCoord + vec2(pixelSize.x, -pixelSize.y)).rg * 0.05;
        laplacian += texture2D(u_state, v_texCoord + vec2(-pixelSize.x, pixelSize.y)).rg * 0.05;
        laplacian += texture2D(u_state, v_texCoord + vec2(pixelSize.x, pixelSize.y)).rg * 0.05;

        // Get feed rate from feed map (camera darkness)
        float feedMapValue = texture2D(u_feedMap, v_texCoord).r;

        // Vary feed rate based on camera input
        // Dark areas: LOW feed = patterns grow
        // Light areas: HIGH feed = patterns suppressed
        float feedVariation = 0.02;
        float f = u_feed + (1.0 - feedMapValue) * feedVariation;
        float k = u_kill;

        // Karl Sims' exact Gray-Scott equations
        // DA = 1.0, DB = 0.5 (set in JavaScript)
        float reactionRate = a * b * b;

        float da = laplacian.r - reactionRate + f * (1.0 - a);
        float db = laplacian.g + reactionRate - (k + f) * b;

        // Update with timestep
        a += da * u_timestep;
        b += db * u_timestep;

        // Clamp
        a = clamp(a, 0.0, 1.0);
        b = clamp(b, 0.0, 1.0);

        gl_FragColor = vec4(a, b, 0.0, 1.0);
    }
`;

// Display shader - shows B chemical as black on white
const displayShader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;

    void main() {
        float b = texture2D(u_state, v_texCoord).g;

        // Apply threshold for crisp output
        float pattern = step(0.5, b);

        // Black on white
        vec3 color = vec3(1.0 - pattern);

        gl_FragColor = vec4(color, 1.0);
    }
`;

// Initialize shader
const initShader = `
    precision highp float;
    varying vec2 v_texCoord;

    float random(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
    }

    void main() {
        float rand = random(v_texCoord * 100.0);

        // Initialize: A=1.0, B=random spots (Karl Sims' method)
        float a = 1.0;
        float b = (rand > 0.95) ? 1.0 : 0.0;

        gl_FragColor = vec4(a, b, 0.0, 1.0);
    }
`;

class ReactionDiffusionApp {
    constructor() {
        this.canvas = document.getElementById('glCanvas');
        this.gl = this.canvas.getContext('webgl', {
            preserveDrawingBuffer: true,
            premultipliedAlpha: false
        });

        if (!this.gl) {
            alert('WebGL not supported');
            return;
        }

        // Karl Sims' standard parameters
        this.feed = 0.055;
        this.kill = 0.062;
        this.diffA = 1.0;  // DA
        this.diffB = 0.5;  // DB
        this.timestep = 1.0;
        this.stepsPerFrame = 1;

        this.video = null;
        this.mediaElement = null;
        this.inputSource = 'camera';

        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        this.init();
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.width = window.innerWidth;
        this.height = window.innerHeight;
    }

    compileShader(source, type) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader error:', this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    createProgram(vertexSource, fragmentSource) {
        const vertexShader = this.compileShader(vertexSource, this.gl.VERTEX_SHADER);
        const fragmentShader = this.compileShader(fragmentSource, this.gl.FRAGMENT_SHADER);

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);

        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error('Program error:', this.gl.getProgramInfoLog(program));
            return null;
        }

        return program;
    }

    createTexture(width, height) {
        const texture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, width, height, 0, this.gl.RGBA, this.gl.UNSIGNED_BYTE, null);
        return texture;
    }

    createFramebuffer(texture) {
        const fb = this.gl.createFramebuffer();
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fb);
        this.gl.framebufferTexture2D(this.gl.FRAMEBUFFER, this.gl.COLOR_ATTACHMENT0, this.gl.TEXTURE_2D, texture, 0);
        return fb;
    }

    setupQuad() {
        const positions = new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]);
        const buffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);
        return buffer;
    }

    async setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
            });

            this.video = document.createElement('video');
            this.video.srcObject = stream;
            this.video.autoplay = true;
            this.video.playsInline = true;

            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    resolve();
                };
            });

            this.mediaElement = this.video;
            this.videoTexture = this.createTexture(this.video.videoWidth, this.video.videoHeight);

        } catch (err) {
            console.error('Camera error:', err);
        }
    }

    async setupVideoFile(file) {
        try {
            this.video = document.createElement('video');
            this.video.src = URL.createObjectURL(file);
            this.video.loop = true;
            this.video.autoplay = true;
            this.video.muted = true;

            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    resolve();
                };
            });

            this.mediaElement = this.video;
            this.videoTexture = this.createTexture(this.video.videoWidth, this.video.videoHeight);
        } catch (err) {
            console.error('Video error:', err);
        }
    }

    async setupImageFile(file) {
        try {
            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);

            await new Promise((resolve) => {
                img.onload = () => resolve();
            });

            this.mediaElement = img;
            this.video = null;
            this.videoTexture = this.createTexture(img.width, img.height);
        } catch (err) {
            console.error('Image error:', err);
        }
    }

    async init() {
        this.gl.clearColor(1.0, 1.0, 1.0, 1.0);

        // Create programs
        this.cameraProgram = this.createProgram(vertexShaderSource, cameraProcessShader);
        this.rdProgram = this.createProgram(vertexShaderSource, reactionDiffusionShader);
        this.displayProgram = this.createProgram(vertexShaderSource, displayShader);
        this.initProgram = this.createProgram(vertexShaderSource, initShader);

        // Setup quad
        this.quadBuffer = this.setupQuad();

        // Create ping-pong textures for simulation
        this.simTextures = {
            ping: this.createTexture(this.width, this.height),
            pong: this.createTexture(this.width, this.height)
        };

        this.simFramebuffers = {
            ping: this.createFramebuffer(this.simTextures.ping),
            pong: this.createFramebuffer(this.simTextures.pong)
        };

        // Feed map (camera) texture
        this.feedMapTexture = this.createTexture(this.width, this.height);
        this.feedMapFB = this.createFramebuffer(this.feedMapTexture);

        this.currentBuffer = 'ping';

        // Initialize simulation
        this.initializeSimulation();

        // Setup controls
        this.setupControls();

        // Setup camera
        await this.setupCamera();

        // Start
        this.render();
    }

    initializeSimulation() {
        this.gl.viewport(0, 0, this.width, this.height);
        this.gl.useProgram(this.initProgram);

        const posLoc = this.gl.getAttribLocation(this.initProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        // Initialize both buffers
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.simFramebuffers.ping);
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.simFramebuffers.pong);
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }

    setupControls() {
        const scaleSlider = document.getElementById('scale');
        const scaleValue = document.getElementById('scaleValue');
        const viscositySlider = document.getElementById('viscosity');
        const viscosityValue = document.getElementById('viscosityValue');

        scaleSlider.addEventListener('input', (e) => {
            const scale = parseFloat(e.target.value);
            scaleValue.textContent = scale.toFixed(2);
            // Scale controls diffusion rates
            this.diffA = 0.5 + scale * 1.0;  // 0.5 to 1.5
            this.diffB = this.diffA * 0.5;   // Always half
        });

        viscositySlider.addEventListener('input', (e) => {
            const viscosity = parseFloat(e.target.value);
            viscosityValue.textContent = viscosity.toFixed(2);
            // Viscosity controls steps per frame
            this.stepsPerFrame = Math.floor(1 + (1.0 - viscosity) * 5);
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            this.initializeSimulation();
        });

        const inputSourceSelect = document.getElementById('inputSource');
        const fileInputGroup = document.getElementById('fileInputGroup');
        const fileInput = document.getElementById('fileInput');

        inputSourceSelect.addEventListener('change', (e) => {
            this.inputSource = e.target.value;
            if (this.inputSource === 'camera') {
                fileInputGroup.style.display = 'none';
                this.setupCamera();
            } else {
                fileInputGroup.style.display = 'block';
            }
        });

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            if (this.inputSource === 'video') {
                await this.setupVideoFile(file);
            } else if (this.inputSource === 'image') {
                await this.setupImageFile(file);
            }
        });
    }

    updateVideoTexture() {
        if (!this.mediaElement) return;

        if (this.mediaElement.tagName === 'VIDEO') {
            if (this.mediaElement.readyState >= this.mediaElement.HAVE_CURRENT_DATA) {
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
                this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, this.mediaElement);
            }
        } else if (this.mediaElement.tagName === 'IMG' && this.mediaElement.complete) {
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
            this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.RGBA, this.gl.RGBA, this.gl.UNSIGNED_BYTE, this.mediaElement);
        }
    }

    processFeedMap() {
        if (!this.mediaElement) return;

        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.feedMapFB);
        this.gl.viewport(0, 0, this.width, this.height);
        this.gl.useProgram(this.cameraProgram);

        const posLoc = this.gl.getAttribLocation(this.cameraProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
        this.gl.uniform1i(this.gl.getUniformLocation(this.cameraProgram, 'u_videoTexture'), 0);

        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }

    simulateStep() {
        this.gl.viewport(0, 0, this.width, this.height);
        this.gl.useProgram(this.rdProgram);

        const posLoc = this.gl.getAttribLocation(this.rdProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        // Set uniforms
        this.gl.uniform2f(this.gl.getUniformLocation(this.rdProgram, 'u_resolution'), this.width, this.height);
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_feed'), this.feed);
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_kill'), this.kill);
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_timestep'), this.timestep);

        // Bind state texture
        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.simTextures[this.currentBuffer]);
        this.gl.uniform1i(this.gl.getUniformLocation(this.rdProgram, 'u_state'), 0);

        // Bind feed map
        this.gl.activeTexture(this.gl.TEXTURE1);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.feedMapTexture);
        this.gl.uniform1i(this.gl.getUniformLocation(this.rdProgram, 'u_feedMap'), 1);

        // Render to other buffer
        const nextBuffer = this.currentBuffer === 'ping' ? 'pong' : 'ping';
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.simFramebuffers[nextBuffer]);
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

        this.currentBuffer = nextBuffer;
    }

    display() {
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        this.gl.useProgram(this.displayProgram);

        const posLoc = this.gl.getAttribLocation(this.displayProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.simTextures[this.currentBuffer]);
        this.gl.uniform1i(this.gl.getUniformLocation(this.displayProgram, 'u_state'), 0);

        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }

    render() {
        // Update video
        this.updateVideoTexture();

        // Process feed map
        if (this.mediaElement) {
            this.processFeedMap();
        }

        // Run simulation steps
        for (let i = 0; i < this.stepsPerFrame; i++) {
            this.simulateStep();
        }

        // Display
        this.display();

        requestAnimationFrame(() => this.render());
    }
}

window.addEventListener('load', () => {
    console.log('Starting Karl Sims Reaction-Diffusion...');
    new ReactionDiffusionApp();
});
