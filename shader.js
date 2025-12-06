// Vertex shader - simple passthrough
const vertexShaderSource = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;

    void main() {
        v_texCoord = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

// Camera processing shader - inverts video and extracts CMY channels
const cameraProcessShader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_videoTexture;

    void main() {
        vec2 uv = vec2(v_texCoord.x, 1.0 - v_texCoord.y); // Flip Y
        vec4 color = texture2D(u_videoTexture, uv);

        // Invert the RGB channels
        // Red inverted -> Cyan concentration
        // Green inverted -> Magenta concentration
        // Blue inverted -> Yellow concentration
        vec3 inverted = 1.0 - color.rgb;

        // Output CMY channels as RGB for processing
        gl_FragColor = vec4(inverted, 1.0);
    }
`;

// Reaction-Diffusion shader using Gray-Scott model
const reactionDiffusionShader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;
    uniform sampler2D u_cameraData;
    uniform vec2 u_resolution;
    uniform float u_feed;
    uniform float u_kill;
    uniform float u_diffA;
    uniform float u_diffB;
    uniform float u_cameraInfluence;
    uniform int u_channel; // 0 = Cyan, 1 = Magenta, 2 = Yellow

    void main() {
        vec2 pixel = 1.0 / u_resolution;

        // Sample current state
        vec2 state = texture2D(u_state, v_texCoord).rg;
        float a = state.r;
        float b = state.g;

        // Laplacian kernel for diffusion (3x3)
        vec2 laplacian = vec2(0.0);

        // Center weight
        laplacian += texture2D(u_state, v_texCoord).rg * -1.0;

        // Adjacent cells (4-neighbors)
        laplacian += texture2D(u_state, v_texCoord + vec2(pixel.x, 0.0)).rg * 0.2;
        laplacian += texture2D(u_state, v_texCoord + vec2(-pixel.x, 0.0)).rg * 0.2;
        laplacian += texture2D(u_state, v_texCoord + vec2(0.0, pixel.y)).rg * 0.2;
        laplacian += texture2D(u_state, v_texCoord + vec2(0.0, -pixel.y)).rg * 0.2;

        // Diagonal cells
        laplacian += texture2D(u_state, v_texCoord + vec2(pixel.x, pixel.y)).rg * 0.05;
        laplacian += texture2D(u_state, v_texCoord + vec2(-pixel.x, pixel.y)).rg * 0.05;
        laplacian += texture2D(u_state, v_texCoord + vec2(pixel.x, -pixel.y)).rg * 0.05;
        laplacian += texture2D(u_state, v_texCoord + vec2(-pixel.x, -pixel.y)).rg * 0.05;

        // Get camera influence for this channel
        vec3 cameraColor = texture2D(u_cameraData, v_texCoord).rgb;
        float channelInfluence = 0.0;

        if (u_channel == 0) {
            channelInfluence = cameraColor.r; // Cyan from inverted red
        } else if (u_channel == 1) {
            channelInfluence = cameraColor.g; // Magenta from inverted green
        } else {
            channelInfluence = cameraColor.b; // Yellow from inverted blue
        }

        // Gray-Scott reaction-diffusion equations
        float abb = a * b * b;
        float da = u_diffA * laplacian.r - abb + u_feed * (1.0 - a);
        float db = u_diffB * laplacian.g + abb - (u_kill + u_feed) * b;

        // Add camera influence to feed the B chemical
        db += channelInfluence * u_cameraInfluence * 0.01;

        // Update state
        a += da;
        b += db;

        // Clamp values
        a = clamp(a, 0.0, 1.0);
        b = clamp(b, 0.0, 1.0);

        gl_FragColor = vec4(a, b, 0.0, 1.0);
    }
`;

// Compositing shader - combines CMY layers using subtractive color mixing
const compositingShader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_cyanLayer;
    uniform sampler2D u_magentaLayer;
    uniform sampler2D u_yellowLayer;

    void main() {
        // Get the B chemical (the visible pattern) from each layer
        float cyan = texture2D(u_cyanLayer, v_texCoord).g;
        float magenta = texture2D(u_magentaLayer, v_texCoord).g;
        float yellow = texture2D(u_yellowLayer, v_texCoord).g;

        // Subtractive color mixing (CMY to RGB)
        // Cyan absorbs red, Magenta absorbs green, Yellow absorbs blue
        // RGB = 1 - CMY
        vec3 rgb;
        rgb.r = 1.0 - min(1.0, cyan + magenta);      // Red = 1 - (Cyan + Magenta)
        rgb.g = 1.0 - min(1.0, cyan + yellow);       // Green = 1 - (Cyan + Yellow)
        rgb.b = 1.0 - min(1.0, magenta + yellow);    // Blue = 1 - (Magenta + Yellow)

        // Boost contrast for better visibility
        rgb = pow(rgb, vec3(0.8));

        gl_FragColor = vec4(rgb, 1.0);
    }
`;

// Initialize shader - creates initial random state
const initShader = `
    precision highp float;
    varying vec2 v_texCoord;

    float random(vec2 st) {
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
    }

    void main() {
        float rand = random(v_texCoord * 10.0 + vec2(gl_FragCoord.xy));

        // Initialize with mostly A chemical (1.0) and some random B spots
        float a = 1.0;
        float b = rand > 0.95 ? 1.0 : 0.0;

        gl_FragColor = vec4(a, b, 0.0, 1.0);
    }
`;

class ReactionDiffusionApp {
    constructor() {
        this.canvas = document.getElementById('glCanvas');
        this.gl = this.canvas.createContext('webgl', {
            preserveDrawingBuffer: true,
            premultipliedAlpha: false
        });

        if (!this.gl) {
            this.showError('WebGL not supported');
            return;
        }

        // Parameters
        this.params = {
            feed: 0.055,
            kill: 0.062,
            diffA: 1.0,
            diffB: 0.5,
            cameraInfluence: 0.5
        };

        this.resolution = 512; // Fixed resolution for simulation
        this.video = null;
        this.videoTexture = null;

        this.init();
    }

    showError(message) {
        const errorDiv = document.getElementById('error');
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }

    compileShader(source, type) {
        const shader = this.gl.createShader(type);
        this.gl.shaderSource(shader, source);
        this.gl.compileShader(shader);

        if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
            console.error('Shader compilation error:', this.gl.getShaderInfoLog(shader));
            this.gl.deleteShader(shader);
            return null;
        }

        return shader;
    }

    createProgram(vertexSource, fragmentSource) {
        const vertexShader = this.compileShader(vertexSource, this.gl.VERTEX_SHADER);
        const fragmentShader = this.compileShader(fragmentSource, this.gl.FRAGMENT_SHADER);

        if (!vertexShader || !fragmentShader) return null;

        const program = this.gl.createProgram();
        this.gl.attachShader(program, vertexShader);
        this.gl.attachShader(program, fragmentShader);
        this.gl.linkProgram(program);

        if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
            console.error('Program linking error:', this.gl.getProgramInfoLog(program));
            return null;
        }

        return program;
    }

    createFramebuffer(texture) {
        const fb = this.gl.createFramebuffer();
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, fb);
        this.gl.framebufferTexture2D(
            this.gl.FRAMEBUFFER,
            this.gl.COLOR_ATTACHMENT0,
            this.gl.TEXTURE_2D,
            texture,
            0
        );
        return fb;
    }

    createTexture(width, height, data = null) {
        const texture = this.gl.createTexture();
        this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
        this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);
        this.gl.texImage2D(
            this.gl.TEXTURE_2D,
            0,
            this.gl.RGBA,
            width,
            height,
            0,
            this.gl.RGBA,
            this.gl.UNSIGNED_BYTE,
            data
        );
        return texture;
    }

    setupQuad() {
        const positions = new Float32Array([
            -1, -1,
             1, -1,
            -1,  1,
             1,  1
        ]);

        const buffer = this.gl.createBuffer();
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, positions, this.gl.STATIC_DRAW);

        return buffer;
    }

    async setupCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
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

            this.videoTexture = this.createTexture(this.video.videoWidth, this.video.videoHeight);

        } catch (err) {
            this.showError('Camera access denied: ' + err.message);
            console.error('Camera error:', err);
        }
    }

    async init() {
        // Set canvas size
        this.canvas.width = this.resolution;
        this.canvas.height = this.resolution;

        // Create programs
        this.cameraProgram = this.createProgram(vertexShaderSource, cameraProcessShader);
        this.rdProgram = this.createProgram(vertexShaderSource, reactionDiffusionShader);
        this.compositingProgram = this.createProgram(vertexShaderSource, compositingShader);
        this.initProgram = this.createProgram(vertexShaderSource, initShader);

        // Setup quad
        this.quadBuffer = this.setupQuad();

        // Create textures and framebuffers for 3 CMY layers (ping-pong)
        this.layers = {
            cyan: {
                ping: this.createTexture(this.resolution, this.resolution),
                pong: this.createTexture(this.resolution, this.resolution)
            },
            magenta: {
                ping: this.createTexture(this.resolution, this.resolution),
                pong: this.createTexture(this.resolution, this.resolution)
            },
            yellow: {
                ping: this.createTexture(this.resolution, this.resolution),
                pong: this.createTexture(this.resolution, this.resolution)
            }
        };

        this.framebuffers = {
            cyan: {
                ping: this.createFramebuffer(this.layers.cyan.ping),
                pong: this.createFramebuffer(this.layers.cyan.pong)
            },
            magenta: {
                ping: this.createFramebuffer(this.layers.magenta.ping),
                pong: this.createFramebuffer(this.layers.magenta.pong)
            },
            yellow: {
                ping: this.createFramebuffer(this.layers.yellow.ping),
                pong: this.createFramebuffer(this.layers.yellow.pong)
            }
        };

        // Camera data texture
        this.cameraDataTexture = this.createTexture(this.resolution, this.resolution);
        this.cameraDataFB = this.createFramebuffer(this.cameraDataTexture);

        this.currentBuffer = 'ping';

        // Setup camera
        await this.setupCamera();

        // Initialize layers
        this.initializeLayers();

        // Setup controls
        this.setupControls();

        // Start render loop
        this.render();
    }

    initializeLayers() {
        this.gl.viewport(0, 0, this.resolution, this.resolution);

        const posLoc = this.gl.getAttribLocation(this.initProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        this.gl.useProgram(this.initProgram);

        // Initialize each layer
        ['cyan', 'magenta', 'yellow'].forEach(layer => {
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffers[layer].ping);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffers[layer].pong);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        });
    }

    resetSimulation() {
        this.initializeLayers();
    }

    setupControls() {
        const controls = {
            feedRate: document.getElementById('feedRate'),
            killRate: document.getElementById('killRate'),
            diffA: document.getElementById('diffA'),
            diffB: document.getElementById('diffB'),
            cameraInfluence: document.getElementById('cameraInfluence')
        };

        const values = {
            feedRate: document.getElementById('feedValue'),
            killRate: document.getElementById('killValue'),
            diffA: document.getElementById('diffAValue'),
            diffB: document.getElementById('diffBValue'),
            cameraInfluence: document.getElementById('influenceValue')
        };

        controls.feedRate.addEventListener('input', (e) => {
            this.params.feed = parseFloat(e.target.value);
            values.feedRate.textContent = this.params.feed.toFixed(3);
        });

        controls.killRate.addEventListener('input', (e) => {
            this.params.kill = parseFloat(e.target.value);
            values.killRate.textContent = this.params.kill.toFixed(3);
        });

        controls.diffA.addEventListener('input', (e) => {
            this.params.diffA = parseFloat(e.target.value);
            values.diffA.textContent = this.params.diffA.toFixed(2);
        });

        controls.diffB.addEventListener('input', (e) => {
            this.params.diffB = parseFloat(e.target.value);
            values.diffB.textContent = this.params.diffB.toFixed(2);
        });

        controls.cameraInfluence.addEventListener('input', (e) => {
            this.params.cameraInfluence = parseFloat(e.target.value);
            values.cameraInfluence.textContent = this.params.cameraInfluence.toFixed(2);
        });

        document.getElementById('resetBtn').addEventListener('click', () => {
            this.resetSimulation();
        });

        const controlsDiv = document.getElementById('controls');
        const toggleBtn = document.getElementById('toggleControls');
        let controlsVisible = true;

        toggleBtn.addEventListener('click', () => {
            controlsVisible = !controlsVisible;
            if (controlsVisible) {
                controlsDiv.style.display = 'block';
                toggleBtn.textContent = 'Hide Controls';
            } else {
                controlsDiv.style.display = 'none';
                toggleBtn.style.position = 'fixed';
                toggleBtn.style.top = '10px';
                toggleBtn.style.left = '10px';
                toggleBtn.textContent = 'Show Controls';
            }
        });
    }

    updateVideoTexture() {
        if (this.video && this.video.readyState >= this.video.HAVE_CURRENT_DATA) {
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
            this.gl.texImage2D(
                this.gl.TEXTURE_2D,
                0,
                this.gl.RGBA,
                this.gl.RGBA,
                this.gl.UNSIGNED_BYTE,
                this.video
            );
        }
    }

    processCameraData() {
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.cameraDataFB);
        this.gl.viewport(0, 0, this.resolution, this.resolution);

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

    simulateReactionDiffusion() {
        this.gl.viewport(0, 0, this.resolution, this.resolution);
        this.gl.useProgram(this.rdProgram);

        const posLoc = this.gl.getAttribLocation(this.rdProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        // Set uniforms
        this.gl.uniform2f(
            this.gl.getUniformLocation(this.rdProgram, 'u_resolution'),
            this.resolution,
            this.resolution
        );
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_feed'), this.params.feed);
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_kill'), this.params.kill);
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_diffA'), this.params.diffA);
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_diffB'), this.params.diffB);
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_cameraInfluence'), this.params.cameraInfluence);

        // Bind camera data
        this.gl.activeTexture(this.gl.TEXTURE1);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.cameraDataTexture);
        this.gl.uniform1i(this.gl.getUniformLocation(this.rdProgram, 'u_cameraData'), 1);

        const nextBuffer = this.currentBuffer === 'ping' ? 'pong' : 'ping';
        const layers = ['cyan', 'magenta', 'yellow'];

        layers.forEach((layer, index) => {
            // Set current state texture
            this.gl.activeTexture(this.gl.TEXTURE0);
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.layers[layer][this.currentBuffer]);
            this.gl.uniform1i(this.gl.getUniformLocation(this.rdProgram, 'u_state'), 0);

            // Set channel
            this.gl.uniform1i(this.gl.getUniformLocation(this.rdProgram, 'u_channel'), index);

            // Render to next buffer
            this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffers[layer][nextBuffer]);
            this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
        });

        this.currentBuffer = nextBuffer;
    }

    composite() {
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);

        this.gl.useProgram(this.compositingProgram);

        const posLoc = this.gl.getAttribLocation(this.compositingProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        // Bind the three layers
        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.layers.cyan[this.currentBuffer]);
        this.gl.uniform1i(this.gl.getUniformLocation(this.compositingProgram, 'u_cyanLayer'), 0);

        this.gl.activeTexture(this.gl.TEXTURE1);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.layers.magenta[this.currentBuffer]);
        this.gl.uniform1i(this.gl.getUniformLocation(this.compositingProgram, 'u_magentaLayer'), 1);

        this.gl.activeTexture(this.gl.TEXTURE2);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.layers.yellow[this.currentBuffer]);
        this.gl.uniform1i(this.gl.getUniformLocation(this.compositingProgram, 'u_yellowLayer'), 2);

        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }

    render() {
        // Update video texture
        this.updateVideoTexture();

        // Process camera data
        this.processCameraData();

        // Run simulation multiple times per frame for smoother results
        for (let i = 0; i < 2; i++) {
            this.simulateReactionDiffusion();
        }

        // Composite final image
        this.composite();

        requestAnimationFrame(() => this.render());
    }
}

// Start the app when page loads
window.addEventListener('load', () => {
    new ReactionDiffusionApp();
});
