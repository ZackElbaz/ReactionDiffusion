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

        // Modulate feed/kill rates based on camera input
        // Where color is present, promote B growth; where absent, promote B decay
        float localFeed = u_feed + channelInfluence * u_cameraInfluence * 0.01;
        float localKill = u_kill + (1.0 - channelInfluence) * u_cameraInfluence * 0.01;

        // Standard Gray-Scott reaction-diffusion equations with local parameters
        float abb = a * b * b;
        float da = u_diffA * laplacian.r - abb + localFeed * (1.0 - a);
        float db = u_diffB * laplacian.g + abb - (localKill + localFeed) * b;

        // Update state
        a += da;
        b += db;

        // Clamp values to valid range
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

        // Proper subtractive color mixing (CMY inks on white paper)
        // Each ink absorbs specific wavelengths:
        // - Cyan absorbs Red and Magenta absorbs Red = darker red absorption
        // - Magenta absorbs Green and Yellow absorbs Green = darker green absorption
        // - Cyan absorbs Blue and Yellow absorbs Blue = darker blue absorption
        vec3 rgb = vec3(1.0); // Start with white

        // Multiply by what each ink DOESN'T absorb (transmits)
        rgb.r *= (1.0 - cyan) * (1.0 - magenta);     // Red absorbed by Cyan and Magenta
        rgb.g *= (1.0 - magenta) * (1.0 - yellow);   // Green absorbed by Magenta and Yellow
        rgb.b *= (1.0 - cyan) * (1.0 - yellow);      // Blue absorbed by Cyan and Yellow

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
        this.gl = this.canvas.getContext('webgl', {
            preserveDrawingBuffer: true,
            premultipliedAlpha: false
        });

        if (!this.gl) {
            this.showError('WebGL not supported');
            return;
        }

        // Parameters (Gray-Scott coral/mitosis pattern range)
        this.params = {
            feed: 0.055,
            kill: 0.062,
            diffA: 1.0,
            diffB: 0.5,
            cameraInfluence: 0.5  // Balance between camera influence and natural RD patterns
        };

        this.video = null;
        this.videoTexture = null;
        this.inputSource = 'camera'; // 'camera', 'video', or 'image'
        this.mediaElement = null; // Can be video or image element

        // Set canvas to fill window
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        this.init();
    }

    resizeCanvas() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        // Set canvas to fill window
        this.canvas.width = width;
        this.canvas.height = height;

        // Store dimensions for aspect ratio calculations
        this.canvasWidth = width;
        this.canvasHeight = height;

        // Use max resolution for better quality (matching input resolution)
        this.resolution = Math.max(width, height);
        console.log(`Canvas: ${width}x${height}, Simulation: ${this.resolution}`);
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
            console.log('🎥 Requesting camera access...');
            console.log('You should see a browser permission prompt now.');

            // Check if getUserMedia is supported
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                throw new Error('Camera API not supported in this browser');
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });

            console.log('✅ Camera access granted!');

            this.video = document.createElement('video');
            this.video.srcObject = stream;
            this.video.autoplay = true;
            this.video.playsInline = true;

            await new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    console.log('Video playing:', this.video.videoWidth, 'x', this.video.videoHeight);
                    resolve();
                };
            });

            this.mediaElement = this.video;
            this.videoTexture = this.createTexture(this.video.videoWidth, this.video.videoHeight);

        } catch (err) {
            console.error('Camera error:', err);
            let errorMsg = 'Camera Error: ';

            if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
                errorMsg += 'Permission denied. Please allow camera access and reload the page.';
            } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
                errorMsg += 'No camera found on this device.';
            } else if (err.name === 'NotReadableError') {
                errorMsg += 'Camera is already in use by another application.';
            } else if (err.name === 'SecurityError') {
                errorMsg += 'Camera blocked. Try using http://localhost instead of file://.';
            } else {
                errorMsg += err.message;
            }

            this.showError(errorMsg + '\n\nClick Reset to try again.');
        }
    }

    async setupVideoFile(file) {
        try {
            console.log('Loading video file:', file.name);

            this.video = document.createElement('video');
            this.video.src = URL.createObjectURL(file);
            this.video.loop = true;
            this.video.autoplay = true;
            this.video.playsInline = true;
            this.video.muted = true;

            await new Promise((resolve, reject) => {
                this.video.onloadedmetadata = () => {
                    this.video.play();
                    console.log('Video file loaded:', this.video.videoWidth, 'x', this.video.videoHeight);
                    resolve();
                };
                this.video.onerror = () => reject(new Error('Failed to load video file'));
            });

            this.mediaElement = this.video;
            this.videoTexture = this.createTexture(this.video.videoWidth, this.video.videoHeight);

        } catch (err) {
            console.error('Video file error:', err);
            this.showError('Failed to load video file: ' + err.message);
        }
    }

    async setupImageFile(file) {
        try {
            console.log('Loading image file:', file.name);

            const img = document.createElement('img');
            img.src = URL.createObjectURL(file);

            await new Promise((resolve, reject) => {
                img.onload = () => {
                    console.log('Image file loaded:', img.width, 'x', img.height);
                    resolve();
                };
                img.onerror = () => reject(new Error('Failed to load image file'));
            });

            this.mediaElement = img;
            this.video = null; // Clear video reference
            this.videoTexture = this.createTexture(img.width, img.height);

        } catch (err) {
            console.error('Image file error:', err);
            this.showError('Failed to load image file: ' + err.message);
        }
    }

    stopCurrentInput() {
        // Stop camera stream if active
        if (this.video && this.video.srcObject) {
            const stream = this.video.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            this.video.srcObject = null;
        }

        // Clear media element
        if (this.video) {
            this.video.pause();
            this.video = null;
        }

        this.mediaElement = null;
    }

    async init() {
        // Set clear color to white for subtractive color mixing
        this.gl.clearColor(1.0, 1.0, 1.0, 1.0);

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

        // Initialize layers
        this.initializeLayers();

        // Setup controls (must be before camera setup to bind events first)
        this.setupControls();

        // Setup camera by default (after controls are ready)
        await this.setupCamera();

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

        // Input source selector
        const inputSourceSelect = document.getElementById('inputSource');
        const fileInputGroup = document.getElementById('fileInputGroup');
        const fileInput = document.getElementById('fileInput');
        const manualFileBtn = document.getElementById('manualFileBtn');

        inputSourceSelect.addEventListener('change', (e) => {
            this.inputSource = e.target.value;
            console.log('📋 Input source changed to:', this.inputSource);

            // Stop any current input first
            this.stopCurrentInput();
            document.getElementById('error').style.display = 'none';

            if (this.inputSource === 'camera') {
                fileInputGroup.style.display = 'none';
                console.log('📷 Switching to camera mode - requesting camera access...');
                this.setupCamera();
            } else {
                fileInputGroup.style.display = 'block';
                fileInput.value = ''; // Clear previous selection
                console.log('📁 File input mode activated. Click the button or file input to select a', this.inputSource);

                // Try to automatically trigger file selector
                // This may be blocked by browser security, so we also provide a manual button
                setTimeout(() => {
                    console.log('⚡ Attempting to auto-open file selector...');
                    try {
                        fileInput.click();
                    } catch (err) {
                        console.log('⚠️ Auto-open blocked. Please click the "Choose File" button.');
                    }
                }, 100);
            }
        });

        // Manual file button (fallback if auto-trigger doesn't work)
        manualFileBtn.addEventListener('click', () => {
            console.log('👆 Manual file button clicked');
            fileInput.click();
        });

        fileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) {
                console.log('No file selected. You can select', this.inputSource, 'again from the dropdown or click the file input.');
                return;
            }

            console.log('File selected:', file.name, '(' + (file.size / 1024 / 1024).toFixed(2) + ' MB)');

            this.stopCurrentInput();
            document.getElementById('error').style.display = 'none';

            if (this.inputSource === 'video') {
                await this.setupVideoFile(file);
            } else if (this.inputSource === 'image') {
                await this.setupImageFile(file);
            }
        });

        document.getElementById('resetBtn').addEventListener('click', async () => {
            this.resetSimulation();
            // Also retry camera if it failed
            if (this.inputSource === 'camera' && (!this.video || !this.video.srcObject)) {
                document.getElementById('error').style.display = 'none';
                await this.setupCamera();
            }
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
        if (!this.mediaElement) return;

        // For video elements, check if ready
        if (this.mediaElement.tagName === 'VIDEO') {
            if (this.mediaElement.readyState >= this.mediaElement.HAVE_CURRENT_DATA) {
                this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
                this.gl.texImage2D(
                    this.gl.TEXTURE_2D,
                    0,
                    this.gl.RGBA,
                    this.gl.RGBA,
                    this.gl.UNSIGNED_BYTE,
                    this.mediaElement
                );
            }
        }
        // For image elements, update once (will keep updating same image)
        else if (this.mediaElement.tagName === 'IMG' && this.mediaElement.complete) {
            this.gl.bindTexture(this.gl.TEXTURE_2D, this.videoTexture);
            this.gl.texImage2D(
                this.gl.TEXTURE_2D,
                0,
                this.gl.RGBA,
                this.gl.RGBA,
                this.gl.UNSIGNED_BYTE,
                this.mediaElement
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

        // Clear to white for subtractive color mixing
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

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

        // Process camera data (only if we have media)
        if (this.mediaElement || this.videoTexture) {
            this.processCameraData();
        }

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
    console.log('🚀 Page loaded, initializing CMY Reaction Diffusion...');
    console.log('💡 Check the controls panel on the top-left of the screen');
    console.log('📱 Select your input source from the dropdown: Camera, Video File, or Image File');
    try {
        new ReactionDiffusionApp();
        console.log('✅ App initialized successfully!');
    } catch (err) {
        console.error('❌ Failed to initialize app:', err);
    }
});
