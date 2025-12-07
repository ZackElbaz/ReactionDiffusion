// Vertex shader - simple passthrough
const vertexShaderSource = `
    attribute vec2 a_position;
    varying vec2 v_texCoord;

    void main() {
        v_texCoord = a_position * 0.5 + 0.5;
        gl_Position = vec4(a_position, 0.0, 1.0);
    }
`;

// Camera processing shader - converts to greyscale
const cameraProcessShader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_videoTexture;

    void main() {
        vec2 uv = vec2(v_texCoord.x, 1.0 - v_texCoord.y); // Flip Y
        vec4 color = texture2D(u_videoTexture, uv);

        // Convert to greyscale (luminance)
        float grey = dot(color.rgb, vec3(0.299, 0.587, 0.114));

        // Invert so darker areas = higher value (patterns form in dark areas)
        float inverted = 1.0 - grey;

        gl_FragColor = vec4(inverted, inverted, inverted, 1.0);
    }
`;

// Reaction-Diffusion shader using Gray-Scott model (Karl Sims' math)
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

        // Get camera darkness (darker = more patterns)
        float darkness = texture2D(u_cameraData, v_texCoord).r;

        // Spatially-varying feed rate based on camera darkness
        // Dark areas = LOW feed rate = patterns grow and persist
        // Light areas = HIGH feed rate = patterns are suppressed/killed
        float feedRange = 0.08; // Stronger camera influence
        float localFeed = u_feed + (1.0 - darkness) * feedRange;

        // Radial flow (advection) - push patterns outward from center
        vec2 center = vec2(0.5, 0.5);
        vec2 toCenter = v_texCoord - center;
        float dist = length(toCenter);
        vec2 flowDir = normalize(toCenter);
        float flowStrength = 0.0003; // Subtle outward flow

        // Sample slightly inward for outward advection effect
        vec2 advectUV = v_texCoord - flowDir * flowStrength;
        vec2 advectedState = texture2D(u_state, advectUV).rg;

        // Mix original and advected state
        a = mix(a, advectedState.r, 0.3);
        b = mix(b, advectedState.g, 0.3);

        // Pure Gray-Scott equations (Karl Sims' math)
        float abb = a * b * b;
        float da = u_diffA * laplacian.r - abb + localFeed * (1.0 - a);
        float db = u_diffB * laplacian.g + abb - (u_kill + localFeed) * b;

        // Update state
        a += da;
        b += db;

        // Clamp to valid range (keep CONTINUOUS values for proper simulation)
        a = clamp(a, 0.0, 1.0);
        b = clamp(b, 0.0, 1.0);

        // Output continuous values (threshold happens in compositing shader)
        gl_FragColor = vec4(a, b, 0.0, 1.0);
    }
`;

// Compositing shader - outputs crisp black/white pattern
const compositingShader = `
    precision highp float;
    varying vec2 v_texCoord;
    uniform sampler2D u_state;

    void main() {
        // Get the B chemical (the visible pattern) - continuous value from simulation
        float b = texture2D(u_state, v_texCoord).g;

        // Apply binary threshold for crisp black/white output
        // Values above 0.5 become black (1.0), below become white (0.0)
        float pattern = step(0.5, b);

        // Invert: pattern=1 (black), pattern=0 (white)
        vec3 rgb = vec3(1.0 - pattern);

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
        // Create multiple random seeds for better distribution
        float rand1 = random(v_texCoord * 100.0 + vec2(0.0, 0.0));
        float rand2 = random(v_texCoord * 100.0 + vec2(123.456, 789.012));

        // Initialize with mostly A chemical (1.0) and random B spots
        float a = 1.0;
        // More random B seeds (5% instead of 2%) for better pattern nucleation
        float b = (rand1 > 0.95 || rand2 > 0.97) ? 1.0 : 0.0;

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

        // Parameters for crisp maze-like patterns
        this.params = {
            thickness: 0.5,      // Controls line thickness (via diffusion rates)
            viscosity: 0.5,      // Controls pattern movement speed
            cameraInfluence: 0.95  // Strong influence - patterns follow camera closely
        };

        // Internal Gray-Scott parameters (calculated from thickness/viscosity)
        this.updateGrayScottParams();

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

    updateGrayScottParams() {
        // Fixed feed/kill parameters for worm-like patterns
        this.feed = 0.05303;
        this.kill = 0.06235;

        // Thickness slider controls diffusion rates (line thickness)
        // Use standard Gray-Scott ratio: Da/Db ≈ 2.0
        // Lower values = crisper, more defined patterns
        const thicknessScale = this.params.thickness;
        this.diffA = 0.9 + thicknessScale * 0.3;   // Range: 0.9 - 1.2
        this.diffB = 0.45 + thicknessScale * 0.15; // Range: 0.45 - 0.6 (keeps 2:1 ratio)

        // Viscosity controls simulation speed (iterations per frame)
        // Low viscosity = more iterations = faster pattern evolution
        // High viscosity = fewer iterations = slower pattern evolution
        this.iterationsPerFrame = Math.max(1, Math.floor(1 + (1.0 - this.params.viscosity) * 5));
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

        // Create textures and framebuffers for single RD layer (ping-pong)
        this.stateTextures = {
            ping: this.createTexture(this.resolution, this.resolution),
            pong: this.createTexture(this.resolution, this.resolution)
        };

        this.framebuffers = {
            ping: this.createFramebuffer(this.stateTextures.ping),
            pong: this.createFramebuffer(this.stateTextures.pong)
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

        // Initialize ping buffer
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffers.ping);
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

        // Initialize pong buffer
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffers.pong);
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }

    resetSimulation() {
        this.initializeLayers();
    }

    setupControls() {
        const controls = {
            thickness: document.getElementById('thickness'),
            viscosity: document.getElementById('viscosity')
        };

        const values = {
            thickness: document.getElementById('thicknessValue'),
            viscosity: document.getElementById('viscosityValue')
        };

        controls.thickness.addEventListener('input', (e) => {
            this.params.thickness = parseFloat(e.target.value);
            values.thickness.textContent = this.params.thickness.toFixed(2);
            this.updateGrayScottParams();
        });

        controls.viscosity.addEventListener('input', (e) => {
            this.params.viscosity = parseFloat(e.target.value);
            values.viscosity.textContent = this.params.viscosity.toFixed(2);
            this.updateGrayScottParams();
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
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_feed'), this.feed);
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_kill'), this.kill);
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_diffA'), this.diffA);
        this.gl.uniform1f(this.gl.getUniformLocation(this.rdProgram, 'u_diffB'), this.diffB);

        // Bind camera data
        this.gl.activeTexture(this.gl.TEXTURE1);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.cameraDataTexture);
        this.gl.uniform1i(this.gl.getUniformLocation(this.rdProgram, 'u_cameraData'), 1);

        const nextBuffer = this.currentBuffer === 'ping' ? 'pong' : 'ping';

        // Set current state texture
        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.stateTextures[this.currentBuffer]);
        this.gl.uniform1i(this.gl.getUniformLocation(this.rdProgram, 'u_state'), 0);

        // Render to next buffer
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, this.framebuffers[nextBuffer]);
        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);

        this.currentBuffer = nextBuffer;
    }

    composite() {
        this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
        this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);

        // Clear to white background
        this.gl.clear(this.gl.COLOR_BUFFER_BIT);

        this.gl.useProgram(this.compositingProgram);

        const posLoc = this.gl.getAttribLocation(this.compositingProgram, 'a_position');
        this.gl.enableVertexAttribArray(posLoc);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
        this.gl.vertexAttribPointer(posLoc, 2, this.gl.FLOAT, false, 0, 0);

        // Bind the state texture
        this.gl.activeTexture(this.gl.TEXTURE0);
        this.gl.bindTexture(this.gl.TEXTURE_2D, this.stateTextures[this.currentBuffer]);
        this.gl.uniform1i(this.gl.getUniformLocation(this.compositingProgram, 'u_state'), 0);

        this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    }

    render() {
        // Update video texture
        this.updateVideoTexture();

        // Process camera data (only if we have media)
        if (this.mediaElement || this.videoTexture) {
            this.processCameraData();
        }

        // Run simulation based on viscosity (lower viscosity = more iterations = faster)
        for (let i = 0; i < this.iterationsPerFrame; i++) {
            this.simulateReactionDiffusion();
        }

        // Composite final image
        this.composite();

        requestAnimationFrame(() => this.render());
    }
}

// Start the app when page loads
window.addEventListener('load', () => {
    console.log('🚀 Page loaded, initializing Greyscale Reaction Diffusion...');
    console.log('💡 Check the controls panel on the top-left of the screen');
    console.log('📱 Select your input source from the dropdown: Camera, Video File, or Image File');
    console.log('🎨 Patterns form in darker areas of the input');
    try {
        new ReactionDiffusionApp();
        console.log('✅ App initialized successfully!');
    } catch (err) {
        console.error('❌ Failed to initialize app:', err);
    }
});
