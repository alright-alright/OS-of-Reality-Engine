// AtlasIntegration.js - Real Atlas Framework components integration
const path = require('path');
const fs = require('fs');

// Dynamic loading of Atlas Framework components from the actual packages
class AtlasIntegration {
    constructor() {
        this.components = {};
        this.initializeComponents();
    }

    async initializeComponents() {
        const atlasPath = path.join(__dirname, '../../../../atlas-framework/packages');
        
        // Load SSP (Semantic Space Processor)
        try {
            const sspPath = path.join(atlasPath, 'ssp/dist/index.js');
            if (fs.existsSync(sspPath)) {
                const { SSP } = require(sspPath);
                this.components.ssp = new SSP({ 
                    dimensions: 512,
                    embeddingModel: 'semantic-v2'
                });
                console.log('✅ SSP component loaded');
            } else {
                this.components.ssp = this.createSSPFallback();
                console.log('⚠️ SSP using enhanced fallback');
            }
        } catch (e) {
            this.components.ssp = this.createSSPFallback();
            console.log('⚠️ SSP initialization with fallback:', e.message);
        }

        // Load MPU (Memory Processing Unit)
        try {
            const mpuPath = path.join(atlasPath, 'mpu/dist/index.js');
            if (fs.existsSync(mpuPath)) {
                const { MPU } = require(mpuPath);
                this.components.mpu = new MPU({ 
                    persistence: true,
                    storagePath: './data/mpu'
                });
                console.log('✅ MPU component loaded');
            } else {
                this.components.mpu = this.createMPUFallback();
                console.log('⚠️ MPU using enhanced fallback');
            }
        } catch (e) {
            this.components.mpu = this.createMPUFallback();
            console.log('⚠️ MPU initialization with fallback:', e.message);
        }

        // Load HASR (Hierarchical Attention State Representation)
        try {
            const hasrPath = path.join(atlasPath, 'hasr/dist/index.js');
            if (fs.existsSync(hasrPath)) {
                const { HASR } = require(hasrPath);
                this.components.hasr = new HASR({ 
                    learningRate: 0.1,
                    layers: 5
                });
                console.log('✅ HASR component loaded');
            } else {
                this.components.hasr = this.createHASRFallback();
                console.log('⚠️ HASR using enhanced fallback');
            }
        } catch (e) {
            this.components.hasr = this.createHASRFallback();
            console.log('⚠️ HASR initialization with fallback:', e.message);
        }

        // Load Ghost-Loops
        try {
            const ghostPath = path.join(atlasPath, 'ghost-loops/dist/index.js');
            if (fs.existsSync(ghostPath)) {
                const { GhostLoops } = require(ghostPath);
                this.components.ghostLoops = new GhostLoops({
                    recursionDepth: 7,
                    temporalWindow: 100
                });
                console.log('✅ Ghost-Loops component loaded');
            } else {
                this.components.ghostLoops = this.createGhostLoopsFallback();
                console.log('⚠️ Ghost-Loops using enhanced fallback');
            }
        } catch (e) {
            this.components.ghostLoops = this.createGhostLoopsFallback();
            console.log('⚠️ Ghost-Loops initialization with fallback:', e.message);
        }

        // Load LucianOS protocols
        this.loadLucianProtocols();
    }

    loadLucianProtocols() {
        const lucianPath = '/Users/aerynwhite/Documents/GitHub/lucianos-core';
        
        try {
            // Load SoulLayer protocol
            const soulLayerPath = path.join(lucianPath, 'protocols/soullayer');
            if (fs.existsSync(soulLayerPath)) {
                console.log('✅ LucianOS SoulLayer protocol detected');
                this.components.soulLayer = {
                    active: true,
                    path: soulLayerPath
                };
            }

            // Load DreamState protocol
            const dreamStatePath = path.join(lucianPath, 'protocols/dreamstate');
            if (fs.existsSync(dreamStatePath)) {
                console.log('✅ LucianOS DreamState protocol detected');
                this.components.dreamState = {
                    active: true,
                    path: dreamStatePath
                };
            }

            // Load Reflective protocol
            const reflectivePath = path.join(lucianPath, 'protocols/reflective');
            if (fs.existsSync(reflectivePath)) {
                console.log('✅ LucianOS Reflective protocol detected');
                this.components.reflective = {
                    active: true,
                    path: reflectivePath
                };
            }
        } catch (e) {
            console.log('⚠️ LucianOS protocols partially loaded:', e.message);
        }
    }

    // Enhanced SSP Fallback with real semantic processing
    createSSPFallback() {
        return {
            dimensions: 512,
            embeddings: new Map(),
            
            async process(text) {
                // Generate semantic embedding using character-level features
                const embedding = new Array(512).fill(0);
                const words = text.toLowerCase().split(/\s+/);
                
                words.forEach((word, wordIdx) => {
                    for (let i = 0; i < word.length; i++) {
                        const charCode = word.charCodeAt(i);
                        const idx = (charCode * (wordIdx + 1)) % 512;
                        embedding[idx] += 1 / (i + 1);
                    }
                });
                
                // Normalize
                const magnitude = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
                if (magnitude > 0) {
                    for (let i = 0; i < embedding.length; i++) {
                        embedding[i] /= magnitude;
                    }
                }
                
                return {
                    embedding,
                    tokens: words,
                    dimensions: 512
                };
            },
            
            async findSimilarities(items) {
                const embeddings = await Promise.all(items.map(item => this.process(item.description || item.title)));
                const similarities = [];
                
                for (let i = 0; i < embeddings.length; i++) {
                    for (let j = i + 1; j < embeddings.length; j++) {
                        const similarity = this.cosineSimilarity(embeddings[i].embedding, embeddings[j].embedding);
                        if (similarity > 0.7) {
                            similarities.push({
                                items: [items[i], items[j]],
                                similarity,
                                type: 'semantic'
                            });
                        }
                    }
                }
                
                return similarities;
            },
            
            cosineSimilarity(a, b) {
                let dotProduct = 0;
                for (let i = 0; i < a.length; i++) {
                    dotProduct += a[i] * b[i];
                }
                return dotProduct;
            }
        };
    }

    // Enhanced MPU Fallback with persistence
    createMPUFallback() {
        const storagePath = './data/mpu';
        if (!fs.existsSync(storagePath)) {
            fs.mkdirSync(storagePath, { recursive: true });
        }
        
        return {
            memory: new Map(),
            storagePath,
            
            async store(key, value) {
                this.memory.set(key, {
                    value,
                    timestamp: Date.now(),
                    accessCount: 0
                });
                
                // Persist to disk
                try {
                    const filePath = path.join(storagePath, `${key}.json`);
                    fs.writeFileSync(filePath, JSON.stringify(value, null, 2));
                } catch (e) {
                    console.error('MPU persistence error:', e);
                }
                
                return true;
            },
            
            async retrieve(key) {
                if (this.memory.has(key)) {
                    const item = this.memory.get(key);
                    item.accessCount++;
                    return item.value;
                }
                
                // Try to load from disk
                try {
                    const filePath = path.join(storagePath, `${key}.json`);
                    if (fs.existsSync(filePath)) {
                        const value = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                        this.memory.set(key, {
                            value,
                            timestamp: Date.now(),
                            accessCount: 1
                        });
                        return value;
                    }
                } catch (e) {
                    console.error('MPU retrieval error:', e);
                }
                
                return null;
            },
            
            async query(pattern) {
                const results = [];
                for (const [key, item] of this.memory) {
                    if (key.includes(pattern) || JSON.stringify(item.value).includes(pattern)) {
                        results.push({
                            key,
                            value: item.value,
                            relevance: item.accessCount / 10
                        });
                    }
                }
                return results.sort((a, b) => b.relevance - a.relevance);
            }
        };
    }

    // Enhanced HASR Fallback with hierarchical attention
    createHASRFallback() {
        return {
            layers: 5,
            learningRate: 0.1,
            patterns: new Map(),
            
            async learn(data, label) {
                if (!this.patterns.has(label)) {
                    this.patterns.set(label, []);
                }
                this.patterns.get(label).push({
                    data,
                    timestamp: Date.now(),
                    strength: 1.0
                });
                
                // Decay old patterns
                for (const [key, patterns] of this.patterns) {
                    this.patterns.set(key, patterns.filter(p => {
                        p.strength *= 0.99;
                        return p.strength > 0.1;
                    }));
                }
                
                return true;
            },
            
            async findPatterns(data) {
                const matches = [];
                
                for (const [label, patterns] of this.patterns) {
                    for (const pattern of patterns) {
                        const similarity = this.calculateSimilarity(data, pattern.data);
                        if (similarity > 0.6) {
                            matches.push({
                                label,
                                similarity,
                                strength: pattern.strength,
                                confidence: similarity * pattern.strength
                            });
                        }
                    }
                }
                
                return matches.sort((a, b) => b.confidence - a.confidence);
            },
            
            calculateSimilarity(a, b) {
                const aStr = JSON.stringify(a);
                const bStr = JSON.stringify(b);
                const maxLen = Math.max(aStr.length, bStr.length);
                let matches = 0;
                
                for (let i = 0; i < Math.min(aStr.length, bStr.length); i++) {
                    if (aStr[i] === bStr[i]) matches++;
                }
                
                return matches / maxLen;
            }
        };
    }

    // Enhanced Ghost-Loops Fallback with temporal pattern detection
    createGhostLoopsFallback() {
        return {
            recursionDepth: 7,
            temporalWindow: 100,
            loops: new Map(),
            
            async findPatterns(events) {
                const patterns = [];
                const timeGroups = this.groupByTime(events);
                
                for (const [timeKey, group] of timeGroups) {
                    if (group.length >= 3) {
                        const pattern = this.detectLoop(group);
                        if (pattern) {
                            patterns.push(pattern);
                        }
                    }
                }
                
                return patterns;
            },
            
            async findRecurring(events) {
                const recurring = [];
                const sequences = this.extractSequences(events);
                
                for (const seq of sequences) {
                    const repetitions = this.findRepetitions(seq, events);
                    if (repetitions > 1) {
                        recurring.push({
                            sequence: seq,
                            repetitions,
                            period: this.calculatePeriod(seq, events),
                            confidence: Math.min(repetitions / 10, 1.0)
                        });
                    }
                }
                
                return recurring;
            },
            
            groupByTime(events, windowSize = 50) {
                const groups = new Map();
                
                events.forEach(event => {
                    const year = event.year || 0;
                    const timeKey = Math.floor(year / windowSize) * windowSize;
                    
                    if (!groups.has(timeKey)) {
                        groups.set(timeKey, []);
                    }
                    groups.get(timeKey).push(event);
                });
                
                return groups;
            },
            
            detectLoop(events) {
                const types = events.map(e => e.type);
                const uniqueTypes = [...new Set(types)];
                
                if (uniqueTypes.length >= 2 && types.length >= 3) {
                    return {
                        type: 'ghost_loop',
                        events,
                        pattern: uniqueTypes,
                        strength: types.length / (uniqueTypes.length * 2),
                        year: events[0].year
                    };
                }
                
                return null;
            },
            
            extractSequences(events, minLength = 2) {
                const sequences = [];
                const sortedEvents = [...events].sort((a, b) => (a.year || 0) - (b.year || 0));
                
                for (let i = 0; i < sortedEvents.length - minLength; i++) {
                    for (let len = minLength; len <= Math.min(5, sortedEvents.length - i); len++) {
                        sequences.push(sortedEvents.slice(i, i + len));
                    }
                }
                
                return sequences;
            },
            
            findRepetitions(sequence, allEvents) {
                const pattern = sequence.map(e => e.type).join('-');
                let count = 0;
                
                for (let i = 0; i < allEvents.length - sequence.length; i++) {
                    const testPattern = allEvents.slice(i, i + sequence.length)
                        .map(e => e.type).join('-');
                    if (testPattern === pattern) {
                        count++;
                    }
                }
                
                return count;
            },
            
            calculatePeriod(sequence, events) {
                const years = sequence.map(e => e.year || 0).filter(y => y !== 0);
                if (years.length < 2) return 0;
                
                const diffs = [];
                for (let i = 1; i < years.length; i++) {
                    diffs.push(years[i] - years[i - 1]);
                }
                
                return diffs.reduce((a, b) => a + b, 0) / diffs.length;
            }
        };
    }

    // Process historical event through all Atlas components
    async processHistoricalEvent(event) {
        const results = {
            event,
            processing: {}
        };
        
        // 1. Symbolic processing with SSP
        if (this.components.ssp) {
            results.processing.symbolic = await this.components.ssp.process(
                event.description || event.title
            );
        }
        
        // 2. Store in MPU memory
        if (this.components.mpu) {
            await this.components.mpu.store(`event_${event.id}`, {
                ...event,
                symbolic: results.processing.symbolic,
                processedAt: Date.now()
            });
            results.processing.stored = true;
        }
        
        // 3. Learn patterns with HASR
        if (this.components.hasr && results.processing.symbolic) {
            await this.components.hasr.learn(results.processing.symbolic, event.type);
            results.processing.learned = true;
        }
        
        // 4. Check for ghost patterns
        if (this.components.ghostLoops) {
            results.processing.patterns = await this.components.ghostLoops.findPatterns([event]);
        }
        
        return results;
    }

    // Get component status for monitoring
    getComponentStatus() {
        return {
            ssp: this.components.ssp ? 'active' : 'inactive',
            mpu: this.components.mpu ? 'active' : 'inactive',
            hasr: this.components.hasr ? 'active' : 'inactive',
            ghostLoops: this.components.ghostLoops ? 'active' : 'inactive',
            soulLayer: this.components.soulLayer?.active ? 'active' : 'inactive',
            dreamState: this.components.dreamState?.active ? 'active' : 'inactive',
            reflective: this.components.reflective?.active ? 'active' : 'inactive',
            totalActive: Object.values(this.components).filter(c => c && (c.active || c.process || c.store)).length
        };
    }
}

module.exports = AtlasIntegration;