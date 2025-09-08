// AkashaConnector.js - Integration with AKASHA knowledge preservation system
const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');

class AkashaConnector {
    constructor() {
        this.endpoint = process.env.AKASHA_ENDPOINT || 'http://localhost:8002';
        this.connected = false;
        this.preservationQueue = [];
        this.localStoragePath = './data/akasha';
        this.initializeConnector();
    }

    async initializeConnector() {
        // Ensure local storage exists
        if (!fs.existsSync(this.localStoragePath)) {
            fs.mkdirSync(this.localStoragePath, { recursive: true });
        }

        // Test connection to AKASHA
        await this.testConnection();
        
        // Start preservation queue processor
        this.startQueueProcessor();
    }

    async testConnection() {
        try {
            const response = await fetch(`${this.endpoint}/health`, {
                timeout: 5000
            });
            
            if (response.ok) {
                this.connected = true;
                console.log('✅ AKASHA connection established');
            } else {
                this.connected = false;
                console.log('⚠️ AKASHA not available, using local preservation');
            }
        } catch (error) {
            this.connected = false;
            console.log('⚠️ AKASHA offline, using local fallback');
        }
    }

    async preserveDiscovery(discovery) {
        const preservationData = {
            type: 'historical_discovery',
            subtype: discovery.type,
            content: discovery,
            dimensions: 7, // LucianOS 7D preservation
            metadata: {
                timestamp: Date.now(),
                source: 'ChronoSphere',
                confidence: discovery.confidence || 0,
                preservation_level: this.getPreservationLevel(discovery)
            },
            verification_required: discovery.confidence < 0.7
        };

        if (this.connected) {
            try {
                const response = await fetch(`${this.endpoint}/api/preserve`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(preservationData)
                });

                if (response.ok) {
                    const result = await response.json();
                    console.log(`📦 Preserved to AKASHA: ${result.id}`);
                    return result;
                }
            } catch (error) {
                console.error('AKASHA preservation error:', error);
            }
        }

        // Fallback to local preservation
        return this.preserveLocally(preservationData);
    }

    async preserveLocally(data) {
        const id = `akasha_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const filePath = path.join(this.localStoragePath, `${id}.json`);
        
        try {
            // Add to preservation queue for later sync
            this.preservationQueue.push({
                id,
                data,
                attempts: 0
            });

            // Write to local file
            fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
            
            console.log(`💾 Preserved locally: ${id}`);
            return { id, status: 'local', path: filePath };
        } catch (error) {
            console.error('Local preservation error:', error);
            return null;
        }
    }

    async enrichWithDimensions(event) {
        const enrichmentRequest = {
            content: event,
            target_dimensions: 7,
            enrichment_type: 'historical',
            include_temporal: true,
            include_spatial: true,
            include_causal: true,
            include_semantic: true
        };

        if (this.connected) {
            try {
                const response = await fetch(`${this.endpoint}/api/enrich`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(enrichmentRequest)
                });

                if (response.ok) {
                    const enriched = await response.json();
                    return this.mergeEnrichment(event, enriched);
                }
            } catch (error) {
                console.error('AKASHA enrichment error:', error);
            }
        }

        // Fallback enrichment
        return this.localEnrichment(event);
    }

    localEnrichment(event) {
        // Local 7D enrichment simulation
        return {
            ...event,
            dimensions: {
                temporal: this.extractTemporalDimension(event),
                spatial: this.extractSpatialDimension(event),
                causal: this.extractCausalDimension(event),
                semantic: this.extractSemanticDimension(event),
                entropic: Math.random(), // Entropy dimension
                harmonic: Math.random(), // Harmonic resonance
                quantum: Math.random()   // Quantum coherence
            },
            enriched: true,
            enrichmentTimestamp: Date.now()
        };
    }

    extractTemporalDimension(event) {
        const year = event.year || 0;
        const century = Math.floor(year / 100);
        const decade = Math.floor((year % 100) / 10);
        
        return {
            year,
            century,
            decade,
            epoch: year < -500 ? 'ancient' : year < 500 ? 'classical' : 
                   year < 1500 ? 'medieval' : year < 1900 ? 'modern' : 'contemporary',
            cyclical: Math.sin(year * Math.PI / 500) // 1000-year cycle
        };
    }

    extractSpatialDimension(event) {
        if (!event.coordinates) {
            return { region: 'unknown', connectivity: 0 };
        }

        const { lat, lng } = event.coordinates;
        
        return {
            latitude: lat,
            longitude: lng,
            region: this.classifyRegion(lat, lng),
            hemisphere: lat > 0 ? 'northern' : 'southern',
            continent: this.classifyCon


tinent(lat, lng),
            connectivity: this.calculateConnectivity(lat, lng)
        };
    }

    extractCausalDimension(event) {
        const keywords = (event.description || '').toLowerCase();
        const causes = [];
        const effects = [];

        // Simple causal extraction
        const causalMarkers = {
            causes: ['caused by', 'due to', 'because of', 'resulted from'],
            effects: ['led to', 'caused', 'resulted in', 'triggered']
        };

        causalMarkers.causes.forEach(marker => {
            if (keywords.includes(marker)) {
                causes.push(marker);
            }
        });

        causalMarkers.effects.forEach(marker => {
            if (keywords.includes(marker)) {
                effects.push(marker);
            }
        });

        return {
            hasCauses: causes.length > 0,
            hasEffects: effects.length > 0,
            causalStrength: (causes.length + effects.length) / 6,
            causalType: event.type === 'collapse' ? 'destructive' : 
                       event.type === 'discovery' ? 'creative' : 'neutral'
        };
    }

    extractSemanticDimension(event) {
        const text = `${event.title} ${event.description}`.toLowerCase();
        const words = text.split(/\s+/);
        
        // Simple semantic categorization
        const categories = {
            conflict: ['war', 'battle', 'conflict', 'invasion', 'siege'],
            culture: ['art', 'philosophy', 'religion', 'literature', 'music'],
            technology: ['invention', 'discovery', 'innovation', 'technology'],
            politics: ['empire', 'kingdom', 'dynasty', 'republic', 'democracy'],
            disaster: ['collapse', 'disaster', 'famine', 'plague', 'earthquake']
        };

        const scores = {};
        Object.entries(categories).forEach(([category, keywords]) => {
            scores[category] = keywords.filter(kw => text.includes(kw)).length / keywords.length;
        });

        return {
            categories: scores,
            complexity: words.length / 100,
            uniqueness: new Set(words).size / words.length
        };
    }

    classifyRegion(lat, lng) {
        if (lat > 35 && lng > -15 && lng < 40) return 'Europe';
        if (lat > 20 && lat < 50 && lng > 60 && lng < 140) return 'Asia';
        if (lat < 0 && lng > -20 && lng < 50) return 'Africa';
        if (lat > -60 && lat < 15 && lng > -85 && lng < -30) return 'South_America';
        if (lat > 15 && lng > -130 && lng < -60) return 'North_America';
        if (lat < -10 && lng > 110) return 'Oceania';
        return 'Other';
    }

    classifyContinent(lat, lng) {
        // Simplified continent classification
        if (lng > -30 && lng < 60 && lat > -40 && lat < 75) return 'Afro-Eurasia';
        if (lng > -170 && lng < -30) return 'Americas';
        if (lng > 100 && lat < -10) return 'Australia';
        return 'Unknown';
    }

    calculateConnectivity(lat, lng) {
        // Calculate connectivity based on historical trade routes
        const tradeRoutes = [
            { lat: 30, lng: 35, radius: 15 }, // Mediterranean
            { lat: 25, lng: 80, radius: 20 }, // Indian Ocean
            { lat: 35, lng: 105, radius: 25 }, // Silk Road
            { lat: 0, lng: -70, radius: 15 }, // Amazon
            { lat: 20, lng: -95, radius: 10 } // Mesoamerica
        ];

        let connectivity = 0;
        tradeRoutes.forEach(route => {
            const distance = Math.sqrt(
                Math.pow(lat - route.lat, 2) + 
                Math.pow(lng - route.lng, 2)
            );
            if (distance < route.radius) {
                connectivity += (route.radius - distance) / route.radius;
            }
        });

        return Math.min(connectivity, 1);
    }

    mergeEnrichment(original, enriched) {
        return {
            ...original,
            ...enriched,
            akasha: {
                enriched: true,
                dimensions: enriched.dimensions || {},
                preservationId: enriched.id || null,
                timestamp: Date.now()
            }
        };
    }

    getPreservationLevel(discovery) {
        if (discovery.confidence > 0.9) return 'permanent';
        if (discovery.confidence > 0.7) return 'century';
        if (discovery.confidence > 0.5) return 'decade';
        return 'temporary';
    }

    async query(searchParams) {
        const queryRequest = {
            query: searchParams.query || '',
            filters: {
                type: searchParams.type,
                timeRange: searchParams.timeRange,
                confidence: searchParams.minConfidence || 0.5,
                dimensions: searchParams.dimensions || []
            },
            limit: searchParams.limit || 100
        };

        if (this.connected) {
            try {
                const response = await fetch(`${this.endpoint}/api/query`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(queryRequest)
                });

                if (response.ok) {
                    return await response.json();
                }
            } catch (error) {
                console.error('AKASHA query error:', error);
            }
        }

        // Local query fallback
        return this.queryLocal(searchParams);
    }

    async queryLocal(searchParams) {
        const results = [];
        const files = fs.readdirSync(this.localStoragePath);
        
        for (const file of files) {
            if (file.endsWith('.json')) {
                try {
                    const data = JSON.parse(
                        fs.readFileSync(path.join(this.localStoragePath, file), 'utf8')
                    );
                    
                    if (this.matchesQuery(data, searchParams)) {
                        results.push(data);
                    }
                } catch (error) {
                    console.error(`Error reading ${file}:`, error);
                }
            }
        }

        return results.slice(0, searchParams.limit || 100);
    }

    matchesQuery(data, params) {
        if (params.type && data.content?.type !== params.type) return false;
        
        if (params.minConfidence && 
            (data.content?.confidence || 0) < params.minConfidence) return false;
        
        if (params.query) {
            const searchText = JSON.stringify(data).toLowerCase();
            if (!searchText.includes(params.query.toLowerCase())) return false;
        }
        
        return true;
    }

    startQueueProcessor() {
        // Process preservation queue every 30 seconds
        setInterval(async () => {
            if (this.preservationQueue.length > 0 && this.connected) {
                await this.processQueue();
            }
        }, 30000);
    }

    async processQueue() {
        const toProcess = [...this.preservationQueue];
        this.preservationQueue = [];
        
        for (const item of toProcess) {
            try {
                const response = await fetch(`${this.endpoint}/api/preserve`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(item.data)
                });

                if (response.ok) {
                    console.log(`✅ Synced to AKASHA: ${item.id}`);
                    // Remove local file after successful sync
                    const filePath = path.join(this.localStoragePath, `${item.id}.json`);
                    if (fs.existsSync(filePath)) {
                        fs.unlinkSync(filePath);
                    }
                } else {
                    // Re-add to queue if failed
                    item.attempts++;
                    if (item.attempts < 3) {
                        this.preservationQueue.push(item);
                    }
                }
            } catch (error) {
                console.error('Queue processing error:', error);
                item.attempts++;
                if (item.attempts < 3) {
                    this.preservationQueue.push(item);
                }
            }
        }
    }

    getStatus() {
        return {
            connected: this.connected,
            endpoint: this.endpoint,
            queueSize: this.preservationQueue.length,
            localFiles: fs.readdirSync(this.localStoragePath).length
        };
    }
}

module.exports = AkashaConnector;