// DiscoveryEngine.js - Real discovery algorithms using Atlas and AI
const WikidataService = require('./WikidataService');
const AtlasIntegration = require('./AtlasIntegration');
const AIAnalyzer = require('./AIAnalyzer');

class DiscoveryEngine {
    constructor() {
        this.wikidata = new WikidataService();
        this.atlas = new AtlasIntegration();
        this.ai = new AIAnalyzer();
        this.discoveries = new Map();
        this.initializeEngine();
    }

    async initializeEngine() {
        await this.atlas.initializeComponents();
        console.log('🚀 Discovery Engine initialized with Atlas components');
    }

    // Main discovery method
    async discover(parameters, timeRange) {
        console.log(`🔍 Starting discovery: ${timeRange.start} to ${timeRange.end}`);
        
        // 1. Query real historical events from Wikidata
        const events = await this.wikidata.queryHistoricalEvents({
            startYear: timeRange.start,
            endYear: timeRange.end,
            region: parameters.region || 'global',
            eventTypes: parameters.eventTypes || [],
            limit: parameters.limit || 1000
        });

        console.log(`📊 Processing ${events.length} historical events`);

        // 2. Process events through Atlas components
        const processedEvents = await this.processWithAtlas(events);

        // 3. Discover patterns based on parameters
        const discoveries = {
            synchronicities: [],
            networks: [],
            collapses: [],
            anomalies: [],
            ghostLoops: []
        };

        // Run discovery algorithms in parallel
        const [syncResults, networkResults, collapseResults, anomalyResults, ghostResults] = await Promise.all([
            this.discoverSynchronicities(processedEvents, parameters),
            this.reconstructNetworks(processedEvents, parameters),
            this.detectCollapsePatterns(processedEvents, parameters),
            this.findAnomalies(processedEvents, parameters),
            this.findGhostLoops(processedEvents, parameters)
        ]);

        discoveries.synchronicities = syncResults;
        discoveries.networks = networkResults;
        discoveries.collapses = collapseResults;
        discoveries.anomalies = anomalyResults;
        discoveries.ghostLoops = ghostResults;

        // 4. Store discoveries
        await this.storeDiscoveries(discoveries);

        // 5. Calculate discovery metrics
        const metrics = this.calculateMetrics(discoveries);

        return {
            discoveries: this.formatDiscoveries(discoveries),
            metrics,
            eventsAnalyzed: events.length,
            timestamp: Date.now()
        };
    }

    async processWithAtlas(events) {
        const processed = [];
        
        for (const event of events) {
            const result = await this.atlas.processHistoricalEvent(event);
            processed.push(result);
        }
        
        return processed;
    }

    async discoverSynchronicities(events, params) {
        const timeWindow = params.timeWindowRadius || 50;
        const threshold = params.synchronicityThreshold || 0.75;
        
        // Group events by time window
        const timeGroups = this.groupByTimeWindow(events, timeWindow);
        const synchronicities = [];
        
        for (const [period, group] of timeGroups) {
            if (group.length >= 3) {
                // Check for multi-region patterns
                const regions = this.extractRegions(group);
                
                if (regions.size >= 3) {
                    // Use Atlas SSP to find semantic similarities
                    let similarities = [];
                    if (this.atlas.components.ssp) {
                        similarities = await this.atlas.components.ssp.findSimilarities(
                            group.map(e => e.event)
                        );
                    }
                    
                    // Use AI to analyze synchronicity
                    const aiAnalysis = await this.ai.analyzePattern(
                        group.map(e => e.event),
                        'synchronicity'
                    );
                    
                    if (aiAnalysis.confidence >= threshold) {
                        synchronicities.push({
                            type: 'synchronicity',
                            period,
                            events: group.map(e => ({
                                id: e.event.id,
                                title: e.event.title,
                                location: e.event.coordinates,
                                year: e.event.year
                            })),
                            pattern: aiAnalysis.synchronicities[0]?.type || 'unknown',
                            description: aiAnalysis.explanation,
                            confidence: aiAnalysis.confidence,
                            regions: Array.from(regions),
                            semanticSimilarity: similarities.length > 0 ? 
                                similarities[0].similarity : 0
                        });
                    }
                }
            }
        }
        
        return synchronicities;
    }

    async reconstructNetworks(events, params) {
        const density = params.networkDensity || 0.5;
        const networks = [];
        
        // Use AI to identify network patterns
        const aiAnalysis = await this.ai.analyzePattern(
            events.map(e => e.event),
            'network'
        );
        
        if (aiAnalysis.networks && aiAnalysis.networks.length > 0) {
            for (const network of aiAnalysis.networks) {
                if (network.strength >= density) {
                    // Calculate diffusion velocity
                    const velocity = this.calculateDiffusionVelocity(
                        events.filter(e => network.nodes.includes(e.event.title))
                    );
                    
                    networks.push({
                        type: 'network',
                        networkType: network.type,
                        nodes: network.nodes,
                        period: network.period,
                        evidence: network.evidence,
                        strength: network.strength,
                        velocity,
                        routes: aiAnalysis.routes || [],
                        confidence: aiAnalysis.confidence
                    });
                }
            }
        }
        
        // Use Atlas HASR to find material culture patterns
        if (this.atlas.components.hasr) {
            const patterns = await this.atlas.components.hasr.findPatterns(
                events.map(e => e.processing?.symbolic).filter(Boolean)
            );
            
            patterns.forEach(pattern => {
                if (pattern.confidence >= density) {
                    networks.push({
                        type: 'network',
                        networkType: 'material_culture',
                        pattern: pattern.label,
                        strength: pattern.strength,
                        confidence: pattern.confidence
                    });
                }
            });
        }
        
        return networks;
    }

    async detectCollapsePatterns(events, params) {
        const collapses = [];
        
        // Use AI to detect collapse indicators
        const aiAnalysis = await this.ai.analyzePattern(
            events.map(e => e.event),
            'collapse'
        );
        
        if (aiAnalysis.indicators && aiAnalysis.indicators.length >= 3) {
            // Group by civilization
            const civilizations = this.extractCivilizations(events);
            
            for (const [civ, civEvents] of civilizations) {
                const indicators = aiAnalysis.indicators.filter(ind => 
                    ind.events.some(e => civEvents.some(ce => ce.event.title === e))
                );
                
                if (indicators.length >= 2) {
                    collapses.push({
                        type: 'collapse',
                        civilization: civ,
                        period: this.calculatePeriod(civEvents),
                        indicators: indicators.map(i => ({
                            type: i.type,
                            severity: i.severity,
                            description: i.description
                        })),
                        riskLevel: aiAnalysis.riskLevel,
                        pattern: aiAnalysis.pattern,
                        timeline: aiAnalysis.timeline,
                        parallels: aiAnalysis.parallels || [],
                        confidence: Math.min(indicators.length * 0.25, 0.95)
                    });
                }
            }
        }
        
        return collapses;
    }

    async findAnomalies(events, params) {
        const anomalyThreshold = params.anomalyDetection || 0.7;
        const anomalies = [];
        
        // Use AI to detect anomalies
        const aiAnalysis = await this.ai.analyzePattern(
            events.map(e => e.event),
            'anomaly'
        );
        
        if (aiAnalysis.anomalies) {
            for (const anomaly of aiAnalysis.anomalies) {
                if (anomaly.anomalyScore >= anomalyThreshold) {
                    const event = events.find(e => e.event.title === anomaly.event);
                    
                    anomalies.push({
                        type: 'anomaly',
                        event: {
                            id: event?.event.id,
                            title: anomaly.event,
                            location: event?.event.coordinates,
                            year: event?.event.year
                        },
                        anomalyType: anomaly.type,
                        score: anomaly.anomalyScore,
                        description: anomaly.description,
                        explanations: anomaly.possibleExplanations,
                        confidence: aiAnalysis.confidence
                    });
                }
            }
        }
        
        return anomalies;
    }

    async findGhostLoops(events, params) {
        const ghostLoops = [];
        
        // Use Atlas Ghost-Loops component
        if (this.atlas.components.ghostLoops && this.atlas.components.ghostLoops.findPatterns) {
            const patterns = await this.atlas.components.ghostLoops.findPatterns(
                events.map(e => e.event)
            );
            
            const recurring = await this.atlas.components.ghostLoops.findRecurring(
                events.map(e => e.event)
            );
            
            // Combine ghost loop patterns
            patterns.forEach(pattern => {
                ghostLoops.push({
                    type: 'ghost_loop',
                    subType: 'pattern',
                    events: pattern.events.map(e => ({
                        id: e.id,
                        title: e.title,
                        year: e.year
                    })),
                    pattern: pattern.pattern,
                    strength: pattern.strength,
                    year: pattern.year,
                    confidence: pattern.strength
                });
            });
            
            recurring.forEach(rec => {
                ghostLoops.push({
                    type: 'ghost_loop',
                    subType: 'recurring',
                    sequence: rec.sequence.map(e => e.title),
                    repetitions: rec.repetitions,
                    period: rec.period,
                    confidence: rec.confidence
                });
            });
        } else {
            // Fallback ghost loop detection when Atlas component not available
            const eventGroups = {};
            events.forEach(e => {
                const key = `${e.year}_${e.region || 'global'}`;
                if (!eventGroups[key]) eventGroups[key] = [];
                eventGroups[key].push(e);
            });
            
            Object.entries(eventGroups).forEach(([key, group]) => {
                if (group.length > 3) {
                    ghostLoops.push({
                        type: 'ghost_loop',
                        subType: 'temporal_cluster',
                        year: group[0].year,
                        region: group[0].region,
                        events: group.map(e => e.event),
                        confidence: Math.min(0.9, group.length / 10)
                    });
                }
            });
        }
        
        return ghostLoops;
    }

    // Helper methods
    groupByTimeWindow(events, windowSize) {
        const groups = new Map();
        
        events.forEach(event => {
            const year = event.event.year || 0;
            const period = Math.floor(year / windowSize) * windowSize;
            
            if (!groups.has(period)) {
                groups.set(period, []);
            }
            groups.get(period).push(event);
        });
        
        return groups;
    }

    extractRegions(events) {
        const regions = new Set();
        
        events.forEach(event => {
            if (event.event.coordinates) {
                // Simple region classification based on coordinates
                const lat = event.event.coordinates.lat;
                const lng = event.event.coordinates.lng;
                
                if (lat > 35 && lng > -15 && lng < 40) regions.add('Europe');
                else if (lat > 20 && lat < 50 && lng > 60 && lng < 140) regions.add('Asia');
                else if (lat < 0 && lng > -20 && lng < 50) regions.add('Africa');
                else if (lat > -60 && lat < 15 && lng > -85 && lng < -30) regions.add('South America');
                else if (lat > 15 && lng > -130 && lng < -60) regions.add('North America');
                else if (lat < -10 && lng > 110) regions.add('Oceania');
                else regions.add('Other');
            }
        });
        
        return regions;
    }

    extractCivilizations(events) {
        const civilizations = new Map();
        
        events.forEach(event => {
            const civ = event.event.country || event.event.category || 'Unknown';
            if (!civilizations.has(civ)) {
                civilizations.set(civ, []);
            }
            civilizations.get(civ).push(event);
        });
        
        return civilizations;
    }

    calculateDiffusionVelocity(events) {
        if (events.length < 2) return 0;
        
        const sorted = events.sort((a, b) => (a.event.year || 0) - (b.event.year || 0));
        const first = sorted[0];
        const last = sorted[sorted.length - 1];
        
        if (!first.event.coordinates || !last.event.coordinates) return 0;
        
        const distance = this.calculateDistance(
            first.event.coordinates,
            last.event.coordinates
        );
        const timeSpan = Math.abs((last.event.year || 0) - (first.event.year || 0));
        
        return timeSpan > 0 ? distance / timeSpan : 0; // km/year
    }

    calculateDistance(coord1, coord2) {
        const R = 6371; // Earth's radius in km
        const dLat = (coord2.lat - coord1.lat) * Math.PI / 180;
        const dLon = (coord2.lng - coord1.lng) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                  Math.cos(coord1.lat * Math.PI / 180) * Math.cos(coord2.lat * Math.PI / 180) *
                  Math.sin(dLon/2) * Math.sin(dLon/2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
        return R * c;
    }

    calculatePeriod(events) {
        const years = events.map(e => e.event.year || 0).filter(y => y !== 0);
        if (years.length === 0) return 'Unknown';
        
        const min = Math.min(...years);
        const max = Math.max(...years);
        
        return `${min} to ${max}`;
    }

    async storeDiscoveries(discoveries) {
        if (this.atlas.components.mpu) {
            const timestamp = Date.now();
            
            for (const [type, items] of Object.entries(discoveries)) {
                for (const item of items) {
                    await this.atlas.components.mpu.store(
                        `discovery_${type}_${timestamp}_${Math.random().toString(36).substr(2, 9)}`,
                        item
                    );
                }
            }
        }
        
        // Also store in local cache
        this.discoveries.set(Date.now(), discoveries);
    }

    calculateMetrics(discoveries) {
        const totalDiscoveries = Object.values(discoveries)
            .reduce((sum, arr) => sum + arr.length, 0);
        
        const avgConfidence = Object.values(discoveries)
            .flat()
            .reduce((sum, d) => sum + (d.confidence || 0), 0) / (totalDiscoveries || 1);
        
        return {
            total: totalDiscoveries,
            byType: {
                synchronicities: discoveries.synchronicities.length,
                networks: discoveries.networks.length,
                collapses: discoveries.collapses.length,
                anomalies: discoveries.anomalies.length,
                ghostLoops: discoveries.ghostLoops.length
            },
            averageConfidence: avgConfidence,
            quality: avgConfidence > 0.7 ? 'high' : avgConfidence > 0.5 ? 'medium' : 'low'
        };
    }

    formatDiscoveries(discoveries) {
        const formatted = [];
        
        // Format each discovery type for frontend display
        Object.entries(discoveries).forEach(([type, items]) => {
            items.forEach(item => {
                formatted.push({
                    ...item,
                    id: `${type}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    timestamp: Date.now()
                });
            });
        });
        
        return formatted.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
    }

    // Get engine status
    getStatus() {
        return {
            atlas: this.atlas.getComponentStatus(),
            ai: this.ai.getProviderStatus(),
            discoveriesCount: this.discoveries.size,
            ready: true
        };
    }
}

module.exports = DiscoveryEngine;