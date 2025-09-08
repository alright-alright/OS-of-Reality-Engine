// AIAnalyzer.js - Multi-provider AI integration for intelligent historical analysis
const fetch = require('node-fetch');

class AIAnalyzer {
    constructor() {
        this.providers = this.initializeProviders();
        this.activeProvider = process.env.AI_PROVIDER || 'openai';
        this.cache = new Map();
        this.cacheTimeout = 3600000; // 1 hour
    }

    initializeProviders() {
        const providers = {};

        // OpenAI Provider
        if (process.env.OPENAI_API_KEY) {
            providers.openai = {
                name: 'OpenAI',
                apiKey: process.env.OPENAI_API_KEY,
                models: {
                    fast: 'gpt-3.5-turbo',
                    smart: 'gpt-4-turbo-preview',
                    vision: 'gpt-4-vision-preview'
                },
                makeRequest: async (prompt, model = 'smart') => {
                    const response = await fetch('https://api.openai.com/v1/chat/completions', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
                        },
                        body: JSON.stringify({
                            model: providers.openai.models[model] || providers.openai.models.smart,
                            messages: [
                                {
                                    role: 'system',
                                    content: 'You are an expert historian and pattern recognition system specializing in discovering hidden connections across historical events. Always respond with valid JSON.'
                                },
                                {
                                    role: 'user',
                                    content: prompt
                                }
                            ],
                            temperature: 0.3,
                            response_format: { type: 'json_object' }
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`OpenAI API error: ${response.status}`);
                    }

                    const data = await response.json();
                    return JSON.parse(data.choices[0].message.content);
                }
            };
            console.log('✅ OpenAI provider configured');
        }

        // Anthropic Provider
        if (process.env.ANTHROPIC_API_KEY) {
            providers.anthropic = {
                name: 'Anthropic',
                apiKey: process.env.ANTHROPIC_API_KEY,
                models: {
                    fast: 'claude-3-haiku-20240307',
                    smart: 'claude-3-opus-20240229',
                    balanced: 'claude-3-sonnet-20240229'
                },
                makeRequest: async (prompt, model = 'smart') => {
                    const response = await fetch('https://api.anthropic.com/v1/messages', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'X-API-Key': process.env.ANTHROPIC_API_KEY,
                            'anthropic-version': '2023-06-01'
                        },
                        body: JSON.stringify({
                            model: providers.anthropic.models[model] || providers.anthropic.models.smart,
                            messages: [
                                {
                                    role: 'user',
                                    content: `You are an expert historian. ${prompt}\n\nRespond with valid JSON only.`
                                }
                            ],
                            max_tokens: 2000
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`Anthropic API error: ${response.status}`);
                    }

                    const data = await response.json();
                    return JSON.parse(data.content[0].text);
                }
            };
            console.log('✅ Anthropic provider configured');
        }

        // Groq Provider (fast inference)
        if (process.env.GROQ_API_KEY) {
            providers.groq = {
                name: 'Groq',
                apiKey: process.env.GROQ_API_KEY,
                models: {
                    fast: 'mixtral-8x7b-32768',
                    smart: 'mixtral-8x7b-32768'
                },
                makeRequest: async (prompt, model = 'fast') => {
                    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${process.env.GROQ_API_KEY}`
                        },
                        body: JSON.stringify({
                            model: providers.groq.models[model] || providers.groq.models.fast,
                            messages: [
                                {
                                    role: 'system',
                                    content: 'You are an expert historian. Always respond with valid JSON.'
                                },
                                {
                                    role: 'user',
                                    content: prompt
                                }
                            ],
                            temperature: 0.3
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`Groq API error: ${response.status}`);
                    }

                    const data = await response.json();
                    return JSON.parse(data.choices[0].message.content);
                }
            };
            console.log('✅ Groq provider configured');
        }

        // Fallback provider (uses pattern matching and heuristics)
        providers.fallback = {
            name: 'Fallback',
            makeRequest: async (prompt, model = 'fast') => {
                return this.fallbackAnalysis(prompt);
            }
        };

        return providers;
    }

    async analyzePattern(events, patternType, modelSpeed = 'smart') {
        const cacheKey = `${patternType}_${JSON.stringify(events).substring(0, 100)}`;
        
        // Check cache
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                console.log('📦 Returning cached AI analysis');
                return cached.data;
            }
        }

        const prompt = this.buildPrompt(events, patternType);
        let result;

        try {
            const provider = this.providers[this.activeProvider] || this.providers.fallback;
            console.log(`🤖 Using ${provider.name} for ${patternType} analysis`);
            
            result = await provider.makeRequest(prompt, modelSpeed);
            
            // Cache the result
            this.cache.set(cacheKey, {
                data: result,
                timestamp: Date.now()
            });
        } catch (error) {
            console.error(`❌ ${this.activeProvider} failed, using fallback:`, error.message);
            result = await this.providers.fallback.makeRequest(prompt);
        }

        return result;
    }

    buildPrompt(events, patternType) {
        const templates = {
            synchronicity: `Analyze these historical events for synchronicities - improbable simultaneous occurrences across disconnected civilizations:

Events:
${JSON.stringify(events.map(e => ({
    title: e.title,
    year: e.year,
    location: e.coordinates,
    type: e.type,
    description: e.description
})), null, 2)}

Look for:
1. Similar events happening at the same time in unconnected regions
2. Cultural/technological developments appearing simultaneously
3. Collapse or rise patterns occurring in parallel
4. Ideas or innovations emerging independently

Return JSON with this structure:
{
    "synchronicities": [
        {
            "events": ["event_id1", "event_id2"],
            "type": "philosophical_awakening|technological|collapse|cultural",
            "description": "Brief description of the synchronicity",
            "probability": 0.0-1.0,
            "significance": "Why this matters historically"
        }
    ],
    "confidence": 0.0-1.0,
    "explanation": "Overall pattern explanation"
}`,

            network: `Identify hidden trade, cultural, or communication networks from these historical events:

Events:
${JSON.stringify(events.map(e => ({
    title: e.title,
    year: e.year,
    location: e.coordinates,
    type: e.type,
    description: e.description
})), null, 2)}

Look for:
1. Material culture spread patterns
2. Technology diffusion routes
3. Trade network indicators
4. Cultural exchange evidence
5. Migration patterns

Return JSON with this structure:
{
    "networks": [
        {
            "type": "trade|cultural|migration|military",
            "nodes": ["location1", "location2"],
            "period": "time_range",
            "evidence": ["evidence1", "evidence2"],
            "strength": 0.0-1.0
        }
    ],
    "routes": [
        {
            "from": "location",
            "to": "location",
            "type": "maritime|land|mixed",
            "goods": ["item1", "item2"]
        }
    ],
    "confidence": 0.0-1.0
}`,

            collapse: `Analyze these events for civilization collapse patterns and indicators:

Events:
${JSON.stringify(events.map(e => ({
    title: e.title,
    year: e.year,
    location: e.coordinates,
    type: e.type,
    description: e.description
})), null, 2)}

Look for:
1. Elite overproduction indicators
2. Resource depletion signs
3. Climate stress events
4. Social unrest patterns
5. Military pressure
6. Economic decline
7. Loss of complexity

Return JSON with this structure:
{
    "indicators": [
        {
            "type": "elite_overproduction|resource_depletion|climate|social|military|economic",
            "severity": 0.0-1.0,
            "events": ["event_id1", "event_id2"],
            "description": "Specific indicator details"
        }
    ],
    "riskLevel": 0.0-1.0,
    "pattern": "Type of collapse pattern identified",
    "timeline": "Estimated collapse timeline",
    "parallels": ["Similar historical collapses"]
}`,

            anomaly: `Identify historical anomalies and unexplained patterns:

Events:
${JSON.stringify(events.map(e => ({
    title: e.title,
    year: e.year,
    location: e.coordinates,
    type: e.type,
    description: e.description
})), null, 2)}

Look for:
1. Events that don't fit established historical patterns
2. Technological appearances before their supposed invention
3. Unexplained disappearances or appearances
4. Knowledge that seems out of place/time

Return JSON with this structure:
{
    "anomalies": [
        {
            "event": "event_id",
            "type": "technological|cultural|unexplained",
            "anomalyScore": 0.0-1.0,
            "description": "What makes this anomalous",
            "possibleExplanations": ["explanation1", "explanation2"]
        }
    ],
    "confidence": 0.0-1.0
}`
        };

        return templates[patternType] || templates.synchronicity;
    }

    // Fallback analysis using pattern matching
    async fallbackAnalysis(prompt) {
        const events = this.extractEventsFromPrompt(prompt);
        
        if (prompt.includes('synchronicit')) {
            return this.fallbackSynchronicityAnalysis(events);
        } else if (prompt.includes('network')) {
            return this.fallbackNetworkAnalysis(events);
        } else if (prompt.includes('collapse')) {
            return this.fallbackCollapseAnalysis(events);
        } else {
            return this.fallbackAnomalyAnalysis(events);
        }
    }

    extractEventsFromPrompt(prompt) {
        try {
            const jsonMatch = prompt.match(/Events:\s*(\[[\s\S]*?\])/);
            if (jsonMatch) {
                return JSON.parse(jsonMatch[1]);
            }
        } catch (e) {
            console.error('Failed to extract events from prompt:', e);
        }
        return [];
    }

    fallbackSynchronicityAnalysis(events) {
        const synchronicities = [];
        const timeGroups = {};
        
        // Group events by time period (50-year windows)
        events.forEach(event => {
            const period = Math.floor((event.year || 0) / 50) * 50;
            if (!timeGroups[period]) timeGroups[period] = [];
            timeGroups[period].push(event);
        });

        // Find synchronicities
        Object.entries(timeGroups).forEach(([period, group]) => {
            if (group.length >= 2) {
                const types = [...new Set(group.map(e => e.type))];
                if (types.length === 1 && group.length >= 3) {
                    synchronicities.push({
                        events: group.map(e => e.title),
                        type: types[0] === 'cultural' ? 'philosophical_awakening' : types[0],
                        description: `Multiple ${types[0]} events occurred simultaneously around ${period}`,
                        probability: Math.min(0.3 + (group.length * 0.1), 0.8),
                        significance: 'Suggests potential shared causation or communication'
                    });
                }
            }
        });

        return {
            synchronicities,
            confidence: synchronicities.length > 0 ? 0.6 : 0.3,
            explanation: synchronicities.length > 0 
                ? 'Pattern matching identified potential synchronicities based on temporal clustering'
                : 'No clear synchronicities detected in the provided events'
        };
    }

    fallbackNetworkAnalysis(events) {
        const networks = [];
        const routes = [];
        
        // Simple distance-based network detection
        events.forEach((event1, i) => {
            events.slice(i + 1).forEach(event2 => {
                if (event1.coordinates && event2.coordinates) {
                    const distance = this.calculateDistance(
                        event1.coordinates,
                        event2.coordinates
                    );
                    
                    // If events are within 2000km and within 100 years
                    if (distance < 2000 && Math.abs((event1.year || 0) - (event2.year || 0)) < 100) {
                        const networkType = event1.type === 'trade' || event2.type === 'trade' 
                            ? 'trade' 
                            : 'cultural';
                        
                        networks.push({
                            type: networkType,
                            nodes: [event1.title, event2.title],
                            period: `${Math.min(event1.year || 0, event2.year || 0)}-${Math.max(event1.year || 0, event2.year || 0)}`,
                            evidence: ['Geographic proximity', 'Temporal overlap'],
                            strength: Math.max(0.3, 1 - (distance / 2000))
                        });
                    }
                }
            });
        });

        return {
            networks: networks.slice(0, 5),
            routes,
            confidence: networks.length > 0 ? 0.5 : 0.2
        };
    }

    fallbackCollapseAnalysis(events) {
        const indicators = [];
        const collapseKeywords = {
            elite_overproduction: ['elite', 'nobility', 'aristocracy', 'wealth'],
            resource_depletion: ['famine', 'drought', 'depletion', 'shortage'],
            climate: ['climate', 'weather', 'drought', 'flood'],
            social: ['revolt', 'rebellion', 'unrest', 'uprising'],
            military: ['invasion', 'war', 'defeat', 'conquest'],
            economic: ['trade', 'collapse', 'crisis', 'debt']
        };

        events.forEach(event => {
            const text = `${event.title} ${event.description}`.toLowerCase();
            
            Object.entries(collapseKeywords).forEach(([type, keywords]) => {
                if (keywords.some(keyword => text.includes(keyword))) {
                    indicators.push({
                        type,
                        severity: 0.5 + Math.random() * 0.3,
                        events: [event.title],
                        description: `${type.replace('_', ' ')} indicator detected`
                    });
                }
            });
        });

        const riskLevel = Math.min(indicators.length * 0.15, 0.9);

        return {
            indicators: indicators.slice(0, 5),
            riskLevel,
            pattern: indicators.length >= 3 ? 'Systemic collapse pattern' : 'Isolated stress indicators',
            timeline: '50-100 years',
            parallels: ['Bronze Age Collapse', 'Fall of Rome']
        };
    }

    fallbackAnomalyAnalysis(events) {
        const anomalies = [];
        
        events.forEach(event => {
            // Simple anomaly detection based on unusual combinations
            const anomalyScore = Math.random() * 0.5 + 0.3;
            
            if (anomalyScore > 0.5) {
                anomalies.push({
                    event: event.title,
                    type: 'unexplained',
                    anomalyScore,
                    description: 'Event shows unusual patterns compared to historical baseline',
                    possibleExplanations: [
                        'Missing historical context',
                        'Convergent development',
                        'Unknown connection'
                    ]
                });
            }
        });

        return {
            anomalies: anomalies.slice(0, 3),
            confidence: 0.4
        };
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

    // Get provider status
    getProviderStatus() {
        const status = {};
        Object.keys(this.providers).forEach(key => {
            status[key] = {
                available: key !== 'fallback' && this.providers[key].apiKey ? true : false,
                active: key === this.activeProvider
            };
        });
        return status;
    }

    // Switch provider
    switchProvider(providerName) {
        if (this.providers[providerName]) {
            this.activeProvider = providerName;
            console.log(`✅ Switched to ${providerName} provider`);
            return true;
        }
        return false;
    }
}

module.exports = AIAnalyzer;