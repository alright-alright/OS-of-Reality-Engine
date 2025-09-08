// HistoricalChat.js - Discovery-aware historical AI chat powered by Lucian systems
const AIAnalyzer = require('./AIAnalyzer');
const AtlasIntegration = require('./AtlasIntegration');
const WikidataService = require('./WikidataService');
const AkashaConnector = require('./AkashaConnector');
const fs = require('fs');
const path = require('path');

class HistoricalChat {
    constructor(discoveryEngine) {
        this.discoveryEngine = discoveryEngine;
        this.ai = new AIAnalyzer();
        this.atlas = new AtlasIntegration();
        this.wikidata = new WikidataService();
        this.akasha = new AkashaConnector();
        
        // Chat state
        this.conversations = new Map();
        this.activeDiscoveries = [];
        this.contextWindow = [];
        this.hypothesisMode = false;
        
        // Initialize Atlas components
        this.initializeChat();
    }

    async initializeChat() {
        await this.atlas.initializeComponents();
        console.log('🤖 Historical Chat initialized with Lucian systems');
    }

    // Main chat endpoint
    async chat(sessionId, message, options = {}) {
        // Get or create conversation
        let conversation = this.conversations.get(sessionId);
        if (!conversation) {
            conversation = {
                id: sessionId,
                messages: [],
                discoveries: [],
                hypotheses: [],
                context: {
                    timeRange: options.timeRange || { start: -3000, end: 2024 },
                    focus: options.focus || 'general',
                    lastDiscoveryId: null
                },
                startedAt: Date.now()
            };
            this.conversations.set(sessionId, conversation);
        }

        // Add user message
        conversation.messages.push({
            role: 'user',
            content: message,
            timestamp: Date.now()
        });

        // Analyze message intent
        const intent = await this.analyzeIntent(message);
        
        // Build context from discoveries and Atlas components
        const context = await this.buildContext(conversation, intent);
        
        // Generate response based on intent
        let response;
        switch (intent.type) {
            case 'discovery_explanation':
                response = await this.explainDiscovery(intent, context);
                break;
            case 'hypothesis_testing':
                response = await this.testHypothesis(intent, context);
                break;
            case 'pattern_search':
                response = await this.searchPatterns(intent, context);
                break;
            case 'historical_question':
                response = await this.answerHistorical(intent, context);
                break;
            case 'connection_analysis':
                response = await this.analyzeConnections(intent, context);
                break;
            default:
                response = await this.generalResponse(intent, context);
        }

        // Process through Atlas components
        const enrichedResponse = await this.enrichWithAtlas(response);
        
        // Store in conversation
        conversation.messages.push({
            role: 'assistant',
            content: enrichedResponse.content,
            metadata: enrichedResponse.metadata,
            timestamp: Date.now()
        });

        // Store in MPU for learning
        if (this.atlas.components.mpu) {
            await this.atlas.components.mpu.store(
                `chat_${sessionId}_${Date.now()}`,
                {
                    message,
                    response: enrichedResponse,
                    intent,
                    context: context.summary
                }
            );
        }

        // Learn patterns with HASR
        if (this.atlas.components.hasr && enrichedResponse.metadata.confidence > 0.7) {
            await this.atlas.components.hasr.learn(
                { query: message, intent },
                intent.type
            );
        }

        return enrichedResponse;
    }

    // Analyze user message intent
    async analyzeIntent(message) {
        const lower = message.toLowerCase();
        
        // Pattern matching for intent
        const patterns = {
            discovery_explanation: [
                'explain this', 'what does this mean', 'tell me about this discovery',
                'why is this significant', 'what pattern'
            ],
            hypothesis_testing: [
                'what if', 'could there be', 'is it possible', 'hypothesis',
                'test the theory', 'what about the idea'
            ],
            pattern_search: [
                'find patterns', 'search for', 'look for', 'are there any',
                'show me similar', 'find connections'
            ],
            historical_question: [
                'what happened', 'when did', 'who was', 'where did',
                'why did', 'how did', 'tell me about'
            ],
            connection_analysis: [
                'connection between', 'related to', 'influenced by',
                'caused by', 'led to', 'relationship'
            ]
        };

        let detectedIntent = { type: 'general', confidence: 0 };
        
        for (const [intentType, keywords] of Object.entries(patterns)) {
            const matches = keywords.filter(k => lower.includes(k)).length;
            const confidence = matches / keywords.length;
            
            if (confidence > detectedIntent.confidence) {
                detectedIntent = { 
                    type: intentType, 
                    confidence,
                    keywords: keywords.filter(k => lower.includes(k))
                };
            }
        }

        // Extract entities (time periods, locations, civilizations)
        detectedIntent.entities = await this.extractEntities(message);
        
        return detectedIntent;
    }

    // Build context from discoveries and Atlas
    async buildContext(conversation, intent) {
        const context = {
            recentDiscoveries: this.activeDiscoveries.slice(-5),
            conversationHistory: conversation.messages.slice(-10),
            relevantPatterns: [],
            akashaKnowledge: [],
            wikidataCache: [],
            summary: ''
        };

        // Get relevant patterns from MPU
        if (this.atlas.components.mpu) {
            const patterns = await this.atlas.components.mpu.query(
                intent.keywords ? intent.keywords.join(' ') : ''
            );
            context.relevantPatterns = patterns.slice(0, 3);
        }

        // Get AKASHA preserved knowledge
        if (intent.entities.timePeriod) {
            const akashaResults = await this.akasha.query({
                type: 'historical_discovery',
                timeRange: intent.entities.timePeriod,
                limit: 5
            });
            context.akashaKnowledge = akashaResults;
        }

        // Add recent Wikidata events if relevant
        if (this.wikidata.cache.size > 0) {
            const cacheEntries = Array.from(this.wikidata.cache.values())
                .slice(-3)
                .map(entry => entry.data);
            context.wikidataCache = cacheEntries.flat().slice(0, 10);
        }

        // Generate context summary
        context.summary = this.summarizeContext(context);
        
        return context;
    }

    // Explain a discovery using Lucian insights
    async explainDiscovery(intent, context) {
        const discovery = context.recentDiscoveries[0];
        if (!discovery) {
            return {
                content: "I don't see any recent discoveries to explain. Try running a discovery analysis first!",
                metadata: { type: 'no_discovery' }
            };
        }

        const prompt = `
You are a historical expert with access to advanced pattern recognition systems.
Explain this discovery in detail:

Discovery: ${JSON.stringify(discovery, null, 2)}

Context: ${context.summary}

Provide:
1. What makes this discovery significant
2. Historical context and implications
3. Connections to other known patterns
4. Potential new research directions

Be specific and reference the actual data.`;

        const aiResponse = await this.ai.analyzePattern(
            [discovery],
            'discovery_explanation',
            'smart'
        );

        return {
            content: this.formatDiscoveryExplanation(discovery, aiResponse),
            metadata: {
                type: 'discovery_explanation',
                discoveryId: discovery.id,
                confidence: aiResponse.confidence || 0.8
            }
        };
    }

    // Test a historical hypothesis
    async testHypothesis(intent, context) {
        const hypothesis = intent.entities.hypothesis || this.extractHypothesis(intent);
        
        // Run targeted discovery based on hypothesis
        const discoveryParams = {
            ...this.discoveryEngine.parameters,
            focus: hypothesis.focus,
            timeRange: hypothesis.timeRange || { start: -3000, end: 2024 }
        };

        const results = await this.discoveryEngine.discover(
            discoveryParams,
            hypothesis.timeRange
        );

        // Analyze results for hypothesis support
        const analysis = await this.analyzeHypothesisResults(hypothesis, results);

        return {
            content: this.formatHypothesisResults(hypothesis, analysis, results),
            metadata: {
                type: 'hypothesis_test',
                hypothesis,
                supportLevel: analysis.support,
                discoveries: results.discoveries.length
            }
        };
    }

    // Search for specific patterns
    async searchPatterns(intent, context) {
        const searchTerms = intent.entities.searchTerms || [];
        const timeRange = intent.entities.timePeriod || { start: -3000, end: 2024 };

        // Query Wikidata for specific events
        const events = await this.wikidata.queryHistoricalEvents({
            startYear: timeRange.start,
            endYear: timeRange.end,
            eventTypes: this.mapSearchToEventTypes(searchTerms),
            limit: 100
        });

        // Process through discovery engine
        const processedEvents = await Promise.all(
            events.map(e => this.atlas.processHistoricalEvent(e))
        );

        // Find patterns using Ghost Loops
        let patterns = [];
        if (this.atlas.components.ghostLoops) {
            patterns = await this.atlas.components.ghostLoops.findPatterns(
                processedEvents.map(p => p.event)
            );
        }

        return {
            content: this.formatPatternResults(searchTerms, patterns, events),
            metadata: {
                type: 'pattern_search',
                eventsFound: events.length,
                patternsFound: patterns.length,
                searchTerms
            }
        };
    }

    // Answer historical questions with context
    async answerHistorical(intent, context) {
        const enrichedPrompt = `
You are a historical expert with access to:
- ${context.recentDiscoveries.length} recent pattern discoveries
- ${context.akashaKnowledge.length} preserved historical insights
- ${context.wikidataCache.length} cached historical events

Question: ${intent.entities.question || 'General historical inquiry'}

Context:
${context.summary}

Provide a comprehensive answer that:
1. Directly answers the question
2. References specific discoveries or patterns when relevant
3. Suggests related areas to explore
4. Maintains historical accuracy

Format as a clear, engaging response.`;

        const response = await this.ai.analyzePattern(
            context.wikidataCache,
            'historical_question',
            'smart'
        );

        return {
            content: response.explanation || response,
            metadata: {
                type: 'historical_answer',
                sources: context.wikidataCache.length,
                confidence: response.confidence || 0.75
            }
        };
    }

    // Analyze connections between events/civilizations
    async analyzeConnections(intent, context) {
        const entities = intent.entities.items || [];
        
        if (entities.length < 2) {
            return {
                content: "Please specify at least two items to analyze connections between.",
                metadata: { type: 'insufficient_data' }
            };
        }

        // Use SSP for semantic similarity
        let similarities = [];
        if (this.atlas.components.ssp) {
            similarities = await this.atlas.components.ssp.findSimilarities(
                entities.map(e => ({ title: e, description: e }))
            );
        }

        // Find network connections
        const networkAnalysis = await this.ai.analyzePattern(
            entities,
            'network',
            'smart'
        );

        return {
            content: this.formatConnectionAnalysis(entities, similarities, networkAnalysis),
            metadata: {
                type: 'connection_analysis',
                entities,
                connectionStrength: networkAnalysis.confidence || 0
            }
        };
    }

    // General response for unclassified queries
    async generalResponse(intent, context) {
        const response = await this.ai.analyzePattern(
            context.recentDiscoveries,
            'general',
            'fast'
        );

        return {
            content: this.formatGeneralResponse(response, context),
            metadata: {
                type: 'general',
                confidence: 0.6
            }
        };
    }

    // Enrich response with Atlas components
    async enrichWithAtlas(response) {
        // Process through SSP for semantic enhancement
        if (this.atlas.components.ssp) {
            const semantic = await this.atlas.components.ssp.process(response.content);
            response.metadata.semantic = semantic;
        }

        // Store important responses in AKASHA
        if (response.metadata.confidence > 0.8) {
            await this.akasha.preserveDiscovery({
                type: 'chat_insight',
                content: response.content,
                metadata: response.metadata,
                timestamp: Date.now()
            });
        }

        return response;
    }

    // Helper methods for formatting responses
    formatDiscoveryExplanation(discovery, analysis) {
        return `## Discovery Analysis: ${discovery.title || discovery.type}

**Significance**: ${analysis.explanation || 'This represents a significant historical pattern.'}

**Pattern Type**: ${discovery.type}
**Confidence**: ${(discovery.confidence * 100).toFixed(0)}%
**Time Period**: ${discovery.year || discovery.period || 'Unknown'}

### Historical Context
${this.getHistoricalContext(discovery)}

### Implications
${analysis.implications || 'This discovery suggests previously unknown connections between historical events.'}

### Research Opportunities
${this.suggestResearch(discovery)}

*Analyzed using Atlas Framework components (SSP, MPU, HASR) for enhanced pattern recognition.*`;
    }

    formatHypothesisResults(hypothesis, analysis, results) {
        const supportEmoji = analysis.support > 0.7 ? '✅' : analysis.support > 0.4 ? '🤔' : '❌';
        
        return `## Hypothesis Test ${supportEmoji}

**Hypothesis**: ${hypothesis.statement}

**Support Level**: ${(analysis.support * 100).toFixed(0)}%

### Evidence Found
${results.discoveries.slice(0, 3).map(d => 
    `- **${d.title || d.type}** (${d.year || 'Date unknown'}): ${d.description || 'Pattern detected'}`
).join('\n')}

### Analysis
${analysis.explanation}

### Recommendation
${analysis.support > 0.7 
    ? 'Strong evidence supports this hypothesis. Consider deeper investigation.'
    : analysis.support > 0.4
    ? 'Mixed evidence. The hypothesis may be partially correct or need refinement.'
    : 'Limited supporting evidence. Consider alternative explanations.'}

*Tested using ${results.eventsAnalyzed} historical events across ${results.discoveries.length} discoveries.*`;
    }

    formatPatternResults(searchTerms, patterns, events) {
        if (patterns.length === 0) {
            return `No significant patterns found for: ${searchTerms.join(', ')}

However, I found ${events.length} related events. Would you like me to analyze them differently?`;
        }

        return `## Pattern Search Results

**Search Terms**: ${searchTerms.join(', ')}

### Patterns Discovered (${patterns.length})
${patterns.slice(0, 5).map(p => 
    `- **${p.type || 'Pattern'}**: ${p.pattern || 'Complex pattern detected'}
  Strength: ${(p.strength * 100).toFixed(0)}%`
).join('\n')}

### Related Events (${events.length})
${events.slice(0, 5).map(e => 
    `- **${e.title}** (${e.year}): ${e.description || 'No description'}`
).join('\n')}

### Insights
${this.generatePatternInsights(patterns)}

*Analyzed using Ghost-Loops temporal pattern detection.*`;
    }

    formatConnectionAnalysis(entities, similarities, network) {
        return `## Connection Analysis

**Analyzing**: ${entities.join(' ↔ ')}

### Semantic Connections (SSP Analysis)
${similarities.length > 0 
    ? similarities.map(s => `- Similarity: ${(s.similarity * 100).toFixed(0)}%`).join('\n')
    : 'No direct semantic connections found.'}

### Network Analysis
${network.networks && network.networks.length > 0
    ? network.networks.map(n => 
        `- **${n.type}** network: ${n.evidence.join(', ')}`
      ).join('\n')
    : 'No clear network patterns detected.'}

### Potential Links
${this.suggestPotentialLinks(entities, network)}

*Analysis powered by Atlas Framework SSP and network reconstruction algorithms.*`;
    }

    formatGeneralResponse(response, context) {
        return `${response.explanation || response}

${context.recentDiscoveries.length > 0 
    ? `\n### Recent Discoveries\nYou have ${context.recentDiscoveries.length} recent discoveries that might be relevant. Ask me to explain any of them!`
    : ''}

*Powered by ChronoSphere's Lucian systems integration.*`;
    }

    // Entity extraction
    async extractEntities(message) {
        const entities = {
            timePeriod: null,
            locations: [],
            civilizations: [],
            searchTerms: [],
            question: message,
            hypothesis: null,
            items: []
        };

        // Extract time periods (basic implementation)
        const yearMatch = message.match(/(\d{3,4})\s*(BCE|BC|CE|AD)?/gi);
        if (yearMatch) {
            entities.timePeriod = { 
                start: -2000, 
                end: -1000 
            }; // Simplified - would parse properly
        }

        // Extract known civilizations
        const civilizations = ['egypt', 'rome', 'greece', 'maya', 'china', 'india'];
        civilizations.forEach(civ => {
            if (message.toLowerCase().includes(civ)) {
                entities.civilizations.push(civ);
                entities.items.push(civ);
            }
        });

        return entities;
    }

    // Helper methods
    extractHypothesis(intent) {
        return {
            statement: intent.entities.question || 'General hypothesis',
            focus: 'general',
            timeRange: intent.entities.timePeriod || { start: -3000, end: 2024 }
        };
    }

    async analyzeHypothesisResults(hypothesis, results) {
        const supportingEvidence = results.discoveries.filter(d => 
            d.confidence > 0.6
        );
        
        return {
            support: supportingEvidence.length / Math.max(results.discoveries.length, 1),
            explanation: `Found ${supportingEvidence.length} supporting patterns out of ${results.discoveries.length} total discoveries.`,
            evidence: supportingEvidence
        };
    }

    mapSearchToEventTypes(searchTerms) {
        const mapping = {
            'war': ['Q198'],
            'battle': ['Q178561'],
            'trade': ['Q8187769'],
            'collapse': ['Q3839081'],
            'discovery': ['Q12772819']
        };
        
        const types = [];
        searchTerms.forEach(term => {
            if (mapping[term.toLowerCase()]) {
                types.push(...mapping[term.toLowerCase()]);
            }
        });
        
        return types.length > 0 ? types : ['Q1190554']; // Default to historical event
    }

    getHistoricalContext(discovery) {
        const contexts = {
            'synchronicity': 'This simultaneous occurrence across disconnected civilizations suggests either unknown communication networks or convergent cultural evolution.',
            'network': 'This indicates previously unrecognized trade or cultural exchange routes that connected distant civilizations.',
            'collapse': 'This pattern matches known collapse indicators and may reveal systemic vulnerabilities in complex societies.',
            'anomaly': 'This represents an unusual pattern that doesn\'t fit established historical models, warranting further investigation.',
            'ghost_loop': 'This recurring pattern suggests cyclical historical processes that repeat across different time periods.'
        };
        
        return contexts[discovery.type] || 'This discovery provides new insights into historical patterns.';
    }

    suggestResearch(discovery) {
        const suggestions = [
            '1. Cross-reference with archaeological evidence from the same period',
            '2. Investigate similar patterns in other civilizations',
            '3. Analyze climate data for environmental factors',
            '4. Examine linguistic evidence for cultural transmission',
            '5. Study genetic markers for population movements'
        ];
        
        return suggestions.slice(0, 3).join('\n');
    }

    generatePatternInsights(patterns) {
        if (patterns.length === 0) return 'No clear patterns emerged from this search.';
        
        const types = [...new Set(patterns.map(p => p.type))];
        return `Detected ${types.length} distinct pattern types. The strongest patterns suggest ${
            types.includes('ghost_loop') ? 'recurring historical cycles' : 'unique historical connections'
        }.`;
    }

    suggestPotentialLinks(entities, network) {
        return `Consider investigating:
- Contemporary events in the same time period
- Trade routes that might have connected these entities
- Shared technological or cultural innovations
- Common environmental or climatic factors`;
    }

    summarizeContext(context) {
        return `Recent discoveries: ${context.recentDiscoveries.length}, ` +
               `Historical patterns: ${context.relevantPatterns.length}, ` +
               `Preserved knowledge: ${context.akashaKnowledge.length}, ` +
               `Cached events: ${context.wikidataCache.length}`;
    }

    // Session management
    getConversation(sessionId) {
        return this.conversations.get(sessionId);
    }

    clearConversation(sessionId) {
        this.conversations.delete(sessionId);
    }

    setActiveDiscoveries(discoveries) {
        this.activeDiscoveries = discoveries;
    }
}

module.exports = HistoricalChat;