# 🚨 CODING AGENT IMPLEMENTATION INSTRUCTIONS

## CRITICAL: Transform ChronoSphere from Mock to Production

**Project**: ChronoSphere - Historical Discovery Engine  
**Owner**: AerwareAI (Aeryn)  
**Priority**: HIGH - This is the FIRST real Atlas Framework implementation  
**Purpose**: Prove LucianOS components work in production with real data

---

## 📋 EXECUTIVE SUMMARY

ChronoSphere currently has a **working UI and mock backend**. Your mission is to replace ALL mock implementations with REAL functionality using:

1. **Atlas Framework components** (not approximations)
2. **Real Wikidata SPARQL** queries (13M+ historical events)
3. **AI Provider integration** (OpenAI/Anthropic/Groq)
4. **Actual discovery algorithms** that find real patterns
5. **AKASHA integration** for knowledge preservation

This is NOT a demo - it's a production system to prove LucianOS components work at scale.

---

## 🎯 PRIMARY OBJECTIVES

### 1. ✅ **REPLACE ALL MOCK DATA WITH REAL WIKIDATA**

**Current State**: Mock data in `server.js` lines 110-145 and 150-195  
**Required**: Real SPARQL queries to Wikidata

```javascript
// REPLACE THIS MOCK CODE:
const mockEvents = [
  { id: 'event_1', title: 'Bronze Age Collapse', ... }
];

// WITH REAL WIKIDATA QUERIES:
const WikidataService = require('./services/WikidataService');
const events = await WikidataService.queryHistoricalEvents({
  startYear,
  endYear,
  region,
  eventTypes: ['Q178561', 'Q198', 'Q124757'] // battles, wars, disasters
});
```

**Implementation Requirements**:
- Create `src/backend/services/WikidataService.js`
- Use `node-fetch` for SPARQL endpoint: `https://query.wikidata.org/sparql`
- Implement proper SPARQL queries with error handling
- Cache results in Redis/memory to avoid rate limits
- Return REAL coordinates, dates, and descriptions

**Example SPARQL Query**:
```sparql
SELECT ?event ?eventLabel ?date ?coords ?description WHERE {
  ?event wdt:P31/wdt:P279* wd:Q1190554.  # instance of historical event
  ?event wdt:P585 ?date.                  # point in time
  ?event wdt:P625 ?coords.                # coordinates
  OPTIONAL { ?event schema:description ?description FILTER (lang(?description) = "en") }
  FILTER(?date >= "START_DATE"^^xsd:dateTime && ?date <= "END_DATE"^^xsd:dateTime)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} LIMIT 10000
```

---

### 2. 🧠 **INTEGRATE REAL ATLAS FRAMEWORK COMPONENTS**

**Current State**: No actual Atlas imports  
**Required**: Use REAL Atlas Framework packages

First, check if Atlas packages exist in the repo:
```bash
ls /Users/aerynwhite/Documents/GitHub/atlas-framework/packages/
```

If they exist, install them:
```javascript
// package.json dependencies
"dependencies": {
  "@atlas/ssp": "file:../atlas-framework/packages/ssp",
  "@atlas/mpu": "file:../atlas-framework/packages/mpu",
  "@atlas/hasr": "file:../atlas-framework/packages/hasr",
  "@atlas/ghost-loops": "file:../atlas-framework/packages/ghost-loops",
  "@atlas/chrono-layer": "file:../atlas-framework/packages/chrono-layer"
}
```

Then create `src/backend/services/AtlasIntegration.js`:
```javascript
const { SSP } = require('@atlas/ssp');
const { MPU } = require('@atlas/mpu');
const { HASR } = require('@atlas/hasr');
const { GhostLoops } = require('@atlas/ghost-loops');
const { ChronoLayer } = require('@atlas/chrono-layer');

class AtlasProcessor {
  constructor() {
    this.ssp = new SSP({ dimensions: 512 });
    this.mpu = new MPU({ persistence: true });
    this.hasr = new HASR({ learningRate: 0.1 });
    this.ghostLoops = new GhostLoops();
    this.chronoLayer = new ChronoLayer();
  }

  async processHistoricalEvent(event) {
    // 1. Symbolic processing
    const symbolic = await this.ssp.process(event.description);
    
    // 2. Store in memory
    await this.mpu.store(`event_${event.id}`, {
      ...event,
      symbolic,
      timestamp: Date.now()
    });
    
    // 3. Learn patterns
    await this.hasr.learn(symbolic, event.type);
    
    // 4. Apply temporal coherence
    const smoothed = await this.chronoLayer.smooth(event);
    
    // 5. Check for ghost patterns
    const patterns = await this.ghostLoops.findPatterns([event]);
    
    return { event: smoothed, patterns };
  }
}
```

---

### 3. 🤖 **IMPLEMENT AI PROVIDER FOR ANALYSIS**

**Current State**: No AI integration  
**Required**: Use OpenAI/Anthropic for intelligent analysis

Create `src/backend/services/AIAnalyzer.js`:
```javascript
const OpenAI = require('openai');
const Anthropic = require('@anthropic-ai/sdk');

class AIAnalyzer {
  constructor() {
    // Support multiple providers
    this.providers = {
      openai: new OpenAI({ apiKey: process.env.OPENAI_API_KEY }),
      anthropic: new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY })
    };
    this.activeProvider = process.env.AI_PROVIDER || 'openai';
  }

  async analyzePattern(events, patternType) {
    const prompt = this.buildPrompt(events, patternType);
    
    if (this.activeProvider === 'openai') {
      const response = await this.providers.openai.chat.completions.create({
        model: 'gpt-4-turbo-preview',
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.3,
        response_format: { type: 'json_object' }
      });
      return JSON.parse(response.choices[0].message.content);
    } else if (this.activeProvider === 'anthropic') {
      const response = await this.providers.anthropic.messages.create({
        model: 'claude-3-opus-20240229',
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 2000
      });
      return JSON.parse(response.content);
    }
  }

  buildPrompt(events, patternType) {
    const templates = {
      synchronicity: `Analyze these historical events for synchronicities:
        ${JSON.stringify(events)}
        
        Find improbable simultaneous occurrences across disconnected civilizations.
        Return JSON: { synchronicities: [...], confidence: 0-1, explanation: "..." }`,
      
      network: `Identify hidden trade/cultural networks from these events:
        ${JSON.stringify(events)}
        
        Trace material culture spread and connection patterns.
        Return JSON: { networks: [...], routes: [...], confidence: 0-1 }`,
      
      collapse: `Analyze collapse indicators in these events:
        ${JSON.stringify(events)}
        
        Identify patterns that preceded civilizational decline.
        Return JSON: { indicators: [...], riskLevel: 0-1, pattern: "..." }`
    };
    
    return templates[patternType];
  }
}
```

Add to `.env`:
```env
# AI Provider Configuration
AI_PROVIDER=openai  # or anthropic
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Wikidata Configuration  
WIKIDATA_ENDPOINT=https://query.wikidata.org/sparql
WIKIDATA_CACHE_TTL=3600

# Atlas Framework
ATLAS_PERSISTENCE_PATH=./data/atlas
```

---

### 4. 🔍 **IMPLEMENT REAL DISCOVERY ALGORITHMS**

Replace the mock discovery endpoint with REAL pattern detection:

Create `src/backend/services/DiscoveryEngine.js`:
```javascript
class DiscoveryEngine {
  constructor(atlasProcessor, aiAnalyzer) {
    this.atlas = atlasProcessor;
    this.ai = aiAnalyzer;
  }

  async discoverSynchronicities(events, params) {
    // 1. Group events by time window
    const timeGroups = this.groupByTimeWindow(events, params.timeWindowRadius);
    
    // 2. Find multi-region patterns using Atlas SSP
    const patterns = await Promise.all(
      timeGroups.map(group => this.atlas.ssp.findSimilarities(group))
    );
    
    // 3. Use AI to analyze synchronicity probability
    const synchronicities = [];
    for (const pattern of patterns) {
      if (pattern.regions.length >= 3) {
        const analysis = await this.ai.analyzePattern(pattern.events, 'synchronicity');
        if (analysis.confidence > params.synchronicityThreshold) {
          synchronicities.push({
            ...analysis,
            year: pattern.year,
            events: pattern.events
          });
        }
      }
    }
    
    // 4. Store discoveries in Atlas MPU
    for (const sync of synchronicities) {
      await this.atlas.mpu.store(`sync_${Date.now()}`, sync);
    }
    
    return synchronicities;
  }

  async reconstructNetworks(events, params) {
    // 1. Use Atlas HASR to find material culture patterns
    const materialPatterns = await this.atlas.hasr.findPatterns(
      events.map(e => e.symbolic)
    );
    
    // 2. Calculate diffusion velocities
    const networks = [];
    for (const pattern of materialPatterns) {
      const velocity = this.calculateDiffusionVelocity(pattern);
      if (velocity > 0) {
        // 3. Use AI to assess network probability
        const analysis = await this.ai.analyzePattern(pattern.events, 'network');
        if (analysis.confidence > params.networkDensity) {
          networks.push({
            ...analysis,
            velocity,
            route: this.tracePath(pattern.events)
          });
        }
      }
    }
    
    return networks;
  }

  async detectCollapsePatterns(events, params) {
    // 1. Use Ghost Loops to find recurring patterns
    const recurringPatterns = await this.atlas.ghostLoops.findRecurring(events);
    
    // 2. Identify collapse indicators
    const collapseIndicators = [
      'elite_overproduction',
      'resource_depletion', 
      'climate_stress',
      'social_unrest',
      'military_defeat'
    ];
    
    // 3. Score each civilization for collapse risk
    const collapseRisks = [];
    for (const civ of this.extractCivilizations(events)) {
      const indicators = this.findIndicators(civ.events, collapseIndicators);
      if (indicators.length >= 3) {
        const analysis = await this.ai.analyzePattern(civ.events, 'collapse');
        collapseRisks.push({
          civilization: civ.name,
          period: civ.period,
          indicators,
          risk: analysis.riskLevel,
          pattern: analysis.pattern
        });
      }
    }
    
    return collapseRisks;
  }
}
```

---

### 5. 🌐 **FIX THE GLOBE VISUALIZATION**

The globe isn't rendering properly. Here's the fix:

**Option 1: Use Globe.gl properly**
```bash
npm install globe.gl
```

Update `src/frontend/index.html`:
```javascript
// After globe container is ready
useEffect(() => {
  if (!globeRef.current) return;
  
  // Initialize Globe.gl with proper configuration
  const globe = Globe()
    (document.getElementById('globe'))
    .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
    .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
    .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
    .pointsData(discoveries)
    .pointAltitude(d => d.confidence * 0.5)
    .pointRadius(0.5)
    .pointColor(d => {
      const colors = {
        synchronicity: '#00ff88',
        network: '#00aaff',
        collapse: '#ff0088'
      };
      return colors[d.type] || '#ffffff';
    })
    .pointLabel(d => `
      <div style="background: rgba(0,0,0,0.8); padding: 5px; border-radius: 3px;">
        <div style="color: #00ff88; font-size: 12px;">${d.title}</div>
        <div style="color: #aaa; font-size: 10px;">Confidence: ${(d.confidence * 100).toFixed(0)}%</div>
      </div>
    `);
    
  // Add arc connections between related discoveries
  const arcs = generateConnectionArcs(discoveries);
  globe.arcsData(arcs)
    .arcColor('color')
    .arcDashLength(0.5)
    .arcDashGap(0.1)
    .arcDashAnimateTime(2000);
    
  window.chronoGlobe = globe;
}, []);
```

**Option 2: Use Cesium for professional quality**
```bash
npm install cesium
```

---

### 6. 📦 **AKASHA INTEGRATION**

Connect to AKASHA for knowledge preservation:

Create `src/backend/services/AkashaConnector.js`:
```javascript
class AkashaConnector {
  constructor() {
    this.endpoint = process.env.AKASHA_ENDPOINT || 'http://localhost:8001';
  }

  async preserveDiscovery(discovery) {
    const response = await fetch(`${this.endpoint}/api/preserve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type: 'historical_discovery',
        content: discovery,
        dimensions: 7,
        preservation_level: 'century',
        verification_required: true
      })
    });
    
    return response.json();
  }

  async enrichWithDimensions(event) {
    const response = await fetch(`${this.endpoint}/api/enrich`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: event,
        target_dimensions: 7,
        enrichment_type: 'historical'
      })
    });
    
    return response.json();
  }
}
```

---

## 📊 TESTING REQUIREMENTS

### Unit Tests
Create `tests/` directory with:
- `WikidataService.test.js` - Test SPARQL queries
- `AtlasIntegration.test.js` - Test Atlas components
- `DiscoveryEngine.test.js` - Test pattern detection
- `AIAnalyzer.test.js` - Test AI integration

### Integration Tests
- Full discovery pipeline with real data
- WebSocket real-time updates
- Globe visualization with actual discoveries

### Performance Requirements
- Process 10,000 events in < 30 seconds
- Discovery analysis < 5 seconds
- Globe render with 1000 points smoothly
- WebSocket latency < 100ms

---

## 🚀 DEPLOYMENT CHECKLIST

### Environment Variables Required
```env
# Required
NODE_ENV=production
AI_PROVIDER=openai
OPENAI_API_KEY=sk-...
WIKIDATA_ENDPOINT=https://query.wikidata.org/sparql

# Optional but recommended
ANTHROPIC_API_KEY=sk-ant-...
AKASHA_ENDPOINT=http://localhost:8001
REDIS_URL=redis://localhost:6379
ATLAS_PERSISTENCE_PATH=./data/atlas
```

### Docker Setup
Create `Dockerfile`:
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 8000 8001
CMD ["npm", "start"]
```

### Production Configuration
- Use PM2 for process management
- Implement rate limiting for API endpoints
- Add authentication for sensitive endpoints
- Set up monitoring (Prometheus/Grafana)
- Configure CORS properly
- Implement request logging

---

## ⚠️ CRITICAL NOTES

1. **This is NOT a demo** - Build for production
2. **Use REAL data** - No more mocks
3. **Atlas Framework is required** - Not optional
4. **AI integration is required** - For intelligent analysis
5. **The globe MUST work** - It's the centerpiece
6. **Performance matters** - This proves LucianOS scales

---

## 📝 ACCEPTANCE CRITERIA

The implementation is complete when:

- [ ] Wikidata returns REAL historical events (verified by checking actual dates/locations)
- [ ] Atlas Framework components are properly imported and functioning
- [ ] AI provider analyzes patterns and returns insights
- [ ] Discovery algorithms find at least 5 real synchronicities
- [ ] Globe displays actual discovery locations with animations
- [ ] WebSocket streams real-time updates
- [ ] AKASHA preserves discoveries for long-term storage
- [ ] All tests pass with >80% coverage
- [ ] Documentation is updated with real examples
- [ ] Performance meets requirements

---

## 🎯 SUCCESS METRICS

- **Data Quality**: Real events with Wikipedia/Wikidata sources
- **Discovery Quality**: Patterns that could be published academically
- **Performance**: <5s for complete discovery analysis
- **Reliability**: 99% uptime with proper error handling
- **User Experience**: Smooth globe interaction with 60fps

---

## 💪 YOU'VE GOT THIS!

This is the FIRST real implementation of Atlas Framework. Make it spectacular. The goal is to prove that LucianOS components can work with real data at scale to make actual discoveries.

No shortcuts. No mocks. Real implementation only.

**Contact**: If you need clarification, check the existing code first - the patterns are there, they just need to be made real.

Good luck! 🚀
