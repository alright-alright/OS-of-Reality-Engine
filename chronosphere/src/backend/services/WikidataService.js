// WikidataService.js - Real SPARQL queries to Wikidata for historical events
const fetch = require('node-fetch');

class WikidataService {
    constructor() {
        this.endpoint = process.env.WIKIDATA_ENDPOINT || 'https://query.wikidata.org/sparql';
        this.cache = new Map();
        this.cacheTimeout = parseInt(process.env.WIKIDATA_CACHE_TTL || '3600') * 1000;
    }

    async queryHistoricalEvents({ startYear, endYear, region, eventTypes = [], limit = 1000 }) {
        const cacheKey = `${startYear}_${endYear}_${region}_${eventTypes.join('_')}`;
        
        // Check cache first
        if (this.cache.has(cacheKey)) {
            const cached = this.cache.get(cacheKey);
            if (Date.now() - cached.timestamp < this.cacheTimeout) {
                console.log('📦 Returning cached Wikidata results');
                return cached.data;
            }
        }

        // Build SPARQL query
        const sparql = this.buildHistoricalQuery(startYear, endYear, region, eventTypes, limit);
        
        try {
            console.log(`🔍 Querying Wikidata for events ${startYear} to ${endYear}`);
            const response = await fetch(this.endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/sparql-query',
                    'Accept': 'application/json',
                    'User-Agent': 'ChronoSphere/1.0 (https://github.com/aerware/chronosphere)'
                },
                body: sparql
            });

            if (!response.ok) {
                throw new Error(`Wikidata query failed: ${response.status}`);
            }

            const data = await response.json();
            const events = this.processWikidataResults(data);
            
            // Cache the results
            this.cache.set(cacheKey, {
                data: events,
                timestamp: Date.now()
            });

            console.log(`✅ Found ${events.length} historical events`);
            return events;
        } catch (error) {
            console.error('❌ Wikidata query error:', error);
            // Return fallback data on error
            return this.getFallbackEvents(startYear, endYear);
        }
    }

    buildHistoricalQuery(startYear, endYear, region, eventTypes, limit) {
        // Convert years to date format
        const startDate = `${Math.abs(startYear)}-01-01`;
        const endDate = `${Math.abs(endYear)}-12-31`;
        const startPrefix = startYear < 0 ? '-' : '';
        const endPrefix = endYear < 0 ? '-' : '';

        // Default event types if none specified
        const types = eventTypes.length > 0 ? eventTypes : [
            'Q178561', // battle
            'Q198',    // war
            'Q3839081', // disaster
            'Q1190554', // occurrence
            'Q13418847', // historical event
            'Q2334719', // historical period
            'Q1656682', // event
            'Q3241121', // archaeological culture
            'Q839954'   // archaeological site
        ];

        const typeFilter = types.map(t => `?event wdt:P31/wdt:P279* wd:${t}.`).join(' ');
        
        return `
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?event ?eventLabel ?date ?coords ?description ?typeLabel ?countryLabel ?image WHERE {
  { ${typeFilter} }
  
  # Date constraints
  ?event wdt:P585 ?date.
  FILTER(?date >= "${startPrefix}${startDate}"^^xsd:dateTime && ?date <= "${endPrefix}${endDate}"^^xsd:dateTime)
  
  # Optional properties
  OPTIONAL { ?event wdt:P625 ?coords. }
  OPTIONAL { ?event schema:description ?description FILTER (lang(?description) = "en") }
  OPTIONAL { ?event wdt:P31 ?type. }
  OPTIONAL { ?event wdt:P17 ?country. }
  OPTIONAL { ?event wdt:P18 ?image. }
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?date
LIMIT ${limit}`;
    }

    processWikidataResults(data) {
        if (!data.results || !data.results.bindings) {
            return [];
        }

        return data.results.bindings.map((binding, index) => {
            // Extract coordinates if available
            let coordinates = null;
            if (binding.coords && binding.coords.value) {
                const coordMatch = binding.coords.value.match(/Point\(([-\d.]+) ([-\d.]+)\)/);
                if (coordMatch) {
                    coordinates = {
                        lng: parseFloat(coordMatch[1]),
                        lat: parseFloat(coordMatch[2])
                    };
                }
            }

            // Parse date
            const date = binding.date ? new Date(binding.date.value) : null;
            const year = date ? date.getFullYear() : 0;

            // Determine event type
            const eventType = this.classifyEventType(
                binding.typeLabel?.value || '',
                binding.eventLabel?.value || '',
                binding.description?.value || ''
            );

            return {
                id: binding.event.value.split('/').pop(),
                wikidataUrl: binding.event.value,
                title: binding.eventLabel?.value || 'Unknown Event',
                date: binding.date?.value || '',
                year: year,
                coordinates: coordinates,
                type: eventType,
                category: binding.typeLabel?.value || 'historical event',
                country: binding.countryLabel?.value || '',
                description: binding.description?.value || '',
                image: binding.image?.value || null,
                confidence: 0.7 + Math.random() * 0.3, // Will be replaced with AI confidence
                source: 'wikidata'
            };
        }).filter(event => event.title !== 'Unknown Event');
    }

    classifyEventType(typeLabel, title, description) {
        const text = `${typeLabel} ${title} ${description}`.toLowerCase();
        
        if (text.includes('battle') || text.includes('war') || text.includes('conflict')) {
            return 'conflict';
        } else if (text.includes('collapse') || text.includes('fall') || text.includes('decline')) {
            return 'collapse';
        } else if (text.includes('trade') || text.includes('route') || text.includes('merchant')) {
            return 'trade';
        } else if (text.includes('migration') || text.includes('movement') || text.includes('exodus')) {
            return 'migration';
        } else if (text.includes('discovery') || text.includes('invention') || text.includes('innovation')) {
            return 'discovery';
        } else if (text.includes('philosophy') || text.includes('religion') || text.includes('spiritual')) {
            return 'cultural';
        } else if (text.includes('disaster') || text.includes('earthquake') || text.includes('volcano')) {
            return 'disaster';
        }
        
        return 'event';
    }

    async queryRelatedEvents(eventId, radius = 100) {
        const sparql = `
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>

SELECT ?related ?relatedLabel ?date ?coords WHERE {
  wd:${eventId} wdt:P585 ?mainDate.
  ?related wdt:P585 ?date.
  
  # Find events within time radius (in years)
  FILTER(ABS(YEAR(?date) - YEAR(?mainDate)) <= ${radius})
  FILTER(?related != wd:${eventId})
  
  OPTIONAL { ?related wdt:P625 ?coords. }
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
LIMIT 100`;

        try {
            const response = await fetch(this.endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/sparql-query',
                    'Accept': 'application/json'
                },
                body: sparql
            });

            const data = await response.json();
            return this.processWikidataResults(data);
        } catch (error) {
            console.error('Failed to query related events:', error);
            return [];
        }
    }

    async queryByCoordinates(lat, lng, radiusKm = 500, startYear, endYear) {
        const sparql = `
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>

SELECT ?event ?eventLabel ?date ?coords ?distance WHERE {
  ?event wdt:P31/wdt:P279* wd:Q1190554.
  ?event wdt:P625 ?coords.
  ?event wdt:P585 ?date.
  
  BIND(geof:distance(?coords, "Point(${lng} ${lat})"^^geo:wktLiteral) as ?distance)
  FILTER(?distance <= ${radiusKm})
  FILTER(?date >= "${startYear}-01-01"^^xsd:dateTime && ?date <= "${endYear}-12-31"^^xsd:dateTime)
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?distance
LIMIT 500`;

        try {
            const response = await fetch(this.endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/sparql-query',
                    'Accept': 'application/json'
                },
                body: sparql
            });

            const data = await response.json();
            return this.processWikidataResults(data);
        } catch (error) {
            console.error('Failed to query by coordinates:', error);
            return [];
        }
    }

    getFallbackEvents(startYear, endYear) {
        // Fallback data for when Wikidata is unavailable
        console.log('⚠️ Using fallback historical data');
        return [
            {
                id: 'Q47064',
                title: 'Bronze Age collapse',
                date: `${startYear}-01-01`,
                year: startYear,
                coordinates: { lat: 35.0, lng: 33.0 },
                type: 'collapse',
                category: 'historical period',
                description: 'The Late Bronze Age collapse was a time of widespread societal collapse',
                confidence: 0.89,
                source: 'fallback'
            },
            {
                id: 'Q208823',
                title: 'Sea Peoples',
                date: `${startYear + 50}-01-01`,
                year: startYear + 50,
                coordinates: { lat: 31.0, lng: 35.0 },
                type: 'migration',
                category: 'historical event',
                description: 'The Sea Peoples were a confederacy of naval raiders',
                confidence: 0.73,
                source: 'fallback'
            },
            {
                id: 'Q180299',
                title: 'Trojan War',
                date: `${startYear + 20}-01-01`,
                year: startYear + 20,
                coordinates: { lat: 39.95, lng: 26.24 },
                type: 'conflict',
                category: 'war',
                description: 'Legendary war between the Greeks and Troy',
                confidence: 0.65,
                source: 'fallback'
            }
        ];
    }

    async getEventDetails(eventId) {
        const sparql = `
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX schema: <http://schema.org/>

SELECT ?event ?eventLabel ?date ?coords ?description ?image 
       ?participantLabel ?locationLabel ?causeLabel ?effectLabel WHERE {
  BIND(wd:${eventId} as ?event)
  
  OPTIONAL { ?event wdt:P585 ?date. }
  OPTIONAL { ?event wdt:P625 ?coords. }
  OPTIONAL { ?event schema:description ?description FILTER (lang(?description) = "en") }
  OPTIONAL { ?event wdt:P18 ?image. }
  OPTIONAL { ?event wdt:P710 ?participant. }
  OPTIONAL { ?event wdt:P276 ?location. }
  OPTIONAL { ?event wdt:P828 ?cause. }
  OPTIONAL { ?event wdt:P1542 ?effect. }
  
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}`;

        try {
            const response = await fetch(this.endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/sparql-query',
                    'Accept': 'application/json'
                },
                body: sparql
            });

            const data = await response.json();
            if (data.results && data.results.bindings.length > 0) {
                return this.processWikidataResults(data)[0];
            }
        } catch (error) {
            console.error('Failed to get event details:', error);
        }
        return null;
    }
}

module.exports = WikidataService;