# AI Town Technical Documentation

## Table of Contents
1. [Quick Start Guide](#quick-start-guide)
2. [System Architecture](#system-architecture) 
3. [Core Systems](#core-systems)
4. [Development Guide](#development-guide)
5. [Technical Reference](#technical-reference)
6. [Operations Guide](#operations-guide)

## Quick Start Guide

### Prerequisites
- Node.js 18+
- Local LLM (Ollama) or OpenAI API key
- Basic understanding of React and TypeScript

### Installation
```bash
git clone https://github.com/a16z-infra/ai-town.git
cd ai-town
npm install

# Set up local Convex backend
./convex-local-backend

# Configure LLM
just convex env set OLLAMA_HOST http://localhost:11434
# or
just convex env set OPENAI_API_KEY your_key_here
```

### Project Structure
```
ai-town/
├── convex/                 # Backend code
│   ├── aiTown/            # Core game logic
│   ├── agent/             # AI agent system
│   ├── engine/            # Game engine
│   └── util/              # Utilities
├── src/                   # Frontend code
│   ├── components/        # React components
│   ├── hooks/            # Custom React hooks
│   └── lib/              # Frontend utilities
├── data/                 # Game data and assets
└── docs/                 # Additional documentation
```

## System Architecture

### High-Level Overview
AI Town is a virtual environment where AI characters live, chat, and socialize. The system is built on multiple layers:

1. **Frontend Layer**: React + PixiJS for rendering
2. **Backend Layer**: Convex for game engine and database
3. **AI Layer**: OpenAI/Ollama for character intelligence
4. **State Layer**: Real-time synchronization using Convex

### Core Components
1. **Game Engine** (`convex/aiTown/`)
   - Manages game state
   - Processes inputs
   - Handles real-time updates

2. **Agent System** (`convex/agent/`)
   - Controls AI behavior
   - Manages conversations
   - Handles memory and learning

3. **State Management**
   - World state
   - Player data
   - Conversation tracking
   - Historical data

## Core Systems

### Game Engine
1. **Engine Design**
   ```typescript
   export class Game extends AbstractGame {
     tickDuration = 16;
     stepDuration = 1000;
     maxTicksPerStep = 600;
     maxInputsPerStep = 32;

     world: World;
     historicalLocations: Map<GameId<'players'>, HistoricalObject<Location>>;
     // ... additional properties
   }
   ```

2. **State Processing**
   - 60 ticks/second
   - State updates batched into steps
   - Input processing queue
   - Historical state tracking

### Agent System

1. **Decision Making Flow**
   ```mermaid
   graph TD
      A[Agent Tick] --> B[Check State]
      B --> C[Process Inputs]
      C --> D[Make Decisions]
      D --> E[Execute Actions]
      E --> A
   ```

2. **Memory System**
   ```typescript
   export const memoryFields = {
     playerId,
     description: v.string(),
     embeddingId: v.id('memoryEmbeddings'),
     importance: v.number(),
     lastAccess: v.number(),
     data: v.union(
       // Relationship memories - how an agent feels about another
       v.object({
         type: v.literal('relationship'),
         playerId,
       }),
       // Conversation memories - record of interactions
       v.object({
         type: v.literal('conversation'),
         conversationId,
         playerIds: v.array(playerId),
       }),
       // Reflection memories - insights from multiple memories
       v.object({
         type: v.literal('reflection'),
         relatedMemoryIds: v.array(v.id('memories')),
       })
     ),
   };
   ```

3. **LLM Integration**
   - Conversation generation
   - Memory processing
   - Decision making
   - Personality development

### Graphics System

#### Understanding the Basics
The game world is built like a digital board game, using a grid system where each square (or "tile") represents a small piece of the world. Think of it like building with Lego blocks, where each block is either:
- Part of the background (like grass, paths, or water)
- An object (like trees, buildings, or furniture)
- A character (players and NPCs)

#### How the Map Works
1. **Layers**
   - The map is built in layers, like stacking transparent sheets:
     * Bottom layer: Background tiles (ground, paths)
     * Middle layer: Objects (trees, buildings)
     * Top layer: Characters and moving elements
   - Each layer can be updated independently, making it efficient to change just what's needed

2. **Tile System**
   - The world is divided into a grid of square tiles
   - Each tile is typically 16x16 or 32x32 pixels
   - Tiles are referenced by coordinates (x,y) from the top-left corner
   - The game uses a "tileset" - a single image containing all possible tiles
   - Each tile in the game is just a reference to a position in this tileset

3. **Collision System**
   - Some tiles block movement (like walls)
   - Others allow movement (like paths)
   - This is tracked in the `objmap` layer
   - Characters use this information for pathfinding

#### Customizing the World

1. **Changing the Map**
   - Use the Tiled editor (like Photoshop for game maps)
   - Design your map by placing tiles from the tileset
   - Export two main layers:
     * `bgtiles`: The visual background
     * `objmap`: What players can/can't walk through
   - Convert the map for the game using the provided tool

2. **Adding New Areas**
   - Create new sections in Tiled
   - Add collision information for new objects
   - Update the map configuration in the game
   - Consider pathfinding implications

3. **Changing Appearances**
   - Modify or replace the tileset image
   - Update character sprite sheets
   - Add new animated elements
   - Create new object types

#### How Rendering Works

1. **Display Process**
   - The game uses PixiJS (a 2D rendering engine)
   - Only visible portions of the map are rendered
   - Characters are animated sprites
   - The viewport follows the player smoothly

2. **Animation System**
   - Characters have different animation states (walking, idle)
   - Each state uses specific frames from their sprite sheet
   - Animations are played based on character actions
   - Movement is smoothly interpolated between positions

### Memory System Deep Dive

1. **Memory Types**
   The memory system uses three distinct types of memories to create a rich internal world for each agent:
   - Relationship memories track how an agent feels about other agents
   - Conversation memories store records of interactions and their context
   - Reflection memories represent higher-level insights derived from multiple memories
   
   Each memory has an importance score, embedding for similarity search, and last access time for recency tracking:
   ```typescript
   export const memoryFields = {
     playerId,
     description: v.string(),
     embeddingId: v.id('memoryEmbeddings'),
     importance: v.number(),
     lastAccess: v.number(),
     data: v.union(
       // Relationship memories - how an agent feels about another
       v.object({
         type: v.literal('relationship'),
         playerId,
       }),
       // Conversation memories - record of interactions
       v.object({
         type: v.literal('conversation'),
         conversationId,
         playerIds: v.array(playerId),
       }),
       // Reflection memories - insights from multiple memories
       v.object({
         type: v.literal('reflection'),
         relatedMemoryIds: v.array(v.id('memories')),
       })
     ),
   };
   ```

2. **Memory Formation Process**
   When an event occurs (like a conversation ending), the system creates a new memory through several steps:
   - First, loads the full context of the event
   - Uses LLM to generate a natural language summary
   - Calculates importance score for the memory
   - Creates an embedding for future similarity searching
   - Saves the memory to the database
   - Potentially triggers reflection if enough important memories have accumulated
   
   ```typescript
   // After a conversation ends
   export async function rememberConversation(
     ctx: ActionCtx,
     worldId: Id<'worlds'>,
     agentId: GameId<'agents'>,
     playerId: GameId<'players'>,
     conversationId: GameId<'conversations'>,
   ) {
     // 1. Load conversation data
     const data = await ctx.runQuery(selfInternal.loadConversation, {...});
     
     // 2. Generate memory summary using LLM
     const { content } = await chatCompletion({
       messages: [
         {
           role: 'user',
           content: `You are ${player.name}, summarize the conversation...`,
         }
       ],
     });

     // 3. Calculate importance
     const importance = await calculateImportance(description);

     // 4. Create embedding for future similarity search
     const { embedding } = await fetchEmbedding(description);

     // 5. Save memory
     await ctx.runMutation(selfInternal.insertMemory, {...});

     // 6. Trigger reflection if needed
     await reflectOnMemories(ctx, worldId, playerId);
   }
   ```

3. **Memory Retrieval**
   When an agent needs to recall relevant information, the system:
   - Uses vector similarity search to find potentially relevant memories
   - Ranks memories using a combination of:
     * Relevance (vector similarity)
     * Recency (how recently accessed)
     * Importance (scored during formation)
   - Returns the top N most relevant memories
   
   ```typescript
   export async function searchMemories(
     ctx: ActionCtx,
     playerId: GameId<'players'>,
     searchEmbedding: number[],
     n: number = 3,
   ) {
     // 1. Find relevant memories using vector search
     const candidates = await ctx.vectorSearch('memoryEmbeddings', 'embedding', {
       vector: searchEmbedding,
       filter: (q) => q.eq('playerId', playerId),
       limit: n * MEMORY_OVERFETCH,
     });

     // 2. Rank memories by relevance, recency, and importance
     const rankedMemories = await ctx.runMutation(
       selfInternal.rankAndTouchMemories,
       { candidates, n }
     );

     return rankedMemories;
   }
   ```

4. **Memory Usage in Conversations**
   Before and during conversations, the system:
   - Creates an embedding of the current context
   - Searches for relevant memories about the other participant
   - Includes these memories in the LLM prompt
   - This allows agents to maintain consistent relationships and reference past interactions
   
   ```typescript
   // When starting a conversation
   const embedding = await embeddingsCache.fetch(
     ctx,
     `What you think about ${otherPlayer.name}?`
   );

   const memories = await memory.searchMemories(
     ctx,
     player.id as GameId<'players'>,
     embedding,
     NUM_MEMORIES_TO_SEARCH,
   );

   // Include memories in conversation prompt
   const prompt = [
     `You are ${player.name}, and you're talking to ${otherPlayer.name}.`,
     `Here are relevant memories:`,
     ...memories.map(m => ` - ${m.description}`),
   ];
   ```

5. **Reflection System**
   The reflection system helps agents develop deeper understanding by:
   - Periodically reviewing recent memories
   - Calculating total importance to determine if reflection is needed
   - Using LLM to generate insights from patterns in memories
   - Creating new, higher-level reflection memories
   - This creates a form of "learning" from experience
   
   ```typescript
   async function reflectOnMemories(
     ctx: ActionCtx,
     worldId: Id<'worlds'>,
     playerId: GameId<'players'>,
   ) {
     // 1. Get recent memories
     const { memories, lastReflectionTs } = await ctx.runQuery(
       internal.agent.memory.getReflectionMemories,
       { worldId, playerId, numberOfItems: 100 }
     );

     // 2. Check if reflection is needed
     const sumOfImportanceScore = memories
       .filter((m) => m._creationTime > (lastReflectionTs ?? 0))
       .reduce((acc, curr) => acc + curr.importance, 0);

     if (sumOfImportanceScore > 500) {
       // 3. Generate insights using LLM
       const insights = await generateInsights(memories);
       
       // 4. Save new reflection memories
       await saveReflections(insights);
     }
   }
   ```

6. **Memory Importance Calculation**
   Each memory's importance is calculated by:
   - Sending the memory description to the LLM
   - Asking for a 0-9 rating based on emotional/practical significance
   - Using this score to prioritize memories in retrieval
   - Influencing which memories trigger reflection
   
   ```typescript
   async function calculateImportance(description: string) {
     const { content: importanceRaw } = await chatCompletion({
       messages: [
         {
           role: 'user',
           content: `On the scale of 0 to 9, where 0 is mundane and 9 is extremely poignant, 
                    rate the following memory: ${description}
                    Answer with number only.`,
         },
       ],
       temperature: 0.0,
       max_tokens: 1,
     });
     return parseFloat(importanceRaw);
   }
   ```

7. **Memory Optimization**
   The system uses several strategies to maintain performance:
   - Caches embeddings to avoid recomputing for identical text
   - Uses a scoring system combining multiple factors for memory ranking
   - Throttles memory access updates to reduce database writes
   - Employs vector search for efficient similarity matching
   - Overfetches candidates to improve ranking quality

8. **Memory Integration Flow**
   The complete memory lifecycle follows this pattern:
   1. Event triggers memory formation
   2. System generates description and importance
   3. Creates and caches embedding
   4. Stores memory in database
   5. Checks if reflection is needed
   6. Potentially generates new insights
   7. Updates memory access patterns
   
   ```mermaid
   graph TD
     A[Event Occurs] --> B[Generate Description]
     B --> C[Calculate Importance]
     C --> D[Create Embedding]
     D --> E[Store Memory]
     E --> F[Check Reflection Threshold]
     F -->|Threshold Met| G[Generate Insights]
     G --> H[Store Reflection]
     F -->|Threshold Not Met| I[End]
     H --> I
   ```

## Development Guide

### Common Tasks

1. **Adding New Agent Behaviors**
   ```typescript
   // data/characters.ts
   export const Descriptions = [
     {
       name: "Curious Scholar",
       character: "f1",
       identity: "I am a curious scholar who loves learning about others. " +
                "I'm particularly interested in people's life experiences and " +
                "always try to ask thoughtful questions. I'm gentle and patient " +
                "in conversations, but can get very excited about new discoveries.",
       plan: "I want to learn something new from everyone I meet, " +
             "build meaningful relationships, and share knowledge with others.",
     },
     {
       name: "Town Gossip",
       character: "f2",
       identity: "I'm the town gossip who knows everything about everyone. " +
                "I love sharing stories and connecting people. I'm friendly " +
                "but sometimes can't help spreading interesting rumors.",
       plan: "I want to collect interesting stories about townspeople " +
             "and help connect people who might benefit from knowing each other.",
     }
   ];
   ```

2. **Implementing New Activities**
   ```typescript
   // convex/constants.ts
   export const ACTIVITIES = [
     {
       description: "Reading a fascinating book about local history",
       emoji: "📚",
       duration: 15000, // 15 seconds
     },
     {
       description: "Tending to the community garden",
       emoji: "🌱",
       duration: 10000,
       // Activities can influence conversation topics
       topics: ["gardening", "nature", "community"],
     },
     {
       description: "Writing in personal journal",
       emoji: "📝",
       duration: 8000,
       // Can trigger reflection memories
       reflectionChance: 0.3,
     }
   ];

   // Example activity integration in agent behavior
   if (shouldDoActivity) {
     const activity = selectActivity(agent.identity, currentTime);
     this.startOperation(game, now, 'startActivity', {
       activity,
       location: findAppropriateLocation(game, activity),
     });
   }
   ```

3. **Extending Memory System**
   ```typescript
   // Add new memory types
   export const memoryFields = {
     // ... existing fields
     data: v.union(
       // ... existing types
       // New memory type for activities
       v.object({
         type: v.literal('activity'),
         description: v.string(),
         location: point,
         duration: v.number(),
         outcome: v.string(),
         mood: v.string(),
       }),
       // New memory type for observations
       v.object({
         type: v.literal('observation'),
         targetPlayerId: playerId,
         activity: v.string(),
         insight: v.string(),
       }),
       // New memory type for locations
       v.object({
         type: v.literal('location'),
         location: point,
         description: v.string(),
         significance: v.string(),
       })
     ),
   };

   // Example memory formation for activity
   async function rememberActivity(
     ctx: ActionCtx,
     agent: Agent,
     activity: Activity,
   ) {
     const description = await generateActivityMemory(
       agent.identity,
       activity,
     );
     
     const importance = await calculateImportance(description);
     
     await ctx.runMutation(selfInternal.insertMemory, {
       playerId: agent.id,
       description,
       importance,
       data: {
         type: 'activity',
         description: activity.description,
         location: activity.location,
         duration: activity.duration,
         outcome: await generateOutcome(agent, activity),
         mood: await determineMood(agent, activity),
       }
     });
   }

   // Example memory usage in conversation
   const locationMemories = await searchMemories(ctx, {
     playerId: agent.id,
     type: 'location',
     near: currentLocation,
   });

   const prompt = [
     `You are ${agent.name}, currently at ${currentLocation}.`,
     `You remember these things about this place:`,
     ...locationMemories.map(m => ` - ${m.description}`),
     `How does being here influence your conversation?`
   ];
   ```

4. **Custom Behavior Integration**
   ```typescript
   // Example of combining activities, memories, and behavior
   export const agentBehavior = {
     // Personality-driven activity selection
     selectActivity(agent: Agent, time: number) {
       const preferences = parseIdentity(agent.identity);
       const availableActivities = ACTIVITIES.filter(a => 
         matchesPreferences(a, preferences)
       );
       return weightedRandomChoice(availableActivities);
     },

     // Memory-influenced conversation
     async generateResponse(
       agent: Agent,
       context: ConversationContext,
       memories: Memory[],
     ) {
       const relevantActivities = memories
         .filter(m => m.data.type === 'activity')
         .slice(0, 3);
       
       const conversationStyle = determineStyle(agent.identity);
       
       return await generateResponse(
         agent,
         context,
         memories,
         relevantActivities,
         conversationStyle,
       );
     },

     // Location-aware behavior
     async decideNextAction(
       agent: Agent,
       location: Point,
       memories: Memory[],
     ) {
       const locationMemory = memories.find(m => 
         m.data.type === 'location' && 
         distance(m.data.location, location) < NEARBY_THRESHOLD
       );

       if (locationMemory) {
         return await generateLocationBasedAction(
           agent,
           locationMemory,
         );
       }
       
       return defaultBehavior(agent);
     }
   };
   ```

### Customizing Town Graphics and Map

1. **Map Creation Process**
   - Use [Tiled](https://www.mapeditor.org/) to create/edit maps
   - Export as JSON with two layers:
     - `bgtiles`: Background terrain
     - `objmap`: Collision objects/interactive elements
   - Convert using the provided tool:
   ```bash
   node data/convertMap.js <mapDataPath> <assetPath> <tilesetpxw> <tilesetpxh>
   ```

2. **Map Configuration**
   ```typescript
   // convex/init.ts
   const map = {
     mapwidth: 32,
     mapheight: 32,
     tilesetpath: "/assets/tileset.png",
     tilesetpxw: 256,
     tilesetpxh: 1536,
     tiledim: 16,
     bgtiles: [[...]], // Background tiles
     objmap: [[...]], // Object/collision map
     animatedsprites: [] // Special animated elements
   };
   ```

3. **Tileset Structure**
   - Tilesets should be organized in a grid
   - Each tile is `tileDim x tileDim` pixels
   - Background tiles (`bgtiles`):
     - -1: Empty tile
     - 0+: Index in tileset
   - Object tiles (`objmap`):
     - -1: Walkable space
     - 0+: Collision/interactive object

4. **Adding New Sprites**
   ```typescript
   // data/characters.ts
   export const characters = [
     {
       name: 'f1',
       textureUrl: '/assets/32x32folk.png',
       spritesheetData: {
         frames: {
           // Define animation frames
           "idle_down": { x: 0, y: 0, w: 32, h: 32 },
           "walk_down": { x: 32, y: 0, w: 32, h: 32 },
           // ... more frames
         },
         animations: {
           idle: ["idle_down"],
           walk: ["walk_down"],
           // ... more animations
         }
       },
       speed: 0.1,
     }
   ];
   ```

5. **Map Loading Process**
   ```typescript
   // convex/init.ts
   await ctx.db.insert('maps', {
     worldId,
     width: map.mapwidth,
     height: map.mapheight,
     tileSetUrl: map.tilesetpath,
     tileSetDimX: map.tilesetpxw,
     tileSetDimY: map.tilesetpxh,
     tileDim: map.tiledim,
     bgTiles: map.bgtiles,
     objectTiles: map.objmap,
     animatedSprites: map.animatedsprites,
   });
   ```

6. **Important Considerations**
   - Keep tile dimensions consistent
   - Ensure proper collision mapping
   - Test pathfinding with new layouts
   - Consider performance with large maps
   - Verify asset loading paths

### Best Practices

1. **Code Organization**
   - Separate game logic from AI logic
   - Use type-safe input handlers
   - Maintain clear module boundaries

2. **Performance**
   - Batch database operations
   - Minimize LLM calls
   - Use efficient data structures

3. **Security**
   - Protect API keys
   - Validate user input
   - Rate limit operations

## Technical Reference

### State Management
1. **World State**
   ```typescript
   export class World {
     nextId: number;
     conversations: Map<GameId<'conversations'>, Conversation>;
     players: Map<GameId<'players'>, Player>;
     agents: Map<GameId<'agents'>, Agent>;
     historicalLocations?: Map<GameId<'players'>, ArrayBuffer>;
   }
   ```

2. **Location Tracking**
   ```typescript
   export type Location = {
     x: number;
     y: number;
     dx: number;
     dy: number;
     speed: number;
   };
   ```

### API Documentation
[See detailed API documentation in generated files]

## Operations Guide

### Troubleshooting

1. **Common Issues**
   - LLM integration problems
   - Game engine performance
   - State synchronization issues

2. **Debugging Tools**
   ```typescript
   // Enable debug logging
   export const DEBUG = true;
   
   // Use testing utilities
   just convex run testing:debugCreatePlayers
   ```

3. **Monitoring**
   - Convex Dashboard
   - Browser DevTools
   - LLM provider dashboard

### Performance Optimization

1. **Rendering Optimizations**
   - Texture atlasing
   - Sprite batching
   - Culling of off-screen elements

2. **State Management**
   - Efficient delta updates
   - Minimal state transfer
   - Compressed position history

3. **Memory Management**
   ```typescript
   export const MEMORY_ACCESS_THROTTLE = 300_000; // In ms
   export const MEMORY_OVERFETCH = 10;
   ```
