# Building an Automatic CodeWiki Generator

## Overview
This tool automatically generates comprehensive documentation for a codebase by:
1. Analyzing code structure and relationships
2. Using LLM to generate natural language explanations
3. Creating organized, hierarchical documentation
4. Maintaining code context and examples

## Core Components

### 1. Code Analysis Engine
```typescript
interface CodeAnalyzer {
  // Parse project structure
  analyzeProject(rootDir: string): ProjectStructure;
  
  // Extract key components
  extractDefinitions(): ComponentDefinitions;
  
  // Build dependency graph
  buildDependencyGraph(): DependencyGraph;
  
  // Find entry points
  identifyEntryPoints(): EntryPoints;
}
```

### 2. Documentation Generator
```typescript
interface DocGenerator {
  // Generate high-level overview
  generateSystemOverview(project: ProjectStructure): string;
  
  // Create component documentation
  generateComponentDocs(components: ComponentDefinitions): string;
  
  // Write developer guides
  generateGuides(components: ComponentDefinitions): string;
  
  // Create technical reference
  generateReference(definitions: ComponentDefinitions): string;
}
```

### 3. LLM Integration
```typescript
interface LLMClient {
  // Generate natural language descriptions
  explainComponent(component: Component): string;
  
  // Create how-to guides
  createHowTo(component: Component): string;
  
  // Generate examples
  generateExample(component: Component): string;
  
  // Identify relationships
  explainRelationship(components: Component[]): string;
}
```

## Implementation Steps

### 1. Project Analysis
1. **Documentation Scanner**
   ```typescript
   interface DocumentationScanner {
     // Parse existing documentation
     parseMarkdownFiles(files: string[]): ExistingDocs;
     
     // Extract key concepts
     extractConcepts(docs: ExistingDocs): Concepts;
     
     // Find documentation gaps
     identifyGaps(docs: ExistingDocs, code: CodeAnalysis): Gaps;
   }
   ```

2. **Documentation Integration**
   ```typescript
   interface DocumentationIntegrator {
     // Combine existing docs with code analysis
     mergeDocumentation(
       existingDocs: ExistingDocs,
       codeAnalysis: CodeAnalysis
     ): IntegratedDocs;
     
     // Preserve existing explanations
     preserveExplanations(
       existing: ExistingDocs,
       generated: GeneratedDocs
     ): FinalDocs;
     
     // Update outdated sections
     updateOutdatedSections(
       existing: ExistingDocs,
       codeAnalysis: CodeAnalysis
     ): UpdatedDocs;
   }
   ```

3. **Priority Analysis**
   ```typescript
   interface PriorityAnalyzer {
     // Analyze README importance
     analyzeReadmeStructure(
       readme: string
     ): DocumentationPriorities;
     
     // Extract architecture decisions
     findArchitectureDecisions(
       docs: ExistingDocs
     ): ArchitectureContext;
     
     // Map documentation to code
     mapDocsToCode(
       docs: ExistingDocs,
       code: CodeAnalysis
     ): DocumentationCodeMap;
   }
   ```

4. **File System Scanner**
   - Walk directory tree
   - Identify key files
   - Parse configuration files
   - Build project structure map
   - **Identify documentation files**
   - **Parse markdown content**

5. **Code Parser**
   - Parse source files
   - Extract types and interfaces
   - Identify key functions
   - Map dependencies
   - **Link code to documentation**
   - **Find documented examples**

### 2. Documentation Structure
1. **Standard Sections**
   - Quick Start
   - System Architecture
   - Core Components
   - Development Guide
   - Technical Reference
   - Operations Guide

2. **Component Documentation**
   - Overview
   - Key Interfaces
   - Usage Examples
   - Common Patterns
   - Best Practices

3. **Cross-References**
   - Related Components
   - Dependencies
   - Usage Examples
   - Common Patterns

### 3. LLM Integration

1. **Prompt Engineering**
   ```typescript
   const prompts = {
     systemOverview: `
       Analyze this codebase structure and provide a high-level overview:
       - Main components
       - Key features
       - Architecture patterns
       - Core technologies
       ${codebaseContext}
     `,
     
     componentExplanation: `
       Explain this component in clear, technical terms:
       - Purpose
       - Key functionality
       - Usage patterns
       - Important considerations
       ${componentContext}
     `,
     
     developerGuide: `
       Create a developer guide for this component:
       - Setup steps
       - Common tasks
       - Best practices
       - Troubleshooting
       ${componentContext}
     `
   };
   ```

2. **Context Management**
   - Extract relevant code snippets
   - Maintain type definitions
   - Include usage examples
   - Preserve relationships

3. **Output Processing**
   - Format markdown
   - Add code blocks
   - Create diagrams
   - Generate index

### 4. File Generation

1. **Main Documentation**
   ```typescript
   interface DocumentationBuilder {
     buildTableOfContents(): string;
     buildQuickStart(): string;
     buildArchitecture(): string;
     buildComponents(): string;
     buildGuides(): string;
     buildReference(): string;
   }
   ```

2. **Supporting Files**
   - Component diagrams
   - Code maps
   - Quick reference
   - Index

3. **Navigation**
   - Cross-references
   - Section links
   - Component relationships
   - Search index

## Usage

```bash
# Install
npm install codewiki-generator

# Configure
export OPENAI_API_KEY=your_key_here

# Generate
codewiki-generator generate /path/to/repo
```

## Configuration

```typescript
interface Config {
  // LLM settings
  llm: {
    provider: 'openai' | 'anthropic' | 'local';
    model: string;
    temperature: number;
    maxTokens: number;
  };
  
  // Analysis settings
  analysis: {
    maxDepth: number;
    excludeDirs: string[];
    includePatterns: string[];
    excludePatterns: string[];
  };
  
  // Output settings
  output: {
    format: 'markdown' | 'html' | 'pdf';
    diagrams: boolean;
    codeBlocks: boolean;
    examples: boolean;
  };
  
  // Documentation settings
  documentation: {
    preserveExisting: boolean;
    updateOutdated: boolean;
    generateMissing: boolean;
    includeExamples: boolean;
    markdownFiles: string[];
    docPriority: DocumentationPriority[];
  };
}
```

## Best Practices

1. **Code Analysis**
   - Use AST parsers for accuracy
   - Maintain type information
   - Track dependencies
   - Identify patterns

2. **Documentation Generation**
   - Clear hierarchy
   - Consistent style
   - Practical examples
   - Proper cross-referencing

3. **LLM Integration**
   - Clear prompts
   - Proper context
   - Error handling
   - Rate limiting

## Extension Points

1. **Custom Analyzers**
   - Language-specific parsers
   - Framework detection
   - Pattern recognition
   - Metric collection

2. **Documentation Templates**
   - Custom sections
   - Style guides
   - Output formats
   - Diagram types

3. **LLM Providers**
   - Multiple providers
   - Custom models
   - Local deployment
   - Caching strategies 

### Documentation Sources Priority

1. **Primary Sources**
   ```typescript
   const documentationSources = {
     priority1: [
       'README.md',
       'ARCHITECTURE.md',
       'CONTRIBUTING.md',
       'docs/*.md'
     ],
     priority2: [
       'package.json',
       'tsconfig.json',
       '*/README.md'
     ],
     priority3: [
       '*.md',
       'comments in code'
     ]
   };
   ```

2. **Integration Strategy**
   ```typescript
   interface DocumentationStrategy {
     // Extract from existing docs
     extractFromDocs(files: string[]): ExistingContent;
     
     // Compare with code analysis
     compareWithCode(
       docs: ExistingContent,
       code: CodeAnalysis
     ): Gaps;
     
     // Generate missing sections
     generateMissing(gaps: Gaps): NewContent;
     
     // Preserve existing content
     mergeContent(
       existing: ExistingContent,
       generated: NewContent
     ): FinalContent;
   }
   ```

3. **LLM Integration**
   - **Documentation Context**
     ```typescript
     const documentationContext = {
       // Existing documentation context
       existingDocs: `
         Analyze these existing documentation files:
         ${existingMarkdown}
         
         Consider:
         - Existing explanations
         - Architecture decisions
         - Development patterns
         - Examples
       `,
       
       // Code analysis context
       codeAnalysis: `
         Based on code analysis:
         - Core components: ${components}
         - Key interfaces: ${interfaces}
         - Main patterns: ${patterns}
         
         Identify:
         - Missing documentation
         - Outdated sections
         - Areas needing examples
       `,
       
       // Integration prompt
       integration: `
         Create documentation that:
         1. Preserves existing explanations
         2. Updates outdated sections
         3. Adds missing information
         4. Maintains consistent style
       `
     };
     ```

### Token Management

1. **Budget Allocation**
   ```typescript
   interface TokenBudget {
     // Calculate token usage per component
     calculateTokens(content: string): number;
     
     // Allocate tokens across sections
     allocateTokens(
       totalBudget: number,
       components: Component[]
     ): TokenAllocation;
     
     // Optimize content for token limit
     optimizeContent(
       content: string,
       limit: number
     ): string;
   }
   ```

2. **Importance Ranking**
   ```typescript
   interface ImportanceRanker {
     // Score by reference count
     scoreByReferences(
       component: Component,
       codebase: Codebase
     ): number;
     
     // Score by dependency depth
     scoreByDependencyDepth(
       component: Component,
       dependencies: DependencyGraph
     ): number;
     
     // Score by documentation need
     scoreByDocumentationGaps(
       component: Component,
       existingDocs: ExistingDocs
     ): number;
   }
   ```

3. **Content Selection**
   ```typescript
   interface ContentSelector {
     // Select most important symbols
     selectCriticalSymbols(
       components: Component[],
       tokenBudget: number
     ): Symbol[];
     
     // Prioritize documentation sections
     prioritizeSections(
       sections: Section[],
       tokenBudget: number
     ): Section[];
     
     // Balance code vs explanation
     balanceContent(
       code: CodeContent,
       explanation: ExplanationContent,
       tokenBudget: number
     ): OptimizedContent;
   }
   ```

### Tree-sitter Integration

1. **Parser Configuration**
   ```typescript
   interface TreeSitterConfig {
     // Configure language parsers
     languages: {
       typescript: TreeSitterParser;
       javascript: TreeSitterParser;
       python: TreeSitterParser;
       // ... other languages
     };
     
     // Query patterns for symbol extraction
     queries: {
       classes: string;
       functions: string;
       interfaces: string;
       exports: string;
     };
   }
   ```

2. **Symbol Extraction**
   ```typescript
   interface SymbolExtractor {
     // Extract with tree-sitter
     extractSymbols(
       content: string,
       language: string
     ): Symbol[];
     
     // Find symbol references
     findReferences(
       symbol: Symbol,
       codebase: Codebase
     ): Reference[];
     
     // Extract documentation comments
     extractDocs(
       node: TreeSitterNode
     ): Documentation;
   }
   ```

3. **Code Analysis**
   ```typescript
   interface CodeAnalyzer {
     // Parse with tree-sitter
     parseFile(
       content: string,
       language: string
     ): ParsedFile;
     
     // Extract dependencies
     analyzeDependencies(
       parsedFile: ParsedFile
     ): Dependencies;
     
     // Build symbol graph
     buildSymbolGraph(
       files: ParsedFile[]
     ): SymbolGraph;
   }
   ```

### Progressive Documentation

1. **Usage Analysis**
   ```typescript
   interface UsageAnalyzer {
     // Track documentation access
     trackAccess(
       section: string,
       count: number
     ): void;
     
     // Identify high-traffic sections
     getPopularSections(): Section[];
     
     // Find underutilized docs
     findUnderusedSections(): Section[];
   }
   ```

2. **Documentation Evolution**
   ```typescript
   interface DocEvolution {
     // Expand popular sections
     expandSection(
       section: Section,
       codebase: Codebase
     ): ExpandedSection;
     
     // Add more examples
     addExamples(
       section: Section,
       usage: UsageData
     ): Section;
     
     // Improve clarity
     improveClarity(
       section: Section,
       feedback: UserFeedback
     ): ImprovedSection;
   }
   ```

### Visual Documentation Generation

1. **Flow Diagram Generation**
   ```typescript
   interface FlowDiagramGenerator {
     // Generate mermaid diagrams from code flow
     generateFlowDiagram(
       entryPoints: EntryPoint[],
       dependencies: DependencyGraph
     ): string;
     
     // Create component relationship diagrams
     generateComponentDiagram(
       components: Component[],
       relationships: Relationship[]
     ): string;
     
     // Generate sequence diagrams from function calls
     generateSequenceDiagram(
       functionFlow: FunctionCall[]
     ): string;
   }
   ```

2. **Code Link Management**
   ```typescript
   interface CodeLinkManager {
     // Generate GitHub-style links to code
     generateCodeLink(
       file: string,
       lineStart: number,
       lineEnd?: number
     ): string;
     
     // Track code references
     trackCodeReference(
       reference: CodeReference,
       context: DocumentationContext
     ): void;
     
     // Generate hover previews
     generateHoverPreview(
       reference: CodeReference
     ): string;
   }
   ```

3. **Visual Hierarchy**
   ```typescript
   interface VisualHierarchyBuilder {
     // Build section hierarchy
     buildHierarchy(
       sections: Section[],
       relationships: Relationship[]
     ): DocumentHierarchy;
     
     // Generate navigation structure
     generateNavigation(
       hierarchy: DocumentHierarchy
     ): Navigation;
     
     // Create expandable sections
     createExpandableSection(
       section: Section,
       depth: number
     ): ExpandableSection;
   }
   ```

### Progressive Documentation Flow

1. **Documentation Pipeline**
   ```mermaid
   graph TD
     A[Scan Repository] --> B[Parse Existing Docs]
     B --> C[Analyze Code Structure]
     C --> D[Generate Flow Diagrams]
     D --> E[Create Initial Documentation]
     E --> F[Add Code Links]
     F --> G[Generate Examples]
     G --> H[Add Visual Hierarchy]
   ```

2. **Content Organization**
   ```typescript
   interface ContentOrganizer {
     // Organize by complexity
     organizeByComplexity(
       sections: Section[]
     ): OrganizedSections;
     
     // Create learning paths
     createLearningPath(
       components: Component[]
     ): LearningPath;
     
     // Generate quick references
     generateQuickReference(
       component: Component
     ): QuickReference;
   }
   ```

3. **Example Generation**
   ```typescript
   interface ExampleGenerator {
     // Generate from test files
     generateFromTests(
       testFiles: TestFile[]
     ): Example[];
     
     // Extract from comments
     extractFromComments(
       comments: Comment[]
     ): Example[];
     
     // Create minimal examples
     generateMinimalExample(
       component: Component
     ): Example;
   }
   ```

### Documentation Enhancement

1. **Code Context**
   ```typescript
   interface CodeContextManager {
     // Extract relevant context
     extractContext(
       codeBlock: CodeBlock,
       surroundingLines: number
     ): CodeContext;
     
     // Generate type information
     generateTypeInfo(
       symbol: Symbol
     ): TypeInformation;
     
     // Create usage examples
     generateUsageExample(
       context: CodeContext
     ): Example;
   }
   ```

2. **Visual Aids**
   ```typescript
   interface VisualAidGenerator {
     // Generate architecture diagrams
     generateArchitectureDiagram(
       components: Component[]
     ): Diagram;
     
     // Create data flow diagrams
     generateDataFlowDiagram(
       dataFlow: DataFlow[]
     ): Diagram;
     
     // Generate state diagrams
     generateStateDiagram(
       states: State[]
     ): Diagram;
   }
   ```

3. **Interactive Elements**
   ```typescript
   interface InteractiveElementGenerator {
     // Create expandable code sections
     createExpandableCode(
       code: CodeBlock
     ): ExpandableElement;
     
     // Generate live examples
     createLiveExample(
       example: Example
     ): LiveExample;
     
     // Create API playgrounds
     generateApiPlayground(
       api: ApiDefinition
     ): ApiPlayground;
   }
   ```