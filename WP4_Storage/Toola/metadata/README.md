# FlexRICAN Toola Metadata Graph

This folder contains ontology-oriented metadata and relationship graph files for the FlexRICAN Toola data architecture.

The metadata describe source datasets, curated Toola input tables, calculation dependencies, provenance and output indicators using an OEMetadata-compatible structure and a subject-predicate-object relationship table.

## Toola data relationship graph

The diagram below provides a simplified visual representation of the main data dependencies in Toola.

```mermaid
graph TD

    Ember["Ember Electricity Data Explorer"]
    EnergyMix["Ember EnergyMIX long-format dataset"]
    CarbonRaw["Ember carbon intensity raw exports"]
    CarbonSummary["FlexRICAN carbon intensity summary"]
    ToolaRun["Toola calculation run"]

    Scenario["Toola scenario configuration"]
    Country["Country or economy"]
    Year["Assessment year"]

    Demand["Facility electricity demand profile"]
    CountryCI["Country carbon intensity"]
    GridImport["Grid electricity import"]
    GridEmissions["Annual grid electricity emissions"]

    ElectricityMix["Electricity generation mix"]
    PV["PV generation potential"]
    Wind["Wind generation potential"]
    RESShare["Renewable energy share"]

    BESS["Battery storage sizing"]
    Resilience["Operational resilience indicator"]

    Ember -->|provides| EnergyMix
    Ember -->|provides| CarbonRaw

    EnergyMix -->|contains variable| ElectricityMix
    EnergyMix -->|contains variable| CountryCI
    EnergyMix -->|contextualises| CarbonSummary

    CarbonRaw -->|is processed into| CarbonSummary
    CarbonSummary -->|provides| CountryCI

    Scenario -->|selects| Country
    Scenario -->|selects| Year
    Scenario -->|configures| ToolaRun

    Country -->|determines applicable value of| CountryCI
    Year -->|determines applicable value of| CountryCI

    CarbonSummary -->|provides input to| ToolaRun
    Demand -->|provides input to| ToolaRun
    PV -->|provides input to| ToolaRun
    Wind -->|provides input to| ToolaRun

    GridImport -->|combined with| CountryCI
    Demand -->|is multiplied by| CountryCI
    CountryCI -->|used to compute| GridEmissions
    ToolaRun -->|computes| GridEmissions

    PV -->|contributes to| RESShare
    Wind -->|contributes to| RESShare
    ElectricityMix -->|contextualises| RESShare
    ToolaRun -->|computes| RESShare

    Demand -->|influences| BESS
    PV -->|influences| BESS
    Wind -->|influences| BESS
    Scenario -->|defines constraints for| BESS
    BESS -->|supports| Resilience
    ToolaRun -->|evaluates| Resilience
```

## Files

- `oemetadata.metadata.json` – OEMetadata-compatible metadata record for the Toola relationship graph.
- `toola_data_relationships.csv` – subject-predicate-object relationship table.
- `toola_data_relationships_graph.jsonld` – JSON-LD representation of the Toola relationship graph.
- `toola_data_relationships_README.md` – detailed methodological note.

## Notes

This graph is a lightweight project-specific relationship graph. It is not a replacement for the Open Energy Ontology. It documents how Toola input datasets, assumptions, calculation steps and output indicators are connected for reproducibility, traceability and later validation.
