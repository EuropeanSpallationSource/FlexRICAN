# FlexRICAN Toola data relationships

This package contains the concrete relationship table for the FlexRICAN Toola lightweight ontology/data graph.

## Files

- `toola_data_relationships.csv` – canonical subject-predicate-object edge list.
- `toola_data_relationships_graph.jsonld` – JSON-LD-style graph serialisation derived from the CSV.
- `oemetadata.metadata (1).json` – OEMetadata record exported from the Open Energy Platform OEMetaBuilder; it describes the relationship table schema and provenance.

## Primary key

The CSV uses the composite primary key defined in the OEMetadata record:

```text
subject_id + predicate + object_id
```

## Core columns

```text
subject_id, subject_label, subject_type, predicate, object_id, object_label, object_type, methodological_note, source_reference
```

## Main methodological logic

The relationship graph links:

```text
Ember source platform
→ Ember EnergyMIX long-format dataset
→ curated FlexRICAN carbon intensity summary
→ Toola calculation run
→ Toola output indicators
```

It also includes contextual Toola inputs such as facility demand, PV/wind potential, climate scenarios, technology parameters and BESS sizing dependencies.
