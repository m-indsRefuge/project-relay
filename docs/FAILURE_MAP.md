# Project Relay — Failure Map

**Status:** Living operational document

## Purpose

This document records how Project Relay can fail, what those failures look
like to an operator, how they propagate, how to diagnose them safely, and
what recovery actions should or should not be attempted.

Failure knowledge must live with the software rather than only inside the
development conversation that created it.

---

# Failure Record Template

## FM-XXX — Short failure name

**Component:**  
**Code location:**  
**Responsibility:**  
**Dependencies:**  
**Assumptions:**  

### Failure point

Describe what can fail.

### Observable symptoms

Describe what the operator is likely to actually see.

### Likely causes

- Cause 1
- Cause 2

### First check

Highest-value first diagnostic.

### Evidence

Relevant:

- log;
- receipt;
- state field;
- command;
- database value;
- test.

### Propagation

Describe downstream symptoms that may originate from this failure.

### Recovery

Safest recovery procedure.

### Retry safety

State whether repeating the operation is safe.

### Data risk

Describe possible data/state consequences.

### DO NOT

List actions that could make the incident worse.

### Related tests

List regression or failure-path tests.

### Escalation

State when manual troubleshooting should stop.

---

# M0 Failure Surface

M0 will populate this section as implementation begins.

Expected initial areas include:

- invalid graph state;
- missing state fields;
- wrong conditional routing;
- unexpected terminal state;
- checkpoint write failure;
- SQLite locking;
- checkpoint resume failure;
- thread-ID contamination;
- state-schema serialization problems;
- execution receipt failure.

No M0 milestone is complete until its significant failure boundaries are
documented here.