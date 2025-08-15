# Documentation Structure

## Root Directory (Essential Files Only)
- `README.md` - Project overview and getting started
- `CLAUDE.md` - Development guidance for Claude Code
- `demo.py` - Main demonstration entry point
- `test_gpu_optimization.py` - GPU performance testing

## Documentation Organization

### `/docs/`
Main documentation directory

#### `/docs/TODO.md` ⭐
**COMPREHENSIVE TECHNICAL ROADMAP** - All current tasks and priorities
- Part 1: Brainstem Refactor (4 weeks)
- Part 2: GPU Optimization (5 weeks)  
- Part 3: Documentation Cleanup
- Part 4: Testing Strategy

#### `/docs/assessments/`
Technical evaluations and analyses
- `BRAIN_ASSESSMENT_2025.md` - Team's comprehensive brain evaluation
- `ENGINEERING_ASSESSMENT.md` - Engineering quality assessment
- `SIMPLIFICATION_ANALYSIS.md` - Complexity reduction opportunities

#### `/docs/planning/`
Development plans and roadmaps
- `REFACTOR_AND_TEST_PLAN.md` - Detailed refactoring strategy
- `NEXT_STEPS_ROADMAP.md` - Strategic development roadmap
- `MINIMAL_BRAIN_INTEGRATION.md` - Simplified integration approach

#### `/docs/deployment/`
Deployment guides and checklists
- `ROBOT_DEPLOYMENT_GUIDE.md` - Complete robot deployment instructions
- `DEPLOYMENT_CHECKLIST.md` - Pre-flight checklist for deployment
- `DEPLOYMENT_READY.md` - Deployment readiness summary

#### `/docs/architecture/`
System architecture documentation
- `ARCHITECTURE.md` - Overall system architecture
- `TENSOR_ARCHITECTURE_ANALYSIS.md` - Tensor field architecture details

#### `/docs/optimization/`
Performance optimization documentation
- `GPU_OPTIMIZATION_WEEK1_SUMMARY.md` - GPU migration progress
- `PERFORMANCE_RECOMMENDATIONS.md` - Performance improvement strategies

#### `/docs/archive/`
Old documentation (for reference only)
- Cleanup summaries
- Dead code analysis
- Obsolete scripts

### `/server/`
Brain server documentation
- `README.md` - Server-specific documentation
- `settings.json` - Configuration
- `settings_safe.json` - Safe mode configuration

### `/client_picarx/`
Robot client documentation
- `PICARX_PROTOCOL_MAPPING.md` - How PiCar-X implements the protocol
- `BRAIN_CLIENT_UPGRADE.md` - Client upgrade notes

## Key Documents Summary

### For Development
1. **Start here**: `/docs/TODO.md` - What needs to be done
2. **Architecture**: `/docs/ARCHITECTURE.md` - How it works
3. **Protocol**: `/docs/COMM_PROTOCOL.md` - Communication specification

### For Deployment
1. **Guide**: `/docs/deployment/ROBOT_DEPLOYMENT_GUIDE.md`
2. **Checklist**: `/docs/deployment/DEPLOYMENT_CHECKLIST.md`
3. **Safety**: Use `--safe-mode` flag for first tests

### For Testing
1. **Strategy**: `/docs/planning/REFACTOR_AND_TEST_PLAN.md`
2. **Behavioral**: See Week 3 in `/docs/TODO.md`
3. **Quick test**: `python3 server/tools/testing/behavioral_test_fast.py`

## Document Status

### Active (Being Updated)
- `/docs/TODO.md` ⭐
- `/docs/COMM_PROTOCOL.md`
- `CLAUDE.md`

### Stable (Reference)
- Architecture documents
- Assessment documents
- Deployment guides

### Archived (Historical)
- Everything in `/docs/archive/`
- Old analysis files
- Cleanup scripts

---
*Last Updated: 2025-08-09*
*Organization Complete: Documents properly structured for easy navigation*