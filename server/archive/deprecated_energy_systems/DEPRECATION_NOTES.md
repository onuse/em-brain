# Deprecated Energy Systems

## Why Deprecated

These systems were replaced by the OrganicEnergySystem which provides:
- Energy emerges naturally from field activity (no tracking)
- No artificial modes or thresholds
- Pattern similarity via cosine distance (no hashing)
- Smooth continuous behavioral influence

## Deprecated Files

### unified_energy_system.py
- Replaced by: organic_energy_system.py
- Issues: 
  - Hard mode thresholds (HUNGRY/BALANCED/SATIATED)
  - Pattern hashing instead of similarity
  - Complex state tracking
  - Artificial energy injection/restoration

### async_energy_processor.py
- Replaced by: Inline processing (performance not critical on dev)
- Issues:
  - Added complexity for minimal benefit
  - Development machine has 750ms budget
  - Production will be 10x faster anyway

## Migration Notes

The OrganicEnergySystem is much simpler:
- Energy = field activity intensity
- Behavior = smooth functions of energy
- No background threads needed
- ~150 lines vs ~650 lines total

## Removed Features
- Mode-based behavior switching
- Background energy maintenance
- Pattern hashing for novelty
- Artificial energy targets