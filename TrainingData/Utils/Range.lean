import Mathlib.Tactic.Change
open Lean

instance : LE Position where
  le p₁ p₂ := p₁.line < p₂.line ∨ p₁.line = p₂.line ∧ p₁.column ≤ p₂.column

instance : DecidableRel ((· : Position) ≤ ·) := by
  change DecidableRel fun _ _ => _ ∨ _
  infer_instance

def Range := Position × Position
deriving DecidableEq, Repr, ToString

instance : LE Range where
  le r₁ r₂ := r₂.1 ≤ r₁.1 ∧ r₁.2 ≤ r₂.2

instance : LT Range where
  lt r₁ r₂ := r₁ ≤ r₂ ∧ r₁ ≠ r₂

instance : DecidableRel ((· : Range) ≤ ·) := by
  change DecidableRel fun _ _ => _ ∧ _
  infer_instance

instance : DecidableRel ((· : Range) < ·) := by
  change DecidableRel fun _ _ => _ ∧ _
  infer_instance
