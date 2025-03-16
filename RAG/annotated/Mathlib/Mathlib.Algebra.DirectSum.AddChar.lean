/-- Direct sum of additive characters. -/
@[simps!]
def directSum (ψ : ∀ i, AddChar (G i) R) : AddChar (⨁ i, G i) R :=
  toAddMonoidHomEquiv.symm <| DirectSum.toAddMonoid fun i ↦ toAddMonoidHomEquiv (ψ i)


lemma directSum_injective :
    Injective (directSum : (∀ i, AddChar (G i) R) → AddChar (⨁ i, G i) R) := by
  /-
    ι : Type u_1
    R : Type u_2
    G : ι → Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddCommGroup (G i)
    inst✝ : CommMonoid R
    ⊢ Function.Injective AddChar.directSum
  -/
  refine toAddMonoidHomEquiv.symm.injective.comp <| DirectSum.toAddMonoid_injective.comp ?_
  /-
    ι : Type u_1
    R : Type u_2
    G : ι → Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddCommGroup (G i)
    inst✝ : CommMonoid R
    ⊢ Function.Injective fun ψ i => AddChar.toAddMonoidHomEquiv (ψ i)
  -/
  rintro ψ χ h
  /-
    ι : Type u_1
    R : Type u_2
    G : ι → Type u_3
    inst✝² : DecidableEq ι
    inst✝¹ : (i : ι) → AddCommGroup (G i)
    inst✝ : CommMonoid R
    ψ χ : (i : ι) → AddChar (G i) R
    h : Eq ((fun ψ i => AddChar.toAddMonoidHomEquiv (ψ i)) ψ) ((fun ψ i => AddChar …
    ⊢ Eq ψ χ
  -/
  simpa [funext_iff] using h
  /-
    🎉 no goals
  -/


