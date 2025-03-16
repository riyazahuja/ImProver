/-- Class of composing triples -/
class CompTriple {M N P : Type*} (φ : M → N) (ψ : N → P) (χ : outParam (M → P)) : Prop where
  /-- The maps form a commuting triangle -/
  comp_eq : ψ.comp φ = χ


/-- Class of Id maps -/
class IsId {M : Type*} (σ : M → M) : Prop where
  eq_id : σ = id


instance {M : Type*} : IsId (@id M) where
  eq_id := rfl


instance instComp_id {N P : Type*} {φ : N → N} [IsId φ] {ψ : N → P} :
    CompTriple φ ψ ψ where
                /-
                  N : Type u_1
                  P : Type u_2
                  φ : N → N
                  inst✝ : CompTriple.IsId φ
                  ψ : N → P
                  ⊢ Eq (Function.comp ψ φ) ψ
                -/
  comp_eq := by simp only [IsId.eq_id, Function.comp_id]
                /-
                  🎉 no goals
                -/


instance instId_comp {M N : Type*} {φ : M → N} {ψ : N → N} [IsId ψ] :
    CompTriple φ ψ φ where
                /-
                  M : Type u_1
                  N : Type u_2
                  φ : M → N
                  ψ : N → N
                  inst✝ : CompTriple.IsId ψ
                  ⊢ Eq (Function.comp ψ φ) φ
                -/
  comp_eq := by simp only [IsId.eq_id, Function.id_comp]
                /-
                  🎉 no goals
                -/


/-- `φ`, `ψ` and `ψ ∘ φ` for` a `CompTriple` -/
theorem comp {M N P : Type*}
    {φ : M → N} {ψ : N → P} :
    CompTriple φ ψ  (ψ.comp φ) where
  comp_eq := rfl


lemma comp_inv {M N : Type*} {φ : M → N} {ψ : N → M}
    (h : Function.RightInverse φ ψ) {χ : M → M} [IsId χ] :
    CompTriple φ ψ χ where
                /-
                  M : Type u_1
                  N : Type u_2
                  φ : M → N
                  ψ : N → M
                  h : Function.RightInverse φ ψ
                  χ : M → M
                  inst✝ : CompTriple.IsId χ
                  ⊢ Eq (Function.comp ψ φ) χ
                -/
  comp_eq := by simp only [IsId.eq_id, h.id]
                /-
                  🎉 no goals
                -/


lemma comp_apply {M N P : Type*}
    {φ : M → N} {ψ : N → P} {χ : M → P} (h : CompTriple φ ψ χ) (x : M) :
    ψ (φ x) = χ x := by
  /-
    M : Type u_1
    N : Type u_2
    P : Type u_3
    φ : M → N
    ψ : N → P
    χ : M → P
    h : CompTriple φ ψ χ
    x : M
    ⊢ Eq (ψ (φ x)) (χ x)
  -/
  rw [← h.comp_eq, Function.comp_apply]
  /-
    🎉 no goals
  -/


