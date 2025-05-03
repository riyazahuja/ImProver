/-- Class of composing triples -/
class CompTriple {M N P : Type*} [Monoid M] [Monoid N] [Monoid P]
  (φ : M →* N) (ψ : N →* P) (χ : outParam (M →* P)) : Prop where
  /-- The maps form a commuting triangle -/
  comp_eq : ψ.comp φ = χ


/-- Class of Id maps -/
class IsId (σ : M →* M) : Prop where
  eq_id : σ = MonoidHom.id M


instance instIsId {M : Type*} [Monoid M] : IsId (MonoidHom.id M) where
  eq_id := rfl


instance {σ : M →* M} [h : _root_.CompTriple.IsId σ] : IsId σ  where
              /-
                M : Type u_1
                N : Type u_2
                P : Type u_3
                inst✝² : Monoid M
                inst✝¹ : Monoid N
                inst✝ : Monoid P
                σ : MonoidHom M M
                h : _root_.CompTriple.IsId ⇑σ
                ⊢ Eq σ (MonoidHom.id M)
              -/
  eq_id := by ext; exact _root_.congr_fun h.eq_id _
                   /-
                     🎉 no goals
                   -/


instance instComp_id {N P : Type*} [Monoid N] [Monoid P]
    {φ : N →* N} [IsId φ] {ψ : N →* P} :
    CompTriple φ ψ ψ where
                /-
                  M : Type u_1
                  N✝ : Type u_2
                  P✝ : Type u_3
                  inst✝⁵ : Monoid M
                  inst✝⁴ : Monoid N✝
                  inst✝³ : Monoid P✝
                  N : Type u_4
                  P : Type u_5
                  inst✝² : Monoid N
                  inst✝¹ : Monoid P
                  φ : MonoidHom N N
                  inst✝ : MonoidHom.CompTriple.IsId φ
                  ψ : MonoidHom N P
                  ⊢ Eq (ψ.comp φ) ψ
                -/
  comp_eq := by simp only [IsId.eq_id, MonoidHom.comp_id]
                /-
                  🎉 no goals
                -/


instance instId_comp {M N : Type*} [Monoid M] [Monoid N]
    {φ : M →* N} {ψ : N →* N} [IsId ψ] :
    CompTriple φ ψ φ where
                /-
                  M✝ : Type u_1
                  N✝ : Type u_2
                  P : Type u_3
                  inst✝⁵ : Monoid M✝
                  inst✝⁴ : Monoid N✝
                  inst✝³ : Monoid P
                  M : Type u_4
                  N : Type u_5
                  inst✝² : Monoid M
                  inst✝¹ : Monoid N
                  φ : MonoidHom M N
                  ψ : MonoidHom N N
                  inst✝ : MonoidHom.CompTriple.IsId ψ
                  ⊢ Eq (ψ.comp φ) φ
                -/
  comp_eq := by simp only [IsId.eq_id, MonoidHom.id_comp]
                /-
                  🎉 no goals
                -/


lemma comp_inv {φ : M →* N} {ψ : N →* M} (h : Function.RightInverse φ ψ)
    {χ : M →* M} [IsId χ] :
    CompTriple φ ψ χ where
  comp_eq := by
    /-
      M : Type u_1
      N : Type u_2
      inst✝² : Monoid M
      inst✝¹ : Monoid N
      φ : MonoidHom M N
      ψ : MonoidHom N M
      h : Function.RightInverse ⇑φ ⇑ψ
      χ : MonoidHom M M
      inst✝ : MonoidHom.CompTriple.IsId χ
      ⊢ Eq (ψ.comp φ) χ
    -/
    simp only [IsId.eq_id, ← DFunLike.coe_fn_eq, coe_comp, h.id]
    /-
      M : Type u_1
      N : Type u_2
      inst✝² : Monoid M
      inst✝¹ : Monoid N
      φ : MonoidHom M N
      ψ : MonoidHom N M
      h : Function.RightInverse ⇑φ ⇑ψ
      χ : MonoidHom M M
      inst✝ : MonoidHom.CompTriple.IsId χ
      ⊢ Eq _root_.id ⇑(MonoidHom.id M)
    -/
    rfl
    /-
      🎉 no goals
    -/


instance instRootCompTriple {φ : M →* N} {ψ : N  →* P} {χ : M →* P} [κ : CompTriple φ ψ χ] :
    _root_.CompTriple φ ψ χ where
                /-
                  M : Type u_1
                  N : Type u_2
                  P : Type u_3
                  inst✝² : Monoid M
                  inst✝¹ : Monoid N
                  inst✝ : Monoid P
                  φ : MonoidHom M N
                  ψ : MonoidHom N P
                  χ : MonoidHom M P
                  κ : φ.CompTriple ψ χ
                  ⊢ Eq (Function.comp ⇑ψ ⇑φ) ⇑χ
                -/
  comp_eq := by rw [← MonoidHom.coe_comp, κ.comp_eq]
                /-
                  🎉 no goals
                -/


/-- `φ`, `ψ` and `ψ.comp φ` form a `MonoidHom.CompTriple`

  (to be used with care, because no simplification is done)-/
theorem comp {φ : M →* N} {ψ : N →* P} :
    CompTriple φ ψ (ψ.comp φ) where
  comp_eq := rfl


lemma comp_apply
    {φ : M →* N} {ψ : N →* P} {χ : M →* P} (h : CompTriple φ ψ χ) (x : M) :
    ψ (φ x) = χ x := by
  /-
    M : Type u_1
    N : Type u_2
    P : Type u_3
    inst✝² : Monoid M
    inst✝¹ : Monoid N
    inst✝ : Monoid P
    φ : MonoidHom M N
    ψ : MonoidHom N P
    χ : MonoidHom M P
    h : φ.CompTriple ψ χ
    x : M
    ⊢ Eq (ψ (φ x)) (χ x)
  -/
  rw [← h.comp_eq, MonoidHom.comp_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem comp_assoc {Q : Type*} [Monoid Q]
    {φ₁ : M →* N} {φ₂ : N →* P} {φ₁₂ : M →* P}
    (κ : CompTriple φ₁ φ₂ φ₁₂)
    {φ₃ : P →* Q} {φ₂₃ : N →* Q} (κ' : CompTriple φ₂ φ₃ φ₂₃)
    {φ₁₂₃ : M →* Q} :
    CompTriple φ₁ φ₂₃ φ₁₂₃ ↔ CompTriple φ₁₂ φ₃ φ₁₂₃ := by
  /-
    M : Type u_1
    N : Type u_2
    P : Type u_3
    inst✝³ : Monoid M
    inst✝² : Monoid N
    inst✝¹ : Monoid P
    Q : Type u_4
    inst✝ : Monoid Q
    φ₁ : MonoidHom M N
    φ₂ : MonoidHom N P
    φ₁₂ : MonoidHom M P
    κ : φ₁.CompTriple φ₂ φ₁₂
    φ₃ : MonoidHom P Q
    φ₂₃ : MonoidHom N Q
    κ' : φ₂.CompTriple φ₃ φ₂₃
    φ₁₂₃ : MonoidHom M Q
    ⊢ Iff (φ₁.CompTriple φ₂₃ φ₁₂₃) (φ₁₂.CompTriple φ₃ φ₁₂₃)
  -/
  constructor <;>
    /-
      case mp
      M : Type u_1
      N : Type u_2
      P : Type u_3
      inst✝³ : Monoid M
      inst✝² : Monoid N
      inst✝¹ : Monoid P
      Q : Type u_4
      inst✝ : Monoid Q
      φ₁ : MonoidHom M N
      φ₂ : MonoidHom N P
      φ₁₂ : MonoidHom M P
      κ : φ₁.CompTriple φ₂ φ₁₂
      φ₃ : MonoidHom P Q
      φ₂₃ : MonoidHom N Q
      κ' : φ₂.CompTriple φ₃ φ₂₃
      φ₁₂₃ : MonoidHom M Q
      ⊢ φ₁.CompTriple φ₂₃ φ₁₂₃ → φ₁₂.CompTriple φ₃ φ₁₂₃
    -/
    /-
      case mp.mk
      M : Type u_1
      N : Type u_2
      P : Type u_3
      inst✝³ : Monoid M
      inst✝² : Monoid N
      inst✝¹ : Monoid P
      Q : Type u_4
      inst✝ : Monoid Q
      φ₁ : MonoidHom M N
      φ₂ : MonoidHom N P
      φ₁₂ : MonoidHom M P
      κ : φ₁.CompTriple φ₂ φ₁₂
      φ₃ : MonoidHom P Q
      φ₂₃ : MonoidHom N Q
      κ' : φ₂.CompTriple φ₃ φ₂₃
      φ₁₂₃ : MonoidHom M Q
      h : Eq (φ₂₃.comp φ₁) φ₁₂₃
      ⊢ φ₁₂.CompTriple φ₃ φ₁₂₃
    -/
    /-
      🎉 no goals
    -/
    /-
      case mpr.mk
      M : Type u_1
      N : Type u_2
      P : Type u_3
      inst✝³ : Monoid M
      inst✝² : Monoid N
      inst✝¹ : Monoid P
      Q : Type u_4
      inst✝ : Monoid Q
      φ₁ : MonoidHom M N
      φ₂ : MonoidHom N P
      φ₁₂ : MonoidHom M P
      κ : φ₁.CompTriple φ₂ φ₁₂
      φ₃ : MonoidHom P Q
      φ₂₃ : MonoidHom N Q
      κ' : φ₂.CompTriple φ₃ φ₂₃
      φ₁₂₃ : MonoidHom M Q
      h : Eq (φ₃.comp φ₁₂) φ₁₂₃
      ⊢ φ₁.CompTriple φ₂₃ φ₁₂₃
    -/
    exact ⟨by simp only [← κ.comp_eq, ← h, ← κ'.comp_eq, MonoidHom.comp_assoc]⟩
    /-
      🎉 no goals
    -/


