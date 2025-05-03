/-- A valuation has rank one if it is nontrivial and its image is contained in `ℝ≥0`.
  Note that this class includes the data of an inclusion morphism `Γ₀ → ℝ≥0`. -/
class RankOne (v : Valuation R Γ₀) where
  /-- The inclusion morphism from `Γ₀` to `ℝ≥0`. -/
  hom : Γ₀ →*₀ ℝ≥0
  strictMono' : StrictMono hom
  nontrivial' : ∃ r : R, v r ≠ 0 ∧ v r ≠ 1


lemma strictMono : StrictMono (hom v) := strictMono'


lemma nontrivial : ∃ r : R, v r ≠ 0 ∧ v r ≠ 1 := nontrivial'


/-- If `v` is a rank one valuation and `x : Γ₀` has image `0` under `RankOne.hom v`, then
  `x = 0`. -/
theorem zero_of_hom_zero {x : Γ₀} (hx : hom v x = 0) : x = 0 := by
  /-
    R : Type u_1
    inst✝² : Ring R
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    inst✝ : v.RankOne
    x : Γ₀
    hx : Eq ((Valuation.RankOne.hom v) x) 0
    ⊢ Eq x 0
  -/
  refine (eq_of_le_of_not_lt (zero_le' (a := x)) fun h_lt ↦ ?_).symm
  /-
    R : Type u_1
    inst✝² : Ring R
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    inst✝ : v.RankOne
    x : Γ₀
    hx : Eq ((Valuation.RankOne.hom v) x) 0
    h_lt : LT.lt 0 x
    ⊢ False
  -/
  have hs := strictMono v h_lt
  /-
    R : Type u_1
    inst✝² : Ring R
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    inst✝ : v.RankOne
    x : Γ₀
    hx : Eq ((Valuation.RankOne.hom v) x) 0
    h_lt : LT.lt 0 x
    hs : LT.lt ((Valuation.RankOne.hom v) 0) ((Valuation.RankOne.hom v) x)
    ⊢ False
  -/
  rw [_root_.map_zero, hx] at hs
  /-
    R : Type u_1
    inst✝² : Ring R
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    inst✝ : v.RankOne
    x : Γ₀
    hx : Eq ((Valuation.RankOne.hom v) x) 0
    h_lt : LT.lt 0 x
    hs : LT.lt 0 0
    ⊢ False
  -/
  exact hs.false
  /-
    🎉 no goals
  -/


/-- If `v` is a rank one valuation, then`x : Γ₀` has image `0` under `RankOne.hom v` if and
  only if `x = 0`. -/
theorem hom_eq_zero_iff {x : Γ₀} : RankOne.hom v x = 0 ↔ x = 0 :=
                                            /-
                                              R : Type u_1
                                              inst✝² : Ring R
                                              Γ₀ : Type u_2
                                              inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
                                              v : Valuation R Γ₀
                                              inst✝ : v.RankOne
                                              x : Γ₀
                                              h : Eq x 0
                                              ⊢ Eq ((Valuation.RankOne.hom v) x) 0
                                            -/
  ⟨fun h ↦ zero_of_hom_zero v h, fun h ↦ by rw [h, _root_.map_zero]⟩
                                            /-
                                              🎉 no goals
                                            -/


/-- A nontrivial unit of `Γ₀`, given that there exists a rank one `v : Valuation R Γ₀`. -/
def unit : Γ₀ˣ :=
  Units.mk0 (v (nontrivial v).choose) ((nontrivial v).choose_spec).1


/-- A proof that `RankOne.unit v ≠ 1`. -/
theorem unit_ne_one : unit v ≠ 1 := by
  /-
    R : Type u_1
    inst✝² : Ring R
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    inst✝ : v.RankOne
    ⊢ Ne (Valuation.RankOne.unit v) 1
  -/
  rw [Ne, ← Units.eq_iff, Units.val_one]
  /-
    R : Type u_1
    inst✝² : Ring R
    Γ₀ : Type u_2
    inst✝¹ : LinearOrderedCommGroupWithZero Γ₀
    v : Valuation R Γ₀
    inst✝ : v.RankOne
    ⊢ Not (Eq (↑(Valuation.RankOne.unit v)) 1)
  -/
  exact ((nontrivial v).choose_spec ).2
  /-
    🎉 no goals
  -/


