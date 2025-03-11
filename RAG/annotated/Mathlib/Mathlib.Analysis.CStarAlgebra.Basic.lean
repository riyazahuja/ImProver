local postfix:max "⋆" => star


/-- A normed star group is a normed group with a compatible `star` which is isometric. -/
class NormedStarGroup (E : Type*) [SeminormedAddCommGroup E] [StarAddMonoid E] : Prop where
  norm_star : ∀ x : E, ‖x⋆‖ = ‖x‖


@[simp]
theorem nnnorm_star (x : E) : ‖star x‖₊ = ‖x‖₊ :=
  Subtype.ext <| norm_star _


/-- The `star` map in a normed star group is a normed group homomorphism. -/
def starNormedAddGroupHom : NormedAddGroupHom E E :=
  { starAddEquiv with bound' := ⟨1, fun _ => le_trans (norm_star _).le (one_mul _).symm.le⟩ }


/-- The `star` map in a normed star group is an isometry -/
theorem star_isometry : Isometry (star : E → E) :=
  show Isometry starAddEquiv from
    AddMonoidHomClass.isometry_of_norm starAddEquiv (show ∀ x, ‖x⋆‖ = ‖x‖ from norm_star)


instance (priority := 100) NormedStarGroup.to_continuousStar : ContinuousStar E :=
  ⟨star_isometry.continuous⟩


instance RingHomIsometric.starRingEnd [NormedCommRing E] [StarRing E] [NormedStarGroup E] :
    RingHomIsometric (starRingEnd E) :=
  ⟨@norm_star _ _ _ _⟩


/-- A C*-ring is a normed star ring that satisfies the stronger condition `‖x‖ ^ 2 ≤ ‖x⋆ * x‖`
for every `x`. Note that this condition actually implies equality, as is shown in
`norm_star_mul_self` below. -/
class CStarRing (E : Type*) [NonUnitalNormedRing E] [StarRing E] : Prop where
  norm_mul_self_le : ∀ x : E, ‖x‖ * ‖x‖ ≤ ‖x⋆ * x‖


@[deprecated (since := "2024-08-04")] alias CstarRing := CStarRing


instance : CStarRing ℝ where
  norm_mul_self_le x := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      α : Type u_3
      x : Real
      ⊢ LE.le (HMul.hMul (Norm.norm x) (Norm.norm x)) (Norm.norm (HMul.hMul (Star.st …
    -/
    simp only [Real.norm_eq_abs, abs_mul_abs_self, star, id, norm_mul, le_refl]
    /-
      🎉 no goals
    -/


/-- In a C*-ring, star preserves the norm. -/
instance (priority := 100) to_normedStarGroup : NormedStarGroup E :=
  ⟨by
    /-
      𝕜 : Type u_1
      E : Type u_2
      α : Type u_3
      inst✝² : NonUnitalNormedRing E
      inst✝¹ : StarRing E
      inst✝ : CStarRing E
      ⊢ ∀ (x : E), Eq (Norm.norm (Star.star x)) (Norm.norm x)
    -/
    intro x
    /-
      𝕜 : Type u_1
      E : Type u_2
      α : Type u_3
      inst✝² : NonUnitalNormedRing E
      inst✝¹ : StarRing E
      inst✝ : CStarRing E
      x : E
      ⊢ Eq (Norm.norm (Star.star x)) (Norm.norm x)
    -/
    by_cases htriv : x = 0
      /-
        case pos
        𝕜 : Type u_1
        E : Type u_2
        α : Type u_3
        inst✝² : NonUnitalNormedRing E
        inst✝¹ : StarRing E
        inst✝ : CStarRing E
        x : E
        htriv : Eq x 0
        ⊢ Eq (Norm.norm (Star.star x)) (Norm.norm x)
      -/
    · simp only [htriv, star_zero]
      /-
        🎉 no goals
      -/
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        α : Type u_3
        inst✝² : NonUnitalNormedRing E
        inst✝¹ : StarRing E
        inst✝ : CStarRing E
        x : E
        htriv : Not (Eq x 0)
        ⊢ Eq (Norm.norm (Star.star x)) (Norm.norm x)
      -/
    · have hnt : 0 < ‖x‖ := norm_pos_iff.mpr htriv
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        α : Type u_3
        inst✝² : NonUnitalNormedRing E
        inst✝¹ : StarRing E
        inst✝ : CStarRing E
        x : E
        htriv : Not (Eq x 0)
        hnt : LT.lt 0 (Norm.norm x)
        ⊢ Eq (Norm.norm (Star.star x)) (Norm.norm x)
      -/
      have h₁ : ∀ z : E, ‖z⋆ * z‖ ≤ ‖z⋆‖ * ‖z‖ := fun z => norm_mul_le z⋆ z
      have h₂ : ∀ z : E, 0 < ‖z‖ → ‖z‖ ≤ ‖z⋆‖ := fun z hz => by
        rw [← mul_le_mul_right hz]; exact (CStarRing.norm_mul_self_le z).trans (h₁ z)
      have h₃ : ‖x⋆‖ ≤ ‖x‖ := by
        conv_rhs => rw [← star_star x]
        exact h₂ x⋆ (gt_of_ge_of_gt (h₂ x hnt) hnt)
      /-
        case neg
        𝕜 : Type u_1
        E : Type u_2
        α : Type u_3
        inst✝² : NonUnitalNormedRing E
        inst✝¹ : StarRing E
        inst✝ : CStarRing E
        x : E
        htriv : Not (Eq x 0)
        hnt : LT.lt 0 (Norm.norm x)
        h₁ : ∀ (z : E), LE.le (Norm.norm (HMul.hMul (Star.star z) z)) (HMul.hMul (Norm …
        h₂ : ∀ (z : E), LT.lt 0 (Norm.norm z) → LE.le (Norm.norm z) (Norm.norm (Star.s …
        h₃ : LE.le (Norm.norm (Star.star x)) (Norm.norm x)
        ⊢ Eq (Norm.norm (Star.star x)) (Norm.norm x)
      -/
      exact le_antisymm h₃ (h₂ x hnt)⟩
      /-
        🎉 no goals
      -/


theorem norm_star_mul_self {x : E} : ‖x⋆ * x‖ = ‖x‖ * ‖x‖ :=
                                           /-
                                             E : Type u_2
                                             inst✝² : NonUnitalNormedRing E
                                             inst✝¹ : StarRing E
                                             inst✝ : CStarRing E
                                             x : E
                                             ⊢ LE.le (HMul.hMul (Norm.norm (Star.star x)) (Norm.norm x)) (HMul.hMul (Norm.n …
                                           -/
  le_antisymm ((norm_mul_le _ _).trans (by rw [norm_star])) (CStarRing.norm_mul_self_le x)
                                           /-
                                             🎉 no goals
                                           -/


theorem norm_self_mul_star {x : E} : ‖x * x⋆‖ = ‖x‖ * ‖x‖ := by
  /-
    E : Type u_2
    inst✝² : NonUnitalNormedRing E
    inst✝¹ : StarRing E
    inst✝ : CStarRing E
    x : E
    ⊢ Eq (Norm.norm (HMul.hMul x (Star.star x))) (HMul.hMul (Norm.norm x) (Norm.no …
  -/
  nth_rw 1 [← star_star x]
  /-
    E : Type u_2
    inst✝² : NonUnitalNormedRing E
    inst✝¹ : StarRing E
    inst✝ : CStarRing E
    x : E
    ⊢ Eq (Norm.norm (HMul.hMul (Star.star (Star.star x)) (Star.star x))) (HMul.hMu …
  -/
  simp only [norm_star_mul_self, norm_star]
  /-
    🎉 no goals
  -/


                                                                  /-
                                                                    E : Type u_2
                                                                    inst✝² : NonUnitalNormedRing E
                                                                    inst✝¹ : StarRing E
                                                                    inst✝ : CStarRing E
                                                                    x : E
                                                                    ⊢ Eq (Norm.norm (HMul.hMul (Star.star x) x)) (HMul.hMul (Norm.norm (Star.star  …
                                                                  -/
theorem norm_star_mul_self' {x : E} : ‖x⋆ * x‖ = ‖x⋆‖ * ‖x‖ := by rw [norm_star_mul_self, norm_star]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


theorem nnnorm_self_mul_star {x : E} : ‖x * x⋆‖₊ = ‖x‖₊ * ‖x‖₊ :=
  Subtype.ext norm_self_mul_star


theorem nnnorm_star_mul_self {x : E} : ‖x⋆ * x‖₊ = ‖x‖₊ * ‖x‖₊ :=
  Subtype.ext norm_star_mul_self


@[simp]
theorem star_mul_self_eq_zero_iff (x : E) : x⋆ * x = 0 ↔ x = 0 := by
  /-
    E : Type u_2
    inst✝² : NonUnitalNormedRing E
    inst✝¹ : StarRing E
    inst✝ : CStarRing E
    x : E
    ⊢ Iff (Eq (HMul.hMul (Star.star x) x) 0) (Eq x 0)
  -/
  rw [← norm_eq_zero, norm_star_mul_self]
  /-
    E : Type u_2
    inst✝² : NonUnitalNormedRing E
    inst✝¹ : StarRing E
    inst✝ : CStarRing E
    x : E
    ⊢ Iff (Eq (HMul.hMul (Norm.norm x) (Norm.norm x)) 0) (Eq x 0)
  -/
  exact mul_self_eq_zero.trans norm_eq_zero
  /-
    🎉 no goals
  -/


theorem star_mul_self_ne_zero_iff (x : E) : x⋆ * x ≠ 0 ↔ x ≠ 0 := by
  /-
    E : Type u_2
    inst✝² : NonUnitalNormedRing E
    inst✝¹ : StarRing E
    inst✝ : CStarRing E
    x : E
    ⊢ Iff (Ne (HMul.hMul (Star.star x) x) 0) (Ne x 0)
  -/
  simp only [Ne, star_mul_self_eq_zero_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem mul_star_self_eq_zero_iff (x : E) : x * x⋆ = 0 ↔ x = 0 := by
  /-
    E : Type u_2
    inst✝² : NonUnitalNormedRing E
    inst✝¹ : StarRing E
    inst✝ : CStarRing E
    x : E
    ⊢ Iff (Eq (HMul.hMul x (Star.star x)) 0) (Eq x 0)
  -/
  simpa only [star_eq_zero, star_star] using @star_mul_self_eq_zero_iff _ _ _ _ (star x)
  /-
    🎉 no goals
  -/


theorem mul_star_self_ne_zero_iff (x : E) : x * x⋆ ≠ 0 ↔ x ≠ 0 := by
  /-
    E : Type u_2
    inst✝² : NonUnitalNormedRing E
    inst✝¹ : StarRing E
    inst✝ : CStarRing E
    x : E
    ⊢ Iff (Ne (HMul.hMul x (Star.star x)) 0) (Ne x 0)
  -/
  simp only [Ne, mul_star_self_eq_zero_iff]
  /-
    🎉 no goals
  -/


/-- This instance exists to short circuit type class resolution because of problems with
inference involving Π-types. -/
instance _root_.Pi.starRing' : StarRing (∀ i, R i) :=
  inferInstance


instance _root_.Prod.cstarRing : CStarRing (R₁ × R₂) where
  norm_mul_self_le x := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      α : Type u_3
      ι : Type u_4
      R₁ : Type u_5
      R₂ : Type u_6
      R : ι → Type u_7
      inst✝⁹ : NonUnitalNormedRing R₁
      inst✝⁸ : StarRing R₁
      inst✝⁷ : CStarRing R₁
      inst✝⁶ : NonUnitalNormedRing R₂
      inst✝⁵ : StarRing R₂
      inst✝⁴ : CStarRing R₂
      inst✝³ : (i : ι) → NonUnitalNormedRing (R i)
      inst✝² : (i : ι) → StarRing (R i)
      inst✝¹ : Fintype ι
      inst✝ : ∀ (i : ι), CStarRing (R i)
      x : Prod R₁ R₂
      ⊢ LE.le (HMul.hMul (Norm.norm x) (Norm.norm x)) (Norm.norm (HMul.hMul (Star.st …
    -/
    dsimp only [norm]
    /-
      𝕜 : Type u_1
      E : Type u_2
      α : Type u_3
      ι : Type u_4
      R₁ : Type u_5
      R₂ : Type u_6
      R : ι → Type u_7
      inst✝⁹ : NonUnitalNormedRing R₁
      inst✝⁸ : StarRing R₁
      inst✝⁷ : CStarRing R₁
      inst✝⁶ : NonUnitalNormedRing R₂
      inst✝⁵ : StarRing R₂
      inst✝⁴ : CStarRing R₂
      inst✝³ : (i : ι) → NonUnitalNormedRing (R i)
      inst✝² : (i : ι) → StarRing (R i)
      inst✝¹ : Fintype ι
      inst✝ : ∀ (i : ι), CStarRing (R i)
      x : Prod R₁ R₂
      ⊢ LE.le (HMul.hMul (Max.max (Norm.norm x.1) (Norm.norm x.2)) (Max.max (Norm.no …
    -/
    simp only [Prod.fst_mul, Prod.fst_star, Prod.snd_mul, Prod.snd_star, norm_star_mul_self, ← sq]
    /-
      𝕜 : Type u_1
      E : Type u_2
      α : Type u_3
      ι : Type u_4
      R₁ : Type u_5
      R₂ : Type u_6
      R : ι → Type u_7
      inst✝⁹ : NonUnitalNormedRing R₁
      inst✝⁸ : StarRing R₁
      inst✝⁷ : CStarRing R₁
      inst✝⁶ : NonUnitalNormedRing R₂
      inst✝⁵ : StarRing R₂
      inst✝⁴ : CStarRing R₂
      inst✝³ : (i : ι) → NonUnitalNormedRing (R i)
      inst✝² : (i : ι) → StarRing (R i)
      inst✝¹ : Fintype ι
      inst✝ : ∀ (i : ι), CStarRing (R i)
      x : Prod R₁ R₂
      ⊢ LE.le (HPow.hPow (Max.max (Norm.norm x.1) (Norm.norm x.2)) 2) (Max.max (HPow …
    -/
    rw [le_sup_iff]
    /-
      𝕜 : Type u_1
      E : Type u_2
      α : Type u_3
      ι : Type u_4
      R₁ : Type u_5
      R₂ : Type u_6
      R : ι → Type u_7
      inst✝⁹ : NonUnitalNormedRing R₁
      inst✝⁸ : StarRing R₁
      inst✝⁷ : CStarRing R₁
      inst✝⁶ : NonUnitalNormedRing R₂
      inst✝⁵ : StarRing R₂
      inst✝⁴ : CStarRing R₂
      inst✝³ : (i : ι) → NonUnitalNormedRing (R i)
      inst✝² : (i : ι) → StarRing (R i)
      inst✝¹ : Fintype ι
      inst✝ : ∀ (i : ι), CStarRing (R i)
      x : Prod R₁ R₂
      ⊢ Or (LE.le (HPow.hPow (Max.max (Norm.norm x.1) (Norm.norm x.2)) 2) (HPow.hPow …
    -/
                                                     /-
                                                       🎉 no goals
                                                     -/
    rcases le_total ‖x.fst‖ ‖x.snd‖ with (h | h) <;> simp [h]
                                                     /-
                                                       🎉 no goals
                                                     -/


instance _root_.Pi.cstarRing : CStarRing (∀ i, R i) where
  norm_mul_self_le x := by
    /-
      𝕜 : Type u_1
      E : Type u_2
      α : Type u_3
      ι : Type u_4
      R₁ : Type u_5
      R₂ : Type u_6
      R : ι → Type u_7
      inst✝⁹ : NonUnitalNormedRing R₁
      inst✝⁸ : StarRing R₁
      inst✝⁷ : CStarRing R₁
      inst✝⁶ : NonUnitalNormedRing R₂
      inst✝⁵ : StarRing R₂
      inst✝⁴ : CStarRing R₂
      inst✝³ : (i : ι) → NonUnitalNormedRing (R i)
      inst✝² : (i : ι) → StarRing (R i)
      inst✝¹ : Fintype ι
      inst✝ : ∀ (i : ι), CStarRing (R i)
      x : (i : ι) → R i
      ⊢ LE.le (HMul.hMul (Norm.norm x) (Norm.norm x)) (Norm.norm (HMul.hMul (Star.st …
    -/
    refine le_of_eq (Eq.symm ?_)
    /-
      𝕜 : Type u_1
      E : Type u_2
      α : Type u_3
      ι : Type u_4
      R₁ : Type u_5
      R₂ : Type u_6
      R : ι → Type u_7
      inst✝⁹ : NonUnitalNormedRing R₁
      inst✝⁸ : StarRing R₁
      inst✝⁷ : CStarRing R₁
      inst✝⁶ : NonUnitalNormedRing R₂
      inst✝⁵ : StarRing R₂
      inst✝⁴ : CStarRing R₂
      inst✝³ : (i : ι) → NonUnitalNormedRing (R i)
      inst✝² : (i : ι) → StarRing (R i)
      inst✝¹ : Fintype ι
      inst✝ : ∀ (i : ι), CStarRing (R i)
      x : (i : ι) → R i
      ⊢ Eq (Norm.norm (HMul.hMul (Star.star x) x)) (HMul.hMul (Norm.norm x) (Norm.no …
    -/
    simp only [norm, Pi.mul_apply, Pi.star_apply, nnnorm_star_mul_self, ← sq]
    /-
      𝕜 : Type u_1
      E : Type u_2
      α : Type u_3
      ι : Type u_4
      R₁ : Type u_5
      R₂ : Type u_6
      R : ι → Type u_7
      inst✝⁹ : NonUnitalNormedRing R₁
      inst✝⁸ : StarRing R₁
      inst✝⁷ : CStarRing R₁
      inst✝⁶ : NonUnitalNormedRing R₂
      inst✝⁵ : StarRing R₂
      inst✝⁴ : CStarRing R₂
      inst✝³ : (i : ι) → NonUnitalNormedRing (R i)
      inst✝² : (i : ι) → StarRing (R i)
      inst✝¹ : Fintype ι
      inst✝ : ∀ (i : ι), CStarRing (R i)
      x : (i : ι) → R i
      ⊢ Eq (↑(Finset.univ.sup fun b => HPow.hPow (NNNorm.nnnorm (x b)) 2)) (HPow.hPo …
    -/
    norm_cast
    exact
      (Finset.comp_sup_eq_sup_comp_of_is_total (fun x : NNReal => x ^ 2)
          (fun x y h => by simpa only [sq] using mul_le_mul' h h) (by simp)).symm


instance _root_.Pi.cstarRing' : CStarRing (ι → R₁) :=
  Pi.cstarRing


@[simp, nolint simpNF]
theorem norm_one [Nontrivial E] : ‖(1 : E)‖ = 1 := by
  /-
    E : Type u_2
    inst✝³ : NormedRing E
    inst✝² : StarRing E
    inst✝¹ : CStarRing E
    inst✝ : Nontrivial E
    ⊢ Eq (Norm.norm 1) 1
  -/
  have : 0 < ‖(1 : E)‖ := norm_pos_iff.mpr one_ne_zero
  /-
    E : Type u_2
    inst✝³ : NormedRing E
    inst✝² : StarRing E
    inst✝¹ : CStarRing E
    inst✝ : Nontrivial E
    this : LT.lt 0 (Norm.norm 1)
    ⊢ Eq (Norm.norm 1) 1
  -/
  rw [← mul_left_inj' this.ne', ← norm_star_mul_self, mul_one, star_one, one_mul]
  /-
    🎉 no goals
  -/

-- see Note [lower instance priority]

instance (priority := 100) [Nontrivial E] : NormOneClass E :=
  ⟨norm_one⟩


theorem norm_coe_unitary [Nontrivial E] (U : unitary E) : ‖(U : E)‖ = 1 := by
  rw [← sq_eq_sq₀ (norm_nonneg _) zero_le_one, one_pow 2, sq, ← CStarRing.norm_star_mul_self,
    unitary.coe_star_mul_self, CStarRing.norm_one]


@[simp]
theorem norm_of_mem_unitary [Nontrivial E] {U : E} (hU : U ∈ unitary E) : ‖U‖ = 1 :=
  norm_coe_unitary ⟨U, hU⟩


@[simp]
theorem norm_coe_unitary_mul (U : unitary E) (A : E) : ‖(U : E) * A‖ = ‖A‖ := by
  /-
    E : Type u_2
    inst✝² : NormedRing E
    inst✝¹ : StarRing E
    inst✝ : CStarRing E
    U : Subtype fun x => Membership.mem (unitary E) x
    A : E
    ⊢ Eq (Norm.norm (HMul.hMul (↑U) A)) (Norm.norm A)
  -/
  nontriviality E
  /-
    E : Type u_2
    inst✝² : NormedRing E
    inst✝¹ : StarRing E
    inst✝ : CStarRing E
    U : Subtype fun x => Membership.mem (unitary E) x
    A : E
    a✝ : Nontrivial E
    ⊢ Eq (Norm.norm (HMul.hMul (↑U) A)) (Norm.norm A)
  -/
  refine le_antisymm ?_ ?_
  · calc
      _ ≤ ‖(U : E)‖ * ‖A‖ := norm_mul_le _ _
      _ = ‖A‖ := by rw [norm_coe_unitary, one_mul]
  · calc
      _ = ‖(U : E)⋆ * U * A‖ := by rw [unitary.coe_star_mul_self U, one_mul]
      _ ≤ ‖(U : E)⋆‖ * ‖(U : E) * A‖ := by
        rw [mul_assoc]
        exact norm_mul_le _ _
      _ = ‖(U : E) * A‖ := by rw [norm_star, norm_coe_unitary, one_mul]


@[simp]
theorem norm_unitary_smul (U : unitary E) (A : E) : ‖U • A‖ = ‖A‖ :=
  norm_coe_unitary_mul U A


theorem norm_mem_unitary_mul {U : E} (A : E) (hU : U ∈ unitary E) : ‖U * A‖ = ‖A‖ :=
  norm_coe_unitary_mul ⟨U, hU⟩ A


@[simp]
theorem norm_mul_coe_unitary (A : E) (U : unitary E) : ‖A * U‖ = ‖A‖ :=
  calc
                                 /-
                                   E : Type u_2
                                   inst✝² : NormedRing E
                                   inst✝¹ : StarRing E
                                   inst✝ : CStarRing E
                                   A : E
                                   U : Subtype fun x => Membership.mem (unitary E) x
                                   ⊢ Eq (Norm.norm (HMul.hMul A ↑U)) (Norm.norm (Star.star (HMul.hMul (Star.star  …
                                 -/
    _ = ‖((U : E)⋆ * A⋆)⋆‖ := by simp only [star_star, star_mul]
                                 /-
                                   🎉 no goals
                                 -/
                              /-
                                E : Type u_2
                                inst✝² : NormedRing E
                                inst✝¹ : StarRing E
                                inst✝ : CStarRing E
                                A : E
                                U : Subtype fun x => Membership.mem (unitary E) x
                                ⊢ Eq (Norm.norm (Star.star (HMul.hMul (Star.star ↑U) (Star.star A)))) (Norm.no …
                              -/
    _ = ‖(U : E)⋆ * A⋆‖ := by rw [norm_star]
                              /-
                                🎉 no goals
                              -/
    _ = ‖A⋆‖ := norm_mem_unitary_mul (star A) (unitary.star_mem U.prop)
    _ = ‖A‖ := norm_star _


theorem norm_mul_mem_unitary (A : E) {U : E} (hU : U ∈ unitary E) : ‖A * U‖ = ‖A‖ :=
  norm_mul_coe_unitary A ⟨U, hU⟩


theorem IsSelfAdjoint.nnnorm_pow_two_pow [NormedRing E] [StarRing E] [CStarRing E] {x : E}
    (hx : IsSelfAdjoint x) (n : ℕ) : ‖x ^ 2 ^ n‖₊ = ‖x‖₊ ^ 2 ^ n := by
  /-
    E : Type u_2
    inst✝² : NormedRing E
    inst✝¹ : StarRing E
    inst✝ : CStarRing E
    x : E
    hx : IsSelfAdjoint x
    n : Nat
    ⊢ Eq (NNNorm.nnnorm (HPow.hPow x (HPow.hPow 2 n))) (HPow.hPow (NNNorm.nnnorm x …
  -/
  induction' n with k hk
    /-
      case zero
      E : Type u_2
      inst✝² : NormedRing E
      inst✝¹ : StarRing E
      inst✝ : CStarRing E
      x : E
      hx : IsSelfAdjoint x
      ⊢ Eq (NNNorm.nnnorm (HPow.hPow x (HPow.hPow 2 0))) (HPow.hPow (NNNorm.nnnorm x …
    -/
  · simp only [pow_zero, pow_one]
    /-
      🎉 no goals
    -/
    /-
      case succ
      E : Type u_2
      inst✝² : NormedRing E
      inst✝¹ : StarRing E
      inst✝ : CStarRing E
      x : E
      hx : IsSelfAdjoint x
      k : Nat
      hk : Eq (NNNorm.nnnorm (HPow.hPow x (HPow.hPow 2 k))) (HPow.hPow (NNNorm.nnnor …
      ⊢ Eq (NNNorm.nnnorm (HPow.hPow x (HPow.hPow 2 (HAdd.hAdd k 1)))) (HPow.hPow (N …
    -/
  · rw [pow_succ', pow_mul', sq]
    /-
      case succ
      E : Type u_2
      inst✝² : NormedRing E
      inst✝¹ : StarRing E
      inst✝ : CStarRing E
      x : E
      hx : IsSelfAdjoint x
      k : Nat
      hk : Eq (NNNorm.nnnorm (HPow.hPow x (HPow.hPow 2 k))) (HPow.hPow (NNNorm.nnnor …
      ⊢ Eq (NNNorm.nnnorm (HMul.hMul (HPow.hPow x (HPow.hPow 2 k)) (HPow.hPow x (HPo …
    -/
    nth_rw 1 [← selfAdjoint.mem_iff.mp hx]
    /-
      case succ
      E : Type u_2
      inst✝² : NormedRing E
      inst✝¹ : StarRing E
      inst✝ : CStarRing E
      x : E
      hx : IsSelfAdjoint x
      k : Nat
      hk : Eq (NNNorm.nnnorm (HPow.hPow x (HPow.hPow 2 k))) (HPow.hPow (NNNorm.nnnor …
      ⊢ Eq (NNNorm.nnnorm (HMul.hMul (HPow.hPow (Star.star x) (HPow.hPow 2 k)) (HPow …
    -/
    rw [← star_pow, CStarRing.nnnorm_star_mul_self, ← sq, hk, pow_mul']
    /-
      🎉 no goals
    -/


theorem selfAdjoint.nnnorm_pow_two_pow [NormedRing E] [StarRing E] [CStarRing E] (x : selfAdjoint E)
    (n : ℕ) : ‖x ^ 2 ^ n‖₊ = ‖x‖₊ ^ 2 ^ n :=
  x.prop.nnnorm_pow_two_pow _


/-- `star` bundled as a linear isometric equivalence -/
def starₗᵢ : E ≃ₗᵢ⋆[𝕜] E :=
  { starAddEquiv with
    map_smul' := star_smul
    norm_map' := norm_star }


@[simp]
theorem coe_starₗᵢ : (starₗᵢ 𝕜 : E → E) = star :=
  rfl


theorem starₗᵢ_apply {x : E} : starₗᵢ 𝕜 x = star x :=
  rfl


@[simp]
theorem starₗᵢ_toContinuousLinearEquiv :
    (starₗᵢ 𝕜 : E ≃ₗᵢ⋆[𝕜] E).toContinuousLinearEquiv = (starL 𝕜 : E ≃L⋆[𝕜] E) :=
  ContinuousLinearEquiv.ext rfl


instance toNormedAlgebra {𝕜 A : Type*} [NormedField 𝕜] [StarRing 𝕜] [SeminormedRing A] [StarRing A]
    [NormedAlgebra 𝕜 A] [StarModule 𝕜 A] (S : StarSubalgebra 𝕜 A) : NormedAlgebra 𝕜 S :=
  NormedAlgebra.induced 𝕜 S A S.subtype


instance to_cstarRing {R A} [CommRing R] [StarRing R] [NormedRing A] [StarRing A] [CStarRing A]
    [Algebra R A] [StarModule R A] (S : StarSubalgebra R A) : CStarRing S where
  norm_mul_self_le x := @CStarRing.norm_mul_self_le A _ _ _ x


