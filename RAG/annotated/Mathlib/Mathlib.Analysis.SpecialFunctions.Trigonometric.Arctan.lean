theorem tan_add {x y : ℝ}
    (h : ((∀ k : ℤ, x ≠ (2 * k + 1) * π / 2) ∧ ∀ l : ℤ, y ≠ (2 * l + 1) * π / 2) ∨
      (∃ k : ℤ, x = (2 * k + 1) * π / 2) ∧ ∃ l : ℤ, y = (2 * l + 1) * π / 2) :
    tan (x + y) = (tan x + tan y) / (1 - tan x * tan y) := by
  simpa only [← Complex.ofReal_inj, Complex.ofReal_sub, Complex.ofReal_add, Complex.ofReal_div,
    Complex.ofReal_mul, Complex.ofReal_tan] using
    @Complex.tan_add (x : ℂ) (y : ℂ) (by convert h <;> norm_cast)


theorem tan_add' {x y : ℝ}
    (h : (∀ k : ℤ, x ≠ (2 * k + 1) * π / 2) ∧ ∀ l : ℤ, y ≠ (2 * l + 1) * π / 2) :
    tan (x + y) = (tan x + tan y) / (1 - tan x * tan y) :=
  tan_add (Or.inl h)


theorem tan_two_mul {x : ℝ} : tan (2 * x) = 2 * tan x / (1 - tan x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (Real.tan (HMul.hMul 2 x)) (HDiv.hDiv (HMul.hMul 2 (Real.tan x)) (HSub.hS …
  -/
  have := @Complex.tan_two_mul x
  /-
    x : Real
    this : Eq (Complex.tan (HMul.hMul 2 ↑x)) (HDiv.hDiv (HMul.hMul 2 (Complex.tan  …
    ⊢ Eq (Real.tan (HMul.hMul 2 x)) (HDiv.hDiv (HMul.hMul 2 (Real.tan x)) (HSub.hS …
  -/
  norm_cast at *
  /-
    🎉 no goals
  -/


theorem tan_int_mul_pi_div_two (n : ℤ) : tan (n * π / 2) = 0 :=
                          /-
                            n : Int
                            ⊢ Exists fun k => Eq (HDiv.hDiv (HMul.hMul (↑k) Real.pi) 2) (HDiv.hDiv (HMul.h …
                          -/
  tan_eq_zero_iff.mpr (by use n)
                          /-
                            🎉 no goals
                          -/


theorem continuousOn_tan : ContinuousOn tan {x | cos x ≠ 0} := by
  suffices ContinuousOn (fun x => sin x / cos x) {x | cos x ≠ 0} by
    have h_eq : (fun x => sin x / cos x) = tan := by ext1 x; rw [tan_eq_sin_div_cos]
    rwa [h_eq] at this
  /-
    ⊢ ContinuousOn (fun x => HDiv.hDiv (Real.sin x) (Real.cos x)) (setOf fun x =>  …
  -/
  exact continuousOn_sin.div continuousOn_cos fun x => id
  /-
    🎉 no goals
  -/


@[continuity]
theorem continuous_tan : Continuous fun x : {x | cos x ≠ 0} => tan x :=
  continuousOn_iff_continuous_restrict.1 continuousOn_tan


theorem continuousOn_tan_Ioo : ContinuousOn tan (Ioo (-(π / 2)) (π / 2)) := by
  /-
    ⊢ ContinuousOn Real.tan (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Re …
  -/
  refine ContinuousOn.mono continuousOn_tan fun x => ?_
  /-
    x : Real
    ⊢ Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.pi 2 …
  -/
  simp only [and_imp, mem_Ioo, mem_setOf_eq, Ne]
  /-
    x : Real
    ⊢ LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) x → LT.lt x (HDiv.hDiv Real.pi 2) → No …
  -/
  rw [cos_eq_zero_iff]
  /-
    x : Real
    ⊢ LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) x → LT.lt x (HDiv.hDiv Real.pi 2) → No …
  -/
  rintro hx_gt hx_lt ⟨r, hxr_eq⟩
  /-
    case intro
    x : Real
    hx_gt : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) x
    hx_lt : LT.lt x (HDiv.hDiv Real.pi 2)
    r : Int
    hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
    ⊢ False
  -/
  rcases le_or_lt 0 r with h | h
    /-
      case intro.inl
      x : Real
      hx_gt : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) x
      hx_lt : LT.lt x (HDiv.hDiv Real.pi 2)
      r : Int
      hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
      h : LE.le 0 r
      ⊢ False
    -/
  · rw [lt_iff_not_ge] at hx_lt
    /-
      case intro.inl
      x : Real
      hx_gt : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) x
      hx_lt : Not (GE.ge x (HDiv.hDiv Real.pi 2))
      r : Int
      hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
      h : LE.le 0 r
      ⊢ False
    -/
    refine hx_lt ?_
    /-
      case intro.inl
      x : Real
      hx_gt : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) x
      hx_lt : Not (GE.ge x (HDiv.hDiv Real.pi 2))
      r : Int
      hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
      h : LE.le 0 r
      ⊢ GE.ge x (HDiv.hDiv Real.pi 2)
    -/
    rw [hxr_eq, ← one_mul (π / 2), mul_div_assoc, ge_iff_le, mul_le_mul_right (half_pos pi_pos)]
    /-
      case intro.inl
      x : Real
      hx_gt : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) x
      hx_lt : Not (GE.ge x (HDiv.hDiv Real.pi 2))
      r : Int
      hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
      h : LE.le 0 r
      ⊢ LE.le 1 (HAdd.hAdd (HMul.hMul 2 ↑r) 1)
    -/
    simp [h]
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      x : Real
      hx_gt : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) x
      hx_lt : LT.lt x (HDiv.hDiv Real.pi 2)
      r : Int
      hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
      h : LT.lt r 0
      ⊢ False
    -/
  · rw [lt_iff_not_ge] at hx_gt
    /-
      case intro.inr
      x : Real
      hx_gt : Not (GE.ge (Neg.neg (HDiv.hDiv Real.pi 2)) x)
      hx_lt : LT.lt x (HDiv.hDiv Real.pi 2)
      r : Int
      hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
      h : LT.lt r 0
      ⊢ False
    -/
    refine hx_gt ?_
    rw [hxr_eq, ← one_mul (π / 2), mul_div_assoc, ge_iff_le, neg_mul_eq_neg_mul,
      mul_le_mul_right (half_pos pi_pos)]
    /-
      case intro.inr
      x : Real
      hx_gt : Not (GE.ge (Neg.neg (HDiv.hDiv Real.pi 2)) x)
      hx_lt : LT.lt x (HDiv.hDiv Real.pi 2)
      r : Int
      hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
      h : LT.lt r 0
      ⊢ LE.le (HAdd.hAdd (HMul.hMul 2 ↑r) 1) (-1)
    -/
    have hr_le : r ≤ -1 := by rwa [Int.lt_iff_add_one_le, ← le_neg_iff_add_nonpos_right] at h
    /-
      case intro.inr
      x : Real
      hx_gt : Not (GE.ge (Neg.neg (HDiv.hDiv Real.pi 2)) x)
      hx_lt : LT.lt x (HDiv.hDiv Real.pi 2)
      r : Int
      hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
      h : LT.lt r 0
      hr_le : LE.le r (-1)
      ⊢ LE.le (HAdd.hAdd (HMul.hMul 2 ↑r) 1) (-1)
    -/
    rw [← le_sub_iff_add_le, mul_comm, ← le_div_iff₀]
      /-
        case intro.inr
        x : Real
        hx_gt : Not (GE.ge (Neg.neg (HDiv.hDiv Real.pi 2)) x)
        hx_lt : LT.lt x (HDiv.hDiv Real.pi 2)
        r : Int
        hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
        h : LT.lt r 0
        hr_le : LE.le r (-1)
        ⊢ LE.le (↑r) (HDiv.hDiv (HSub.hSub (-1) 1) 2)
      -/
    · norm_num
      /-
        case intro.inr
        x : Real
        hx_gt : Not (GE.ge (Neg.neg (HDiv.hDiv Real.pi 2)) x)
        hx_lt : LT.lt x (HDiv.hDiv Real.pi 2)
        r : Int
        hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
        h : LT.lt r 0
        hr_le : LE.le r (-1)
        ⊢ LE.le (↑r) (-1)
      -/
      rw [← Int.cast_one, ← Int.cast_neg]; norm_cast
                                           /-
                                             🎉 no goals
                                           -/
      /-
        case intro.inr
        x : Real
        hx_gt : Not (GE.ge (Neg.neg (HDiv.hDiv Real.pi 2)) x)
        hx_lt : LT.lt x (HDiv.hDiv Real.pi 2)
        r : Int
        hxr_eq : Eq x (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑r) 1) Real.pi) 2)
        h : LT.lt r 0
        hr_le : LE.le r (-1)
        ⊢ LT.lt 0 2
      -/
    · exact zero_lt_two
      /-
        🎉 no goals
      -/


theorem surjOn_tan : SurjOn tan (Ioo (-(π / 2)) (π / 2)) univ :=
  have := neg_lt_self pi_div_two_pos
  continuousOn_tan_Ioo.surjOn_of_tendsto (nonempty_Ioo.2 this)
        /-
          this : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.pi 2)
          ⊢ Filter.Tendsto (fun x => Real.tan ↑x) Filter.atBot Filter.atBot
        -/
    (by rw [tendsto_comp_coe_Ioo_atBot this]; exact tendsto_tan_neg_pi_div_two)
                                              /-
                                                🎉 no goals
                                              -/
        /-
          this : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.pi 2)
          ⊢ Filter.Tendsto (fun x => Real.tan ↑x) Filter.atTop Filter.atTop
        -/
    (by rw [tendsto_comp_coe_Ioo_atTop this]; exact tendsto_tan_pi_div_two)
                                              /-
                                                🎉 no goals
                                              -/


theorem tan_surjective : Function.Surjective tan := fun _ => surjOn_tan.subset_range trivial


theorem image_tan_Ioo : tan '' Ioo (-(π / 2)) (π / 2) = univ :=
  univ_subset_iff.1 surjOn_tan


/-- `Real.tan` as an `OrderIso` between `(-(π / 2), π / 2)` and `ℝ`. -/
def tanOrderIso : Ioo (-(π / 2)) (π / 2) ≃o ℝ :=
  (strictMonoOn_tan.orderIso _ _).trans <|
    (OrderIso.setCongr _ _ image_tan_Ioo).trans OrderIso.Set.univ


/-- Inverse of the `tan` function, returns values in the range `-π / 2 < arctan x` and
`arctan x < π / 2` -/
@[pp_nodot]
noncomputable def arctan (x : ℝ) : ℝ :=
  tanOrderIso.symm x


@[simp]
theorem tan_arctan (x : ℝ) : tan (arctan x) = x :=
  tanOrderIso.apply_symm_apply x


theorem arctan_mem_Ioo (x : ℝ) : arctan x ∈ Ioo (-(π / 2)) (π / 2) :=
  Subtype.coe_prop _


@[simp]
theorem range_arctan : range arctan = Ioo (-(π / 2)) (π / 2) :=
  ((EquivLike.surjective _).range_comp _).trans Subtype.range_coe


theorem arctan_tan {x : ℝ} (hx₁ : -(π / 2) < x) (hx₂ : x < π / 2) : arctan (tan x) = x :=
  Subtype.ext_iff.1 <| tanOrderIso.symm_apply_apply ⟨x, hx₁, hx₂⟩


theorem cos_arctan_pos (x : ℝ) : 0 < cos (arctan x) :=
  cos_pos_of_mem_Ioo <| arctan_mem_Ioo x


theorem cos_sq_arctan (x : ℝ) : cos (arctan x) ^ 2 = 1 / (1 + x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (HPow.hPow (Real.cos (Real.arctan x)) 2) (HDiv.hDiv 1 (HAdd.hAdd 1 (HPow. …
  -/
  rw_mod_cast [one_div, ← inv_one_add_tan_sq (cos_arctan_pos x).ne', tan_arctan]
  /-
    🎉 no goals
  -/


theorem sin_arctan (x : ℝ) : sin (arctan x) = x / √(1 + x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (Real.sin (Real.arctan x)) (HDiv.hDiv x (HAdd.hAdd 1 (HPow.hPow x 2)).sqrt)
  -/
  rw_mod_cast [← tan_div_sqrt_one_add_tan_sq (cos_arctan_pos x), tan_arctan]
  /-
    🎉 no goals
  -/


theorem cos_arctan (x : ℝ) : cos (arctan x) = 1 / √(1 + x ^ 2) := by
  /-
    x : Real
    ⊢ Eq (Real.cos (Real.arctan x)) (HDiv.hDiv 1 (HAdd.hAdd 1 (HPow.hPow x 2)).sqrt)
  -/
  rw_mod_cast [one_div, ← inv_sqrt_one_add_tan_sq (cos_arctan_pos x), tan_arctan]
  /-
    🎉 no goals
  -/


theorem arctan_lt_pi_div_two (x : ℝ) : arctan x < π / 2 :=
  (arctan_mem_Ioo x).2


theorem neg_pi_div_two_lt_arctan (x : ℝ) : -(π / 2) < arctan x :=
  (arctan_mem_Ioo x).1


theorem arctan_eq_arcsin (x : ℝ) : arctan x = arcsin (x / √(1 + x ^ 2)) :=
  Eq.symm <| arcsin_eq_of_sin_eq (sin_arctan x) (mem_Icc_of_Ioo <| arctan_mem_Ioo x)


theorem arcsin_eq_arctan {x : ℝ} (h : x ∈ Ioo (-(1 : ℝ)) 1) :
    arcsin x = arctan (x / √(1 - x ^ 2)) := by
  rw_mod_cast [arctan_eq_arcsin, div_pow, sq_sqrt, one_add_div, div_div, ← sqrt_mul,
                                                            /-
                                                              case hb
                                                              x : Real
                                                              h : Membership.mem (Set.Ioo (↑(Int.negSucc 0)) 1) x
                                                              ⊢ Ne (HSub.hSub 1 (HPow.hPow x 2)) 0
                                                            -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
                                                                          /-
                                                                            🎉 no goals
                                                                          -/
    mul_div_cancel₀, sub_add_cancel, sqrt_one, div_one] <;> simp at h <;> nlinarith [h.1, h.2]
                                                                          /-
                                                                            🎉 no goals
                                                                          -/


@[simp]
                                         /-
                                           ⊢ Eq (Real.arctan 0) 0
                                         -/
theorem arctan_zero : arctan 0 = 0 := by simp [arctan_eq_arcsin]
                                         /-
                                           🎉 no goals
                                         -/


@[mono]
theorem arctan_strictMono : StrictMono arctan := tanOrderIso.symm.strictMono


@[gcongr]
lemma arctan_lt_arctan {x y : ℝ} (hxy : x < y) : arctan x < arctan y := arctan_strictMono hxy


@[gcongr]
lemma arctan_le_arctan {x y : ℝ} (hxy : x ≤ y) : arctan x ≤ arctan y :=
  arctan_strictMono.monotone hxy


theorem arctan_injective : arctan.Injective := arctan_strictMono.injective


@[simp]
theorem arctan_eq_zero_iff {x : ℝ} : arctan x = 0 ↔ x = 0 :=
             /-
               x : Real
               ⊢ Iff (Eq (Real.arctan x) 0) (Eq (Real.arctan x) (Real.arctan 0))
             -/
  .trans (by rw [arctan_zero]) arctan_injective.eq_iff
             /-
               🎉 no goals
             -/


theorem tendsto_arctan_atTop : Tendsto arctan atTop (𝓝[<] (π / 2)) :=
  tendsto_Ioo_atTop.mp tanOrderIso.symm.tendsto_atTop


theorem tendsto_arctan_atBot : Tendsto arctan atBot (𝓝[>] (-(π / 2))) :=
  tendsto_Ioo_atBot.mp tanOrderIso.symm.tendsto_atBot


theorem arctan_eq_of_tan_eq {x y : ℝ} (h : tan x = y) (hx : x ∈ Ioo (-(π / 2)) (π / 2)) :
    arctan y = x :=
                                      /-
                                        x y : Real
                                        h : Eq (Real.tan x) y
                                        hx : Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.p …
                                        ⊢ Eq (Real.tan (Real.arctan y)) (Real.tan x)
                                      -/
  injOn_tan (arctan_mem_Ioo _) hx (by rw [tan_arctan, h])
                                      /-
                                        🎉 no goals
                                      -/


@[simp]
theorem arctan_one : arctan 1 = π / 4 :=
                                            /-
                                              ⊢ Membership.mem (Set.Ioo (Neg.neg (HDiv.hDiv Real.pi 2)) (HDiv.hDiv Real.pi 2 …
                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  arctan_eq_of_tan_eq tan_pi_div_four <| by constructor <;> linarith [pi_pos]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
                                                           /-
                                                             x : Real
                                                             ⊢ Eq (Real.arctan (Neg.neg x)) (Neg.neg (Real.arctan x))
                                                           -/
theorem arctan_neg (x : ℝ) : arctan (-x) = -arctan x := by simp [arctan_eq_arcsin, neg_div]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem arctan_eq_arccos {x : ℝ} (h : 0 ≤ x) : arctan x = arccos (√(1 + x ^ 2))⁻¹ := by
  /-
    x : Real
    h : LE.le 0 x
    ⊢ Eq (Real.arctan x) (Real.arccos (Inv.inv (HAdd.hAdd 1 (HPow.hPow x 2)).sqrt))
  -/
  rw [arctan_eq_arcsin, arccos_eq_arcsin]; swap; · exact inv_nonneg.2 (sqrt_nonneg _)
                                                   /-
                                                     🎉 no goals
                                                   -/
  /-
    x : Real
    h : LE.le 0 x
    ⊢ Eq (Real.arcsin (HDiv.hDiv x (HAdd.hAdd 1 (HPow.hPow x 2)).sqrt)) (Real.arcs …
  -/
  congr 1
  rw_mod_cast [← sqrt_inv, sq_sqrt, ← one_div, one_sub_div, add_sub_cancel_left, sqrt_div,
    sqrt_sq h]
  /-
    case e_a.hx
    x : Real
    h : LE.le 0 x
    ⊢ LE.le 0 (HPow.hPow x 2)
  -/
  all_goals positivity
  /-
    🎉 no goals
  -/

-- The junk values for `arccos` and `sqrt` make this true even for `1 < x`.

theorem arccos_eq_arctan {x : ℝ} (h : 0 < x) : arccos x = arctan (√(1 - x ^ 2) / x) := by
  /-
    x : Real
    h : LT.lt 0 x
    ⊢ Eq (Real.arccos x) (Real.arctan (HDiv.hDiv (HSub.hSub 1 (HPow.hPow x 2)).sqr …
  -/
  rw [arccos, eq_comm]
  /-
    x : Real
    h : LT.lt 0 x
    ⊢ Eq (Real.arctan (HDiv.hDiv (HSub.hSub 1 (HPow.hPow x 2)).sqrt x)) (HSub.hSub …
  -/
  refine arctan_eq_of_tan_eq ?_ ⟨?_, ?_⟩
    /-
      case refine_1
      x : Real
      h : LT.lt 0 x
      ⊢ Eq (Real.tan (HSub.hSub (HDiv.hDiv Real.pi 2) (Real.arcsin x))) (HDiv.hDiv ( …
    -/
  · rw_mod_cast [tan_pi_div_two_sub, tan_arcsin, inv_div]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      x : Real
      h : LT.lt 0 x
      ⊢ LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) (HSub.hSub (HDiv.hDiv Real.pi 2) (Real …
    -/
  · linarith only [arcsin_le_pi_div_two x, pi_pos]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      x : Real
      h : LT.lt 0 x
      ⊢ LT.lt (HSub.hSub (HDiv.hDiv Real.pi 2) (Real.arcsin x)) (HDiv.hDiv Real.pi 2)
    -/
  · linarith only [arcsin_pos.2 h]
    /-
      🎉 no goals
    -/


theorem arctan_inv_of_pos {x : ℝ} (h : 0 < x) : arctan x⁻¹ = π / 2 - arctan x := by
  /-
    x : Real
    h : LT.lt 0 x
    ⊢ Eq (Real.arctan (Inv.inv x)) (HSub.hSub (HDiv.hDiv Real.pi 2) (Real.arctan x))
  -/
  rw [← arctan_tan (x := _ - _), tan_pi_div_two_sub, tan_arctan]
    /-
      case hx₁
      x : Real
      h : LT.lt 0 x
      ⊢ LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) (HSub.hSub (HDiv.hDiv Real.pi 2) (Real …
    -/
  · norm_num
    /-
      case hx₁
      x : Real
      h : LT.lt 0 x
      ⊢ LT.lt (Real.arctan x) Real.pi
    -/
    exact (arctan_lt_pi_div_two x).trans (half_lt_self_iff.mpr pi_pos)
    /-
      🎉 no goals
    -/
    /-
      case hx₂
      x : Real
      h : LT.lt 0 x
      ⊢ LT.lt (HSub.hSub (HDiv.hDiv Real.pi 2) (Real.arctan x)) (HDiv.hDiv Real.pi 2)
    -/
  · rw [sub_lt_self_iff, ← arctan_zero]
    /-
      case hx₂
      x : Real
      h : LT.lt 0 x
      ⊢ LT.lt (Real.arctan 0) (Real.arctan x)
    -/
    exact tanOrderIso.symm.strictMono h
    /-
      🎉 no goals
    -/


theorem arctan_inv_of_neg {x : ℝ} (h : x < 0) : arctan x⁻¹ = -(π / 2) - arctan x := by
  /-
    x : Real
    h : LT.lt x 0
    ⊢ Eq (Real.arctan (Inv.inv x)) (HSub.hSub (Neg.neg (HDiv.hDiv Real.pi 2)) (Rea …
  -/
  have := arctan_inv_of_pos (neg_pos.mpr h)
  /-
    x : Real
    h : LT.lt x 0
    this : Eq (Real.arctan (Inv.inv (Neg.neg x))) (HSub.hSub (HDiv.hDiv Real.pi 2) …
    ⊢ Eq (Real.arctan (Inv.inv x)) (HSub.hSub (Neg.neg (HDiv.hDiv Real.pi 2)) (Rea …
  -/
  rwa [inv_neg, arctan_neg, neg_eq_iff_eq_neg, neg_sub', arctan_neg, neg_neg] at this
  /-
    🎉 no goals
  -/


lemma arctan_ne_mul_pi_div_two {x : ℝ} : ∀ (k : ℤ), arctan x ≠ (2 * k + 1) * π / 2 := by
  /-
    x : Real
    ⊢ ∀ (k : Int), Ne (Real.arctan x) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul  …
  -/
  by_contra!
  /-
    x : Real
    this : Exists fun k => Eq (Real.arctan x) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HM …
    ⊢ False
  -/
  obtain ⟨k, h⟩ := this
  /-
    case intro
    x : Real
    k : Int
    h : Eq (Real.arctan x) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) 1) Re …
    ⊢ False
  -/
  obtain ⟨lb, ub⟩ := arctan_mem_Ioo x
  /-
    case intro.intro
    x : Real
    k : Int
    h : Eq (Real.arctan x) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) 1) Re …
    lb : LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) (Real.arctan x)
    ub : LT.lt (Real.arctan x) (HDiv.hDiv Real.pi 2)
    ⊢ False
  -/
  rw [h, neg_eq_neg_one_mul, mul_div_assoc, mul_lt_mul_right (by positivity)] at lb
  /-
    case intro.intro
    x : Real
    k : Int
    h : Eq (Real.arctan x) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) 1) Re …
    lb : LT.lt (-1) (HAdd.hAdd (HMul.hMul 2 ↑k) 1)
    ub : LT.lt (Real.arctan x) (HDiv.hDiv Real.pi 2)
    ⊢ False
  -/
  rw [h, ← one_mul (π / 2), mul_div_assoc, mul_lt_mul_right (by positivity)] at ub
  /-
    case intro.intro
    x : Real
    k : Int
    h : Eq (Real.arctan x) (HDiv.hDiv (HMul.hMul (HAdd.hAdd (HMul.hMul 2 ↑k) 1) Re …
    lb : LT.lt (-1) (HAdd.hAdd (HMul.hMul 2 ↑k) 1)
    ub : LT.lt (HAdd.hAdd (HMul.hMul 2 ↑k) 1) 1
    ⊢ False
  -/
  norm_cast at lb ub; change -1 < _ at lb; omega
                                           /-
                                             🎉 no goals
                                           -/


lemma arctan_add_arctan_lt_pi_div_two {x y : ℝ} (h : x * y < 1) : arctan x + arctan y < π / 2 := by
  /-
    x y : Real
    h : LT.lt (HMul.hMul x y) 1
    ⊢ LT.lt (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HDiv.hDiv Real.pi 2)
  -/
  cases' le_or_lt y 0 with hy hy
    /-
      case inl
      x y : Real
      h : LT.lt (HMul.hMul x y) 1
      hy : LE.le y 0
      ⊢ LT.lt (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HDiv.hDiv Real.pi 2)
    -/
  · rw [← add_zero (π / 2), ← arctan_zero]
    /-
      case inl
      x y : Real
      h : LT.lt (HMul.hMul x y) 1
      hy : LE.le y 0
      ⊢ LT.lt (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HAdd.hAdd (HDiv.hDiv Real …
    -/
    exact add_lt_add_of_lt_of_le (arctan_lt_pi_div_two _) (tanOrderIso.symm.monotone hy)
    /-
      🎉 no goals
    -/
    /-
      case inr
      x y : Real
      h : LT.lt (HMul.hMul x y) 1
      hy : LT.lt 0 y
      ⊢ LT.lt (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HDiv.hDiv Real.pi 2)
    -/
  · rw [← lt_div_iff₀ hy, ← inv_eq_one_div] at h
    /-
      case inr
      x y : Real
      h : LT.lt x (Inv.inv y)
      hy : LT.lt 0 y
      ⊢ LT.lt (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HDiv.hDiv Real.pi 2)
    -/
    replace h : arctan x < arctan y⁻¹ := tanOrderIso.symm.strictMono h
    /-
      case inr
      x y : Real
      hy : LT.lt 0 y
      h : LT.lt (Real.arctan x) (Real.arctan (Inv.inv y))
      ⊢ LT.lt (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HDiv.hDiv Real.pi 2)
    -/
    rwa [arctan_inv_of_pos hy, lt_tsub_iff_right] at h
    /-
      🎉 no goals
    -/


theorem arctan_add {x y : ℝ} (h : x * y < 1) :
    arctan x + arctan y = arctan ((x + y) / (1 - x * y)) := by
  /-
    x y : Real
    h : LT.lt (HMul.hMul x y) 1
    ⊢ Eq (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (Real.arctan (HDiv.hDiv (HAdd …
  -/
  rw [← arctan_tan (x := _ + _)]
    /-
      x y : Real
      h : LT.lt (HMul.hMul x y) 1
      ⊢ Eq (Real.arctan (Real.tan (HAdd.hAdd (Real.arctan x) (Real.arctan y)))) (Rea …
    -/
  · congr
    /-
      case e_x
      x y : Real
      h : LT.lt (HMul.hMul x y) 1
      ⊢ Eq (Real.tan (HAdd.hAdd (Real.arctan x) (Real.arctan y))) (HDiv.hDiv (HAdd.h …
    -/
    conv_rhs => rw [← tan_arctan x, ← tan_arctan y]
    /-
      case e_x
      x y : Real
      h : LT.lt (HMul.hMul x y) 1
      ⊢ Eq (Real.tan (HAdd.hAdd (Real.arctan x) (Real.arctan y))) (HDiv.hDiv (HAdd.h …
    -/
    exact tan_add' ⟨arctan_ne_mul_pi_div_two, arctan_ne_mul_pi_div_two⟩
    /-
      🎉 no goals
    -/
    /-
      case hx₁
      x y : Real
      h : LT.lt (HMul.hMul x y) 1
      ⊢ LT.lt (Neg.neg (HDiv.hDiv Real.pi 2)) (HAdd.hAdd (Real.arctan x) (Real.arcta …
    -/
  · rw [neg_lt, neg_add, ← arctan_neg, ← arctan_neg]
    /-
      case hx₁
      x y : Real
      h : LT.lt (HMul.hMul x y) 1
      ⊢ LT.lt (HAdd.hAdd (Real.arctan (Neg.neg x)) (Real.arctan (Neg.neg y))) (HDiv. …
    -/
    rw [← neg_mul_neg] at h
    /-
      case hx₁
      x y : Real
      h : LT.lt (HMul.hMul (Neg.neg x) (Neg.neg y)) 1
      ⊢ LT.lt (HAdd.hAdd (Real.arctan (Neg.neg x)) (Real.arctan (Neg.neg y))) (HDiv. …
    -/
    exact arctan_add_arctan_lt_pi_div_two h
    /-
      🎉 no goals
    -/
    /-
      case hx₂
      x y : Real
      h : LT.lt (HMul.hMul x y) 1
      ⊢ LT.lt (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HDiv.hDiv Real.pi 2)
    -/
  · exact arctan_add_arctan_lt_pi_div_two h
    /-
      🎉 no goals
    -/


theorem arctan_add_eq_add_pi {x y : ℝ} (h : 1 < x * y) (hx : 0 < x) :
    arctan x + arctan y = arctan ((x + y) / (1 - x * y)) + π := by
  have hy : 0 < y := by
    have := mul_pos_iff.mp (zero_lt_one.trans h)
    simpa [hx, hx.asymm]
  /-
    x y : Real
    h : LT.lt 1 (HMul.hMul x y)
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    ⊢ Eq (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HAdd.hAdd (Real.arctan (HDiv …
  -/
  have k := arctan_add (mul_inv x y ▸ inv_lt_one_of_one_lt₀ h)
  rw [arctan_inv_of_pos hx, arctan_inv_of_pos hy, show _ + _ = π - (arctan x + arctan y) by ring,
    sub_eq_iff_eq_add, ← sub_eq_iff_eq_add', sub_eq_add_neg, ← arctan_neg, add_comm] at k
  /-
    x y : Real
    h : LT.lt 1 (HMul.hMul x y)
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    k : Eq (HAdd.hAdd (Real.arctan (Neg.neg (HDiv.hDiv (HAdd.hAdd (Inv.inv x) (Inv …
    ⊢ Eq (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HAdd.hAdd (Real.arctan (HDiv …
  -/
  convert k.symm using 3
  /-
    case h.e'_3.h.e'_5.h.e'_1
    x y : Real
    h : LT.lt 1 (HMul.hMul x y)
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    k : Eq (HAdd.hAdd (Real.arctan (Neg.neg (HDiv.hDiv (HAdd.hAdd (Inv.inv x) (Inv …
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd x y) (HSub.hSub 1 (HMul.hMul x y))) (Neg.neg (HDiv. …
  -/
  field_simp
  /-
    case h.e'_3.h.e'_5.h.e'_1
    x y : Real
    h : LT.lt 1 (HMul.hMul x y)
    hx : LT.lt 0 x
    hy : LT.lt 0 y
    k : Eq (HAdd.hAdd (Real.arctan (Neg.neg (HDiv.hDiv (HAdd.hAdd (Inv.inv x) (Inv …
    ⊢ Eq (HDiv.hDiv (HAdd.hAdd x y) (HSub.hSub 1 (HMul.hMul x y))) (HDiv.hDiv (HAd …
  -/
  rw [show -x + -y = -(x + y) by ring, show x * y - 1 = -(1 - x * y) by ring, neg_div_neg_eq]
  /-
    🎉 no goals
  -/


theorem arctan_add_eq_sub_pi {x y : ℝ} (h : 1 < x * y) (hx : x < 0) :
    arctan x + arctan y = arctan ((x + y) / (1 - x * y)) - π := by
  /-
    x y : Real
    h : LT.lt 1 (HMul.hMul x y)
    hx : LT.lt x 0
    ⊢ Eq (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HSub.hSub (Real.arctan (HDiv …
  -/
  rw [← neg_mul_neg] at h
  /-
    x y : Real
    h : LT.lt 1 (HMul.hMul (Neg.neg x) (Neg.neg y))
    hx : LT.lt x 0
    ⊢ Eq (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HSub.hSub (Real.arctan (HDiv …
  -/
  have k := arctan_add_eq_add_pi h (neg_pos.mpr hx)
  /-
    x y : Real
    h : LT.lt 1 (HMul.hMul (Neg.neg x) (Neg.neg y))
    hx : LT.lt x 0
    k : Eq (HAdd.hAdd (Real.arctan (Neg.neg x)) (Real.arctan (Neg.neg y))) (HAdd.h …
    ⊢ Eq (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HSub.hSub (Real.arctan (HDiv …
  -/
  rw [show _ / _ = -((x + y) / (1 - x * y)) by ring, ← neg_inj] at k
  /-
    x y : Real
    h : LT.lt 1 (HMul.hMul (Neg.neg x) (Neg.neg y))
    hx : LT.lt x 0
    k : Eq (Neg.neg (HAdd.hAdd (Real.arctan (Neg.neg x)) (Real.arctan (Neg.neg y)) …
    ⊢ Eq (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HSub.hSub (Real.arctan (HDiv …
  -/
  simp only [arctan_neg, neg_add, neg_neg, ← sub_eq_add_neg _ π] at k
  /-
    x y : Real
    h : LT.lt 1 (HMul.hMul (Neg.neg x) (Neg.neg y))
    hx : LT.lt x 0
    k : Eq (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HSub.hSub (Real.arctan (HD …
    ⊢ Eq (HAdd.hAdd (Real.arctan x) (Real.arctan y)) (HSub.hSub (Real.arctan (HDiv …
  -/
  exact k
  /-
    🎉 no goals
  -/


theorem two_mul_arctan {x : ℝ} (h₁ : -1 < x) (h₂ : x < 1) :
    2 * arctan x = arctan (2 * x / (1 - x ^ 2)) := by
  /-
    x : Real
    h₁ : LT.lt (-1) x
    h₂ : LT.lt x 1
    ⊢ Eq (HMul.hMul 2 (Real.arctan x)) (Real.arctan (HDiv.hDiv (HMul.hMul 2 x) (HS …
  -/
  rw [two_mul, arctan_add (by nlinarith)]; congr 1; ring
                                                    /-
                                                      🎉 no goals
                                                    -/


theorem two_mul_arctan_add_pi {x : ℝ} (h : 1 < x) :
    2 * arctan x = arctan (2 * x / (1 - x ^ 2)) + π := by
  /-
    x : Real
    h : LT.lt 1 x
    ⊢ Eq (HMul.hMul 2 (Real.arctan x)) (HAdd.hAdd (Real.arctan (HDiv.hDiv (HMul.hM …
  -/
  rw [two_mul, arctan_add_eq_add_pi (by nlinarith) (by linarith)]; congr 2; ring
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem two_mul_arctan_sub_pi {x : ℝ} (h : x < -1) :
    2 * arctan x = arctan (2 * x / (1 - x ^ 2)) - π := by
  /-
    x : Real
    h : LT.lt x (-1)
    ⊢ Eq (HMul.hMul 2 (Real.arctan x)) (HSub.hSub (Real.arctan (HDiv.hDiv (HMul.hM …
  -/
  rw [two_mul, arctan_add_eq_sub_pi (by nlinarith) (by linarith)]; congr 2; ring
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


theorem arctan_inv_2_add_arctan_inv_3 : arctan 2⁻¹ + arctan 3⁻¹ = π / 4 := by
  /-
    ⊢ Eq (HAdd.hAdd (Real.arctan (Inv.inv 2)) (Real.arctan (Inv.inv 3))) (HDiv.hDi …
  -/
                      /-
                        🎉 no goals
                      -/
  rw [arctan_add] <;> norm_num
                      /-
                        🎉 no goals
                      -/


theorem two_mul_arctan_inv_2_sub_arctan_inv_7 : 2 * arctan 2⁻¹ - arctan 7⁻¹ = π / 4 := by
  /-
    ⊢ Eq (HSub.hSub (HMul.hMul 2 (Real.arctan (Inv.inv 2))) (Real.arctan (Inv.inv  …
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  rw [two_mul_arctan, ← arctan_one, sub_eq_iff_eq_add, arctan_add] <;> norm_num
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


theorem two_mul_arctan_inv_3_add_arctan_inv_7 : 2 * arctan 3⁻¹ + arctan 7⁻¹ = π / 4 := by
  /-
    ⊢ Eq (HAdd.hAdd (HMul.hMul 2 (Real.arctan (Inv.inv 3))) (Real.arctan (Inv.inv  …
  -/
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
                                      /-
                                        🎉 no goals
                                      -/
  rw [two_mul_arctan, arctan_add] <;> norm_num
                                      /-
                                        🎉 no goals
                                      -/


/-- **John Machin's 1706 formula**, which he used to compute π to 100 decimal places. -/
theorem four_mul_arctan_inv_5_sub_arctan_inv_239 : 4 * arctan 5⁻¹ - arctan 239⁻¹ = π / 4 := by
  rw [show 4 * arctan _ = 2 * (2 * _) by ring, two_mul_arctan, two_mul_arctan, ← arctan_one,
                                       /-
                                         ⊢ Eq (Real.arctan (HDiv.hDiv (HMul.hMul 2 (HDiv.hDiv (HMul.hMul 2 (Inv.inv 5)) …
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
                                       /-
                                         🎉 no goals
                                       -/
    sub_eq_iff_eq_add, arctan_add] <;> norm_num
                                       /-
                                         🎉 no goals
                                       -/


@[continuity]
theorem continuous_arctan : Continuous arctan :=
  continuous_subtype_val.comp tanOrderIso.toHomeomorph.continuous_invFun


theorem continuousAt_arctan {x : ℝ} : ContinuousAt arctan x :=
  continuous_arctan.continuousAt


/-- `Real.tan` as a `PartialHomeomorph` between `(-(π / 2), π / 2)` and the whole line. -/
def tanPartialHomeomorph : PartialHomeomorph ℝ ℝ where
  toFun := tan
  invFun := arctan
  source := Ioo (-(π / 2)) (π / 2)
  target := univ
  map_source' := mapsTo_univ _ _
  map_target' y _ := arctan_mem_Ioo y
  left_inv' _ hx := arctan_tan hx.1 hx.2
  right_inv' y _ := tan_arctan y
  open_source := isOpen_Ioo
  open_target := isOpen_univ
  continuousOn_toFun := continuousOn_tan_Ioo
  continuousOn_invFun := continuous_arctan.continuousOn


@[simp]
theorem coe_tanPartialHomeomorph : ⇑tanPartialHomeomorph = tan :=
  rfl


@[simp]
theorem coe_tanPartialHomeomorph_symm : ⇑tanPartialHomeomorph.symm = arctan :=
  rfl


