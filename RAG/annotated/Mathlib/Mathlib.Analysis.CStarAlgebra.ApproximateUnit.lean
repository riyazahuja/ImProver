local notation "σₙ" => quasispectrum

local notation "σ" => spectrum


lemma CFC.monotoneOn_one_sub_one_add_inv :
    MonotoneOn (cfcₙ (fun x : ℝ≥0 ↦ 1 - (1 + x)⁻¹)) (Set.Ici (0 : A)) := by
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    ⊢ MonotoneOn (cfcₙ fun x => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 x))) (Set.Ici 0)
  -/
  intro a ha b hb hab
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a : A
    ha : Membership.mem (Set.Ici 0) a
    b : A
    hb : Membership.mem (Set.Ici 0) b
    hab : LE.le a b
    ⊢ LE.le (cfcₙ (fun x => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 x))) a) (cfcₙ (fun x …
  -/
  simp only [Set.mem_Ici] at ha hb
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a b : A
    hab : LE.le a b
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ LE.le (cfcₙ (fun x => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 x))) a) (cfcₙ (fun x …
  -/
  rw [← inr_le_iff .., nnreal_cfcₙ_eq_cfc_inr a _, nnreal_cfcₙ_eq_cfc_inr b _]
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a b : A
    hab : LE.le a b
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ LE.le (cfc (fun x => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 x))) ↑a) (cfc (fun x  …
  -/
  rw [← inr_le_iff a b (.of_nonneg ha) (.of_nonneg hb)] at hab
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a b : A
    hab : LE.le ↑a ↑b
    ha : LE.le 0 a
    hb : LE.le 0 b
    ⊢ LE.le (cfc (fun x => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 x))) ↑a) (cfc (fun x  …
  -/
  rw [← inr_nonneg_iff] at ha hb
  have h_cfc_one_sub (c : A⁺¹) (hc : 0 ≤ c := by cfc_tac) :
      cfc (fun x : ℝ≥0 ↦ 1 - (1 + x)⁻¹) c = 1 - cfc (·⁻¹ : ℝ≥0 → ℝ≥0) (1 + c) := by
    rw [cfc_tsub _ _ _ (fun x _ ↦ by simp) (hg := by fun_prop (disch := intro _ _; positivity)),
      cfc_const_one ℝ≥0 c, cfc_comp' (·⁻¹) (1 + ·) c ?_, cfc_add .., cfc_const_one ℝ≥0 c,
      cfc_id' ℝ≥0 c]
    exact continuousOn_id.inv₀ (Set.forall_mem_image.mpr fun x _ ↦ by dsimp only [id]; positivity)
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a b : A
    hab : LE.le ↑a ↑b
    ha : LE.le 0 ↑a
    hb : LE.le 0 ↑b
    h_cfc_one_sub : ∀ (c : Unitization Complex A), autoParam (LE.le 0 c) _auto✝ →  …
    ⊢ LE.le (cfc (fun x => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 x))) ↑a) (cfc (fun x  …
  -/
  rw [h_cfc_one_sub (a : A⁺¹), h_cfc_one_sub (b : A⁺¹)]
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a b : A
    hab : LE.le ↑a ↑b
    ha : LE.le 0 ↑a
    hb : LE.le 0 ↑b
    h_cfc_one_sub : ∀ (c : Unitization Complex A), autoParam (LE.le 0 c) _auto✝ →  …
    ⊢ LE.le (HSub.hSub 1 (cfc (fun x => Inv.inv x) (HAdd.hAdd 1 ↑a))) (HSub.hSub 1 …
  -/
  gcongr
  /-
    case h
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    a b : A
    hab : LE.le ↑a ↑b
    ha : LE.le 0 ↑a
    hb : LE.le 0 ↑b
    h_cfc_one_sub : ∀ (c : Unitization Complex A), autoParam (LE.le 0 c) _auto✝ →  …
    ⊢ LE.le (cfc (fun x => Inv.inv x) (HAdd.hAdd 1 ↑b)) (cfc (fun x => Inv.inv x)  …
  -/
  rw [← CFC.rpow_neg_one_eq_cfc_inv, ← CFC.rpow_neg_one_eq_cfc_inv]
  exact rpow_neg_one_le_rpow_neg_one (add_nonneg zero_le_one ha) (by gcongr) <|
    isUnit_of_le isUnit_one zero_le_one <| le_add_of_nonneg_right ha


lemma Set.InvOn.one_sub_one_add_inv : Set.InvOn (fun x ↦ 1 - (1 + x)⁻¹) (fun x ↦ x * (1 - x)⁻¹)
    {x : ℝ≥0 | x < 1} {x : ℝ≥0 | x < 1} := by
  have : (fun x : ℝ≥0 ↦ x * (1 + x)⁻¹) = fun x ↦ 1 - (1 + x)⁻¹ := by
    ext x : 1
    field_simp
    simp [tsub_mul, inv_mul_cancel₀]
  /-
    this : Eq (fun x => HMul.hMul x (Inv.inv (HAdd.hAdd 1 x))) fun x => HSub.hSub  …
    ⊢ Set.InvOn (fun x => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 x))) (fun x => HMul.hM …
  -/
  rw [← this]
  /-
    this : Eq (fun x => HMul.hMul x (Inv.inv (HAdd.hAdd 1 x))) fun x => HSub.hSub  …
    ⊢ Set.InvOn (fun x => HMul.hMul x (Inv.inv (HAdd.hAdd 1 x))) (fun x => HMul.hM …
  -/
  constructor <;> intro x (hx : x < 1)
    /-
      case left
      this : Eq (fun x => HMul.hMul x (Inv.inv (HAdd.hAdd 1 x))) fun x => HSub.hSub  …
      x : NNReal
      hx : LT.lt x 1
      ⊢ Eq ((fun x => HMul.hMul x (Inv.inv (HAdd.hAdd 1 x))) ((fun x => HMul.hMul x  …
    -/
  · have : 0 < 1 - x := tsub_pos_of_lt hx
    /-
      case left
      this✝ : Eq (fun x => HMul.hMul x (Inv.inv (HAdd.hAdd 1 x))) fun x => HSub.hSub …
      x : NNReal
      hx : LT.lt x 1
      this : LT.lt 0 (HSub.hSub 1 x)
      ⊢ Eq ((fun x => HMul.hMul x (Inv.inv (HAdd.hAdd 1 x))) ((fun x => HMul.hMul x  …
    -/
    field_simp [tsub_add_cancel_of_le hx.le, tsub_tsub_cancel_of_le hx.le]
    /-
      🎉 no goals
    -/
    /-
      case right
      this : Eq (fun x => HMul.hMul x (Inv.inv (HAdd.hAdd 1 x))) fun x => HSub.hSub  …
      x : NNReal
      hx : LT.lt x 1
      ⊢ Eq ((fun x => HMul.hMul x (Inv.inv (HSub.hSub 1 x))) ((fun x => HMul.hMul x  …
    -/
  · field_simp [mul_tsub]
    /-
      🎉 no goals
    -/


lemma norm_cfcₙ_one_sub_one_add_inv_lt_one (a : A) :
    ‖cfcₙ (fun x : ℝ≥0 ↦ 1 - (1 + x)⁻¹) a‖ < 1 :=
                                                               /-
                                                                 A : Type u_1
                                                                 inst✝² : NonUnitalCStarAlgebra A
                                                                 inst✝¹ : PartialOrder A
                                                                 inst✝ : StarOrderedRing A
                                                                 a : A
                                                                 x : NNReal
                                                                 x✝ : Membership.mem (quasispectrum NNReal a) x
                                                                 ⊢ LT.lt 0 (Inv.inv (HAdd.hAdd 1 x))
                                                               -/
  nnnorm_cfcₙ_nnreal_lt fun x _ ↦ tsub_lt_self zero_lt_one (by positivity)
                                                               /-
                                                                 🎉 no goals
                                                               -/

-- the calls to `fun_prop` with a discharger set off the linter

set_option linter.style.multiGoal false in
lemma CStarAlgebra.directedOn_nonneg_ball :
    DirectedOn (· ≤ ·) ({x : A | 0 ≤ x} ∩ Metric.ball 0 1) := by
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    ⊢ DirectedOn (fun x1 x2 => LE.le x1 x2) (Inter.inter (setOf fun x => LE.le 0 x …
  -/
  let f : ℝ≥0 → ℝ≥0 := fun x => 1 - (1 + x)⁻¹
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    f : NNReal → NNReal := fun x => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 x))
    ⊢ DirectedOn (fun x1 x2 => LE.le x1 x2) (Inter.inter (setOf fun x => LE.le 0 x …
  -/
  let g : ℝ≥0 → ℝ≥0 := fun x => x * (1 - x)⁻¹
  suffices ∀ a b : A, 0 ≤ a → 0 ≤ b → ‖a‖ < 1 → ‖b‖ < 1 →
      a ≤ cfcₙ f (cfcₙ g a + cfcₙ g b) by
    rintro a ⟨(ha₁ : 0 ≤ a), ha₂⟩ b ⟨(hb₁ : 0 ≤ b), hb₂⟩
    simp only [Metric.mem_ball, dist_zero_right] at ha₂ hb₂
    refine ⟨cfcₙ f (cfcₙ g a + cfcₙ g b), ⟨by simp, ?_⟩, ?_, ?_⟩
    · simpa only [Metric.mem_ball, dist_zero_right] using norm_cfcₙ_one_sub_one_add_inv_lt_one _
    · exact this a b ha₁ hb₁ ha₂ hb₂
    · exact add_comm (cfcₙ g a) (cfcₙ g b) ▸ this b a hb₁ ha₁ hb₂ ha₂
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    f : NNReal → NNReal := fun x => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 x))
    g : NNReal → NNReal := fun x => HMul.hMul x (Inv.inv (HSub.hSub 1 x))
    ⊢ ∀ (a b : A), LE.le 0 a → LE.le 0 b → LT.lt (Norm.norm a) 1 → LT.lt (Norm.nor …
  -/
  rintro a b ha₁ - ha₂ -
  calc
    a = cfcₙ (f ∘ g) a := by
      conv_lhs => rw [← cfcₙ_id ℝ≥0 a]
      refine cfcₙ_congr (Set.InvOn.one_sub_one_add_inv.1.eqOn.symm.mono fun x hx ↦ ?_)
      exact lt_of_le_of_lt (le_nnnorm_of_mem_quasispectrum hx) ha₂
    _ = cfcₙ f (cfcₙ g a) := by
      rw [cfcₙ_comp f g a ?_ (by simp [f, tsub_self]) ?_ (by simp [g]) ha₁]
      · fun_prop (disch := intro _ _; positivity)
      · have (x) (hx : x ∈ σₙ ℝ≥0 a) :  1 - x ≠ 0 := by
          refine tsub_pos_of_lt ?_ |>.ne'
          exact lt_of_le_of_lt (le_nnnorm_of_mem_quasispectrum hx) ha₂
        fun_prop (disch := assumption)
    _ ≤ cfcₙ f (cfcₙ g a + cfcₙ g b) := by
      have hab' : cfcₙ g a ≤ cfcₙ g a + cfcₙ g b := le_add_of_nonneg_right cfcₙ_nonneg_of_predicate
      exact CFC.monotoneOn_one_sub_one_add_inv cfcₙ_nonneg_of_predicate
        (cfcₙ_nonneg_of_predicate.trans hab') hab'


/-- An *increasing approximate unit* in a C⋆-algebra is an approximate unit contained in the
closed unit ball of nonnegative elements. -/
structure Filter.IsIncreasingApproximateUnit (l : Filter A) extends l.IsApproximateUnit : Prop where
  eventually_nonneg : ∀ᶠ x in l, 0 ≤ x
  eventually_norm : ∀ᶠ x in l, ‖x‖ ≤ 1


omit [StarOrderedRing A] in
lemma eventually_nnnorm {l : Filter A} (hl : l.IsIncreasingApproximateUnit) :
    ∀ᶠ x in l, ‖x‖₊ ≤ 1 :=
  hl.eventually_norm


lemma eventually_isSelfAdjoint {l : Filter A} (hl : l.IsIncreasingApproximateUnit) :
    ∀ᶠ x in l, IsSelfAdjoint x :=
  hl.eventually_nonneg.mp <| .of_forall fun _ ↦ IsSelfAdjoint.of_nonneg


lemma eventually_star_eq {l : Filter A} (hl : l.IsIncreasingApproximateUnit) :
    ∀ᶠ x in l, star x = x :=
  hl.eventually_isSelfAdjoint.mp <| .of_forall fun _ ↦ IsSelfAdjoint.star_eq


open Submodule in
/-- To show that `l` is a one-sided approximate unit for `A`, it suffices to verify it only for
`m : A` with `0 ≤ m` and `‖m‖ < 1`. -/
lemma tendsto_mul_right_of_forall_nonneg_tendsto {l : Filter A}
    (h : ∀ m, 0 ≤ m → ‖m‖ < 1 → Tendsto (· * m) l (𝓝 m)) (m : A) :
    Tendsto (· * m) l (𝓝 m) := by
  obtain ⟨n, c, x, rfl⟩ := mem_span_set'.mp <| by
    show m ∈ span ℂ ({x | 0 ≤ x} ∩ ball 0 1)
    simp [span_nonneg_inter_unitBall]
  /-
    case intro.intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    l : Filter A
    h : ∀ (m : A), LE.le 0 m → LT.lt (Norm.norm m) 1 → Filter.Tendsto (fun x => HM …
    n : Nat
    c : Fin n → Complex
    x : Fin n → ↑(Inter.inter (setOf fun x => LE.le 0 x) (Metric.ball 0 1))
    ⊢ Filter.Tendsto (fun x_1 => HMul.hMul x_1 (Finset.univ.sum fun i => HSMul.hSM …
  -/
  simp_rw [Finset.mul_sum]
  /-
    case intro.intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    l : Filter A
    h : ∀ (m : A), LE.le 0 m → LT.lt (Norm.norm m) 1 → Filter.Tendsto (fun x => HM …
    n : Nat
    c : Fin n → Complex
    x : Fin n → ↑(Inter.inter (setOf fun x => LE.le 0 x) (Metric.ball 0 1))
    ⊢ Filter.Tendsto (fun x_1 => Finset.univ.sum fun i => HMul.hMul x_1 (HSMul.hSM …
  -/
  refine tendsto_finset_sum _ fun i _ ↦ ?_
  /-
    case intro.intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    l : Filter A
    h : ∀ (m : A), LE.le 0 m → LT.lt (Norm.norm m) 1 → Filter.Tendsto (fun x => HM …
    n : Nat
    c : Fin n → Complex
    x : Fin n → ↑(Inter.inter (setOf fun x => LE.le 0 x) (Metric.ball 0 1))
    i : Fin n
    x✝ : Membership.mem Finset.univ i
    ⊢ Filter.Tendsto (fun x_1 => HMul.hMul x_1 (HSMul.hSMul (c i) ↑(x i))) l (nhds …
  -/
  simp_rw [mul_smul_comm]
  /-
    case intro.intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    l : Filter A
    h : ∀ (m : A), LE.le 0 m → LT.lt (Norm.norm m) 1 → Filter.Tendsto (fun x => HM …
    n : Nat
    c : Fin n → Complex
    x : Fin n → ↑(Inter.inter (setOf fun x => LE.le 0 x) (Metric.ball 0 1))
    i : Fin n
    x✝ : Membership.mem Finset.univ i
    ⊢ Filter.Tendsto (fun x_1 => HSMul.hSMul (c i) (HMul.hMul x_1 ↑(x i))) l (nhds …
  -/
  exact tendsto_const_nhds.smul <| h (x i) (x i).2.1 <| by simpa using (x i).2.2
  /-
    🎉 no goals
  -/


omit [PartialOrder A] in
/-- Multiplication on the left by `m` tends to `𝓝 m` if and only if multiplication on the right
does, provided the elements are eventually selfadjoint along the filter `l`. -/
lemma tendsto_mul_left_iff_tendsto_mul_right {l : Filter A} (hl : ∀ᶠ x in l, IsSelfAdjoint x) :
    (∀ m, Tendsto (m * ·) l (𝓝 m)) ↔ (∀ m, Tendsto (· * m) l (𝓝 m)) := by
  /-
    A : Type u_1
    inst✝ : NonUnitalCStarAlgebra A
    l : Filter A
    hl : Filter.Eventually (fun x => IsSelfAdjoint x) l
    ⊢ Iff (∀ (m : A), Filter.Tendsto (fun x => HMul.hMul m x) l (nhds m)) (∀ (m :  …
  -/
  refine ⟨fun h m ↦ ?_, fun h m ↦ ?_⟩
  all_goals
    apply (star_star m ▸ (continuous_star.tendsto _ |>.comp <| h (star m))).congr'
    filter_upwards [hl] with x hx
    simp [hx.star_eq]


/-- The sections of positive strict contractions form a filter basis. -/
lemma isBasis_nonneg_sections :
    IsBasis (fun x : A ↦ 0 ≤ x ∧ ‖x‖ < 1) ({x | · ≤ x}) where
                     /-
                       A : Type u_1
                       inst✝² : NonUnitalCStarAlgebra A
                       inst✝¹ : PartialOrder A
                       inst✝ : StarOrderedRing A
                       ⊢ And (LE.le 0 0) (LT.lt (Norm.norm 0) 1)
                     -/
  nonempty := ⟨0, by simp⟩
                     /-
                       🎉 no goals
                     -/
  inter {x y} hx hy := by
    /-
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      x y : A
      hx : And (LE.le 0 x) (LT.lt (Norm.norm x) 1)
      hy : And (LE.le 0 y) (LT.lt (Norm.norm y) 1)
      ⊢ Exists fun k => And (And (LE.le 0 k) (LT.lt (Norm.norm k) 1)) (HasSubset.Sub …
    -/
    peel directedOn_nonneg_ball x (by simpa) y (by simpa) with z hz
    /-
      case h
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      x y : A
      hx : And (LE.le 0 x) (LT.lt (Norm.norm x) 1)
      hy : And (LE.le 0 y) (LT.lt (Norm.norm y) 1)
      z : A
      hz : And (Membership.mem (Inter.inter (setOf fun x => LE.le 0 x) (Metric.ball  …
      ⊢ And (And (LE.le 0 z) (LT.lt (Norm.norm z) 1)) (HasSubset.Subset (setOf fun x …
    -/
    exact ⟨by simpa using hz.1, fun a ha ↦ ⟨hz.2.1.trans ha, hz.2.2.trans ha⟩⟩
    /-
      🎉 no goals
    -/


/-- The canonical approximate unit in a C⋆-algebra generated by the basis of sets
`{x | a ≤ x} ∩ closedBall 0 1` for `0 ≤ a`. See also `CStarAlgebra.hasBasis_approximateUnit`. -/
def approximateUnit : Filter A :=
  (isBasis_nonneg_sections A).filter ⊓ 𝓟 (closedBall 0 1)


/-- The canonical approximate unit in a C⋆-algebra has a basis of sets
`{x | a ≤ x} ∩ closedBall 0 1` for `0 ≤ a`. -/
lemma hasBasis_approximateUnit :
    (approximateUnit A).HasBasis (fun x : A ↦ 0 ≤ x ∧ ‖x‖ < 1) ({x | · ≤ x} ∩ closedBall 0 1) :=
  isBasis_nonneg_sections A |>.hasBasis.inf_principal (closedBall 0 1)


/-- This is a common reasoning sequence in C⋆-algebra theory. If `0 ≤ x ≤ y ≤ 1`, then the norm of
`z - y * z` is controlled by the norm of `star z * (1 - x) * z`, which is advantageous because the
latter is nonnegative. This is a key step in establishing the existence of an increasing approximate
unit in general C⋆-algebras. -/
lemma nnnorm_sub_mul_self_le {A : Type*} [CStarAlgebra A] [PartialOrder A] [StarOrderedRing A]
    {x y : A} (z : A) (hx₀ : 0 ≤ x) (hy : y ∈ Set.Icc x 1) {c : ℝ≥0}
    (h : ‖star z * (1 - x) * z‖₊ ≤ c ^ 2) :
    ‖z - y * z‖₊ ≤ c := by
  /-
    A : Type u_2
    inst✝² : CStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hy : Membership.mem (Set.Icc x 1) y
    c : NNReal
    h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub z (HMul.hMul y z))) c
  -/
  nth_rw 1 [← one_mul z]
  /-
    A : Type u_2
    inst✝² : CStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hy : Membership.mem (Set.Icc x 1) y
    c : NNReal
    h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub (HMul.hMul 1 z) (HMul.hMul y z))) c
  -/
  rw [← sqrt_sq c, le_sqrt_iff_sq_le, ← sub_mul, sq, ← CStarRing.nnnorm_star_mul_self]
  /-
    A : Type u_2
    inst✝² : CStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hy : Membership.mem (Set.Icc x 1) y
    c : NNReal
    h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
    ⊢ LE.le (NNNorm.nnnorm (HMul.hMul (Star.star (HMul.hMul (HSub.hSub 1 y) z)) (H …
  -/
  simp only [star_mul, star_sub, star_one]
  /-
    A : Type u_2
    inst✝² : CStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hy : Membership.mem (Set.Icc x 1) y
    c : NNReal
    h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
    ⊢ LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 (Star. …
  -/
  have hy₀ : y ∈ Set.Icc 0 1 := ⟨hx₀.trans hy.1, hy.2⟩
  /-
    A : Type u_2
    inst✝² : CStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hy : Membership.mem (Set.Icc x 1) y
    c : NNReal
    h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
    hy₀ : Membership.mem (Set.Icc 0 1) y
    ⊢ LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 (Star. …
  -/
  have hy' : 1 - y ∈ Set.Icc 0 1 := Set.sub_mem_Icc_zero_iff_right.mpr hy₀
  /-
    A : Type u_2
    inst✝² : CStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hy : Membership.mem (Set.Icc x 1) y
    c : NNReal
    h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
    hy₀ : Membership.mem (Set.Icc 0 1) y
    hy' : Membership.mem (Set.Icc 0 1) (HSub.hSub 1 y)
    ⊢ LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 (Star. …
  -/
  rw [hy₀.1.star_eq, ← mul_assoc, mul_assoc (star _), ← sq]
  /-
    A : Type u_2
    inst✝² : CStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hy : Membership.mem (Set.Icc x 1) y
    c : NNReal
    h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
    hy₀ : Membership.mem (Set.Icc 0 1) y
    hy' : Membership.mem (Set.Icc 0 1) (HSub.hSub 1 y)
    ⊢ LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HPow.hPow (HSub.hS …
  -/
  refine nnnorm_le_nnnorm_of_nonneg_of_le (conjugate_nonneg (pow_nonneg hy'.1 2) _) ?_ |>.trans h
  /-
    A : Type u_2
    inst✝² : CStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hy : Membership.mem (Set.Icc x 1) y
    c : NNReal
    h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
    hy₀ : Membership.mem (Set.Icc 0 1) y
    hy' : Membership.mem (Set.Icc 0 1) (HSub.hSub 1 y)
    ⊢ LE.le (HMul.hMul (HMul.hMul (Star.star z) (HPow.hPow (HSub.hSub 1 y) 2)) z)  …
  -/
  refine conjugate_le_conjugate ?_ _
  /-
    A : Type u_2
    inst✝² : CStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hy : Membership.mem (Set.Icc x 1) y
    c : NNReal
    h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
    hy₀ : Membership.mem (Set.Icc 0 1) y
    hy' : Membership.mem (Set.Icc 0 1) (HSub.hSub 1 y)
    ⊢ LE.le (HPow.hPow (HSub.hSub 1 y) 2) (HSub.hSub 1 x)
  -/
  trans (1 - y)
    /-
      A : Type u_2
      inst✝² : CStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      x y z : A
      hx₀ : LE.le 0 x
      hy : Membership.mem (Set.Icc x 1) y
      c : NNReal
      h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
      hy₀ : Membership.mem (Set.Icc 0 1) y
      hy' : Membership.mem (Set.Icc 0 1) (HSub.hSub 1 y)
      ⊢ LE.le (HPow.hPow (HSub.hSub 1 y) 2) (HSub.hSub 1 y)
    -/
  · simpa using pow_antitone hy'.1 hy'.2 one_le_two
    /-
      🎉 no goals
    -/
    /-
      A : Type u_2
      inst✝² : CStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      x y z : A
      hx₀ : LE.le 0 x
      hy : Membership.mem (Set.Icc x 1) y
      c : NNReal
      h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
      hy₀ : Membership.mem (Set.Icc 0 1) y
      hy' : Membership.mem (Set.Icc 0 1) (HSub.hSub 1 y)
      ⊢ LE.le (HSub.hSub 1 y) (HSub.hSub 1 x)
    -/
  · gcongr
    /-
      case h
      A : Type u_2
      inst✝² : CStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      x y z : A
      hx₀ : LE.le 0 x
      hy : Membership.mem (Set.Icc x 1) y
      c : NNReal
      h : LE.le (NNNorm.nnnorm (HMul.hMul (HMul.hMul (Star.star z) (HSub.hSub 1 x))  …
      hy₀ : Membership.mem (Set.Icc 0 1) y
      hy' : Membership.mem (Set.Icc 0 1) (HSub.hSub 1 y)
      ⊢ LE.le x y
    -/
    exact hy.1
    /-
      🎉 no goals
    -/


/-- A variant of `nnnorm_sub_mul_self_le` which uses `‖·‖` instead of `‖·‖₊`. -/
lemma norm_sub_mul_self_le {A : Type*} [CStarAlgebra A] [PartialOrder A] [StarOrderedRing A]
    {x y : A} (z : A) (hx₀ : 0 ≤ x) (hy : y ∈ Set.Icc x 1)
    {c : ℝ} (hc : 0 ≤ c) (h : ‖star z * (1 - x) * z‖ ≤ c ^ 2) :
    ‖z - y * z‖ ≤ c :=
  nnnorm_sub_mul_self_le z hx₀ hy h (c := ⟨c, hc⟩)


variable {A} in
/-- A variant of `norm_sub_mul_self_le` for non-unital algebras that passes to the unitization. -/
lemma norm_sub_mul_self_le_of_inr {x y : A} (z : A) (hx₀ : 0 ≤ x) (hxy : x ≤ y) (hy₁ : ‖y‖ ≤ 1)
    {c : ℝ} (hc : 0 ≤ c) (h : ‖star (z : A⁺¹) * (1 - x) * z‖ ≤ c ^ 2) :
    ‖z - y * z‖ ≤ c := by
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hxy : LE.le x y
    hy₁ : LE.le (Norm.norm y) 1
    c : Real
    hc : LE.le 0 c
    h : LE.le (Norm.norm (HMul.hMul (HMul.hMul (Star.star ↑z) (HSub.hSub 1 ↑x)) ↑z …
    ⊢ LE.le (Norm.norm (HSub.hSub z (HMul.hMul y z))) c
  -/
  rw [← norm_inr (𝕜 := ℂ), inr_sub, inr_mul]
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    x y z : A
    hx₀ : LE.le 0 x
    hxy : LE.le x y
    hy₁ : LE.le (Norm.norm y) 1
    c : Real
    hc : LE.le 0 c
    h : LE.le (Norm.norm (HMul.hMul (HMul.hMul (Star.star ↑z) (HSub.hSub 1 ↑x)) ↑z …
    ⊢ LE.le (Norm.norm (HSub.hSub (↑z) (HMul.hMul ↑y ↑z))) c
  -/
  refine norm_sub_mul_self_le _ ?_ ?_ hc h
    /-
      case refine_1
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      x y z : A
      hx₀ : LE.le 0 x
      hxy : LE.le x y
      hy₁ : LE.le (Norm.norm y) 1
      c : Real
      hc : LE.le 0 c
      h : LE.le (Norm.norm (HMul.hMul (HMul.hMul (Star.star ↑z) (HSub.hSub 1 ↑x)) ↑z …
      ⊢ LE.le 0 ↑x
    -/
  · rwa [inr_nonneg_iff]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      x y z : A
      hx₀ : LE.le 0 x
      hxy : LE.le x y
      hy₁ : LE.le (Norm.norm y) 1
      c : Real
      hc : LE.le 0 c
      h : LE.le (Norm.norm (HMul.hMul (HMul.hMul (Star.star ↑z) (HSub.hSub 1 ↑x)) ↑z …
      ⊢ Membership.mem (Set.Icc (↑x) 1) ↑y
    -/
  · have hy := hx₀.trans hxy
    rw [Set.mem_Icc, inr_le_iff _ _ hx₀.isSelfAdjoint hy.isSelfAdjoint,
      ← norm_le_one_iff_of_nonneg _, norm_inr]
    /-
      case refine_2
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      x y z : A
      hx₀ : LE.le 0 x
      hxy : LE.le x y
      hy₁ : LE.le (Norm.norm y) 1
      c : Real
      hc : LE.le 0 c
      h : LE.le (Norm.norm (HMul.hMul (HMul.hMul (Star.star ↑z) (HSub.hSub 1 ↑x)) ↑z …
      hy : LE.le 0 y
      ⊢ And (LE.le x y) (LE.le (Norm.norm y) 1)
    -/
    exact ⟨hxy, hy₁⟩
    /-
      🎉 no goals
    -/


variable {A} in
/-- This shows `CStarAlgebra.approximateUnit` is a one-sided approximate unit, but this is marked
`private` because it is only used to prove `CStarAlgebra.increasingApproximateUnit`. -/
private lemma tendsto_mul_right_approximateUnit (m : A) :
    Tendsto (· * m) (approximateUnit A) (𝓝 m) := by
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m : A
    ⊢ Filter.Tendsto (fun x => HMul.hMul x m) (CStarAlgebra.approximateUnit A) (nh …
  -/
  refine tendsto_mul_right_of_forall_nonneg_tendsto (fun m hm₁ hm₂ ↦ ?_) m
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    ⊢ Filter.Tendsto (fun x => HMul.hMul x m) (CStarAlgebra.approximateUnit A) (nh …
  -/
  rw [(hasBasis_approximateUnit A).tendsto_iff nhds_basis_closedBall]
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    ⊢ ∀ (ib : Real), LT.lt 0 ib → Exists fun ia => And (And (LE.le 0 ia) (LT.lt (N …
  -/
  intro ε hε
  /-
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    ε : Real
    hε : LT.lt 0 ε
    ⊢ Exists fun ia => And (And (LE.le 0 ia) (LT.lt (Norm.norm ia) 1)) (∀ (x : A), …
  -/
  lift ε to ℝ≥0 using hε.le
  /-
    case intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    ε : NNReal
    hε : LT.lt 0 ↑ε
    ⊢ Exists fun ia => And (And (LE.le 0 ia) (LT.lt (Norm.norm ia) 1)) (∀ (x : A), …
  -/
  rw [coe_pos] at hε
  refine ⟨cfcₙ (fun y : ℝ≥0 ↦ 1 - (1 + y)⁻¹) (ε⁻¹ ^ 2 • m),
    ⟨cfcₙ_nonneg_of_predicate, norm_cfcₙ_one_sub_one_add_inv_lt_one (ε⁻¹ ^ 2 • m)⟩, ?_⟩
  /-
    case intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    ε : NNReal
    hε : LT.lt 0 ε
    ⊢ ∀ (x : A), Membership.mem (Inter.inter (setOf fun x => LE.le (cfcₙ (fun y => …
  -/
  rintro x ⟨(hx₁ : _ ≤ x), hx₂⟩
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    ε : NNReal
    hε : LT.lt 0 ε
    x : A
    hx₁ : LE.le (cfcₙ (fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))) (HSMul.hSMu …
    hx₂ : Membership.mem (Metric.closedBall 0 1) x
    ⊢ Membership.mem (Metric.closedBall m ↑ε) (HMul.hMul x m)
  -/
  simp only [mem_closedBall, dist_eq_norm', zero_sub, norm_neg] at hx₂ ⊢
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    ε : NNReal
    hε : LT.lt 0 ε
    x : A
    hx₁ : LE.le (cfcₙ (fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))) (HSMul.hSMu …
    hx₂ : LE.le (Norm.norm x) 1
    ⊢ LE.le (Norm.norm (HSub.hSub m (HMul.hMul x m))) ↑ε
  -/
  rw [← coe_nnnorm, coe_le_coe]
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    ε : NNReal
    hε : LT.lt 0 ε
    x : A
    hx₁ : LE.le (cfcₙ (fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))) (HSMul.hSMu …
    hx₂ : LE.le (Norm.norm x) 1
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub m (HMul.hMul x m))) ε
  -/
  have hx₀ : 0 ≤ x := cfcₙ_nonneg_of_predicate.trans hx₁
  rw [← inr_le_iff _ _ (.of_nonneg cfcₙ_nonneg_of_predicate) (.of_nonneg hx₀),
    nnreal_cfcₙ_eq_cfc_inr _ _ (by simp [tsub_self]), inr_smul] at hx₁
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    ε : NNReal
    hε : LT.lt 0 ε
    x : A
    hx₁ : LE.le (cfc (fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))) (HSMul.hSMul …
    hx₂ : LE.le (Norm.norm x) 1
    hx₀ : LE.le 0 x
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub m (HMul.hMul x m))) ε
  -/
  rw [← norm_inr (𝕜 := ℂ)] at hm₂ hx₂
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm ↑m) 1
    ε : NNReal
    hε : LT.lt 0 ε
    x : A
    hx₁ : LE.le (cfc (fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))) (HSMul.hSMul …
    hx₂ : LE.le (Norm.norm ↑x) 1
    hx₀ : LE.le 0 x
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub m (HMul.hMul x m))) ε
  -/
  rw [← inr_nonneg_iff] at hx₀ hm₁
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 ↑m
    hm₂ : LT.lt (Norm.norm ↑m) 1
    ε : NNReal
    hε : LT.lt 0 ε
    x : A
    hx₁ : LE.le (cfc (fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))) (HSMul.hSMul …
    hx₂ : LE.le (Norm.norm ↑x) 1
    hx₀ : LE.le 0 ↑x
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub m (HMul.hMul x m))) ε
  -/
  rw [← nnnorm_inr (𝕜 := ℂ), inr_sub, inr_mul]
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝ m : A
    hm₁ : LE.le 0 ↑m
    hm₂ : LT.lt (Norm.norm ↑m) 1
    ε : NNReal
    hε : LT.lt 0 ε
    x : A
    hx₁ : LE.le (cfc (fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))) (HSMul.hSMul …
    hx₂ : LE.le (Norm.norm ↑x) 1
    hx₀ : LE.le 0 ↑x
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub (↑m) (HMul.hMul ↑x ↑m))) ε
  -/
  generalize (x : A⁺¹) = x, (m : A⁺¹) = m at *
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝¹ m✝ : A
    ε : NNReal
    hε : LT.lt 0 ε
    x✝ : A
    x m : Unitization Complex A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    hx₁ : LE.le (cfc (fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))) (HSMul.hSMul …
    hx₂ : LE.le (Norm.norm x) 1
    hx₀ : LE.le 0 x
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub m (HMul.hMul x m))) ε
  -/
  set g : ℝ≥0 → ℝ≥0 := fun y ↦ 1 - (1 + y)⁻¹
  have hg : Continuous g := by
    rw [continuous_iff_continuousOn_univ]
    fun_prop (disch := intro _ _; positivity)
  have hg' : ContinuousOn (fun y ↦ (1 + ε⁻¹ ^ 2 • y)⁻¹) (spectrum ℝ≥0 m) :=
    ContinuousOn.inv₀ (by fun_prop) fun _ _ ↦ by positivity
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝¹ m✝ : A
    ε : NNReal
    hε : LT.lt 0 ε
    x✝ : A
    x m : Unitization Complex A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    hx₂ : LE.le (Norm.norm x) 1
    hx₀ : LE.le 0 x
    g : NNReal → NNReal := fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))
    hx₁ : LE.le (cfc g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) m)) x
    hg : Continuous g
    hg' : ContinuousOn (fun y => Inv.inv (HAdd.hAdd 1 (HSMul.hSMul (HPow.hPow (Inv …
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub m (HMul.hMul x m))) ε
  -/
  have hx : x ∈ Set.Icc 0 1 := mem_Icc_iff_norm_le_one.mpr ⟨hx₀, hx₂⟩
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝¹ m✝ : A
    ε : NNReal
    hε : LT.lt 0 ε
    x✝ : A
    x m : Unitization Complex A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    hx₂ : LE.le (Norm.norm x) 1
    hx₀ : LE.le 0 x
    g : NNReal → NNReal := fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))
    hx₁ : LE.le (cfc g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) m)) x
    hg : Continuous g
    hg' : ContinuousOn (fun y => Inv.inv (HAdd.hAdd 1 (HSMul.hSMul (HPow.hPow (Inv …
    hx : Membership.mem (Set.Icc 0 1) x
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub m (HMul.hMul x m))) ε
  -/
  have hx' : x ∈ Set.Icc _ 1 := ⟨hx₁, hx.2⟩
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝¹ m✝ : A
    ε : NNReal
    hε : LT.lt 0 ε
    x✝ : A
    x m : Unitization Complex A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    hx₂ : LE.le (Norm.norm x) 1
    hx₀ : LE.le 0 x
    g : NNReal → NNReal := fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))
    hx₁ : LE.le (cfc g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) m)) x
    hg : Continuous g
    hg' : ContinuousOn (fun y => Inv.inv (HAdd.hAdd 1 (HSMul.hSMul (HPow.hPow (Inv …
    hx : Membership.mem (Set.Icc 0 1) x
    hx' : Membership.mem (Set.Icc (cfc g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) m) …
    ⊢ LE.le (NNNorm.nnnorm (HSub.hSub m (HMul.hMul x m))) ε
  -/
  refine nnnorm_sub_mul_self_le m cfc_nonneg_of_predicate hx' ?_
  suffices star m * (1 - cfc g (ε⁻¹ ^ 2 • m)) * m =
      cfc (fun y : ℝ≥0 ↦ y * (1 + ε⁻¹ ^ 2 • y)⁻¹ * y) m by
    rw [this]
    refine nnnorm_cfc_nnreal_le fun y hy ↦ ?_
    field_simp
    calc
      y * ε ^ 2 * y / (ε ^ 2 + y) ≤ ε ^ 2 * 1 := by
        rw [mul_div_assoc]
        gcongr
        · refine mul_le_of_le_one_left (zero_le _) ?_
          have hm' := hm₂.le
          rw [norm_le_one_iff_of_nonneg m hm₁, ← cfc_id' ℝ≥0 m, ← cfc_one (R := ℝ≥0) m,
            cfc_nnreal_le_iff _ _ _ (QuasispectrumRestricts.nnreal_of_nonneg hm₁)] at hm'
          exact hm' y hy
        · exact div_le_one (by positivity) |>.mpr le_add_self
      _ = ε ^ 2 := mul_one _
  rw [cfc_mul _ _ m (continuousOn_id' _ |>.mul hg') (continuousOn_id' _),
    cfc_mul _ _ m (continuousOn_id' _) hg', cfc_id' .., hm₁.star_eq]
  /-
    case intro.intro
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝¹ m✝ : A
    ε : NNReal
    hε : LT.lt 0 ε
    x✝ : A
    x m : Unitization Complex A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    hx₂ : LE.le (Norm.norm x) 1
    hx₀ : LE.le 0 x
    g : NNReal → NNReal := fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))
    hx₁ : LE.le (cfc g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) m)) x
    hg : Continuous g
    hg' : ContinuousOn (fun y => Inv.inv (HAdd.hAdd 1 (HSMul.hSMul (HPow.hPow (Inv …
    hx : Membership.mem (Set.Icc 0 1) x
    hx' : Membership.mem (Set.Icc (cfc g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) m) …
    ⊢ Eq (HMul.hMul (HMul.hMul m (HSub.hSub 1 (cfc g (HSMul.hSMul (HPow.hPow (Inv. …
  -/
  congr
  rw [← cfc_one (R := ℝ≥0) m, ← cfc_comp_smul _ _ _ hg.continuousOn hm₁,
    ← cfc_tsub _ _ m (by simp [g]) hm₁ (by fun_prop) (Continuous.continuousOn <| by fun_prop)]
  /-
    case intro.intro.e_a.e_a
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝¹ m✝ : A
    ε : NNReal
    hε : LT.lt 0 ε
    x✝ : A
    x m : Unitization Complex A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    hx₂ : LE.le (Norm.norm x) 1
    hx₀ : LE.le 0 x
    g : NNReal → NNReal := fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))
    hx₁ : LE.le (cfc g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) m)) x
    hg : Continuous g
    hg' : ContinuousOn (fun y => Inv.inv (HAdd.hAdd 1 (HSMul.hSMul (HPow.hPow (Inv …
    hx : Membership.mem (Set.Icc 0 1) x
    hx' : Membership.mem (Set.Icc (cfc g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) m) …
    ⊢ Eq (cfc (fun x => HSub.hSub (1 x) (g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2)  …
  -/
  refine cfc_congr (fun y _ ↦ ?_)
  /-
    case intro.intro.e_a.e_a
    A : Type u_1
    inst✝² : NonUnitalCStarAlgebra A
    inst✝¹ : PartialOrder A
    inst✝ : StarOrderedRing A
    m✝¹ m✝ : A
    ε : NNReal
    hε : LT.lt 0 ε
    x✝¹ : A
    x m : Unitization Complex A
    hm₁ : LE.le 0 m
    hm₂ : LT.lt (Norm.norm m) 1
    hx₂ : LE.le (Norm.norm x) 1
    hx₀ : LE.le 0 x
    g : NNReal → NNReal := fun y => HSub.hSub 1 (Inv.inv (HAdd.hAdd 1 y))
    hx₁ : LE.le (cfc g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) m)) x
    hg : Continuous g
    hg' : ContinuousOn (fun y => Inv.inv (HAdd.hAdd 1 (HSMul.hSMul (HPow.hPow (Inv …
    hx : Membership.mem (Set.Icc 0 1) x
    hx' : Membership.mem (Set.Icc (cfc g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) m) …
    y : NNReal
    x✝ : Membership.mem (spectrum NNReal m) y
    ⊢ Eq (HSub.hSub (1 y) (g (HSMul.hSMul (HPow.hPow (Inv.inv ε) 2) y))) (Inv.inv  …
  -/
  simp [g, tsub_tsub_cancel_of_le]
  /-
    🎉 no goals
  -/


/-- The filter `CStarAlgebra.approximateUnit` generated by the sections
`{x | a ≤ x} ∩ closedBall 0 1` for `0 ≤ a` forms an increasing approximate unit. -/
lemma increasingApproximateUnit :
    IsIncreasingApproximateUnit (approximateUnit A) where
  tendsto_mul_left := by
    /-
      A : Type u_1
      inst✝² : NonUnitalCStarAlgebra A
      inst✝¹ : PartialOrder A
      inst✝ : StarOrderedRing A
      ⊢ ∀ (m : A), Filter.Tendsto (fun x => HMul.hMul m x) (CStarAlgebra.approximate …
    -/
    rw [tendsto_mul_left_iff_tendsto_mul_right]
      /-
        A : Type u_1
        inst✝² : NonUnitalCStarAlgebra A
        inst✝¹ : PartialOrder A
        inst✝ : StarOrderedRing A
        ⊢ ∀ (m : A), Filter.Tendsto (fun x => HMul.hMul x m) (CStarAlgebra.approximate …
      -/
    · exact tendsto_mul_right_approximateUnit
      /-
        🎉 no goals
      -/
      /-
        A : Type u_1
        inst✝² : NonUnitalCStarAlgebra A
        inst✝¹ : PartialOrder A
        inst✝ : StarOrderedRing A
        ⊢ Filter.Eventually (fun x => IsSelfAdjoint x) (CStarAlgebra.approximateUnit A)
      -/
    · rw [(hasBasis_approximateUnit A).eventually_iff]
      /-
        A : Type u_1
        inst✝² : NonUnitalCStarAlgebra A
        inst✝¹ : PartialOrder A
        inst✝ : StarOrderedRing A
        ⊢ Exists fun i => And (And (LE.le 0 i) (LT.lt (Norm.norm i) 1)) (∀ ⦃x : A⦄, Me …
      -/
      peel (hasBasis_approximateUnit A).ex_mem with x hx
      /-
        case h
        A : Type u_1
        inst✝² : NonUnitalCStarAlgebra A
        inst✝¹ : PartialOrder A
        inst✝ : StarOrderedRing A
        x : A
        hx : And (LE.le 0 x) (LT.lt (Norm.norm x) 1)
        ⊢ And (And (LE.le 0 x) (LT.lt (Norm.norm x) 1)) (∀ ⦃x_1 : A⦄, Membership.mem ( …
      -/
      exact ⟨hx, fun y hy ↦ (hx.1.trans hy.1).isSelfAdjoint⟩
      /-
        🎉 no goals
      -/
  tendsto_mul_right := tendsto_mul_right_approximateUnit
  eventually_nonneg := .filter_mono inf_le_left <|
                                                                   /-
                                                                     A : Type u_1
                                                                     inst✝² : NonUnitalCStarAlgebra A
                                                                     inst✝¹ : PartialOrder A
                                                                     inst✝ : StarOrderedRing A
                                                                     ⊢ And (And (LE.le 0 0) (LT.lt (Norm.norm 0) 1)) (∀ ⦃x : A⦄, Membership.mem (se …
                                                                   -/
    (isBasis_nonneg_sections A).hasBasis.eventually_iff.mpr ⟨0, by simp⟩
                                                                   /-
                                                                     🎉 no goals
                                                                   -/
                             /-
                               A : Type u_1
                               inst✝² : NonUnitalCStarAlgebra A
                               inst✝¹ : PartialOrder A
                               inst✝ : StarOrderedRing A
                               i✝ : A
                               hx : And (LE.le 0 i✝) (LT.lt (Norm.norm i✝) 1)
                               ⊢ Membership.mem (Metric.closedBall 0 1) i✝
                             -/
                                                     /-
                                                       A : Type u_1
                                                       inst✝² : NonUnitalCStarAlgebra A
                                                       inst✝¹ : PartialOrder A
                                                       inst✝ : StarOrderedRing A
                                                       ⊢ Filter.Eventually (fun x => LE.le (Norm.norm x) 1) (Filter.principal (Metric …
                                                     -/
                             /-
                               🎉 no goals
                             -/
  eventually_norm := .filter_mono inf_le_right <| by simp
                                                     /-
                                                       🎉 no goals
                                                     -/
  neBot := hasBasis_approximateUnit A |>.neBot_iff.mpr
    fun hx ↦ ⟨_, ⟨le_rfl, by simpa using hx.2.le⟩⟩


