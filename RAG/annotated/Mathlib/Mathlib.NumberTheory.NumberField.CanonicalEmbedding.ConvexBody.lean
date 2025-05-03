/-- The convex body defined by `f`: the set of points `x : E` such that `‖x w‖ < f w` for all
infinite places `w`. -/
abbrev convexBodyLT : Set (mixedSpace K) :=
  (Set.univ.pi (fun w : { w : InfinitePlace K // IsReal w } => ball 0 (f w))) ×ˢ
  (Set.univ.pi (fun w : { w : InfinitePlace K // IsComplex w } => ball 0 (f w)))


theorem convexBodyLT_mem {x : K} :
    mixedEmbedding K x ∈ (convexBodyLT K f) ↔ ∀ w : InfinitePlace K, w x < f w := by
  simp_rw [mixedEmbedding, RingHom.prod_apply, Set.mem_prod, Set.mem_pi, Set.mem_univ,
    forall_true_left, mem_ball_zero_iff, Pi.ringHom_apply, ← Complex.norm_real,
    embedding_of_isReal_apply, Subtype.forall, ← forall₂_or_left, ← not_isReal_iff_isComplex, em,
    forall_true_left, norm_embedding_eq]


theorem convexBodyLT_neg_mem (x : mixedSpace K) (hx : x ∈ (convexBodyLT K f)) :
    -x ∈ (convexBodyLT K f) := by
  simp only [Set.mem_prod, Prod.fst_neg, Set.mem_pi, Set.mem_univ, Pi.neg_apply,
    mem_ball_zero_iff, norm_neg, Real.norm_eq_abs, forall_true_left, Subtype.forall,
    Prod.snd_neg, Complex.norm_eq_abs] at hx ⊢
  /-
    K : Type u_1
    inst✝ : Field K
    f : NumberField.InfinitePlace K → NNReal
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : And (∀ (a : NumberField.InfinitePlace K) (b : a.IsReal), LT.lt (abs (x.1  …
    ⊢ And (∀ (a : NumberField.InfinitePlace K) (b : a.IsReal), LT.lt (abs (x.1 ⟨a, …
  -/
  exact hx
  /-
    🎉 no goals
  -/


theorem convexBodyLT_convex : Convex ℝ (convexBodyLT K f) :=
  Convex.prod (convex_pi (fun _ _ => convex_ball _ _)) (convex_pi (fun _ _ => convex_ball _ _))


/-- The fudge factor that appears in the formula for the volume of `convexBodyLT`. -/
noncomputable abbrev convexBodyLTFactor : ℝ≥0 :=
  (2 : ℝ≥0) ^ nrRealPlaces K * NNReal.pi ^ nrComplexPlaces K


theorem convexBodyLTFactor_ne_zero : convexBodyLTFactor K ≠ 0 :=
  mul_ne_zero (pow_ne_zero _ two_ne_zero) (pow_ne_zero _ pi_ne_zero)


theorem one_le_convexBodyLTFactor : 1 ≤ convexBodyLTFactor K :=
  one_le_mul (one_le_pow₀ one_le_two) (one_le_pow₀ (one_le_two.trans Real.two_le_pi))


/-- The volume of `(ConvexBodyLt K f)` where `convexBodyLT K f` is the set of points `x`
such that `‖x w‖ < f w` for all infinite places `w`. -/
theorem convexBodyLT_volume :
    volume (convexBodyLT K f) = (convexBodyLTFactor K) * ∏ w, (f w) ^ (mult w) := by
  calc
    _ = (∏ x : {w // InfinitePlace.IsReal w}, ENNReal.ofReal (2 * (f x.val))) *
          ∏ x : {w // InfinitePlace.IsComplex w}, ENNReal.ofReal (f x.val) ^ 2 * NNReal.pi := by
      simp_rw [volume_eq_prod, prod_prod, volume_pi, pi_pi, Real.volume_ball, Complex.volume_ball]
    _ = ((2 : ℝ≥0) ^ nrRealPlaces K
          * (∏ x : {w // InfinitePlace.IsReal w}, ENNReal.ofReal (f x.val)))
          * ((∏ x : {w // IsComplex w}, ENNReal.ofReal (f x.val) ^ 2) *
            NNReal.pi ^ nrComplexPlaces K) := by
      simp_rw [ofReal_mul (by norm_num : 0 ≤ (2 : ℝ)), Finset.prod_mul_distrib, Finset.prod_const,
        Finset.card_univ, ofReal_ofNat, ofReal_coe_nnreal, coe_ofNat]
    _ = (convexBodyLTFactor K) * ((∏ x : {w // InfinitePlace.IsReal w}, .ofReal (f x.val)) *
        (∏ x : {w // IsComplex w}, ENNReal.ofReal (f x.val) ^ 2)) := by
      simp_rw [convexBodyLTFactor, coe_mul, ENNReal.coe_pow]
      ring
    _ = (convexBodyLTFactor K) * ∏ w, (f w) ^ (mult w) := by
      simp_rw [mult, pow_ite, pow_one, Finset.prod_ite, ofReal_coe_nnreal, not_isReal_iff_isComplex,
        coe_mul, coe_finset_prod, ENNReal.coe_pow]
      congr 2
      · refine (Finset.prod_subtype (Finset.univ.filter _) ?_ (fun w => (f w : ℝ≥0∞))).symm
        exact fun _ => by simp only [Finset.mem_univ, forall_true_left, Finset.mem_filter, true_and]
      · refine (Finset.prod_subtype (Finset.univ.filter _) ?_ (fun w => (f w : ℝ≥0∞) ^ 2)).symm
        exact fun _ => by simp only [Finset.mem_univ, forall_true_left, Finset.mem_filter, true_and]


/-- This is a technical result: quite often, we want to impose conditions at all infinite places
but one and choose the value at the remaining place so that we can apply
`exists_ne_zero_mem_ringOfIntegers_lt`. -/
theorem adjust_f {w₁ : InfinitePlace K} (B : ℝ≥0) (hf : ∀ w, w ≠ w₁ → f w ≠ 0) :
    ∃ g : InfinitePlace K → ℝ≥0, (∀ w, w ≠ w₁ → g w = f w) ∧ ∏ w, (g w) ^ mult w = B := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    f : NumberField.InfinitePlace K → NNReal
    inst✝ : NumberField K
    w₁ : NumberField.InfinitePlace K
    B : NNReal
    hf : ∀ (w : NumberField.InfinitePlace K), Ne w w₁ → Ne (f w) 0
    ⊢ Exists fun g => And (∀ (w : NumberField.InfinitePlace K), Ne w w₁ → Eq (g w) …
  -/
  let S := ∏ w ∈ Finset.univ.erase w₁, (f w) ^ mult w
  /-
    K : Type u_1
    inst✝¹ : Field K
    f : NumberField.InfinitePlace K → NNReal
    inst✝ : NumberField K
    w₁ : NumberField.InfinitePlace K
    B : NNReal
    hf : ∀ (w : NumberField.InfinitePlace K), Ne w w₁ → Ne (f w) 0
    S : NNReal := (Finset.univ.erase w₁).prod fun w => HPow.hPow (f w) w.mult
    ⊢ Exists fun g => And (∀ (w : NumberField.InfinitePlace K), Ne w w₁ → Eq (g w) …
  -/
  refine ⟨Function.update f w₁ ((B * S⁻¹) ^ (mult w₁ : ℝ)⁻¹), ?_, ?_⟩
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      f : NumberField.InfinitePlace K → NNReal
      inst✝ : NumberField K
      w₁ : NumberField.InfinitePlace K
      B : NNReal
      hf : ∀ (w : NumberField.InfinitePlace K), Ne w w₁ → Ne (f w) 0
      S : NNReal := (Finset.univ.erase w₁).prod fun w => HPow.hPow (f w) w.mult
      ⊢ ∀ (w : NumberField.InfinitePlace K), Ne w w₁ → Eq (Function.update f w₁ (HPo …
    -/
  · exact fun w hw => Function.update_of_ne hw _ f
    /-
      🎉 no goals
    -/
  · rw [← Finset.mul_prod_erase Finset.univ _ (Finset.mem_univ w₁), Function.update_self,
      Finset.prod_congr rfl fun w hw => by rw [Function.update_of_ne (Finset.ne_of_mem_erase hw)],
      ← NNReal.rpow_natCast, ← NNReal.rpow_mul, inv_mul_cancel₀, NNReal.rpow_one, mul_assoc,
      inv_mul_cancel₀, mul_one]
      /-
        case refine_2
        K : Type u_1
        inst✝¹ : Field K
        f : NumberField.InfinitePlace K → NNReal
        inst✝ : NumberField K
        w₁ : NumberField.InfinitePlace K
        B : NNReal
        hf : ∀ (w : NumberField.InfinitePlace K), Ne w w₁ → Ne (f w) 0
        S : NNReal := (Finset.univ.erase w₁).prod fun w => HPow.hPow (f w) w.mult
        ⊢ Ne S 0
      -/
    · rw [Finset.prod_ne_zero_iff]
      /-
        case refine_2
        K : Type u_1
        inst✝¹ : Field K
        f : NumberField.InfinitePlace K → NNReal
        inst✝ : NumberField K
        w₁ : NumberField.InfinitePlace K
        B : NNReal
        hf : ∀ (w : NumberField.InfinitePlace K), Ne w w₁ → Ne (f w) 0
        S : NNReal := (Finset.univ.erase w₁).prod fun w => HPow.hPow (f w) w.mult
        ⊢ ∀ (a : NumberField.InfinitePlace K), Membership.mem (Finset.univ.erase w₁) a …
      -/
      exact fun w hw => pow_ne_zero _ (hf w (Finset.ne_of_mem_erase hw))
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        K : Type u_1
        inst✝¹ : Field K
        f : NumberField.InfinitePlace K → NNReal
        inst✝ : NumberField K
        w₁ : NumberField.InfinitePlace K
        B : NNReal
        hf : ∀ (w : NumberField.InfinitePlace K), Ne w w₁ → Ne (f w) 0
        S : NNReal := (Finset.univ.erase w₁).prod fun w => HPow.hPow (f w) w.mult
        ⊢ Ne (↑w₁.mult) 0
      -/
                               /-
                                 🎉 no goals
                               -/
    · rw [mult]; split_ifs <;> norm_num
                               /-
                                 🎉 no goals
                               -/


/-- A version of `convexBodyLT` with an additional condition at a fixed complex place. This is
needed to ensure the element constructed is not real, see for example
`exists_primitive_element_lt_of_isComplex`.
-/
abbrev convexBodyLT' : Set (mixedSpace K) :=
  (Set.univ.pi (fun w : { w : InfinitePlace K // IsReal w } ↦ ball 0 (f w))) ×ˢ
  (Set.univ.pi (fun w : { w : InfinitePlace K // IsComplex w } ↦
    if w = w₀ then {x | |x.re| < 1 ∧ |x.im| < (f w : ℝ) ^ 2} else ball 0 (f w)))


theorem convexBodyLT'_mem {x : K} :
    mixedEmbedding K x ∈ convexBodyLT' K f w₀ ↔
      (∀ w : InfinitePlace K, w ≠ w₀ → w x < f w) ∧
      |(w₀.val.embedding x).re| < 1 ∧ |(w₀.val.embedding x).im| < (f w₀ : ℝ) ^ 2 := by
  simp_rw [mixedEmbedding, RingHom.prod_apply, Set.mem_prod, Set.mem_pi, Set.mem_univ,
    forall_true_left, Pi.ringHom_apply, mem_ball_zero_iff, ← Complex.norm_real,
    embedding_of_isReal_apply, norm_embedding_eq, Subtype.forall]
  /-
    K : Type u_1
    inst✝ : Field K
    f : NumberField.InfinitePlace K → NNReal
    w₀ : Subtype fun w => w.IsComplex
    x : K
    ⊢ Iff (And (∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a) …
  -/
  refine ⟨fun ⟨h₁, h₂⟩ ↦ ⟨fun w h_ne ↦ ?_, ?_⟩, fun ⟨h₁, h₂⟩ ↦ ⟨fun w hw ↦ ?_, fun w hw ↦ ?_⟩⟩
    /-
      case refine_1
      K : Type u_1
      inst✝ : Field K
      f : NumberField.InfinitePlace K → NNReal
      w₀ : Subtype fun w => w.IsComplex
      x : K
      x✝ : And (∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a))  …
      h₁ : ∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a)
      h₂ : ∀ (a : NumberField.InfinitePlace K) (b : a.IsComplex), Membership.mem (it …
      w : NumberField.InfinitePlace K
      h_ne : Ne w ↑w₀
      ⊢ LT.lt (w x) ↑(f w)
    -/
  · by_cases hw : IsReal w
      /-
        case pos
        K : Type u_1
        inst✝ : Field K
        f : NumberField.InfinitePlace K → NNReal
        w₀ : Subtype fun w => w.IsComplex
        x : K
        x✝ : And (∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a))  …
        h₁ : ∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a)
        h₂ : ∀ (a : NumberField.InfinitePlace K) (b : a.IsComplex), Membership.mem (it …
        w : NumberField.InfinitePlace K
        h_ne : Ne w ↑w₀
        hw : w.IsReal
        ⊢ LT.lt (w x) ↑(f w)
      -/
    · exact norm_embedding_eq w _ ▸ h₁ w hw
      /-
        🎉 no goals
      -/
      /-
        case neg
        K : Type u_1
        inst✝ : Field K
        f : NumberField.InfinitePlace K → NNReal
        w₀ : Subtype fun w => w.IsComplex
        x : K
        x✝ : And (∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a))  …
        h₁ : ∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a)
        h₂ : ∀ (a : NumberField.InfinitePlace K) (b : a.IsComplex), Membership.mem (it …
        w : NumberField.InfinitePlace K
        h_ne : Ne w ↑w₀
        hw : Not w.IsReal
        ⊢ LT.lt (w x) ↑(f w)
      -/
    · specialize h₂ w (not_isReal_iff_isComplex.mp hw)
      rw [apply_ite (w.embedding x ∈ ·), Set.mem_setOf_eq,
        mem_ball_zero_iff, norm_embedding_eq] at h₂
      /-
        case neg
        K : Type u_1
        inst✝ : Field K
        f : NumberField.InfinitePlace K → NNReal
        w₀ : Subtype fun w => w.IsComplex
        x : K
        x✝ : And (∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a))  …
        h₁ : ∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a)
        w : NumberField.InfinitePlace K
        h_ne : Ne w ↑w₀
        hw : Not w.IsReal
        h₂ : ite (Eq ⟨w, ⋯⟩ w₀) (And (LT.lt (abs (w.embedding x).re) 1) (LT.lt (abs (w …
        ⊢ LT.lt (w x) ↑(f w)
      -/
      rwa [if_neg (by exact Subtype.coe_ne_coe.1 h_ne)] at h₂
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      K : Type u_1
      inst✝ : Field K
      f : NumberField.InfinitePlace K → NNReal
      w₀ : Subtype fun w => w.IsComplex
      x : K
      x✝ : And (∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a))  …
      h₁ : ∀ (a : NumberField.InfinitePlace K), a.IsReal → LT.lt (a x) ↑(f a)
      h₂ : ∀ (a : NumberField.InfinitePlace K) (b : a.IsComplex), Membership.mem (it …
      ⊢ And (LT.lt (abs ((↑w₀).embedding x).re) 1) (LT.lt (abs ((↑w₀).embedding x).i …
    -/
  · simpa [if_true] using h₂ w₀.val w₀.prop
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      K : Type u_1
      inst✝ : Field K
      f : NumberField.InfinitePlace K → NNReal
      w₀ : Subtype fun w => w.IsComplex
      x : K
      x✝ : And (∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w))  …
      h₁ : ∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w)
      h₂ : And (LT.lt (abs ((↑w₀).embedding x).re) 1) (LT.lt (abs ((↑w₀).embedding x …
      w : NumberField.InfinitePlace K
      hw : w.IsReal
      ⊢ LT.lt (w x) ↑(f w)
    -/
  · exact h₁ w (ne_of_isReal_isComplex hw w₀.prop)
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      K : Type u_1
      inst✝ : Field K
      f : NumberField.InfinitePlace K → NNReal
      w₀ : Subtype fun w => w.IsComplex
      x : K
      x✝ : And (∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w))  …
      h₁ : ∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w)
      h₂ : And (LT.lt (abs ((↑w₀).embedding x).re) 1) (LT.lt (abs ((↑w₀).embedding x …
      w : NumberField.InfinitePlace K
      hw : w.IsComplex
      ⊢ Membership.mem (ite (Eq ⟨w, hw⟩ w₀) (setOf fun x => And (LT.lt (abs x.re) 1) …
    -/
  · by_cases h_ne : w = w₀
      /-
        case pos
        K : Type u_1
        inst✝ : Field K
        f : NumberField.InfinitePlace K → NNReal
        w₀ : Subtype fun w => w.IsComplex
        x : K
        x✝ : And (∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w))  …
        h₁ : ∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w)
        h₂ : And (LT.lt (abs ((↑w₀).embedding x).re) 1) (LT.lt (abs ((↑w₀).embedding x …
        w : NumberField.InfinitePlace K
        hw : w.IsComplex
        h_ne : Eq w ↑w₀
        ⊢ Membership.mem (ite (Eq ⟨w, hw⟩ w₀) (setOf fun x => And (LT.lt (abs x.re) 1) …
      -/
    · simpa [h_ne]
      /-
        🎉 no goals
      -/
      /-
        case neg
        K : Type u_1
        inst✝ : Field K
        f : NumberField.InfinitePlace K → NNReal
        w₀ : Subtype fun w => w.IsComplex
        x : K
        x✝ : And (∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w))  …
        h₁ : ∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w)
        h₂ : And (LT.lt (abs ((↑w₀).embedding x).re) 1) (LT.lt (abs ((↑w₀).embedding x …
        w : NumberField.InfinitePlace K
        hw : w.IsComplex
        h_ne : Not (Eq w ↑w₀)
        ⊢ Membership.mem (ite (Eq ⟨w, hw⟩ w₀) (setOf fun x => And (LT.lt (abs x.re) 1) …
      -/
    · rw [if_neg (by exact Subtype.coe_ne_coe.1 h_ne)]
      /-
        case neg
        K : Type u_1
        inst✝ : Field K
        f : NumberField.InfinitePlace K → NNReal
        w₀ : Subtype fun w => w.IsComplex
        x : K
        x✝ : And (∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w))  …
        h₁ : ∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w)
        h₂ : And (LT.lt (abs ((↑w₀).embedding x).re) 1) (LT.lt (abs ((↑w₀).embedding x …
        w : NumberField.InfinitePlace K
        hw : w.IsComplex
        h_ne : Not (Eq w ↑w₀)
        ⊢ Membership.mem (Metric.ball 0 ↑(f w)) (w.embedding x)
      -/
      rw [mem_ball_zero_iff, norm_embedding_eq]
      /-
        case neg
        K : Type u_1
        inst✝ : Field K
        f : NumberField.InfinitePlace K → NNReal
        w₀ : Subtype fun w => w.IsComplex
        x : K
        x✝ : And (∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w))  …
        h₁ : ∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w x) ↑(f w)
        h₂ : And (LT.lt (abs ((↑w₀).embedding x).re) 1) (LT.lt (abs ((↑w₀).embedding x …
        w : NumberField.InfinitePlace K
        hw : w.IsComplex
        h_ne : Not (Eq w ↑w₀)
        ⊢ LT.lt (w x) ↑(f w)
      -/
      exact h₁ w h_ne
      /-
        🎉 no goals
      -/


theorem convexBodyLT'_neg_mem (x : mixedSpace K) (hx : x ∈ convexBodyLT' K f w₀) :
    -x ∈ convexBodyLT' K f w₀ := by
  simp only [Set.mem_prod, Set.mem_pi, Set.mem_univ, mem_ball, dist_zero_right, Real.norm_eq_abs,
    true_implies, Subtype.forall, Prod.fst_neg, Pi.neg_apply, norm_neg, Prod.snd_neg] at hx ⊢
  /-
    K : Type u_1
    inst✝ : Field K
    f : NumberField.InfinitePlace K → NNReal
    w₀ : Subtype fun w => w.IsComplex
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : And (∀ (a : NumberField.InfinitePlace K) (b : a.IsReal), LT.lt (abs (x.1  …
    ⊢ And (∀ (a : NumberField.InfinitePlace K) (b : a.IsReal), LT.lt (abs (x.1 ⟨a, …
  -/
  convert hx using 3
  /-
    case h.e'_2.h.h.a
    K : Type u_1
    inst✝ : Field K
    f : NumberField.InfinitePlace K → NNReal
    w₀ : Subtype fun w => w.IsComplex
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : And (∀ (a : NumberField.InfinitePlace K) (b : a.IsReal), LT.lt (abs (x.1  …
    a✝¹ : NumberField.InfinitePlace K
    a✝ : a✝¹.IsComplex
    ⊢ Iff (Membership.mem (ite (Eq ⟨a✝¹, a✝⟩ w₀) (setOf fun x => And (LT.lt (abs x …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


theorem convexBodyLT'_convex : Convex ℝ (convexBodyLT' K f w₀) := by
  /-
    K : Type u_1
    inst✝ : Field K
    f : NumberField.InfinitePlace K → NNReal
    w₀ : Subtype fun w => w.IsComplex
    ⊢ Convex Real (NumberField.mixedEmbedding.convexBodyLT' K f w₀)
  -/
  refine Convex.prod (convex_pi (fun _ _ => convex_ball _ _)) (convex_pi (fun _ _ => ?_))
  /-
    K : Type u_1
    inst✝ : Field K
    f : NumberField.InfinitePlace K → NNReal
    w₀ x✝¹ : Subtype fun w => w.IsComplex
    x✝ : Membership.mem Set.univ x✝¹
    ⊢ Convex Real (ite (Eq x✝¹ w₀) (setOf fun x => And (LT.lt (abs x.re) 1) (LT.lt …
  -/
  split_ifs
    /-
      case pos
      K : Type u_1
      inst✝ : Field K
      f : NumberField.InfinitePlace K → NNReal
      w₀ x✝¹ : Subtype fun w => w.IsComplex
      x✝ : Membership.mem Set.univ x✝¹
      h✝ : Eq x✝¹ w₀
      ⊢ Convex Real (setOf fun x => And (LT.lt (abs x.re) 1) (LT.lt (abs x.im) (HPow …
    -/
  · simp_rw [abs_lt]
    refine Convex.inter ((convex_halfSpace_re_gt _).inter (convex_halfSpace_re_lt _))
      ((convex_halfSpace_im_gt _).inter (convex_halfSpace_im_lt _))
    /-
      case neg
      K : Type u_1
      inst✝ : Field K
      f : NumberField.InfinitePlace K → NNReal
      w₀ x✝¹ : Subtype fun w => w.IsComplex
      x✝ : Membership.mem Set.univ x✝¹
      h✝ : Not (Eq x✝¹ w₀)
      ⊢ Convex Real (Metric.ball 0 ↑(f ↑x✝¹))
    -/
  · exact convex_ball _ _
    /-
      🎉 no goals
    -/


/-- The fudge factor that appears in the formula for the volume of `convexBodyLT'`. -/
noncomputable abbrev convexBodyLT'Factor : ℝ≥0 :=
  (2 : ℝ≥0) ^ (nrRealPlaces K + 2) * NNReal.pi ^ (nrComplexPlaces K - 1)


theorem convexBodyLT'Factor_ne_zero : convexBodyLT'Factor K ≠ 0 :=
  mul_ne_zero (pow_ne_zero _ two_ne_zero) (pow_ne_zero _ pi_ne_zero)


theorem one_le_convexBodyLT'Factor : 1 ≤ convexBodyLT'Factor K :=
  one_le_mul (one_le_pow₀ one_le_two) (one_le_pow₀ (one_le_two.trans Real.two_le_pi))


theorem convexBodyLT'_volume :
    volume (convexBodyLT' K f w₀) = convexBodyLT'Factor K * ∏ w, (f w) ^ (mult w) := by
  have vol_box : ∀ B : ℝ≥0, volume {x : ℂ | |x.re| < 1 ∧ |x.im| < B^2} = 4*B^2 := by
    intro B
    rw [← (Complex.volume_preserving_equiv_real_prod.symm).measure_preimage]
    · simp_rw [Set.preimage_setOf_eq, Complex.measurableEquivRealProd_symm_apply]
      rw [show {a : ℝ × ℝ | |a.1| < 1 ∧ |a.2| < B ^ 2} =
        Set.Ioo (-1 : ℝ) (1 : ℝ) ×ˢ Set.Ioo (- (B : ℝ) ^ 2) ((B : ℝ) ^ 2) by
          ext; simp_rw [Set.mem_setOf_eq, Set.mem_prod, Set.mem_Ioo, abs_lt]]
      simp_rw [volume_eq_prod, prod_prod, Real.volume_Ioo, sub_neg_eq_add, one_add_one_eq_two,
        ← two_mul, ofReal_mul zero_le_two, ofReal_pow (coe_nonneg B), ofReal_ofNat,
        ofReal_coe_nnreal, ← mul_assoc, show (2 : ℝ≥0∞) * 2 = 4 by norm_num]
    · refine (MeasurableSet.inter ?_ ?_).nullMeasurableSet
      · exact measurableSet_lt (measurable_norm.comp Complex.measurable_re) measurable_const
      · exact measurableSet_lt (measurable_norm.comp Complex.measurable_im) measurable_const
  calc
    _ = (∏ x : {w // InfinitePlace.IsReal w}, ENNReal.ofReal (2 * (f x.val))) *
          ((∏ x ∈ Finset.univ.erase  w₀, ENNReal.ofReal (f x.val) ^ 2 * pi) *
          (4 * (f w₀) ^ 2)) := by
      simp_rw [volume_eq_prod, prod_prod, volume_pi, pi_pi, Real.volume_ball]
      rw [← Finset.prod_erase_mul _ _ (Finset.mem_univ w₀)]
      congr 2
      · refine Finset.prod_congr rfl (fun w' hw' ↦ ?_)
        rw [if_neg (Finset.ne_of_mem_erase hw'), Complex.volume_ball]
      · simpa only [ite_true] using vol_box (f w₀)
    _ = ((2 : ℝ≥0) ^ nrRealPlaces K *
          (∏ x : {w // InfinitePlace.IsReal w}, ENNReal.ofReal (f x.val))) *
            ((∏ x ∈ Finset.univ.erase  w₀, ENNReal.ofReal (f x.val) ^ 2) *
              ↑pi ^ (nrComplexPlaces K - 1) * (4 * (f w₀) ^ 2)) := by
      simp_rw [ofReal_mul (by norm_num : 0 ≤ (2 : ℝ)), Finset.prod_mul_distrib, Finset.prod_const,
        Finset.card_erase_of_mem (Finset.mem_univ _), Finset.card_univ, ofReal_ofNat,
        ofReal_coe_nnreal, coe_ofNat]
    _ = convexBodyLT'Factor K * (∏ x : {w // InfinitePlace.IsReal w}, ENNReal.ofReal (f x.val))
        * (∏ x : {w // IsComplex w}, ENNReal.ofReal (f x.val) ^ 2) := by
      rw [show (4 : ℝ≥0∞) = (2 : ℝ≥0) ^ 2 by norm_num, convexBodyLT'Factor, pow_add,
        ← Finset.prod_erase_mul _ _ (Finset.mem_univ w₀), ofReal_coe_nnreal]
      simp_rw [coe_mul, ENNReal.coe_pow]
      ring
    _ = convexBodyLT'Factor K * ∏ w, (f w) ^ (mult w) := by
      simp_rw [mult, pow_ite, pow_one, Finset.prod_ite, ofReal_coe_nnreal, not_isReal_iff_isComplex,
        coe_mul, coe_finset_prod, ENNReal.coe_pow, mul_assoc]
      congr 3
      · refine (Finset.prod_subtype (Finset.univ.filter _) ?_ (fun w => (f w : ℝ≥0∞))).symm
        exact fun _ => by simp only [Finset.mem_univ, forall_true_left, Finset.mem_filter, true_and]
      · refine (Finset.prod_subtype (Finset.univ.filter _) ?_ (fun w => (f w : ℝ≥0∞) ^ 2)).symm
        exact fun _ => by simp only [Finset.mem_univ, forall_true_left, Finset.mem_filter, true_and]


/-- The function that sends `x : mixedSpace K` to `∑ w, ‖x.1 w‖ + 2 * ∑ w, ‖x.2 w‖`. It defines a
norm and it used to define `convexBodySum`. -/
noncomputable abbrev convexBodySumFun (x : mixedSpace K) : ℝ := ∑ w, mult w * normAtPlace w x


theorem convexBodySumFun_apply (x : mixedSpace K) :
    convexBodySumFun x = ∑ w,  mult w * normAtPlace w x := rfl


theorem convexBodySumFun_apply' (x : mixedSpace K) :
    convexBodySumFun x = ∑ w, ‖x.1 w‖ + 2 * ∑ w, ‖x.2 w‖ := by
  simp_rw [convexBodySumFun_apply, ← Finset.sum_add_sum_compl {w | IsReal w}.toFinset,
    Set.toFinset_setOf, Finset.compl_filter, not_isReal_iff_isComplex, ← Finset.subtype_univ,
    ← Finset.univ.sum_subtype_eq_sum_filter, Finset.mul_sum]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Eq (HAdd.hAdd ((Finset.subtype NumberField.InfinitePlace.IsReal Finset.univ) …
  -/
  congr
    /-
      case e_a.e_f
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.mixedEmbedding.mixedSpace K
      ⊢ Eq (fun x_1 => HMul.hMul (↑(↑x_1).mult) ((NumberField.mixedEmbedding.normAtP …
    -/
  · ext w
    /-
      case e_a.e_f.h
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.mixedEmbedding.mixedSpace K
      w : Subtype NumberField.InfinitePlace.IsReal
      ⊢ Eq (HMul.hMul (↑(↑w).mult) ((NumberField.mixedEmbedding.normAtPlace ↑w) x))  …
    -/
    rw [mult, if_pos w.prop, normAtPlace_apply_isReal, Nat.cast_one, one_mul]
    /-
      🎉 no goals
    -/
    /-
      case e_a.e_f
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.mixedEmbedding.mixedSpace K
      ⊢ Eq (fun x_1 => HMul.hMul (↑(↑x_1).mult) ((NumberField.mixedEmbedding.normAtP …
    -/
  · ext w
    rw [mult, if_neg (not_isReal_iff_isComplex.mpr w.prop), normAtPlace_apply_isComplex,
      Nat.cast_ofNat]


theorem convexBodySumFun_nonneg (x : mixedSpace K) :
    0 ≤ convexBodySumFun x :=
  Finset.sum_nonneg (fun _ _ => mul_nonneg (Nat.cast_pos.mpr mult_pos).le (normAtPlace_nonneg _ _))


theorem convexBodySumFun_neg (x : mixedSpace K) :
    convexBodySumFun (- x) = convexBodySumFun x := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Eq (NumberField.mixedEmbedding.convexBodySumFun (Neg.neg x)) (NumberField.mi …
  -/
  simp_rw [convexBodySumFun, normAtPlace_neg]
  /-
    🎉 no goals
  -/


theorem convexBodySumFun_add_le (x y : mixedSpace K) :
    convexBodySumFun (x + y) ≤ convexBodySumFun x + convexBodySumFun y := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x y : NumberField.mixedEmbedding.mixedSpace K
    ⊢ LE.le (NumberField.mixedEmbedding.convexBodySumFun (HAdd.hAdd x y)) (HAdd.hA …
  -/
  simp_rw [convexBodySumFun, ← Finset.sum_add_distrib, ← mul_add]
  exact Finset.sum_le_sum
    fun _ _ ↦ mul_le_mul_of_nonneg_left (normAtPlace_add_le _ x y) (Nat.cast_pos.mpr mult_pos).le


theorem convexBodySumFun_smul (c : ℝ) (x : mixedSpace K) :
    convexBodySumFun (c • x) = |c| * convexBodySumFun x := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    c : Real
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Eq (NumberField.mixedEmbedding.convexBodySumFun (HSMul.hSMul c x)) (HMul.hMu …
  -/
  simp_rw [convexBodySumFun, normAtPlace_smul, ← mul_assoc, mul_comm, Finset.mul_sum, mul_assoc]
  /-
    🎉 no goals
  -/


theorem convexBodySumFun_eq_zero_iff (x : mixedSpace K) :
    convexBodySumFun x = 0 ↔ x = 0 := by
  rw [← forall_normAtPlace_eq_zero_iff, convexBodySumFun, Finset.sum_eq_zero_iff_of_nonneg
    fun _ _ ↦ mul_nonneg (Nat.cast_pos.mpr mult_pos).le (normAtPlace_nonneg _ _)]
  conv =>
    enter [1, w, hw]
    rw [mul_left_mem_nonZeroDivisors_eq_zero_iff
      (mem_nonZeroDivisors_iff_ne_zero.mpr <| Nat.cast_ne_zero.mpr mult_ne_zero)]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (∀ (w : NumberField.InfinitePlace K), Membership.mem Finset.univ w → Eq  …
  -/
  simp_rw [Finset.mem_univ, true_implies]
  /-
    🎉 no goals
  -/


theorem norm_le_convexBodySumFun (x : mixedSpace K) : ‖x‖ ≤ convexBodySumFun x := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ LE.le (Norm.norm x) (NumberField.mixedEmbedding.convexBodySumFun x)
  -/
  rw [norm_eq_sup'_normAtPlace]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ LE.le (Finset.univ.sup' ⋯ fun w => (NumberField.mixedEmbedding.normAtPlace w …
  -/
  refine (Finset.sup'_le_iff _ _).mpr fun w _ ↦ ?_
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    w : NumberField.InfinitePlace K
    x✝ : Membership.mem Finset.univ w
    ⊢ LE.le ((NumberField.mixedEmbedding.normAtPlace w) x) (NumberField.mixedEmbed …
  -/
  rw [convexBodySumFun_apply, ← Finset.univ.add_sum_erase _ (Finset.mem_univ w)]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    w : NumberField.InfinitePlace K
    x✝ : Membership.mem Finset.univ w
    ⊢ LE.le ((NumberField.mixedEmbedding.normAtPlace w) x) (HAdd.hAdd (HMul.hMul ( …
  -/
  refine le_add_of_le_of_nonneg  ?_ ?_
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.mixedEmbedding.mixedSpace K
      w : NumberField.InfinitePlace K
      x✝ : Membership.mem Finset.univ w
      ⊢ LE.le ((NumberField.mixedEmbedding.normAtPlace w) x) (HMul.hMul (↑w.mult) (( …
    -/
  · exact le_mul_of_one_le_left (normAtPlace_nonneg w x) one_le_mult
    /-
      🎉 no goals
    -/
  · exact Finset.sum_nonneg (fun _ _ => mul_nonneg (Nat.cast_pos.mpr mult_pos).le
      (normAtPlace_nonneg _ _))


theorem convexBodySumFun_continuous :
    Continuous (convexBodySumFun : mixedSpace K → ℝ) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Continuous NumberField.mixedEmbedding.convexBodySumFun
  -/
  refine continuous_finset_sum Finset.univ fun w ↦ ?_
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w : NumberField.InfinitePlace K
    ⊢ Membership.mem Finset.univ w → Continuous fun a => HMul.hMul (↑w.mult) ((Num …
  -/
  obtain hw | hw := isReal_or_isComplex w
  all_goals
  · simp only [normAtPlace_apply_isReal, normAtPlace_apply_isComplex, hw]
    fun_prop


/-- The convex body equal to the set of points `x : mixedSpace K` such that
  `∑ w real, ‖x w‖ + 2 * ∑ w complex, ‖x w‖ ≤ B`. -/
abbrev convexBodySum : Set (mixedSpace K)  := { x | convexBodySumFun x ≤ B }


theorem convexBodySum_volume_eq_zero_of_le_zero {B} (hB : B ≤ 0) :
    volume (convexBodySum K B) = 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    hB : LE.le B 0
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (NumberField.mixedEmbedding.convexBody …
  -/
  obtain hB | hB := lt_or_eq_of_le hB
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB✝ : LE.le B 0
      hB : LT.lt B 0
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (NumberField.mixedEmbedding.convexBody …
    -/
  · suffices convexBodySum K B = ∅ by rw [this, measure_empty]
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB✝ : LE.le B 0
      hB : LT.lt B 0
      ⊢ Eq (NumberField.mixedEmbedding.convexBodySum K B) EmptyCollection.emptyColle …
    -/
    ext x
    /-
      case inl.h
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB✝ : LE.le B 0
      hB : LT.lt B 0
      x : NumberField.mixedEmbedding.mixedSpace K
      ⊢ Iff (Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) x) (Membe …
    -/
    refine ⟨fun hx => ?_, fun h => h.elim⟩
    /-
      case inl.h
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB✝ : LE.le B 0
      hB : LT.lt B 0
      x : NumberField.mixedEmbedding.mixedSpace K
      hx : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) x
      ⊢ Membership.mem EmptyCollection.emptyCollection x
    -/
    rw [Set.mem_setOf] at hx
    /-
      case inl.h
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB✝ : LE.le B 0
      hB : LT.lt B 0
      x : NumberField.mixedEmbedding.mixedSpace K
      hx : LE.le (NumberField.mixedEmbedding.convexBodySumFun x) B
      ⊢ Membership.mem EmptyCollection.emptyCollection x
    -/
    linarith [convexBodySumFun_nonneg x]
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB✝ : LE.le B 0
      hB : Eq B 0
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (NumberField.mixedEmbedding.convexBody …
    -/
  · suffices convexBodySum K B = { 0 } by rw [this, measure_singleton]
    /-
      case inr
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB✝ : LE.le B 0
      hB : Eq B 0
      ⊢ Eq (NumberField.mixedEmbedding.convexBodySum K B) (Singleton.singleton 0)
    -/
    ext
    /-
      case inr.h
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB✝ : LE.le B 0
      hB : Eq B 0
      x✝ : NumberField.mixedEmbedding.mixedSpace K
      ⊢ Iff (Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) x✝) (Memb …
    -/
    rw [convexBodySum, Set.mem_setOf_eq, Set.mem_singleton_iff, hB, ← convexBodySumFun_eq_zero_iff]
    /-
      case inr.h
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB✝ : LE.le B 0
      hB : Eq B 0
      x✝ : NumberField.mixedEmbedding.mixedSpace K
      ⊢ Iff (LE.le (NumberField.mixedEmbedding.convexBodySumFun x✝) 0) (Eq (NumberFi …
    -/
    exact (convexBodySumFun_nonneg _).le_iff_eq
    /-
      🎉 no goals
    -/


theorem convexBodySum_mem {x : K} :
    mixedEmbedding K x ∈ (convexBodySum K B) ↔
      ∑ w : InfinitePlace K, (mult w) * w.val x ≤ B := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    x : K
    ⊢ Iff (Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) ((NumberF …
  -/
  simp_rw [Set.mem_setOf_eq, convexBodySumFun, normAtPlace_apply]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    x : K
    ⊢ Iff (LE.le (Finset.univ.sum fun x_1 => HMul.hMul (↑x_1.mult) (x_1 x)) B) (LE …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem convexBodySum_neg_mem {x : mixedSpace K} (hx : x ∈ (convexBodySum K B)) :
    -x ∈ (convexBodySum K B) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) x
    ⊢ Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) (Neg.neg x)
  -/
  rw [Set.mem_setOf, convexBodySumFun_neg]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) x
    ⊢ LE.le (NumberField.mixedEmbedding.convexBodySumFun x) B
  -/
  exact hx
  /-
    🎉 no goals
  -/


theorem convexBodySum_convex : Convex ℝ (convexBodySum K B) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    ⊢ Convex Real (NumberField.mixedEmbedding.convexBodySum K B)
  -/
  refine Convex_subadditive_le (fun _ _ => convexBodySumFun_add_le _ _) (fun c x h => ?_) B
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B c : Real
    x : NumberField.mixedEmbedding.mixedSpace K
    h : LE.le 0 c
    ⊢ LE.le (NumberField.mixedEmbedding.convexBodySumFun (HSMul.hSMul c x)) (HMul. …
  -/
  convert le_of_eq (convexBodySumFun_smul c x)
  /-
    case h.e'_4.h.e'_5
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B c : Real
    x : NumberField.mixedEmbedding.mixedSpace K
    h : LE.le 0 c
    ⊢ Eq c (abs c)
  -/
  exact (abs_eq_self.mpr h).symm
  /-
    🎉 no goals
  -/


theorem convexBodySum_isBounded : Bornology.IsBounded (convexBodySum K B) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    ⊢ Bornology.IsBounded (NumberField.mixedEmbedding.convexBodySum K B)
  -/
  refine Metric.isBounded_iff.mpr ⟨B + B, fun x hx y hy => ?_⟩
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) x
    y : NumberField.mixedEmbedding.mixedSpace K
    hy : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) y
    ⊢ LE.le (Dist.dist x y) (HAdd.hAdd B B)
  -/
  refine le_trans (norm_sub_le x y) (add_le_add ?_ ?_)
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      x : NumberField.mixedEmbedding.mixedSpace K
      hx : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) x
      y : NumberField.mixedEmbedding.mixedSpace K
      hy : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) y
      ⊢ LE.le (Norm.norm x) B
    -/
  · exact le_trans (norm_le_convexBodySumFun x) hx
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      x : NumberField.mixedEmbedding.mixedSpace K
      hx : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) x
      y : NumberField.mixedEmbedding.mixedSpace K
      hy : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) y
      ⊢ LE.le (Norm.norm y) B
    -/
  · exact le_trans (norm_le_convexBodySumFun y) hy
    /-
      🎉 no goals
    -/


theorem convexBodySum_compact : IsCompact (convexBodySum K B) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    ⊢ IsCompact (NumberField.mixedEmbedding.convexBodySum K B)
  -/
  rw [Metric.isCompact_iff_isClosed_bounded]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    ⊢ And (IsClosed (NumberField.mixedEmbedding.convexBodySum K B)) (Bornology.IsB …
  -/
  refine ⟨?_, convexBodySum_isBounded K B⟩
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    ⊢ IsClosed (NumberField.mixedEmbedding.convexBodySum K B)
  -/
  convert IsClosed.preimage (convexBodySumFun_continuous K) (isClosed_Icc : IsClosed (Set.Icc 0 B))
  /-
    case h.e'_3
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    ⊢ Eq (NumberField.mixedEmbedding.convexBodySum K B) (Set.preimage NumberField. …
  -/
  ext
  /-
    case h.e'_3.h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    x✝ : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) x✝) (Memb …
  -/
  simp [convexBodySumFun_nonneg]
  /-
    🎉 no goals
  -/


/-- The fudge factor that appears in the formula for the volume of `convexBodyLt`. -/
noncomputable abbrev convexBodySumFactor : ℝ≥0 :=
  (2 : ℝ≥0) ^ nrRealPlaces K * (NNReal.pi / 2) ^ nrComplexPlaces K / (finrank ℚ K).factorial


theorem convexBodySumFactor_ne_zero : convexBodySumFactor K ≠ 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Ne (NumberField.mixedEmbedding.convexBodySumFactor K) 0
  -/
  refine div_ne_zero ?_ <| Nat.cast_ne_zero.mpr (Nat.factorial_ne_zero _)
  exact mul_ne_zero (pow_ne_zero _ two_ne_zero)
    (pow_ne_zero _ (div_ne_zero NNReal.pi_ne_zero two_ne_zero))


open MeasureTheory MeasureTheory.Measure Real in
theorem convexBodySum_volume :
    volume (convexBodySum K B) = (convexBodySumFactor K) * (.ofReal B) ^ (finrank ℚ K) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (NumberField.mixedEmbedding.convexBody …
  -/
  obtain hB | hB := le_or_lt B 0
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB : LE.le B 0
      ⊢ Eq (MeasureTheory.MeasureSpace.volume (NumberField.mixedEmbedding.convexBody …
    -/
  · rw [convexBodySum_volume_eq_zero_of_le_zero K hB, ofReal_eq_zero.mpr hB, zero_pow, mul_zero]
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      B : Real
      hB : LE.le B 0
      ⊢ Ne (Module.finrank Rat K) 0
    -/
    exact finrank_pos.ne'
    /-
      🎉 no goals
    -/
  · suffices volume (convexBodySum K 1) = (convexBodySumFactor K) by
      rw [mul_comm]
      convert addHaar_smul volume B (convexBodySum K 1)
      · simp_rw [← Set.preimage_smul_inv₀ (ne_of_gt hB), Set.preimage_setOf_eq, convexBodySumFun,
        normAtPlace_smul, abs_inv, abs_eq_self.mpr (le_of_lt hB), ← mul_assoc, mul_comm, mul_assoc,
        ← Finset.mul_sum, inv_mul_le_iff₀ hB, mul_one]
      · rw [abs_pow, ofReal_pow (abs_nonneg _), abs_eq_self.mpr (le_of_lt hB),
          mixedEmbedding.finrank]
      · exact this.symm
    rw [MeasureTheory.measure_le_eq_lt _ ((convexBodySumFun_eq_zero_iff 0).mpr rfl)
      convexBodySumFun_neg convexBodySumFun_add_le
      (fun hx => (convexBodySumFun_eq_zero_iff _).mp hx)
      (fun r x => le_of_eq (convexBodySumFun_smul r x))]
    rw [measure_lt_one_eq_integral_div_gamma (g := fun x : (mixedSpace K) => convexBodySumFun x)
      volume ((convexBodySumFun_eq_zero_iff 0).mpr rfl) convexBodySumFun_neg convexBodySumFun_add_le
      (fun hx => (convexBodySumFun_eq_zero_iff _).mp hx)
      (fun r x => le_of_eq (convexBodySumFun_smul r x)) zero_lt_one]
    simp_rw [mixedEmbedding.finrank, div_one, Gamma_nat_eq_factorial, ofReal_div_of_pos
      (Nat.cast_pos.mpr (Nat.factorial_pos _)), Real.rpow_one, ofReal_natCast]
    suffices ∫ x : mixedSpace K, exp (-convexBodySumFun x) =
        (2 : ℝ) ^ nrRealPlaces K * (π / 2) ^ nrComplexPlaces K by
      rw [this, convexBodySumFactor, ofReal_mul (by positivity), ofReal_pow zero_le_two,
        ofReal_pow (by positivity), ofReal_div_of_pos zero_lt_two, ofReal_ofNat,
        ← NNReal.coe_real_pi, ofReal_coe_nnreal, coe_div (Nat.cast_ne_zero.mpr
        (Nat.factorial_ne_zero _)), coe_mul, coe_pow, coe_pow, coe_ofNat, coe_div two_ne_zero,
        coe_ofNat, coe_natCast]
    calc
      _ = (∫ x : {w : InfinitePlace K // IsReal w} → ℝ, ∏ w, exp (- ‖x w‖)) *
              (∫ x : {w : InfinitePlace K // IsComplex w} → ℂ, ∏ w, exp (- 2 * ‖x w‖)) := by
        simp_rw [convexBodySumFun_apply', neg_add, ← neg_mul, Finset.mul_sum,
          ← Finset.sum_neg_distrib, exp_add, exp_sum, ← integral_prod_mul, volume_eq_prod]
      _ = (∫ x : ℝ, exp (-|x|)) ^ nrRealPlaces K *
              (∫ x : ℂ, Real.exp (-2 * ‖x‖)) ^ nrComplexPlaces K := by
        rw [integral_fintype_prod_eq_pow _ (fun x => exp (- ‖x‖)), integral_fintype_prod_eq_pow _
          (fun x => exp (- 2 * ‖x‖))]
        simp_rw [norm_eq_abs]
      _ =  (2 * Gamma (1 / 1 + 1)) ^ nrRealPlaces K *
              (π * (2 : ℝ) ^ (-(2 : ℝ) / 1) * Gamma (2 / 1 + 1)) ^ nrComplexPlaces K := by
        rw [integral_comp_abs (f := fun x => exp (- x)), ← integral_exp_neg_rpow zero_lt_one,
          ← Complex.integral_exp_neg_mul_rpow le_rfl zero_lt_two]
        simp_rw [Real.rpow_one]
      _ = (2 : ℝ) ^ nrRealPlaces K * (π / 2) ^ nrComplexPlaces K := by
        simp_rw [div_one, one_add_one_eq_two, Gamma_add_one two_ne_zero, Gamma_two, mul_one,
          mul_assoc, ← Real.rpow_add_one two_ne_zero, show (-2 : ℝ) + 1 = -1 by norm_num,
          Real.rpow_neg_one, div_eq_mul_inv]


/-- The bound that appears in **Minkowski Convex Body theorem**, see
`MeasureTheory.exists_ne_zero_mem_lattice_of_measure_mul_two_pow_lt_measure`. See
`NumberField.mixedEmbedding.volume_fundamentalDomain_idealLatticeBasis_eq` and
`NumberField.mixedEmbedding.volume_fundamentalDomain_latticeBasis` for the computation of
`volume (fundamentalDomain (idealLatticeBasis K))`. -/
noncomputable def minkowskiBound : ℝ≥0∞ :=
  volume (fundamentalDomain (fractionalIdealLatticeBasis K I)) *
    (2 : ℝ≥0∞) ^ (finrank ℝ (mixedSpace K))


theorem volume_fundamentalDomain_fractionalIdealLatticeBasis :
    volume (fundamentalDomain (fractionalIdealLatticeBasis K I)) =
      .ofReal (FractionalIdeal.absNorm I.1) *  volume (fundamentalDomain (latticeBasis K)) := by
  let e : (Module.Free.ChooseBasisIndex ℤ I) ≃ (Module.Free.ChooseBasisIndex ℤ (𝓞 K)) := by
    refine Fintype.equivOfCardEq ?_
    rw [← finrank_eq_card_chooseBasisIndex, ← finrank_eq_card_chooseBasisIndex,
      fractionalIdeal_rank]
  rw [← fundamentalDomain_reindex (fractionalIdealLatticeBasis K I) e,
    measure_fundamentalDomain ((fractionalIdealLatticeBasis K I).reindex e)]
  · rw [show (fractionalIdealLatticeBasis K I).reindex e = (mixedEmbedding K) ∘
        (basisOfFractionalIdeal K I) ∘ e.symm by
      ext1; simp only [Basis.coe_reindex, Function.comp_apply, fractionalIdealLatticeBasis_apply]]
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      e : Equiv (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem ( …
      ⊢ Eq (HMul.hMul (ENNReal.ofReal (abs ((Basis.det ?b₀) (Function.comp (⇑(Number …
    -/
    rw [mixedEmbedding.det_basisOfFractionalIdeal_eq_norm]
    /-
      🎉 no goals
    -/


theorem minkowskiBound_lt_top : minkowskiBound K I < ⊤ := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    ⊢ LT.lt (NumberField.mixedEmbedding.minkowskiBound K I) Top.top
  -/
  refine ENNReal.mul_lt_top ?_ ?_
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      ⊢ LT.lt (MeasureTheory.MeasureSpace.volume (ZSpan.fundamentalDomain (NumberFie …
    -/
  · exact (fundamentalDomain_isBounded _).measure_lt_top
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      ⊢ LT.lt (HPow.hPow 2 (Module.finrank Real (NumberField.mixedEmbedding.mixedSpa …
    -/
  · exact ENNReal.pow_lt_top (lt_top_iff_ne_top.mpr ENNReal.two_ne_top) _
    /-
      🎉 no goals
    -/


theorem minkowskiBound_pos : 0 < minkowskiBound K I := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    ⊢ LT.lt 0 (NumberField.mixedEmbedding.minkowskiBound K I)
  -/
  refine zero_lt_iff.mpr (mul_ne_zero ?_ ?_)
    /-
      case refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      ⊢ Ne (MeasureTheory.MeasureSpace.volume (ZSpan.fundamentalDomain (NumberField. …
    -/
  · exact ZSpan.measure_fundamentalDomain_ne_zero _
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      ⊢ Ne (HPow.hPow 2 (Module.finrank Real (NumberField.mixedEmbedding.mixedSpace  …
    -/
  · exact ENNReal.pow_ne_zero two_ne_zero _
    /-
      🎉 no goals
    -/


/-- Let `I` be a fractional ideal of `K`. Assume that `f : InfinitePlace K → ℝ≥0` is such that
`minkowskiBound K I < volume (convexBodyLT K f)` where `convexBodyLT K f` is the set of
points `x` such that `‖x w‖ < f w` for all infinite places `w` (see `convexBodyLT_volume` for
the computation of this volume), then there exists a nonzero algebraic number `a` in `I` such
that `w a < f w` for all infinite places `w`. -/
theorem exists_ne_zero_mem_ideal_lt (h : minkowskiBound K I < volume (convexBodyLT K f)) :
    ∃ a ∈ (I : FractionalIdeal (𝓞 K)⁰ K), a ≠ 0 ∧ ∀ w : InfinitePlace K, w a < f w := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (∀ (w : NumberFiel …
  -/
  have h_fund := ZSpan.isAddFundamentalDomain' (fractionalIdealLatticeBasis K I) volume
  have : Countable (span ℤ (Set.range (fractionalIdealLatticeBasis K I))).toAddSubgroup := by
    change Countable (span ℤ (Set.range (fractionalIdealLatticeBasis K I)))
    infer_instance
  obtain ⟨⟨x, hx⟩, h_nz, h_mem⟩ := exists_ne_zero_mem_lattice_of_measure_mul_two_pow_lt_measure
    h_fund (convexBodyLT_neg_mem K f) (convexBodyLT_convex K f) h
  /-
    case intro.mk.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
    h_nz : Ne ⟨x, hx⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodyLT K f) ↑⟨x, hx⟩
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (∀ (w : NumberFiel …
  -/
  rw [mem_toAddSubgroup, mem_span_fractionalIdealLatticeBasis] at hx
  /-
    case intro.mk.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    x : NumberField.mixedEmbedding.mixedSpace K
    hx✝ : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddi …
    hx : Membership.mem (Set.image ⇑(NumberField.mixedEmbedding K) ↑↑I) x
    h_nz : Ne ⟨x, hx✝⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodyLT K f) ↑⟨x, hx✝⟩
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (∀ (w : NumberFiel …
  -/
  obtain ⟨a, ha, rfl⟩ := hx
  /-
    case intro.mk.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    a : K
    ha : Membership.mem (↑↑I) a
    hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
    h_nz : Ne ⟨(NumberField.mixedEmbedding K) a, hx⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodyLT K f) ↑⟨(Number …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (∀ (w : NumberFiel …
  -/
  exact ⟨a, ha, by simpa using h_nz, (convexBodyLT_mem K f).mp h_mem⟩
  /-
    🎉 no goals
  -/


/-- A version of `exists_ne_zero_mem_ideal_lt` where the absolute value of the real part of `a` is
smaller than `1` at some fixed complex place. This is useful to ensure that `a` is not real. -/
theorem exists_ne_zero_mem_ideal_lt' (w₀ : {w : InfinitePlace K // IsComplex w})
    (h : minkowskiBound K I < volume (convexBodyLT' K f w₀)) :
    ∃ a ∈ (I : FractionalIdeal (𝓞 K)⁰ K), a ≠ 0 ∧ (∀ w : InfinitePlace K, w ≠ w₀ → w a < f w) ∧
      |(w₀.val.embedding a).re| < 1 ∧ |(w₀.val.embedding a).im| < (f w₀ : ℝ) ^ 2 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    w₀ : Subtype fun w => w.IsComplex
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (And (∀ (w : Numbe …
  -/
  have h_fund := ZSpan.isAddFundamentalDomain' (fractionalIdealLatticeBasis K I) volume
  have : Countable (span ℤ (Set.range (fractionalIdealLatticeBasis K I))).toAddSubgroup := by
    change Countable (span ℤ (Set.range (fractionalIdealLatticeBasis K I)))
    infer_instance
  obtain ⟨⟨x, hx⟩, h_nz, h_mem⟩ := exists_ne_zero_mem_lattice_of_measure_mul_two_pow_lt_measure
    h_fund (convexBodyLT'_neg_mem K f w₀) (convexBodyLT'_convex K f w₀) h
  /-
    case intro.mk.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    w₀ : Subtype fun w => w.IsComplex
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
    h_nz : Ne ⟨x, hx⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodyLT' K f w₀) ↑⟨x,  …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (And (∀ (w : Numbe …
  -/
  rw [mem_toAddSubgroup, mem_span_fractionalIdealLatticeBasis] at hx
  /-
    case intro.mk.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    w₀ : Subtype fun w => w.IsComplex
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    x : NumberField.mixedEmbedding.mixedSpace K
    hx✝ : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddi …
    hx : Membership.mem (Set.image ⇑(NumberField.mixedEmbedding K) ↑↑I) x
    h_nz : Ne ⟨x, hx✝⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodyLT' K f w₀) ↑⟨x,  …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (And (∀ (w : Numbe …
  -/
  obtain ⟨a, ha, rfl⟩ := hx
  /-
    case intro.mk.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    w₀ : Subtype fun w => w.IsComplex
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    a : K
    ha : Membership.mem (↑↑I) a
    hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
    h_nz : Ne ⟨(NumberField.mixedEmbedding K) a, hx⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodyLT' K f w₀) ↑⟨(Nu …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (And (∀ (w : Numbe …
  -/
  exact ⟨a, ha, by simpa using h_nz, (convexBodyLT'_mem K f w₀).mp h_mem⟩
  /-
    🎉 no goals
  -/


/-- A version of `exists_ne_zero_mem_ideal_lt` for the ring of integers of `K`. -/
theorem exists_ne_zero_mem_ringOfIntegers_lt (h : minkowskiBound K ↑1 < volume (convexBodyLT K f)) :
    ∃ a : 𝓞 K, a ≠ 0 ∧ ∀ w : InfinitePlace K, w a < f w := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Measu …
    ⊢ Exists fun a => And (Ne a 0) (∀ (w : NumberField.InfinitePlace K), LT.lt (w  …
  -/
  obtain ⟨_, h_mem, h_nz, h_bd⟩ := exists_ne_zero_mem_ideal_lt K ↑1 h
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Measu …
    w✝ : K
    h_mem : Membership.mem (↑1) w✝
    h_nz : Ne w✝ 0
    h_bd : ∀ (w : NumberField.InfinitePlace K), LT.lt (w w✝) ↑(f w)
    ⊢ Exists fun a => And (Ne a 0) (∀ (w : NumberField.InfinitePlace K), LT.lt (w  …
  -/
  obtain ⟨a, rfl⟩ := (FractionalIdeal.mem_one_iff _).mp h_mem
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Measu …
    a : NumberField.RingOfIntegers K
    h_mem : Membership.mem (↑1) ((algebraMap (NumberField.RingOfIntegers K) K) a)
    h_nz : Ne ((algebraMap (NumberField.RingOfIntegers K) K) a) 0
    h_bd : ∀ (w : NumberField.InfinitePlace K), LT.lt (w ((algebraMap (NumberField …
    ⊢ Exists fun a => And (Ne a 0) (∀ (w : NumberField.InfinitePlace K), LT.lt (w  …
  -/
  exact ⟨a, RingOfIntegers.coe_ne_zero_iff.mp h_nz, h_bd⟩
  /-
    🎉 no goals
  -/


/-- A version of `exists_ne_zero_mem_ideal_lt'` for the ring of integers of `K`. -/
theorem exists_ne_zero_mem_ringOfIntegers_lt' (w₀ : {w : InfinitePlace K // IsComplex w})
    (h : minkowskiBound K ↑1 < volume (convexBodyLT' K f w₀)) :
    ∃ a : 𝓞 K, a ≠ 0 ∧ (∀ w : InfinitePlace K, w ≠ w₀ → w a < f w) ∧
      |(w₀.val.embedding a).re| < 1 ∧ |(w₀.val.embedding a).im| < (f w₀ : ℝ) ^ 2 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    w₀ : Subtype fun w => w.IsComplex
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Measu …
    ⊢ Exists fun a => And (Ne a 0) (And (∀ (w : NumberField.InfinitePlace K), Ne w …
  -/
  obtain ⟨_, h_mem, h_nz, h_bd⟩ := exists_ne_zero_mem_ideal_lt' K ↑1 w₀ h
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    w₀ : Subtype fun w => w.IsComplex
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Measu …
    w✝ : K
    h_mem : Membership.mem (↑1) w✝
    h_nz : Ne w✝ 0
    h_bd : And (∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w w✝) ↑(f w …
    ⊢ Exists fun a => And (Ne a 0) (And (∀ (w : NumberField.InfinitePlace K), Ne w …
  -/
  obtain ⟨a, rfl⟩ := (FractionalIdeal.mem_one_iff _).mp h_mem
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    f : NumberField.InfinitePlace K → NNReal
    w₀ : Subtype fun w => w.IsComplex
    h : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Measu …
    a : NumberField.RingOfIntegers K
    h_mem : Membership.mem (↑1) ((algebraMap (NumberField.RingOfIntegers K) K) a)
    h_nz : Ne ((algebraMap (NumberField.RingOfIntegers K) K) a) 0
    h_bd : And (∀ (w : NumberField.InfinitePlace K), Ne w ↑w₀ → LT.lt (w ((algebra …
    ⊢ Exists fun a => And (Ne a 0) (And (∀ (w : NumberField.InfinitePlace K), Ne w …
  -/
  exact ⟨a, RingOfIntegers.coe_ne_zero_iff.mp h_nz, h_bd⟩
  /-
    🎉 no goals
  -/


theorem exists_primitive_element_lt_of_isReal {w₀ : InfinitePlace K} (hw₀ : IsReal w₀) {B : ℝ≥0}
    (hB : minkowskiBound K ↑1 < convexBodyLTFactor K * B) :
    ∃ a : 𝓞 K, ℚ⟮(a : K)⟯ = ⊤ ∧
      ∀ w : InfinitePlace K, w a < max B 1 := by
  have : minkowskiBound K ↑1 < volume (convexBodyLT K (fun w ↦ if w = w₀ then B else 1)) := by
    rw [convexBodyLT_volume, ← Finset.prod_erase_mul _ _ (Finset.mem_univ w₀)]
    simp_rw [ite_pow, one_pow]
    rw [Finset.prod_ite_eq']
    simp_rw [Finset.not_mem_erase, ite_false, mult, hw₀, ite_true, one_mul, pow_one]
    exact hB
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w₀ : NumberField.InfinitePlace K
    hw₀ : w₀.IsReal
    B : NNReal
    hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
    this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
    ⊢ Exists fun a => And (Eq (IntermediateField.adjoin Rat (Singleton.singleton ↑ …
  -/
  obtain ⟨a, h_nz, h_le⟩ := exists_ne_zero_mem_ringOfIntegers_lt K this
  /-
    case intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w₀ : NumberField.InfinitePlace K
    hw₀ : w₀.IsReal
    B : NNReal
    hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
    this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
    a : NumberField.RingOfIntegers K
    h_nz : Ne a 0
    h_le : ∀ (w : NumberField.InfinitePlace K), LT.lt (w ↑a) ↑(ite (Eq w w₀) B 1)
    ⊢ Exists fun a => And (Eq (IntermediateField.adjoin Rat (Singleton.singleton ↑ …
  -/
  refine ⟨a, ?_, fun w ↦ lt_of_lt_of_le (h_le w) ?_⟩
  · exact is_primitive_element_of_infinitePlace_lt h_nz
      (fun w h_ne ↦ by convert (if_neg h_ne) ▸ h_le w) (Or.inl hw₀)
    /-
      case intro.intro.refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w₀ : NumberField.InfinitePlace K
      hw₀ : w₀.IsReal
      B : NNReal
      hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
      this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
      a : NumberField.RingOfIntegers K
      h_nz : Ne a 0
      h_le : ∀ (w : NumberField.InfinitePlace K), LT.lt (w ↑a) ↑(ite (Eq w w₀) B 1)
      w : NumberField.InfinitePlace K
      ⊢ LE.le ↑(ite (Eq w w₀) B 1) ↑(Max.max B 1)
    -/
                  /-
                    🎉 no goals
                  -/
  · split_ifs <;> simp
                  /-
                    🎉 no goals
                  -/


theorem exists_primitive_element_lt_of_isComplex {w₀ : InfinitePlace K} (hw₀ : IsComplex w₀)
    {B : ℝ≥0} (hB : minkowskiBound K ↑1 < convexBodyLT'Factor K * B) :
    ∃ a : 𝓞 K, ℚ⟮(a : K)⟯ = ⊤ ∧
      ∀ w : InfinitePlace K, w a < Real.sqrt (1 + B ^ 2) := by
  have : minkowskiBound K ↑1 <
      volume (convexBodyLT' K (fun w ↦ if w = w₀ then NNReal.sqrt B else 1) ⟨w₀, hw₀⟩) := by
    rw [convexBodyLT'_volume, ← Finset.prod_erase_mul _ _ (Finset.mem_univ w₀)]
    simp_rw [ite_pow, one_pow]
    rw [Finset.prod_ite_eq']
    simp_rw [Finset.not_mem_erase, ite_false, mult, not_isReal_iff_isComplex.mpr hw₀,
      ite_true, ite_false, one_mul, NNReal.sq_sqrt]
    exact hB
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w₀ : NumberField.InfinitePlace K
    hw₀ : w₀.IsComplex
    B : NNReal
    hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
    this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
    ⊢ Exists fun a => And (Eq (IntermediateField.adjoin Rat (Singleton.singleton ↑ …
  -/
  obtain ⟨a, h_nz, h_le, h_le₀⟩ := exists_ne_zero_mem_ringOfIntegers_lt' K ⟨w₀, hw₀⟩ this
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w₀ : NumberField.InfinitePlace K
    hw₀ : w₀.IsComplex
    B : NNReal
    hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
    this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
    a : NumberField.RingOfIntegers K
    h_nz : Ne a 0
    h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
    h_le₀ : And (LT.lt (abs ((↑⟨w₀, hw₀⟩).embedding ↑a).re) 1) (LT.lt (abs ((↑⟨w₀, …
    ⊢ Exists fun a => And (Eq (IntermediateField.adjoin Rat (Singleton.singleton ↑ …
  -/
  refine ⟨a, ?_, fun w ↦ ?_⟩
  · exact is_primitive_element_of_infinitePlace_lt h_nz
      (fun w h_ne ↦ by convert if_neg h_ne ▸ h_le w h_ne) (Or.inr h_le₀.1)
    /-
      case intro.intro.intro.refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w₀ : NumberField.InfinitePlace K
      hw₀ : w₀.IsComplex
      B : NNReal
      hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
      this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
      a : NumberField.RingOfIntegers K
      h_nz : Ne a 0
      h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
      h_le₀ : And (LT.lt (abs ((↑⟨w₀, hw₀⟩).embedding ↑a).re) 1) (LT.lt (abs ((↑⟨w₀, …
      w : NumberField.InfinitePlace K
      ⊢ LT.lt (w ↑a) (HAdd.hAdd 1 (HPow.hPow (↑B) 2)).sqrt
    -/
  · by_cases h_eq : w = w₀
      /-
        case pos
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        w₀ : NumberField.InfinitePlace K
        hw₀ : w₀.IsComplex
        B : NNReal
        hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
        this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
        a : NumberField.RingOfIntegers K
        h_nz : Ne a 0
        h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
        h_le₀ : And (LT.lt (abs ((↑⟨w₀, hw₀⟩).embedding ↑a).re) 1) (LT.lt (abs ((↑⟨w₀, …
        w : NumberField.InfinitePlace K
        h_eq : Eq w w₀
        ⊢ LT.lt (w ↑a) (HAdd.hAdd 1 (HPow.hPow (↑B) 2)).sqrt
      -/
    · rw [if_pos rfl] at h_le₀
      /-
        case pos
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        w₀ : NumberField.InfinitePlace K
        hw₀ : w₀.IsComplex
        B : NNReal
        hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
        this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
        a : NumberField.RingOfIntegers K
        h_nz : Ne a 0
        h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
        h_le₀ : And (LT.lt (abs ((↑⟨w₀, hw₀⟩).embedding ↑a).re) 1) (LT.lt (abs ((↑⟨w₀, …
        w : NumberField.InfinitePlace K
        h_eq : Eq w w₀
        ⊢ LT.lt (w ↑a) (HAdd.hAdd 1 (HPow.hPow (↑B) 2)).sqrt
      -/
      dsimp only at h_le₀
      rw [h_eq, ← norm_embedding_eq, Real.lt_sqrt (norm_nonneg _), ← Complex.re_add_im
        (embedding w₀ _), Complex.norm_eq_abs, Complex.abs_add_mul_I, Real.sq_sqrt (by positivity)]
      /-
        case pos
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        w₀ : NumberField.InfinitePlace K
        hw₀ : w₀.IsComplex
        B : NNReal
        hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
        this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
        a : NumberField.RingOfIntegers K
        h_nz : Ne a 0
        h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
        h_le₀ : And (LT.lt (abs (w₀.embedding ↑a).re) 1) (LT.lt (abs (w₀.embedding ↑a) …
        w : NumberField.InfinitePlace K
        h_eq : Eq w w₀
        ⊢ LT.lt (HAdd.hAdd (HPow.hPow (w₀.embedding ↑a).re 2) (HPow.hPow (w₀.embedding …
      -/
      refine add_lt_add ?_ ?_
        /-
          case pos.refine_1
          K : Type u_1
          inst✝¹ : Field K
          inst✝ : NumberField K
          w₀ : NumberField.InfinitePlace K
          hw₀ : w₀.IsComplex
          B : NNReal
          hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
          this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
          a : NumberField.RingOfIntegers K
          h_nz : Ne a 0
          h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
          h_le₀ : And (LT.lt (abs (w₀.embedding ↑a).re) 1) (LT.lt (abs (w₀.embedding ↑a) …
          w : NumberField.InfinitePlace K
          h_eq : Eq w w₀
          ⊢ LT.lt (HPow.hPow (w₀.embedding ↑a).re 2) 1
        -/
      · rw [← sq_abs, sq_lt_one_iff₀ (abs_nonneg _)]
        /-
          case pos.refine_1
          K : Type u_1
          inst✝¹ : Field K
          inst✝ : NumberField K
          w₀ : NumberField.InfinitePlace K
          hw₀ : w₀.IsComplex
          B : NNReal
          hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
          this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
          a : NumberField.RingOfIntegers K
          h_nz : Ne a 0
          h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
          h_le₀ : And (LT.lt (abs (w₀.embedding ↑a).re) 1) (LT.lt (abs (w₀.embedding ↑a) …
          w : NumberField.InfinitePlace K
          h_eq : Eq w w₀
          ⊢ LT.lt (abs (w₀.embedding ↑a).re) 1
        -/
        exact h_le₀.1
        /-
          🎉 no goals
        -/
        /-
          case pos.refine_2
          K : Type u_1
          inst✝¹ : Field K
          inst✝ : NumberField K
          w₀ : NumberField.InfinitePlace K
          hw₀ : w₀.IsComplex
          B : NNReal
          hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
          this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
          a : NumberField.RingOfIntegers K
          h_nz : Ne a 0
          h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
          h_le₀ : And (LT.lt (abs (w₀.embedding ↑a).re) 1) (LT.lt (abs (w₀.embedding ↑a) …
          w : NumberField.InfinitePlace K
          h_eq : Eq w w₀
          ⊢ LT.lt (HPow.hPow (w₀.embedding ↑a).im 2) (HPow.hPow (↑B) 2)
        -/
      · rw [sq_lt_sq, NNReal.abs_eq, ← NNReal.sq_sqrt B]
        /-
          case pos.refine_2
          K : Type u_1
          inst✝¹ : Field K
          inst✝ : NumberField K
          w₀ : NumberField.InfinitePlace K
          hw₀ : w₀.IsComplex
          B : NNReal
          hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
          this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
          a : NumberField.RingOfIntegers K
          h_nz : Ne a 0
          h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
          h_le₀ : And (LT.lt (abs (w₀.embedding ↑a).re) 1) (LT.lt (abs (w₀.embedding ↑a) …
          w : NumberField.InfinitePlace K
          h_eq : Eq w w₀
          ⊢ LT.lt (abs (w₀.embedding ↑a).im) ↑(HPow.hPow (NNReal.sqrt B) 2)
        -/
        exact h_le₀.2
        /-
          🎉 no goals
        -/
      /-
        case neg
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        w₀ : NumberField.InfinitePlace K
        hw₀ : w₀.IsComplex
        B : NNReal
        hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
        this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
        a : NumberField.RingOfIntegers K
        h_nz : Ne a 0
        h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
        h_le₀ : And (LT.lt (abs ((↑⟨w₀, hw₀⟩).embedding ↑a).re) 1) (LT.lt (abs ((↑⟨w₀, …
        w : NumberField.InfinitePlace K
        h_eq : Not (Eq w w₀)
        ⊢ LT.lt (w ↑a) (HAdd.hAdd 1 (HPow.hPow (↑B) 2)).sqrt
      -/
    · refine lt_of_lt_of_le (if_neg h_eq ▸ h_le w h_eq) ?_
      /-
        case neg
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        w₀ : NumberField.InfinitePlace K
        hw₀ : w₀.IsComplex
        B : NNReal
        hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
        this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
        a : NumberField.RingOfIntegers K
        h_nz : Ne a 0
        h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
        h_le₀ : And (LT.lt (abs ((↑⟨w₀, hw₀⟩).embedding ↑a).re) 1) (LT.lt (abs ((↑⟨w₀, …
        w : NumberField.InfinitePlace K
        h_eq : Not (Eq w w₀)
        ⊢ LE.le (↑1) (HAdd.hAdd 1 (HPow.hPow (↑B) 2)).sqrt
      -/
      rw [NNReal.coe_one, Real.le_sqrt' zero_lt_one, one_pow]
      /-
        case neg
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        w₀ : NumberField.InfinitePlace K
        hw₀ : w₀.IsComplex
        B : NNReal
        hB : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (HMul.hMul ↑(Number …
        this : LT.lt (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Me …
        a : NumberField.RingOfIntegers K
        h_nz : Ne a 0
        h_le : ∀ (w : NumberField.InfinitePlace K), Ne w ↑⟨w₀, hw₀⟩ → LT.lt (w ↑a) ↑(i …
        h_le₀ : And (LT.lt (abs ((↑⟨w₀, hw₀⟩).embedding ↑a).re) 1) (LT.lt (abs ((↑⟨w₀, …
        w : NumberField.InfinitePlace K
        h_eq : Not (Eq w w₀)
        ⊢ LE.le 1 (HAdd.hAdd 1 (HPow.hPow (↑B) 2))
      -/
      norm_num
      /-
        🎉 no goals
      -/


/-- Let `I` be a fractional ideal of `K`. Assume that `B : ℝ` is such that
`minkowskiBound K I < volume (convexBodySum K B)` where `convexBodySum K B` is the set of points
`x` such that `∑ w real, ‖x w‖ + 2 * ∑ w complex, ‖x w‖ ≤ B` (see `convexBodySum_volume` for
the computation of this volume), then there exists a nonzero algebraic number `a` in `I` such
that `|Norm a| < (B / d) ^ d` where `d` is the degree of `K`. -/
theorem exists_ne_zero_mem_ideal_of_norm_le {B : ℝ}
    (h : (minkowskiBound K I) ≤ volume (convexBodySum K B)) :
    ∃ a ∈ (I : FractionalIdeal (𝓞 K)⁰ K), a ≠ 0 ∧
      |Algebra.norm ℚ (a : K)| ≤ (B / finrank ℚ K) ^ finrank ℚ K := by
  have hB : 0 ≤ B := by
    contrapose! h
    rw [convexBodySum_volume_eq_zero_of_le_zero K (le_of_lt h)]
    exact minkowskiBound_pos K I
  -- Some inequalities that will be useful later on
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    hB : LE.le 0 B
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (LE.le (↑(abs ((Al …
  -/
  have h1 : 0 < (finrank ℚ K : ℝ)⁻¹ := inv_pos.mpr (Nat.cast_pos.mpr finrank_pos)
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    hB : LE.le 0 B
    h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (LE.le (↑(abs ((Al …
  -/
  have h2 : 0 ≤ B / (finrank ℚ K) := div_nonneg hB (Nat.cast_nonneg _)
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    hB : LE.le 0 B
    h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
    h2 : LE.le 0 (HDiv.hDiv B ↑(Module.finrank Rat K))
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (LE.le (↑(abs ((Al …
  -/
  have h_fund := ZSpan.isAddFundamentalDomain' (fractionalIdealLatticeBasis K I) volume
  have : Countable (span ℤ (Set.range (fractionalIdealLatticeBasis K I))).toAddSubgroup := by
    change Countable (span ℤ (Set.range (fractionalIdealLatticeBasis K I)))
    infer_instance
  obtain ⟨⟨x, hx⟩, h_nz, h_mem⟩ := exists_ne_zero_mem_lattice_of_measure_mul_two_pow_le_measure
      h_fund (fun _ ↦ convexBodySum_neg_mem K B) (convexBodySum_convex K B)
      (convexBodySum_compact K B) h
  /-
    case intro.mk.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    hB : LE.le 0 B
    h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
    h2 : LE.le 0 (HDiv.hDiv B ↑(Module.finrank Rat K))
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
    h_nz : Ne ⟨x, hx⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) ↑⟨x, hx⟩
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (LE.le (↑(abs ((Al …
  -/
  rw [mem_toAddSubgroup, mem_span_fractionalIdealLatticeBasis] at hx
  /-
    case intro.mk.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    hB : LE.le 0 B
    h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
    h2 : LE.le 0 (HDiv.hDiv B ↑(Module.finrank Rat K))
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    x : NumberField.mixedEmbedding.mixedSpace K
    hx✝ : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddi …
    hx : Membership.mem (Set.image ⇑(NumberField.mixedEmbedding K) ↑↑I) x
    h_nz : Ne ⟨x, hx✝⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) ↑⟨x, hx✝⟩
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (LE.le (↑(abs ((Al …
  -/
  obtain ⟨a, ha, rfl⟩ := hx
  /-
    case intro.mk.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    hB : LE.le 0 B
    h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
    h2 : LE.le 0 (HDiv.hDiv B ↑(Module.finrank Rat K))
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    a : K
    ha : Membership.mem (↑↑I) a
    hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
    h_nz : Ne ⟨(NumberField.mixedEmbedding K) a, hx⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) ↑⟨(Numbe …
    ⊢ Exists fun a => And (Membership.mem (↑I) a) (And (Ne a 0) (LE.le (↑(abs ((Al …
  -/
  refine ⟨a, ha, by simpa using h_nz, ?_⟩
  rw [← rpow_natCast, ← rpow_le_rpow_iff (by simp only [Rat.cast_abs, abs_nonneg])
      (rpow_nonneg h2 _) h1, ← rpow_mul h2,  mul_inv_cancel₀ (Nat.cast_ne_zero.mpr
      (ne_of_gt finrank_pos)), rpow_one, le_div_iff₀' (Nat.cast_pos.mpr finrank_pos)]
  /-
    case intro.mk.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    hB : LE.le 0 B
    h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
    h2 : LE.le 0 (HDiv.hDiv B ↑(Module.finrank Rat K))
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    a : K
    ha : Membership.mem (↑↑I) a
    hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
    h_nz : Ne ⟨(NumberField.mixedEmbedding K) a, hx⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) ↑⟨(Numbe …
    ⊢ LE.le (HMul.hMul (↑(Module.finrank Rat K)) (HPow.hPow (↑(abs ((Algebra.norm  …
  -/
  refine le_trans ?_ ((convexBodySum_mem K B).mp h_mem)
  /-
    case intro.mk.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
    hB : LE.le 0 B
    h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
    h2 : LE.le 0 (HDiv.hDiv B ↑(Module.finrank Rat K))
    h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
    this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
    a : K
    ha : Membership.mem (↑↑I) a
    hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
    h_nz : Ne ⟨(NumberField.mixedEmbedding K) a, hx⟩ 0
    h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) ↑⟨(Numbe …
    ⊢ LE.le (HMul.hMul (↑(Module.finrank Rat K)) (HPow.hPow (↑(abs ((Algebra.norm  …
  -/
  rw [← le_div_iff₀' (Nat.cast_pos.mpr finrank_pos), ← sum_mult_eq, Nat.cast_sum]
  refine le_trans ?_ (geom_mean_le_arith_mean Finset.univ _ _ (fun _ _ => Nat.cast_nonneg _)
    ?_ (fun _ _ => AbsoluteValue.nonneg _ _))
    /-
      case intro.mk.intro.intro.intro.refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      B : Real
      h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
      hB : LE.le 0 B
      h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
      h2 : LE.le 0 (HDiv.hDiv B ↑(Module.finrank Rat K))
      h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
      this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
      a : K
      ha : Membership.mem (↑↑I) a
      hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
      h_nz : Ne ⟨(NumberField.mixedEmbedding K) a, hx⟩ 0
      h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) ↑⟨(Numbe …
      ⊢ LE.le (HPow.hPow (↑(abs ((Algebra.norm Rat) a))) (Inv.inv (Finset.univ.sum f …
    -/
  · simp_rw [← prod_eq_abs_norm, rpow_natCast]
    /-
      case intro.mk.intro.intro.intro.refine_1
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      B : Real
      h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
      hB : LE.le 0 B
      h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
      h2 : LE.le 0 (HDiv.hDiv B ↑(Module.finrank Rat K))
      h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
      this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
      a : K
      ha : Membership.mem (↑↑I) a
      hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
      h_nz : Ne ⟨(NumberField.mixedEmbedding K) a, hx⟩ 0
      h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) ↑⟨(Numbe …
      ⊢ LE.le (HPow.hPow (Finset.univ.prod fun w => HPow.hPow (w a) w.mult) (Inv.inv …
    -/
    exact le_of_eq rfl
    /-
      🎉 no goals
    -/
    /-
      case intro.mk.intro.intro.intro.refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      B : Real
      h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
      hB : LE.le 0 B
      h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
      h2 : LE.le 0 (HDiv.hDiv B ↑(Module.finrank Rat K))
      h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
      this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
      a : K
      ha : Membership.mem (↑↑I) a
      hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
      h_nz : Ne ⟨(NumberField.mixedEmbedding K) a, hx⟩ 0
      h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) ↑⟨(Numbe …
      ⊢ LT.lt 0 (Finset.univ.sum fun i => ↑i.mult)
    -/
  · rw [← Nat.cast_sum, sum_mult_eq, Nat.cast_pos]
    /-
      case intro.mk.intro.intro.intro.refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      B : Real
      h : LE.le (NumberField.mixedEmbedding.minkowskiBound K I) (MeasureTheory.Measu …
      hB : LE.le 0 B
      h1 : LT.lt 0 (Inv.inv ↑(Module.finrank Rat K))
      h2 : LE.le 0 (HDiv.hDiv B ↑(Module.finrank Rat K))
      h_fund : MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem …
      this : Countable (Subtype fun x => Membership.mem (Submodule.span Int (Set.ran …
      a : K
      ha : Membership.mem (↑↑I) a
      hx : Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbeddin …
      h_nz : Ne ⟨(NumberField.mixedEmbedding K) a, hx⟩ 0
      h_mem : Membership.mem (NumberField.mixedEmbedding.convexBodySum K B) ↑⟨(Numbe …
      ⊢ LT.lt 0 (Module.finrank Rat K)
    -/
    exact finrank_pos
    /-
      🎉 no goals
    -/


theorem exists_ne_zero_mem_ringOfIntegers_of_norm_le {B : ℝ}
    (h : (minkowskiBound K ↑1) ≤ volume (convexBodySum K B)) :
    ∃ a : 𝓞 K, a ≠ 0 ∧ |Algebra.norm ℚ (a : K)| ≤ (B / finrank ℚ K) ^ finrank ℚ K := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Measu …
    ⊢ Exists fun a => And (Ne a 0) (LE.le (↑(abs ((Algebra.norm Rat) ↑a))) (HPow.h …
  -/
  obtain ⟨_, h_mem, h_nz, h_bd⟩ := exists_ne_zero_mem_ideal_of_norm_le K ↑1 h
  /-
    case intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Measu …
    w✝ : K
    h_mem : Membership.mem (↑1) w✝
    h_nz : Ne w✝ 0
    h_bd : LE.le (↑(abs ((Algebra.norm Rat) w✝))) (HPow.hPow (HDiv.hDiv B ↑(Module …
    ⊢ Exists fun a => And (Ne a 0) (LE.le (↑(abs ((Algebra.norm Rat) ↑a))) (HPow.h …
  -/
  obtain ⟨a, rfl⟩ := (FractionalIdeal.mem_one_iff _).mp h_mem
  /-
    case intro.intro.intro.intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    B : Real
    h : LE.le (NumberField.mixedEmbedding.minkowskiBound K 1) (MeasureTheory.Measu …
    a : NumberField.RingOfIntegers K
    h_mem : Membership.mem (↑1) ((algebraMap (NumberField.RingOfIntegers K) K) a)
    h_nz : Ne ((algebraMap (NumberField.RingOfIntegers K) K) a) 0
    h_bd : LE.le (↑(abs ((Algebra.norm Rat) ((algebraMap (NumberField.RingOfIntege …
    ⊢ Exists fun a => And (Ne a 0) (LE.le (↑(abs ((Algebra.norm Rat) ↑a))) (HPow.h …
  -/
  exact ⟨a, RingOfIntegers.coe_ne_zero_iff.mp h_nz, h_bd⟩
  /-
    🎉 no goals
  -/


