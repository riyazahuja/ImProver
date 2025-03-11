lemma starOrderedRing_of_sqrt {R : Type*} [PartialOrder R] [NonUnitalRing R] [StarRing R]
    [StarOrderedRing R] [TopologicalSpace R] [ContinuousStar R] [TopologicalRing R]
    (sqrt : R → R) (h_continuousOn : ContinuousOn sqrt {x : R | 0 ≤ x})
    (h_sqrt : ∀ x, 0 ≤ x → star (sqrt x) * sqrt x = x) : StarOrderedRing C(α, R) :=
  StarOrderedRing.of_nonneg_iff' add_le_add_left fun f ↦ by
    /-
      α : Type u_1
      inst✝⁷ : TopologicalSpace α
      R : Type u_2
      inst✝⁶ : PartialOrder R
      inst✝⁵ : NonUnitalRing R
      inst✝⁴ : StarRing R
      inst✝³ : StarOrderedRing R
      inst✝² : TopologicalSpace R
      inst✝¹ : ContinuousStar R
      inst✝ : TopologicalRing R
      sqrt : R → R
      h_continuousOn : ContinuousOn sqrt (setOf fun x => LE.le 0 x)
      h_sqrt : ∀ (x : R), LE.le 0 x → Eq (HMul.hMul (Star.star (sqrt x)) (sqrt x)) x
      f : ContinuousMap α R
      ⊢ Iff (LE.le 0 f) (Exists fun s => Eq f (HMul.hMul (Star.star s) s))
    -/
    constructor
      /-
        case mp
        α : Type u_1
        inst✝⁷ : TopologicalSpace α
        R : Type u_2
        inst✝⁶ : PartialOrder R
        inst✝⁵ : NonUnitalRing R
        inst✝⁴ : StarRing R
        inst✝³ : StarOrderedRing R
        inst✝² : TopologicalSpace R
        inst✝¹ : ContinuousStar R
        inst✝ : TopologicalRing R
        sqrt : R → R
        h_continuousOn : ContinuousOn sqrt (setOf fun x => LE.le 0 x)
        h_sqrt : ∀ (x : R), LE.le 0 x → Eq (HMul.hMul (Star.star (sqrt x)) (sqrt x)) x
        f : ContinuousMap α R
        ⊢ LE.le 0 f → Exists fun s => Eq f (HMul.hMul (Star.star s) s)
      -/
    · intro hf
      /-
        case mp
        α : Type u_1
        inst✝⁷ : TopologicalSpace α
        R : Type u_2
        inst✝⁶ : PartialOrder R
        inst✝⁵ : NonUnitalRing R
        inst✝⁴ : StarRing R
        inst✝³ : StarOrderedRing R
        inst✝² : TopologicalSpace R
        inst✝¹ : ContinuousStar R
        inst✝ : TopologicalRing R
        sqrt : R → R
        h_continuousOn : ContinuousOn sqrt (setOf fun x => LE.le 0 x)
        h_sqrt : ∀ (x : R), LE.le 0 x → Eq (HMul.hMul (Star.star (sqrt x)) (sqrt x)) x
        f : ContinuousMap α R
        hf : LE.le 0 f
        ⊢ Exists fun s => Eq f (HMul.hMul (Star.star s) s)
      -/
      use (mk _ h_continuousOn.restrict).comp ⟨_, map_continuous f |>.codRestrict (by exact hf ·)⟩
      /-
        case h
        α : Type u_1
        inst✝⁷ : TopologicalSpace α
        R : Type u_2
        inst✝⁶ : PartialOrder R
        inst✝⁵ : NonUnitalRing R
        inst✝⁴ : StarRing R
        inst✝³ : StarOrderedRing R
        inst✝² : TopologicalSpace R
        inst✝¹ : ContinuousStar R
        inst✝ : TopologicalRing R
        sqrt : R → R
        h_continuousOn : ContinuousOn sqrt (setOf fun x => LE.le 0 x)
        h_sqrt : ∀ (x : R), LE.le 0 x → Eq (HMul.hMul (Star.star (sqrt x)) (sqrt x)) x
        f : ContinuousMap α R
        hf : LE.le 0 f
        ⊢ Eq f (HMul.hMul (Star.star ({ toFun := (setOf fun x => LE.le 0 x).restrict s …
      -/
      ext x
      /-
        case h.h
        α : Type u_1
        inst✝⁷ : TopologicalSpace α
        R : Type u_2
        inst✝⁶ : PartialOrder R
        inst✝⁵ : NonUnitalRing R
        inst✝⁴ : StarRing R
        inst✝³ : StarOrderedRing R
        inst✝² : TopologicalSpace R
        inst✝¹ : ContinuousStar R
        inst✝ : TopologicalRing R
        sqrt : R → R
        h_continuousOn : ContinuousOn sqrt (setOf fun x => LE.le 0 x)
        h_sqrt : ∀ (x : R), LE.le 0 x → Eq (HMul.hMul (Star.star (sqrt x)) (sqrt x)) x
        f : ContinuousMap α R
        hf : LE.le 0 f
        x : α
        ⊢ Eq (f x) ((HMul.hMul (Star.star ({ toFun := (setOf fun x => LE.le 0 x).restr …
      -/
      exact h_sqrt (f x) (hf x) |>.symm
      /-
        🎉 no goals
      -/
      /-
        case mpr
        α : Type u_1
        inst✝⁷ : TopologicalSpace α
        R : Type u_2
        inst✝⁶ : PartialOrder R
        inst✝⁵ : NonUnitalRing R
        inst✝⁴ : StarRing R
        inst✝³ : StarOrderedRing R
        inst✝² : TopologicalSpace R
        inst✝¹ : ContinuousStar R
        inst✝ : TopologicalRing R
        sqrt : R → R
        h_continuousOn : ContinuousOn sqrt (setOf fun x => LE.le 0 x)
        h_sqrt : ∀ (x : R), LE.le 0 x → Eq (HMul.hMul (Star.star (sqrt x)) (sqrt x)) x
        f : ContinuousMap α R
        ⊢ (Exists fun s => Eq f (HMul.hMul (Star.star s) s)) → LE.le 0 f
      -/
    · rintro ⟨f, rfl⟩
      /-
        case mpr.intro
        α : Type u_1
        inst✝⁷ : TopologicalSpace α
        R : Type u_2
        inst✝⁶ : PartialOrder R
        inst✝⁵ : NonUnitalRing R
        inst✝⁴ : StarRing R
        inst✝³ : StarOrderedRing R
        inst✝² : TopologicalSpace R
        inst✝¹ : ContinuousStar R
        inst✝ : TopologicalRing R
        sqrt : R → R
        h_continuousOn : ContinuousOn sqrt (setOf fun x => LE.le 0 x)
        h_sqrt : ∀ (x : R), LE.le 0 x → Eq (HMul.hMul (Star.star (sqrt x)) (sqrt x)) x
        f : ContinuousMap α R
        ⊢ LE.le 0 (HMul.hMul (Star.star f) f)
      -/
      rw [ContinuousMap.le_def]
      /-
        case mpr.intro
        α : Type u_1
        inst✝⁷ : TopologicalSpace α
        R : Type u_2
        inst✝⁶ : PartialOrder R
        inst✝⁵ : NonUnitalRing R
        inst✝⁴ : StarRing R
        inst✝³ : StarOrderedRing R
        inst✝² : TopologicalSpace R
        inst✝¹ : ContinuousStar R
        inst✝ : TopologicalRing R
        sqrt : R → R
        h_continuousOn : ContinuousOn sqrt (setOf fun x => LE.le 0 x)
        h_sqrt : ∀ (x : R), LE.le 0 x → Eq (HMul.hMul (Star.star (sqrt x)) (sqrt x)) x
        f : ContinuousMap α R
        ⊢ ∀ (a : α), LE.le (0 a) ((HMul.hMul (Star.star f) f) a)
      -/
      exact fun x ↦ star_mul_self_nonneg (f x)
      /-
        🎉 no goals
      -/


open scoped ComplexOrder in
open RCLike in
instance (priority := 100) instStarOrderedRingRCLike {𝕜 : Type*} [RCLike 𝕜] :
    StarOrderedRing C(α, 𝕜) :=
                                                     /-
                                                       α : Type u_1
                                                       inst✝¹ : TopologicalSpace α
                                                       𝕜 : Type u_2
                                                       inst✝ : RCLike 𝕜
                                                       ⊢ ContinuousOn (Function.comp RCLike.ofReal (Function.comp Real.sqrt ⇑RCLike.r …
                                                     -/
  starOrderedRing_of_sqrt ((↑) ∘ Real.sqrt ∘ re) (by fun_prop) fun x hx ↦ by
                                                     /-
                                                       🎉 no goals
                                                     -/
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      𝕜 : Type u_2
      inst✝ : RCLike 𝕜
      x : 𝕜
      hx : LE.le 0 x
      ⊢ Eq (HMul.hMul (Star.star (Function.comp RCLike.ofReal (Function.comp Real.sq …
    -/
    simp only [Function.comp_apply,star_def]
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      𝕜 : Type u_2
      inst✝ : RCLike 𝕜
      x : 𝕜
      hx : LE.le 0 x
      ⊢ Eq (HMul.hMul ((starRingEnd 𝕜) ↑(RCLike.re x).sqrt) ↑(RCLike.re x).sqrt) x
    -/
    obtain hx' := nonneg_iff.mp hx |>.right
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      𝕜 : Type u_2
      inst✝ : RCLike 𝕜
      x : 𝕜
      hx : LE.le 0 x
      hx' : Eq (RCLike.im x) 0
      ⊢ Eq (HMul.hMul ((starRingEnd 𝕜) ↑(RCLike.re x).sqrt) ↑(RCLike.re x).sqrt) x
    -/
    rw [← conj_eq_iff_im, conj_eq_iff_re] at hx'
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      𝕜 : Type u_2
      inst✝ : RCLike 𝕜
      x : 𝕜
      hx : LE.le 0 x
      hx' : Eq (↑(RCLike.re x)) x
      ⊢ Eq (HMul.hMul ((starRingEnd 𝕜) ↑(RCLike.re x).sqrt) ↑(RCLike.re x).sqrt) x
    -/
    rw [conj_ofReal, ← ofReal_mul, Real.mul_self_sqrt, hx']
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      𝕜 : Type u_2
      inst✝ : RCLike 𝕜
      x : 𝕜
      hx : LE.le 0 x
      hx' : Eq (↑(RCLike.re x)) x
      ⊢ LE.le 0 (RCLike.re x)
    -/
    rw [nonneg_iff]
    /-
      α : Type u_1
      inst✝¹ : TopologicalSpace α
      𝕜 : Type u_2
      inst✝ : RCLike 𝕜
      x : 𝕜
      hx : LE.le 0 x
      hx' : Eq (↑(RCLike.re x)) x
      ⊢ And (LE.le 0 (RCLike.re (RCLike.re x))) (Eq (RCLike.im (RCLike.re x)) 0)
    -/
    simpa using nonneg_iff.mp hx |>.left
    /-
      🎉 no goals
    -/


instance instStarOrderedRingReal : StarOrderedRing C(α, ℝ) :=
  instStarOrderedRingRCLike (𝕜 := ℝ)


open scoped ComplexOrder in
open Complex in
instance instStarOrderedRingComplex : StarOrderedRing C(α, ℂ) :=
  instStarOrderedRingRCLike (𝕜 := ℂ)


open NNReal in
instance instStarOrderedRingNNReal : StarOrderedRing C(α, ℝ≥0) :=
  StarOrderedRing.of_le_iff fun f g ↦ by
    /-
      α : Type u_1
      inst✝ : TopologicalSpace α
      f g : ContinuousMap α NNReal
      ⊢ Iff (LE.le f g) (Exists fun s => Eq g (HAdd.hAdd f (HMul.hMul (Star.star s)  …
    -/
    constructor
      /-
        case mp
        α : Type u_1
        inst✝ : TopologicalSpace α
        f g : ContinuousMap α NNReal
        ⊢ LE.le f g → Exists fun s => Eq g (HAdd.hAdd f (HMul.hMul (Star.star s) s))
      -/
    · intro hfg
      /-
        case mp
        α : Type u_1
        inst✝ : TopologicalSpace α
        f g : ContinuousMap α NNReal
        hfg : LE.le f g
        ⊢ Exists fun s => Eq g (HAdd.hAdd f (HMul.hMul (Star.star s) s))
      -/
      use .comp ⟨sqrt, by fun_prop⟩ (g - f)
      /-
        case h
        α : Type u_1
        inst✝ : TopologicalSpace α
        f g : ContinuousMap α NNReal
        hfg : LE.le f g
        ⊢ Eq g (HAdd.hAdd f (HMul.hMul (Star.star ({ toFun := ⇑NNReal.sqrt, continuous …
      -/
      ext1 x
      /-
        case h.h
        α : Type u_1
        inst✝ : TopologicalSpace α
        f g : ContinuousMap α NNReal
        hfg : LE.le f g
        x : α
        ⊢ Eq (g x) ((HAdd.hAdd f (HMul.hMul (Star.star ({ toFun := ⇑NNReal.sqrt, conti …
      -/
      simpa using add_tsub_cancel_of_le (hfg x) |>.symm
      /-
        🎉 no goals
      -/
      /-
        case mpr
        α : Type u_1
        inst✝ : TopologicalSpace α
        f g : ContinuousMap α NNReal
        ⊢ (Exists fun s => Eq g (HAdd.hAdd f (HMul.hMul (Star.star s) s))) → LE.le f g
      -/
    · rintro ⟨s, rfl⟩
      /-
        case mpr.intro
        α : Type u_1
        inst✝ : TopologicalSpace α
        f s : ContinuousMap α NNReal
        ⊢ LE.le f (HAdd.hAdd f (HMul.hMul (Star.star s) s))
      -/
      exact fun _ ↦ by simp
      /-
        🎉 no goals
      -/


instance instStarOrderedRing {R : Type*}
    [TopologicalSpace R] [OrderedCommSemiring R] [NoZeroDivisors R] [StarRing R] [StarOrderedRing R]
    [TopologicalSemiring R] [ContinuousStar R] [StarOrderedRing C(α, R)] :
    StarOrderedRing C(α, R)₀ where
  le_iff f g := by
    /-
      α : Type u_1
      inst✝⁹ : TopologicalSpace α
      inst✝⁸ : Zero α
      R : Type u_2
      inst✝⁷ : TopologicalSpace R
      inst✝⁶ : OrderedCommSemiring R
      inst✝⁵ : NoZeroDivisors R
      inst✝⁴ : StarRing R
      inst✝³ : StarOrderedRing R
      inst✝² : TopologicalSemiring R
      inst✝¹ : ContinuousStar R
      inst✝ : StarOrderedRing (ContinuousMap α R)
      f g : ContinuousMapZero α R
      ⊢ Iff (LE.le f g) (Exists fun p => And (Membership.mem (AddSubmonoid.closure ( …
    -/
    constructor
    · rw [le_def, ← ContinuousMap.coe_coe, ← ContinuousMap.coe_coe g, ← ContinuousMap.le_def,
        StarOrderedRing.le_iff]
      /-
        case mp
        α : Type u_1
        inst✝⁹ : TopologicalSpace α
        inst✝⁸ : Zero α
        R : Type u_2
        inst✝⁷ : TopologicalSpace R
        inst✝⁶ : OrderedCommSemiring R
        inst✝⁵ : NoZeroDivisors R
        inst✝⁴ : StarRing R
        inst✝³ : StarOrderedRing R
        inst✝² : TopologicalSemiring R
        inst✝¹ : ContinuousStar R
        inst✝ : StarOrderedRing (ContinuousMap α R)
        f g : ContinuousMapZero α R
        ⊢ (Exists fun p => And (Membership.mem (AddSubmonoid.closure (Set.range fun s  …
      -/
      rintro ⟨p, hp_mem, hp⟩
      induction hp_mem using AddSubmonoid.closure_induction_left generalizing f g with
      | one => exact ⟨0, zero_mem _, by ext x; congrm($(hp) x)⟩
      | mul_left s s_mem p p_mem hp' =>
        obtain ⟨s, rfl⟩ := s_mem
        simp only at *
        have h₀ : (star s * s + p) 0 = 0 := by simpa using congr($(hp) 0).symm
        rw [← add_assoc] at hp
        have p'₀ : 0 ≤ p 0 := by rw [← StarOrderedRing.nonneg_iff] at p_mem; exact p_mem 0
        have s₉ : (star s * s) 0 = 0 := le_antisymm ((le_add_of_nonneg_right p'₀).trans_eq h₀)
          (star_mul_self_nonneg (s 0))
        have s₀' : s 0 = 0 := by aesop
        let s' : C(α, R)₀ := ⟨s, s₀'⟩
        obtain ⟨p', hp'_mem, rfl⟩ := hp' (f + star s' * s') g hp
        refine ⟨star s' * s' + p', ?_, by rw [add_assoc]⟩
        exact add_mem (AddSubmonoid.subset_closure ⟨s', rfl⟩) hp'_mem
      /-
        case mpr
        α : Type u_1
        inst✝⁹ : TopologicalSpace α
        inst✝⁸ : Zero α
        R : Type u_2
        inst✝⁷ : TopologicalSpace R
        inst✝⁶ : OrderedCommSemiring R
        inst✝⁵ : NoZeroDivisors R
        inst✝⁴ : StarRing R
        inst✝³ : StarOrderedRing R
        inst✝² : TopologicalSemiring R
        inst✝¹ : ContinuousStar R
        inst✝ : StarOrderedRing (ContinuousMap α R)
        f g : ContinuousMapZero α R
        ⊢ (Exists fun p => And (Membership.mem (AddSubmonoid.closure (Set.range fun s  …
      -/
    · rintro ⟨p, hp, rfl⟩
      induction hp using AddSubmonoid.closure_induction generalizing f with
      | mem s s_mem =>
        obtain ⟨s, rfl⟩ := s_mem
        exact fun x ↦ le_add_of_nonneg_right (star_mul_self_nonneg (s x))
      | one => simp
      | mul g₁ g₂ _ _ h₁ h₂ => calc
          f ≤ f + g₁ := h₁ f
          _ ≤ (f + g₁) + g₂ := h₂ (f + g₁)
          _ = f + (g₁ + g₂) := add_assoc _ _ _


instance instStarOrderedRingReal : StarOrderedRing C(α, ℝ)₀ :=
  instStarOrderedRing (R := ℝ)


open scoped ComplexOrder in
instance instStarOrderedRingComplex : StarOrderedRing C(α, ℂ)₀ :=
  instStarOrderedRing (R := ℂ)


instance instStarOrderedRingNNReal : StarOrderedRing C(α, ℝ≥0)₀ :=
  instStarOrderedRing (R := ℝ≥0)


