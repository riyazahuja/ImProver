/-- An integral domain `R` is integral closed if `Rₘ` is integral closed
  for any maximal ideal `m` of `R`. -/
theorem IsIntegrallyClosed.of_localization_maximal {R : Type*} [CommRing R] [IsDomain R]
    (h : ∀ p : Ideal R, p ≠ ⊥ → [p.IsMaximal] → IsIntegrallyClosed (Localization.AtPrime p)) :
    IsIntegrallyClosed R := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    h : ∀ (p : Ideal R), Ne p Bot.bot → ∀ [inst : p.IsMaximal], IsIntegrallyClosed …
    ⊢ IsIntegrallyClosed R
  -/
  by_cases hf : IsField R
    /-
      case pos
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      h : ∀ (p : Ideal R), Ne p Bot.bot → ∀ [inst : p.IsMaximal], IsIntegrallyClosed …
      hf : IsField R
      ⊢ IsIntegrallyClosed R
    -/
  · exact hf.toField.instIsIntegrallyClosed
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    h : ∀ (p : Ideal R), Ne p Bot.bot → ∀ [inst : p.IsMaximal], IsIntegrallyClosed …
    hf : Not (IsField R)
    ⊢ IsIntegrallyClosed R
  -/
  apply (isIntegrallyClosed_iff (FractionRing R)).mpr
  /-
    case neg
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    h : ∀ (p : Ideal R), Ne p Bot.bot → ∀ [inst : p.IsMaximal], IsIntegrallyClosed …
    hf : Not (IsField R)
    ⊢ ∀ {x : FractionRing R}, IsIntegral R x → Exists fun y => Eq ((algebraMap R ( …
  -/
  rintro ⟨x⟩ hx
  /-
    case neg.mk
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    h : ∀ (p : Ideal R), Ne p Bot.bot → ∀ [inst : p.IsMaximal], IsIntegrallyClosed …
    hf : Not (IsField R)
    x✝ : FractionRing R
    x : Prod R (Subtype fun x => Membership.mem (nonZeroDivisors R) x)
    hx : IsIntegral R (Quot.mk (⇑(OreLocalization.oreEqv (nonZeroDivisors R) R)) x)
    ⊢ Exists fun y => Eq ((algebraMap R (FractionRing R)) y) (Quot.mk (⇑(OreLocali …
  -/
  let I : Ideal R := span {x.2.1} / span {x.1}
  have h1 : 1 ∈ I := by
    apply I.eq_top_iff_one.mp
    by_contra hn
    rcases I.exists_le_maximal hn with ⟨p, hpm, hpi⟩
    have hic := h p (Ring.ne_bot_of_isMaximal_of_not_isField hpm hf)
    have hxp : IsIntegral (Localization.AtPrime p) (mk x.1 x.2) := hx.tower_top
    /- `x.1 / x.2.1 ∈ Rₚ` since it is integral over `Rₚ` and `Rₚ` is integrally closed.
      More precisely, `x.1 / x.2.1 = y.1 / y.2.1` where `y.1, y.2.1 ∈ R` and `y.2.1 ∉ p`. -/
    rcases (isIntegrallyClosed_iff (FractionRing R)).mp hic hxp with ⟨⟨y⟩, hy⟩
    /- `y.2.1 ∈ I` since for all `a ∈ Ideal.span {x.1}`, say `a = b * x.1`,
      we have `y.2 * a = b * x.1 * y.2 = b * y.1 * x.2.1 ∈ Ideal.span {x.2.1}`. -/
    have hyi : y.2.1 ∈ I := by
      intro a ha
      rcases mem_span_singleton'.mp ha with ⟨b, hb⟩
      apply mem_span_singleton'.mpr ⟨b * y.1, _⟩
      rw [← hb, ← mul_assoc, mul_comm y.2.1 b, mul_assoc, mul_assoc]
      exact congrArg (HMul.hMul b) <| (mul_comm y.1 x.2.1).trans <|
        NoZeroSMulDivisors.algebraMap_injective R (Localization R⁰) <| mk'_eq_iff_eq.mp <|
          (mk'_eq_algebraMap_mk'_of_submonoid_le _ _ p.primeCompl_le_nonZeroDivisors y.1 y.2).trans
            <| show algebraMap (Localization.AtPrime p) _ (mk' _ y.1 y.2) = mk' _ x.1 x.2
              by simpa only [← mk_eq_mk', ← hy] using by rfl
    -- `y.2.1 ∈ I` implies `y.2.1 ∈ p` since `I ⊆ p`, which contradicts to the choice of `y`.
    exact y.2.2 (hpi hyi)
  /-
    case neg.mk
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    h : ∀ (p : Ideal R), Ne p Bot.bot → ∀ [inst : p.IsMaximal], IsIntegrallyClosed …
    hf : Not (IsField R)
    x✝ : FractionRing R
    x : Prod R (Subtype fun x => Membership.mem (nonZeroDivisors R) x)
    hx : IsIntegral R (Quot.mk (⇑(OreLocalization.oreEqv (nonZeroDivisors R) R)) x)
    I : Ideal R := HDiv.hDiv (Ideal.span (Singleton.singleton ↑x.2)) (Ideal.span ( …
    h1 : Membership.mem I 1
    ⊢ Exists fun y => Eq ((algebraMap R (FractionRing R)) y) (Quot.mk (⇑(OreLocali …
  -/
  rcases mem_span_singleton'.mp (h1 x.1 (mem_span_singleton_self x.1)) with ⟨y, hy⟩
  /-
    case neg.mk.intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    h : ∀ (p : Ideal R), Ne p Bot.bot → ∀ [inst : p.IsMaximal], IsIntegrallyClosed …
    hf : Not (IsField R)
    x✝ : FractionRing R
    x : Prod R (Subtype fun x => Membership.mem (nonZeroDivisors R) x)
    hx : IsIntegral R (Quot.mk (⇑(OreLocalization.oreEqv (nonZeroDivisors R) R)) x)
    I : Ideal R := HDiv.hDiv (Ideal.span (Singleton.singleton ↑x.2)) (Ideal.span ( …
    h1 : Membership.mem I 1
    y : R
    hy : Eq (HMul.hMul y ↑x.2) (HMul.hMul 1 x.1)
    ⊢ Exists fun y => Eq ((algebraMap R (FractionRing R)) y) (Quot.mk (⇑(OreLocali …
  -/
  exact ⟨y, (eq_mk'_of_mul_eq (hy.trans (one_mul x.1))).trans (mk_eq_mk'_apply x.1 x.2).symm⟩
  /-
    🎉 no goals
  -/


theorem isIntegrallyClosed_ofLocalizationMaximal :
    OfLocalizationMaximal fun R _ => ([IsDomain R] → IsIntegrallyClosed R) :=
  fun _ _ h _ ↦ IsIntegrallyClosed.of_localization_maximal fun p _ hpm ↦ h p hpm

