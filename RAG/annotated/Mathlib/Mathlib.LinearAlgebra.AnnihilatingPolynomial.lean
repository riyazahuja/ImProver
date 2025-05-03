/-- `annIdeal R a` is the *annihilating ideal* of all `p : R[X]` such that `p(a) = 0`.

The informal notation `p(a)` stand for `Polynomial.aeval a p`.
Again informally, the annihilating ideal of `a` is
`{ p ∈ R[X] | p(a) = 0 }`. This is an ideal in `R[X]`.
The formal definition uses the kernel of the aeval map. -/
noncomputable def annIdeal (a : A) : Ideal R[X] :=
  RingHom.ker ((aeval a).toRingHom : R[X] →+* A)


/-- It is useful to refer to ideal membership sometimes
 and the annihilation condition other times. -/
theorem mem_annIdeal_iff_aeval_eq_zero {a : A} {p : R[X]} : p ∈ annIdeal R a ↔ aeval a p = 0 :=
  Iff.rfl


/-- `annIdealGenerator 𝕜 a` is the monic generator of `annIdeal 𝕜 a`
if one exists, otherwise `0`.

Since `𝕜[X]` is a principal ideal domain there is a polynomial `g` such that
 `span 𝕜 {g} = annIdeal a`. This picks some generator.
 We prefer the monic generator of the ideal. -/
noncomputable def annIdealGenerator (a : A) : 𝕜[X] :=
  let g := IsPrincipal.generator <| annIdeal 𝕜 a
  g * C g.leadingCoeff⁻¹


@[simp]
theorem annIdealGenerator_eq_zero_iff {a : A} : annIdealGenerator 𝕜 a = 0 ↔ annIdeal 𝕜 a = ⊥ := by
  simp only [annIdealGenerator, mul_eq_zero, IsPrincipal.eq_bot_iff_generator_eq_zero,
    Polynomial.C_eq_zero, inv_eq_zero, Polynomial.leadingCoeff_eq_zero, or_self_iff]


/-- `annIdealGenerator 𝕜 a` is indeed a generator. -/
@[simp]
theorem span_singleton_annIdealGenerator (a : A) :
    Ideal.span {annIdealGenerator 𝕜 a} = annIdeal 𝕜 a := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    ⊢ Eq (Ideal.span (Singleton.singleton (Polynomial.annIdealGenerator 𝕜 a))) (Po …
  -/
  by_cases h : annIdealGenerator 𝕜 a = 0
    /-
      case pos
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Eq (Polynomial.annIdealGenerator 𝕜 a) 0
      ⊢ Eq (Ideal.span (Singleton.singleton (Polynomial.annIdealGenerator 𝕜 a))) (Po …
    -/
  · rw [h, annIdealGenerator_eq_zero_iff.mp h, Set.singleton_zero, Ideal.span_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Not (Eq (Polynomial.annIdealGenerator 𝕜 a) 0)
      ⊢ Eq (Ideal.span (Singleton.singleton (Polynomial.annIdealGenerator 𝕜 a))) (Po …
    -/
  · rw [annIdealGenerator, Ideal.span_singleton_mul_right_unit, Ideal.span_singleton_generator]
    /-
      case neg.h2
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Not (Eq (Polynomial.annIdealGenerator 𝕜 a) 0)
      ⊢ IsUnit (Polynomial.C (Inv.inv (Submodule.IsPrincipal.generator (Polynomial.a …
    -/
    apply Polynomial.isUnit_C.mpr
    /-
      case neg.h2
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Not (Eq (Polynomial.annIdealGenerator 𝕜 a) 0)
      ⊢ IsUnit (Inv.inv (Submodule.IsPrincipal.generator (Polynomial.annIdeal 𝕜 a)). …
    -/
    apply IsUnit.mk0
    /-
      case neg.h2.hx
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Not (Eq (Polynomial.annIdealGenerator 𝕜 a) 0)
      ⊢ Ne (Inv.inv (Submodule.IsPrincipal.generator (Polynomial.annIdeal 𝕜 a)).lead …
    -/
    apply inv_eq_zero.not.mpr
    /-
      case neg.h2.hx
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Not (Eq (Polynomial.annIdealGenerator 𝕜 a) 0)
      ⊢ Not (Eq (Submodule.IsPrincipal.generator (Polynomial.annIdeal 𝕜 a)).leadingC …
    -/
    apply Polynomial.leadingCoeff_eq_zero.not.mpr
    /-
      case neg.h2.hx
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Not (Eq (Polynomial.annIdealGenerator 𝕜 a) 0)
      ⊢ Not (Eq (Submodule.IsPrincipal.generator (Polynomial.annIdeal 𝕜 a)) 0)
    -/
    apply (mul_ne_zero_iff.mp h).1
    /-
      🎉 no goals
    -/


/-- The annihilating ideal generator is a member of the annihilating ideal. -/
theorem annIdealGenerator_mem (a : A) : annIdealGenerator 𝕜 a ∈ annIdeal 𝕜 a :=
  Ideal.mul_mem_right _ _ (Submodule.IsPrincipal.generator_mem _)


theorem mem_iff_eq_smul_annIdealGenerator {p : 𝕜[X]} (a : A) :
    p ∈ annIdeal 𝕜 a ↔ ∃ s : 𝕜[X], p = s • annIdealGenerator 𝕜 a := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    p : Polynomial 𝕜
    a : A
    ⊢ Iff (Membership.mem (Polynomial.annIdeal 𝕜 a) p) (Exists fun s => Eq p (HSMu …
  -/
  simp_rw [@eq_comm _ p, ← mem_span_singleton, ← span_singleton_annIdealGenerator 𝕜 a, Ideal.span]
  /-
    🎉 no goals
  -/


/-- The generator we chose for the annihilating ideal is monic when the ideal is non-zero. -/
theorem monic_annIdealGenerator (a : A) (hg : annIdealGenerator 𝕜 a ≠ 0) :
    Monic (annIdealGenerator 𝕜 a) :=
  monic_mul_leadingCoeff_inv (mul_ne_zero_iff.mp hg).1


theorem annIdealGenerator_aeval_eq_zero (a : A) : aeval a (annIdealGenerator 𝕜 a) = 0 :=
  mem_annIdeal_iff_aeval_eq_zero.mp (annIdealGenerator_mem 𝕜 a)


theorem mem_iff_annIdealGenerator_dvd {p : 𝕜[X]} {a : A} :
    p ∈ annIdeal 𝕜 a ↔ annIdealGenerator 𝕜 a ∣ p := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    p : Polynomial 𝕜
    a : A
    ⊢ Iff (Membership.mem (Polynomial.annIdeal 𝕜 a) p) (Dvd.dvd (Polynomial.annIde …
  -/
  rw [← Ideal.mem_span_singleton, span_singleton_annIdealGenerator]
  /-
    🎉 no goals
  -/


/-- The generator of the annihilating ideal has minimal degree among
 the non-zero members of the annihilating ideal -/
theorem degree_annIdealGenerator_le_of_mem (a : A) (p : 𝕜[X]) (hp : p ∈ annIdeal 𝕜 a)
    (hpn0 : p ≠ 0) : degree (annIdealGenerator 𝕜 a) ≤ degree p :=
  degree_le_of_dvd (mem_iff_annIdealGenerator_dvd.1 hp) hpn0


/-- The generator of the annihilating ideal is the minimal polynomial. -/
theorem annIdealGenerator_eq_minpoly (a : A) : annIdealGenerator 𝕜 a = minpoly 𝕜 a := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    ⊢ Eq (Polynomial.annIdealGenerator 𝕜 a) (minpoly 𝕜 a)
  -/
  by_cases h : annIdealGenerator 𝕜 a = 0
    /-
      case pos
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Eq (Polynomial.annIdealGenerator 𝕜 a) 0
      ⊢ Eq (Polynomial.annIdealGenerator 𝕜 a) (minpoly 𝕜 a)
    -/
  · rw [h, minpoly.eq_zero]
    /-
      case pos
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Eq (Polynomial.annIdealGenerator 𝕜 a) 0
      ⊢ Not (IsIntegral 𝕜 a)
    -/
    rintro ⟨p, p_monic, hp : aeval a p = 0⟩
    /-
      case pos.intro.intro
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Eq (Polynomial.annIdealGenerator 𝕜 a) 0
      p : Polynomial 𝕜
      p_monic : p.Monic
      hp : Eq ((Polynomial.aeval a) p) 0
      ⊢ False
    -/
    refine p_monic.ne_zero (Ideal.mem_bot.mp ?_)
    /-
      case pos.intro.intro
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      h : Eq (Polynomial.annIdealGenerator 𝕜 a) 0
      p : Polynomial 𝕜
      p_monic : p.Monic
      hp : Eq ((Polynomial.aeval a) p) 0
      ⊢ Membership.mem Bot.bot p
    -/
    simpa only [annIdealGenerator_eq_zero_iff.mp h] using mem_annIdeal_iff_aeval_eq_zero.mpr hp
    /-
      🎉 no goals
    -/
  · exact minpoly.unique _ _ (monic_annIdealGenerator _ _ h) (annIdealGenerator_aeval_eq_zero _ _)
      fun q q_monic hq =>
        degree_annIdealGenerator_le_of_mem a q (mem_annIdeal_iff_aeval_eq_zero.mpr hq)
          q_monic.ne_zero


/-- If a monic generates the annihilating ideal, it must match our choice
 of the annihilating ideal generator. -/
theorem monic_generator_eq_minpoly (a : A) (p : 𝕜[X]) (p_monic : p.Monic)
    (p_gen : Ideal.span {p} = annIdeal 𝕜 a) : annIdealGenerator 𝕜 a = p := by
  /-
    𝕜 : Type u_1
    A : Type u_2
    inst✝² : Field 𝕜
    inst✝¹ : Ring A
    inst✝ : Algebra 𝕜 A
    a : A
    p : Polynomial 𝕜
    p_monic : p.Monic
    p_gen : Eq (Ideal.span (Singleton.singleton p)) (Polynomial.annIdeal 𝕜 a)
    ⊢ Eq (Polynomial.annIdealGenerator 𝕜 a) p
  -/
  by_cases h : p = 0
    /-
      case pos
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      p : Polynomial 𝕜
      p_monic : p.Monic
      p_gen : Eq (Ideal.span (Singleton.singleton p)) (Polynomial.annIdeal 𝕜 a)
      h : Eq p 0
      ⊢ Eq (Polynomial.annIdealGenerator 𝕜 a) p
    -/
  · rwa [h, annIdealGenerator_eq_zero_iff, ← p_gen, Ideal.span_singleton_eq_bot.mpr]
    /-
      🎉 no goals
    -/
    /-
      case neg
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      p : Polynomial 𝕜
      p_monic : p.Monic
      p_gen : Eq (Ideal.span (Singleton.singleton p)) (Polynomial.annIdeal 𝕜 a)
      h : Not (Eq p 0)
      ⊢ Eq (Polynomial.annIdealGenerator 𝕜 a) p
    -/
  · rw [← span_singleton_annIdealGenerator, Ideal.span_singleton_eq_span_singleton] at p_gen
    /-
      case neg
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      p : Polynomial 𝕜
      p_monic : p.Monic
      p_gen : Associated p (Polynomial.annIdealGenerator 𝕜 a)
      h : Not (Eq p 0)
      ⊢ Eq (Polynomial.annIdealGenerator 𝕜 a) p
    -/
    rw [eq_comm]
    /-
      case neg
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      p : Polynomial 𝕜
      p_monic : p.Monic
      p_gen : Associated p (Polynomial.annIdealGenerator 𝕜 a)
      h : Not (Eq p 0)
      ⊢ Eq p (Polynomial.annIdealGenerator 𝕜 a)
    -/
    apply eq_of_monic_of_associated p_monic _ p_gen
    /-
      𝕜 : Type u_1
      A : Type u_2
      inst✝² : Field 𝕜
      inst✝¹ : Ring A
      inst✝ : Algebra 𝕜 A
      a : A
      p : Polynomial 𝕜
      p_monic : p.Monic
      p_gen : Associated p (Polynomial.annIdealGenerator 𝕜 a)
      h : Not (Eq p 0)
      ⊢ (Polynomial.annIdealGenerator 𝕜 a).Monic
    -/
    apply monic_annIdealGenerator _ _ ((Associated.ne_zero_iff p_gen).mp h)
    /-
      🎉 no goals
    -/


theorem span_minpoly_eq_annihilator {M} [AddCommGroup M] [Module 𝕜 M] (f : Module.End 𝕜 M) :
    Ideal.span {minpoly 𝕜 f} = Module.annihilator 𝕜[X] (Module.AEval' f) := by
  /-
    𝕜 : Type u_1
    inst✝² : Field 𝕜
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module 𝕜 M
    f : Module.End 𝕜 M
    ⊢ Eq (Ideal.span (Singleton.singleton (minpoly 𝕜 f))) (Module.annihilator (Pol …
  -/
  rw [← annIdealGenerator_eq_minpoly, span_singleton_annIdealGenerator]; ext
  /-
    case h
    𝕜 : Type u_1
    inst✝² : Field 𝕜
    M : Type u_3
    inst✝¹ : AddCommGroup M
    inst✝ : Module 𝕜 M
    f : Module.End 𝕜 M
    x✝ : Polynomial 𝕜
    ⊢ Iff (Membership.mem (Polynomial.annIdeal 𝕜 f) x✝) (Membership.mem (Module.an …
  -/
  rw [mem_annIdeal_iff_aeval_eq_zero, DFunLike.ext_iff, Module.mem_annihilator]; rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


