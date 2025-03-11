/-- The maximal spectrum of a commutative ring `R` is the type of all maximal ideals of `R`. -/
@[ext]
structure MaximalSpectrum where
  asIdeal : Ideal R
  IsMaximal : asIdeal.IsMaximal


instance [Nontrivial R] : Nonempty <| MaximalSpectrum R :=
  let ⟨I, hI⟩ := Ideal.exists_maximal R
  ⟨⟨I, hI⟩⟩


/-- The natural inclusion from the maximal spectrum to the prime spectrum. -/
def toPrimeSpectrum (x : MaximalSpectrum R) : PrimeSpectrum R :=
  ⟨x.asIdeal, x.IsMaximal.isPrime⟩


theorem toPrimeSpectrum_injective : (@toPrimeSpectrum R _).Injective := fun ⟨_, _⟩ ⟨_, _⟩ h => by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    x✝¹ x✝ : MaximalSpectrum R
    asIdeal✝¹ : Ideal R
    IsMaximal✝¹ : asIdeal✝¹.IsMaximal
    asIdeal✝ : Ideal R
    IsMaximal✝ : asIdeal✝.IsMaximal
    h : Eq { asIdeal := asIdeal✝¹, IsMaximal := IsMaximal✝¹ }.toPrimeSpectrum { as …
    ⊢ Eq { asIdeal := asIdeal✝¹, IsMaximal := IsMaximal✝¹ } { asIdeal := asIdeal✝, …
  -/
  simpa only [MaximalSpectrum.mk.injEq] using PrimeSpectrum.ext_iff.mp h
  /-
    🎉 no goals
  -/


/-- An integral domain is equal to the intersection of its localizations at all its maximal ideals
viewed as subalgebras of its field of fractions. -/
theorem iInf_localization_eq_bot : (⨅ v : MaximalSpectrum R,
    Localization.subalgebra.ofField K _ v.asIdeal.primeCompl_le_nonZeroDivisors) = ⊥ := by
  /-
    R : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    ⊢ Eq (iInf fun v => Localization.subalgebra.ofField K v.asIdeal.primeCompl ⋯)  …
  -/
  ext x
  /-
    case h
    R : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x : K
    ⊢ Iff (Membership.mem (iInf fun v => Localization.subalgebra.ofField K v.asIde …
  -/
  rw [Algebra.mem_bot, Algebra.mem_iInf]
  /-
    case h
    R : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x : K
    ⊢ Iff (∀ (i : MaximalSpectrum R), Membership.mem (Localization.subalgebra.ofFi …
  -/
  constructor
    /-
      case h.mp
      R : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_5
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      x : K
      ⊢ (∀ (i : MaximalSpectrum R), Membership.mem (Localization.subalgebra.ofField  …
    -/
  · contrapose
    /-
      case h.mp
      R : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_5
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      x : K
      ⊢ Not (Membership.mem (Set.range ⇑(algebraMap R K)) x) → Not (∀ (i : MaximalSp …
    -/
    intro hrange hlocal
    /-
      case h.mp
      R : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_5
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      x : K
      hrange : Not (Membership.mem (Set.range ⇑(algebraMap R K)) x)
      hlocal : ∀ (i : MaximalSpectrum R), Membership.mem (Localization.subalgebra.of …
      ⊢ False
    -/
    let denom : Ideal R := (1 : Submodule R K).comap (LinearMap.toSpanSingleton R K x)
    /-
      case h.mp
      R : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_5
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      x : K
      hrange : Not (Membership.mem (Set.range ⇑(algebraMap R K)) x)
      hlocal : ∀ (i : MaximalSpectrum R), Membership.mem (Localization.subalgebra.of …
      denom : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R K x) 1
      ⊢ False
    -/
    have hdenom : (1 : R) ∉ denom := by simpa [denom] using hrange
    /-
      case h.mp
      R : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_5
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      x : K
      hrange : Not (Membership.mem (Set.range ⇑(algebraMap R K)) x)
      hlocal : ∀ (i : MaximalSpectrum R), Membership.mem (Localization.subalgebra.of …
      denom : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R K x) 1
      hdenom : Not (Membership.mem denom 1)
      ⊢ False
    -/
    rcases denom.exists_le_maximal (denom.ne_top_iff_one.mpr hdenom) with ⟨max, hmax, hle⟩
    /-
      case h.mp.intro.intro
      R : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_5
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      x : K
      hrange : Not (Membership.mem (Set.range ⇑(algebraMap R K)) x)
      hlocal : ∀ (i : MaximalSpectrum R), Membership.mem (Localization.subalgebra.of …
      denom : Ideal R := Submodule.comap (LinearMap.toSpanSingleton R K x) 1
      hdenom : Not (Membership.mem denom 1)
      max : Ideal R
      hmax : max.IsMaximal
      hle : LE.le denom max
      ⊢ False
    -/
    rcases hlocal ⟨max, hmax⟩ with ⟨n, d, hd, rfl⟩
    exact hd (hle ⟨n, by simp [denom, Algebra.smul_def, mul_left_comm, mul_inv_cancel₀ <|
      (map_ne_zero_iff _ <| IsFractionRing.injective R K).mpr fun h ↦ hd (h ▸ max.zero_mem :)]⟩)
    /-
      case h.mpr
      R : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_5
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      x : K
      ⊢ Membership.mem (Set.range ⇑(algebraMap R K)) x → ∀ (i : MaximalSpectrum R),  …
    -/
  · rintro ⟨y, rfl⟩ ⟨v, hv⟩
    /-
      case h.mpr.intro.mk
      R : Type u_4
      inst✝⁴ : CommRing R
      inst✝³ : IsDomain R
      K : Type u_5
      inst✝² : Field K
      inst✝¹ : Algebra R K
      inst✝ : IsFractionRing R K
      y : R
      v : Ideal R
      hv : v.IsMaximal
      ⊢ Membership.mem (Localization.subalgebra.ofField K { asIdeal := v, IsMaximal  …
    -/
    exact ⟨y, 1, v.ne_top_iff_one.mp hv.ne_top, by rw [map_one, inv_one, mul_one]⟩
    /-
      🎉 no goals
    -/


/-- An integral domain is equal to the intersection of its localizations at all its prime ideals
viewed as subalgebras of its field of fractions. -/
theorem iInf_localization_eq_bot : ⨅ v : PrimeSpectrum R,
    Localization.subalgebra.ofField K _ (v.asIdeal.primeCompl_le_nonZeroDivisors) = ⊥ := by
  /-
    R : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    ⊢ Eq (iInf fun v => Localization.subalgebra.ofField K v.asIdeal.primeCompl ⋯)  …
  -/
  refine bot_unique (.trans (fun _ ↦ ?_) (MaximalSpectrum.iInf_localization_eq_bot R K).le)
  /-
    R : Type u_4
    inst✝⁴ : CommRing R
    inst✝³ : IsDomain R
    K : Type u_5
    inst✝² : Field K
    inst✝¹ : Algebra R K
    inst✝ : IsFractionRing R K
    x✝ : K
    ⊢ Membership.mem (iInf fun v => Localization.subalgebra.ofField K v.asIdeal.pr …
  -/
  simpa only [Algebra.mem_iInf] using fun hx ⟨v, hv⟩ ↦ hx ⟨v, hv.isPrime⟩
  /-
    🎉 no goals
  -/


/-- The product of localizations at all maximal ideals of a commutative semiring. -/
abbrev PiLocalization : Type _ := Π I : MaximalSpectrum R, Localization.AtPrime I.1


/-- The canonical ring homomorphism from a commutative semiring to the product of its
localizations at all maximal ideals. It is always injective. -/
def toPiLocalization : R →+* PiLocalization R := algebraMap R _


theorem toPiLocalization_injective : Function.Injective (toPiLocalization R) := fun r r' eq ↦ by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    r r' : R
    eq : Eq ((MaximalSpectrum.toPiLocalization R) r) ((MaximalSpectrum.toPiLocaliz …
    ⊢ Eq r r'
  -/
  rw [← one_mul r, ← one_mul r']
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    r r' : R
    eq : Eq ((MaximalSpectrum.toPiLocalization R) r) ((MaximalSpectrum.toPiLocaliz …
    ⊢ Eq (HMul.hMul 1 r) (HMul.hMul 1 r')
  -/
  by_contra ne
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    r r' : R
    eq : Eq ((MaximalSpectrum.toPiLocalization R) r) ((MaximalSpectrum.toPiLocaliz …
    ne : Not (Eq (HMul.hMul 1 r) (HMul.hMul 1 r'))
    ⊢ False
  -/
  have ⟨I, mI, hI⟩ := (Module.eqIdeal R r r').exists_le_maximal ((Ideal.ne_top_iff_one _).mpr ne)
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    r r' : R
    eq : Eq ((MaximalSpectrum.toPiLocalization R) r) ((MaximalSpectrum.toPiLocaliz …
    ne : Not (Eq (HMul.hMul 1 r) (HMul.hMul 1 r'))
    I : Ideal R
    mI : I.IsMaximal
    hI : LE.le (Module.eqIdeal R r r') I
    ⊢ False
  -/
  have ⟨s, hs⟩ := (IsLocalization.eq_iff_exists I.primeCompl _).mp (congr_fun eq ⟨I, mI⟩)
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    r r' : R
    eq : Eq ((MaximalSpectrum.toPiLocalization R) r) ((MaximalSpectrum.toPiLocaliz …
    ne : Not (Eq (HMul.hMul 1 r) (HMul.hMul 1 r'))
    I : Ideal R
    mI : I.IsMaximal
    hI : LE.le (Module.eqIdeal R r r') I
    s : Subtype fun x => Membership.mem I.primeCompl x
    hs : Eq (HMul.hMul (↑s) r) (HMul.hMul (↑s) r')
    ⊢ False
  -/
  exact s.2 (hI hs)
  /-
    🎉 no goals
  -/


theorem toPiLocalization_apply_apply {r I} : toPiLocalization R r I = algebraMap R _ r := rfl


/-- Functoriality of `PiLocalization` but restricted to bijective ring homs.
If R and S are commutative rings, surjectivity would be enough. -/
noncomputable def mapPiLocalization : PiLocalization R →+* PiLocalization S :=
  Pi.ringHom fun I ↦ (Localization.localRingHom _ _ f rfl).comp <|
    Pi.evalRingHom _ (⟨_, I.2.comap_bijective f hf⟩ : MaximalSpectrum R)


theorem mapPiLocalization_naturality :
    (mapPiLocalization f hf).comp (toPiLocalization R) =
      (toPiLocalization S).comp f := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Bijective ⇑f
    ⊢ Eq ((MaximalSpectrum.mapPiLocalization f hf).comp (MaximalSpectrum.toPiLocal …
  -/
  ext r I
  /-
    case a.h
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Bijective ⇑f
    r : R
    I : MaximalSpectrum S
    ⊢ Eq (((MaximalSpectrum.mapPiLocalization f hf).comp (MaximalSpectrum.toPiLoca …
  -/
  show Localization.localRingHom _ _ _ rfl (algebraMap _ _ r) = algebraMap _ _ (f r)
  simp_rw [← IsLocalization.mk'_one (M := (I.1.comap f).primeCompl), Localization.localRingHom_mk',
    ← IsLocalization.mk'_one (M := I.1.primeCompl), Submonoid.coe_one, map_one f]
  /-
    case a.h
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Bijective ⇑f
    r : R
    I : MaximalSpectrum S
    ⊢ Eq (IsLocalization.mk' (Localization.AtPrime I.asIdeal) (f r) ⟨1, ⋯⟩) (IsLoc …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mapPiLocalization_id : mapPiLocalization (.id R) Function.bijective_id = .id _ :=
  RingHom.ext fun _ ↦ funext fun _ ↦ congr($(Localization.localRingHom_id _) _)


theorem mapPiLocalization_comp :
    mapPiLocalization (g.comp f) (hg.comp hf) =
      (mapPiLocalization g hg).comp (mapPiLocalization f hf) :=
  RingHom.ext fun _ ↦ funext fun _ ↦ congr($(Localization.localRingHom_comp _ _ _ _ rfl _ rfl) _)


theorem mapPiLocalization_bijective : Function.Bijective (mapPiLocalization f hf) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Bijective ⇑f
    ⊢ Function.Bijective ⇑(MaximalSpectrum.mapPiLocalization f hf)
  -/
  let f := RingEquiv.ofBijective f hf
  let e := RingEquiv.ofRingHom (mapPiLocalization f hf)
    (mapPiLocalization (f.symm : S →+* R) f.symm.bijective) ?_ ?_
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      f✝ : RingHom R S
      hf : Function.Bijective ⇑f✝
      f : RingEquiv R S := RingEquiv.ofBijective f✝ hf
      e : RingEquiv (MaximalSpectrum.PiLocalization R) (MaximalSpectrum.PiLocalizati …
      ⊢ Function.Bijective ⇑(MaximalSpectrum.mapPiLocalization f✝ hf)
    -/
  · exact e.bijective
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      f✝ : RingHom R S
      hf : Function.Bijective ⇑f✝
      f : RingEquiv R S := RingEquiv.ofBijective f✝ hf
      ⊢ Eq ((MaximalSpectrum.mapPiLocalization (↑f) hf).comp (MaximalSpectrum.mapPiL …
    -/
  · rw [← mapPiLocalization_comp]
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      f✝ : RingHom R S
      hf : Function.Bijective ⇑f✝
      f : RingEquiv R S := RingEquiv.ofBijective f✝ hf
      ⊢ Eq (MaximalSpectrum.mapPiLocalization ((↑f).comp ↑f.symm) ⋯) (RingHom.id (Ma …
    -/
    simp_rw [RingEquiv.comp_symm, mapPiLocalization_id]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      f✝ : RingHom R S
      hf : Function.Bijective ⇑f✝
      f : RingEquiv R S := RingEquiv.ofBijective f✝ hf
      ⊢ Eq ((MaximalSpectrum.mapPiLocalization ↑f.symm ⋯).comp (MaximalSpectrum.mapP …
    -/
  · rw [← mapPiLocalization_comp]
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      f✝ : RingHom R S
      hf : Function.Bijective ⇑f✝
      f : RingEquiv R S := RingEquiv.ofBijective f✝ hf
      ⊢ Eq (MaximalSpectrum.mapPiLocalization ((↑f.symm).comp ↑f) ⋯) (RingHom.id (Ma …
    -/
    simp_rw [RingEquiv.symm_comp, mapPiLocalization_id]
    /-
      🎉 no goals
    -/


theorem toPiLocalization_not_surjective_of_infinite [Infinite ι] :
    ¬ Function.Surjective (toPiLocalization (Π i, R i)) := fun surj ↦ by
  /-
    ι : Type u_5
    R : ι → Type u_4
    inst✝² : (i : ι) → CommSemiring (R i)
    inst✝¹ : ∀ (i : ι), Nontrivial (R i)
    inst✝ : Infinite ι
    surj : Function.Surjective ⇑(MaximalSpectrum.toPiLocalization ((i : ι) → R i))
    ⊢ False
  -/
  have ⟨J, max, nmem⟩ := PrimeSpectrum.exists_maximal_nmem_range_sigmaToPi_of_infinite R
  /-
    ι : Type u_5
    R : ι → Type u_4
    inst✝² : (i : ι) → CommSemiring (R i)
    inst✝¹ : ∀ (i : ι), Nontrivial (R i)
    inst✝ : Infinite ι
    surj : Function.Surjective ⇑(MaximalSpectrum.toPiLocalization ((i : ι) → R i))
    J : Ideal ((i : ι) → R i)
    max : J.IsMaximal
    nmem : Not (Membership.mem (Set.range (PrimeSpectrum.sigmaToPi R)) { asIdeal : …
    ⊢ False
  -/
  obtain ⟨r, hr⟩ := surj (Function.update 0 ⟨J, max⟩ 1)
  have : r = 0 := funext fun i ↦ toPiLocalization_injective _ <| funext fun I ↦ by
    replace hr := congr_fun hr ⟨_, I.2.comap_piEvalRingHom⟩
    dsimp only [toPiLocalization_apply_apply, Subtype.coe_mk] at hr
    simp_rw [toPiLocalization_apply_apply,
      ← Localization.AtPrime.mapPiEvalRingHom_algebraMap_apply, hr]
    rw [Function.update_of_ne]; · simp_rw [Pi.zero_apply, map_zero]
    exact fun h ↦ nmem ⟨⟨i, I.1, I.2.isPrime⟩, PrimeSpectrum.ext congr($h.1)⟩
  /-
    case intro
    ι : Type u_5
    R : ι → Type u_4
    inst✝² : (i : ι) → CommSemiring (R i)
    inst✝¹ : ∀ (i : ι), Nontrivial (R i)
    inst✝ : Infinite ι
    surj : Function.Surjective ⇑(MaximalSpectrum.toPiLocalization ((i : ι) → R i))
    J : Ideal ((i : ι) → R i)
    max : J.IsMaximal
    nmem : Not (Membership.mem (Set.range (PrimeSpectrum.sigmaToPi R)) { asIdeal : …
    r : (i : ι) → R i
    hr : Eq ((MaximalSpectrum.toPiLocalization ((i : ι) → R i)) r) (Function.updat …
    this : Eq r 0
    ⊢ False
  -/
  replace hr := congr_fun hr ⟨J, max⟩
  /-
    case intro
    ι : Type u_5
    R : ι → Type u_4
    inst✝² : (i : ι) → CommSemiring (R i)
    inst✝¹ : ∀ (i : ι), Nontrivial (R i)
    inst✝ : Infinite ι
    surj : Function.Surjective ⇑(MaximalSpectrum.toPiLocalization ((i : ι) → R i))
    J : Ideal ((i : ι) → R i)
    max : J.IsMaximal
    nmem : Not (Membership.mem (Set.range (PrimeSpectrum.sigmaToPi R)) { asIdeal : …
    r : (i : ι) → R i
    this : Eq r 0
    hr : Eq ((MaximalSpectrum.toPiLocalization ((i : ι) → R i)) r { asIdeal := J,  …
    ⊢ False
  -/
  rw [this, map_zero, Function.update_self] at hr
  /-
    case intro
    ι : Type u_5
    R : ι → Type u_4
    inst✝² : (i : ι) → CommSemiring (R i)
    inst✝¹ : ∀ (i : ι), Nontrivial (R i)
    inst✝ : Infinite ι
    surj : Function.Surjective ⇑(MaximalSpectrum.toPiLocalization ((i : ι) → R i))
    J : Ideal ((i : ι) → R i)
    max : J.IsMaximal
    nmem : Not (Membership.mem (Set.range (PrimeSpectrum.sigmaToPi R)) { asIdeal : …
    r : (i : ι) → R i
    this : Eq r 0
    hr : Eq (0 { asIdeal := J, IsMaximal := max }) 1
    ⊢ False
  -/
  exact zero_ne_one hr
  /-
    🎉 no goals
  -/


theorem finite_of_toPiLocalization_pi_surjective
    (h : Function.Surjective (toPiLocalization (Π i, R i))) :
    Finite ι := by
  /-
    ι : Type u_5
    R : ι → Type u_4
    inst✝¹ : (i : ι) → CommSemiring (R i)
    inst✝ : ∀ (i : ι), Nontrivial (R i)
    h : Function.Surjective ⇑(MaximalSpectrum.toPiLocalization ((i : ι) → R i))
    ⊢ Finite ι
  -/
  contrapose h; rw [not_finite_iff_infinite] at h
  /-
    ι : Type u_5
    R : ι → Type u_4
    inst✝¹ : (i : ι) → CommSemiring (R i)
    inst✝ : ∀ (i : ι), Nontrivial (R i)
    h : Infinite ι
    ⊢ Not (Function.Surjective ⇑(MaximalSpectrum.toPiLocalization ((i : ι) → R i)))
  -/
  exact toPiLocalization_not_surjective_of_infinite _
  /-
    🎉 no goals
  -/


theorem finite_of_toPiLocalization_surjective
    (surj : Function.Surjective (toPiLocalization R)) :
    Finite (MaximalSpectrum R) := by
  replace surj := mapPiLocalization_bijective _ ⟨toPiLocalization_injective R, surj⟩
    |>.2.comp surj
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    surj✝ : Function.Surjective ⇑(MaximalSpectrum.toPiLocalization R)
    surj : Function.Surjective (Function.comp ⇑(MaximalSpectrum.mapPiLocalization  …
    ⊢ Finite (MaximalSpectrum R)
  -/
  rw [← RingHom.coe_comp, mapPiLocalization_naturality, RingHom.coe_comp] at surj
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    surj✝ : Function.Surjective ⇑(MaximalSpectrum.toPiLocalization R)
    surj : Function.Surjective (Function.comp ⇑(MaximalSpectrum.toPiLocalization ( …
    ⊢ Finite (MaximalSpectrum R)
  -/
  exact finite_of_toPiLocalization_pi_surjective surj.of_comp
  /-
    🎉 no goals
  -/


/-- The product of localizations at all prime ideals of a commutative semiring. -/
abbrev PiLocalization : Type _ := Π p : PrimeSpectrum R, Localization p.asIdeal.primeCompl


/-- The canonical ring homomorphism from a commutative semiring to the product of its
localizations at all prime ideals. It is always injective. -/
def toPiLocalization : R →+* PiLocalization R := algebraMap R _


theorem toPiLocalization_injective : Function.Injective (toPiLocalization R) :=
  fun _ _ eq ↦ MaximalSpectrum.toPiLocalization_injective R <|
    funext fun I ↦ congr_fun eq I.toPrimeSpectrum


/-- The projection from the product of localizations at primes to the product of
localizations at maximal ideals. -/
def piLocalizationToMaximal : PiLocalization R →+* MaximalSpectrum.PiLocalization R :=
  Pi.ringHom fun I ↦ Pi.evalRingHom _ I.toPrimeSpectrum


theorem piLocalizationToMaximal_surjective : Function.Surjective (piLocalizationToMaximal R) :=
  fun r ↦ ⟨fun I ↦ if h : I.1.IsMaximal then r ⟨_, h⟩ else 0, funext fun _ ↦ dif_pos _⟩


/-- If R has Krull dimension ≤ 0, then `piLocalizationToIsMaximal R` is an isomorphism. -/
def piLocalizationToMaximalEquiv (h : ∀ I : Ideal R, I.IsPrime → I.IsMaximal) :
    PiLocalization R ≃+* MaximalSpectrum.PiLocalization R where
  __ := piLocalizationToMaximal R
  invFun := Pi.ringHom fun I ↦ Pi.evalRingHom _ (⟨_, h _ I.2⟩ : MaximalSpectrum R)
  left_inv _ := rfl
  right_inv _ := rfl


theorem piLocalizationToMaximal_bijective (h : ∀ I : Ideal R, I.IsPrime → I.IsMaximal) :
    Function.Bijective (piLocalizationToMaximal R) :=
  (piLocalizationToMaximalEquiv h).bijective


theorem piLocalizationToMaximal_comp_toPiLocalization :
    (piLocalizationToMaximal R).comp (toPiLocalization R) = MaximalSpectrum.toPiLocalization R :=
  rfl


theorem isMaximal_of_toPiLocalization_surjective (surj : Function.Surjective (toPiLocalization R))
    (I : PrimeSpectrum R) : I.1.IsMaximal := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization R)
    I : PrimeSpectrum R
    ⊢ I.asIdeal.IsMaximal
  -/
  have ⟨J, max, le⟩ := I.1.exists_le_maximal I.2.ne_top
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization R)
    I : PrimeSpectrum R
    J : Ideal R
    max : J.IsMaximal
    le : LE.le I.asIdeal J
    ⊢ I.asIdeal.IsMaximal
  -/
  obtain ⟨r, hr⟩ := surj (Function.update 0 ⟨J, max.isPrime⟩ 1)
  /-
    case intro
    R : Type u_1
    inst✝ : CommSemiring R
    surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization R)
    I : PrimeSpectrum R
    J : Ideal R
    max : J.IsMaximal
    le : LE.le I.asIdeal J
    r : R
    hr : Eq ((PrimeSpectrum.toPiLocalization R) r) (Function.update 0 { asIdeal := …
    ⊢ I.asIdeal.IsMaximal
  -/
  by_contra h
  /-
    case intro
    R : Type u_1
    inst✝ : CommSemiring R
    surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization R)
    I : PrimeSpectrum R
    J : Ideal R
    max : J.IsMaximal
    le : LE.le I.asIdeal J
    r : R
    hr : Eq ((PrimeSpectrum.toPiLocalization R) r) (Function.update 0 { asIdeal := …
    h : Not I.asIdeal.IsMaximal
    ⊢ False
  -/
  have hJ : algebraMap _ _ r = _ := (congr_fun hr _).trans (Function.update_self ..)
  /-
    case intro
    R : Type u_1
    inst✝ : CommSemiring R
    surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization R)
    I : PrimeSpectrum R
    J : Ideal R
    max : J.IsMaximal
    le : LE.le I.asIdeal J
    r : R
    hr : Eq ((PrimeSpectrum.toPiLocalization R) r) (Function.update 0 { asIdeal := …
    h : Not I.asIdeal.IsMaximal
    hJ : Eq ((algebraMap R (Localization { asIdeal := J, isPrime := ⋯ }.asIdeal.pr …
    ⊢ False
  -/
  have hI : algebraMap _ _ r = _ := congr_fun hr I
  rw [← IsLocalization.lift_eq (M := J.primeCompl) (S := Localization J.primeCompl), hJ, map_one,
    Function.update_of_ne] at hI
    /-
      case intro
      R : Type u_1
      inst✝ : CommSemiring R
      surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization R)
      I : PrimeSpectrum R
      J : Ideal R
      max : J.IsMaximal
      le : LE.le I.asIdeal J
      r : R
      hr : Eq ((PrimeSpectrum.toPiLocalization R) r) (Function.update 0 { asIdeal := …
      h : Not I.asIdeal.IsMaximal
      hJ : Eq ((algebraMap R (Localization { asIdeal := J, isPrime := ⋯ }.asIdeal.pr …
      hI✝ : Eq ((algebraMap R (Localization I.asIdeal.primeCompl)) r) (Function.upda …
      hI : Eq 1 (0 I)
      ⊢ False
    -/
  · exact one_ne_zero hI
    /-
      🎉 no goals
    -/
    /-
      case intro.h
      R : Type u_1
      inst✝ : CommSemiring R
      surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization R)
      I : PrimeSpectrum R
      J : Ideal R
      max : J.IsMaximal
      le : LE.le I.asIdeal J
      r : R
      hr : Eq ((PrimeSpectrum.toPiLocalization R) r) (Function.update 0 { asIdeal := …
      h : Not I.asIdeal.IsMaximal
      hJ : Eq ((algebraMap R (Localization { asIdeal := J, isPrime := ⋯ }.asIdeal.pr …
      hI✝ : Eq ((algebraMap R (Localization I.asIdeal.primeCompl)) r) (Function.upda …
      hI : Eq 1 (Function.update 0 { asIdeal := J, isPrime := ⋯ } 1 I)
      ⊢ Ne I { asIdeal := J, isPrime := ⋯ }
    -/
  · intro eq; have : I.1 = J := congr_arg (·.1) eq; exact h (this ▸ max)
                                                    /-
                                                      🎉 no goals
                                                    -/
    /-
      case intro.hg
      R : Type u_1
      inst✝ : CommSemiring R
      surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization R)
      I : PrimeSpectrum R
      J : Ideal R
      max : J.IsMaximal
      le : LE.le I.asIdeal J
      r : R
      hr : Eq ((PrimeSpectrum.toPiLocalization R) r) (Function.update 0 { asIdeal := …
      h : Not I.asIdeal.IsMaximal
      hJ : Eq ((algebraMap R (Localization { asIdeal := J, isPrime := ⋯ }.asIdeal.pr …
      hI : Eq ((algebraMap R (Localization I.asIdeal.primeCompl)) r) (Function.updat …
      ⊢ ∀ (y : Subtype fun x => Membership.mem J.primeCompl x), IsUnit ((algebraMap  …
    -/
  · exact fun ⟨s, hs⟩ ↦ IsLocalization.map_units (M := I.1.primeCompl) _ ⟨s, fun h ↦ hs (le h)⟩
    /-
      🎉 no goals
    -/


/-- A ring homomorphism induces a homomorphism between the products of localizations at primes. -/
noncomputable def mapPiLocalization : PiLocalization R →+* PiLocalization S :=
  Pi.ringHom fun I ↦ (Localization.localRingHom _ I.1 f rfl).comp (Pi.evalRingHom _ (f.specComap I))


theorem mapPiLocalization_naturality :
    (mapPiLocalization f).comp (toPiLocalization R) = (toPiLocalization S).comp f := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    ⊢ Eq ((PrimeSpectrum.mapPiLocalization f).comp (PrimeSpectrum.toPiLocalization …
  -/
  ext r I
  /-
    case a.h
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    r : R
    I : PrimeSpectrum S
    ⊢ Eq (((PrimeSpectrum.mapPiLocalization f).comp (PrimeSpectrum.toPiLocalizatio …
  -/
  show Localization.localRingHom _ _ _ rfl (algebraMap _ _ r) = algebraMap _ _ (f r)
  simp_rw [← IsLocalization.mk'_one (M := (I.1.comap f).primeCompl), Localization.localRingHom_mk',
    ← IsLocalization.mk'_one (M := I.1.primeCompl), Submonoid.coe_one, map_one f]
  /-
    case a.h
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    r : R
    I : PrimeSpectrum S
    ⊢ Eq (IsLocalization.mk' (Localization.AtPrime I.asIdeal) (f r) ⟨1, ⋯⟩) (IsLoc …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mapPiLocalization_id : mapPiLocalization (.id R) = .id _ := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    ⊢ Eq (PrimeSpectrum.mapPiLocalization (RingHom.id R)) (RingHom.id (PrimeSpectr …
  -/
  ext; exact congr($(Localization.localRingHom_id _) _)
       /-
         🎉 no goals
       -/


theorem mapPiLocalization_comp (g : S →+* P) :
    mapPiLocalization (g.comp f) = (mapPiLocalization g).comp (mapPiLocalization f) := by
  /-
    R : Type u_1
    S : Type u_2
    P : Type u_3
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : CommSemiring P
    f : RingHom R S
    g : RingHom S P
    ⊢ Eq (PrimeSpectrum.mapPiLocalization (g.comp f)) ((PrimeSpectrum.mapPiLocaliz …
  -/
  ext; exact congr($(Localization.localRingHom_comp _ _ _ _ rfl _ rfl) _)
       /-
         🎉 no goals
       -/


theorem mapPiLocalization_bijective (hf : Function.Bijective f) :
    Function.Bijective (mapPiLocalization f) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f : RingHom R S
    hf : Function.Bijective ⇑f
    ⊢ Function.Bijective ⇑(PrimeSpectrum.mapPiLocalization f)
  -/
  let f := RingEquiv.ofBijective f hf
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : CommSemiring R
    inst✝ : CommSemiring S
    f✝ : RingHom R S
    hf : Function.Bijective ⇑f✝
    f : RingEquiv R S := RingEquiv.ofBijective f✝ hf
    ⊢ Function.Bijective ⇑(PrimeSpectrum.mapPiLocalization f✝)
  -/
  let e := RingEquiv.ofRingHom (mapPiLocalization (f : R →+* S)) (mapPiLocalization f.symm) ?_ ?_
    /-
      case refine_3
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      f✝ : RingHom R S
      hf : Function.Bijective ⇑f✝
      f : RingEquiv R S := RingEquiv.ofBijective f✝ hf
      e : RingEquiv (PrimeSpectrum.PiLocalization R) (PrimeSpectrum.PiLocalization S …
      ⊢ Function.Bijective ⇑(PrimeSpectrum.mapPiLocalization f✝)
    -/
  · exact e.bijective
    /-
      🎉 no goals
    -/
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      f✝ : RingHom R S
      hf : Function.Bijective ⇑f✝
      f : RingEquiv R S := RingEquiv.ofBijective f✝ hf
      ⊢ Eq ((PrimeSpectrum.mapPiLocalization ↑f).comp (PrimeSpectrum.mapPiLocalizati …
    -/
  · rw [← mapPiLocalization_comp, RingEquiv.comp_symm, mapPiLocalization_id]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝¹ : CommSemiring R
      inst✝ : CommSemiring S
      f✝ : RingHom R S
      hf : Function.Bijective ⇑f✝
      f : RingEquiv R S := RingEquiv.ofBijective f✝ hf
      ⊢ Eq ((PrimeSpectrum.mapPiLocalization ↑f.symm).comp (PrimeSpectrum.mapPiLocal …
    -/
  · rw [← mapPiLocalization_comp, RingEquiv.symm_comp, mapPiLocalization_id]
    /-
      🎉 no goals
    -/


theorem toPiLocalization_not_surjective_of_infinite [Infinite ι] :
    ¬ Function.Surjective (toPiLocalization (Π i, R i)) :=
  fun surj ↦ MaximalSpectrum.toPiLocalization_not_surjective_of_infinite R <| by
    /-
      ι : Type u_5
      R : ι → Type u_4
      inst✝² : (i : ι) → CommSemiring (R i)
      inst✝¹ : ∀ (i : ι), Nontrivial (R i)
      inst✝ : Infinite ι
      surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization ((i : ι) → R i))
      ⊢ Function.Surjective ⇑(MaximalSpectrum.toPiLocalization ((i : ι) → R i))
    -/
    rw [← piLocalizationToMaximal_comp_toPiLocalization]
    /-
      ι : Type u_5
      R : ι → Type u_4
      inst✝² : (i : ι) → CommSemiring (R i)
      inst✝¹ : ∀ (i : ι), Nontrivial (R i)
      inst✝ : Infinite ι
      surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization ((i : ι) → R i))
      ⊢ Function.Surjective ⇑((PrimeSpectrum.piLocalizationToMaximal ((i : ι) → R i) …
    -/
    exact (piLocalizationToMaximal_surjective _).comp surj
    /-
      🎉 no goals
    -/


theorem finite_of_toPiLocalization_pi_surjective
    (h : Function.Surjective (toPiLocalization (Π i, R i))) :
    Finite ι := by
  /-
    ι : Type u_5
    R : ι → Type u_4
    inst✝¹ : (i : ι) → CommSemiring (R i)
    inst✝ : ∀ (i : ι), Nontrivial (R i)
    h : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization ((i : ι) → R i))
    ⊢ Finite ι
  -/
  contrapose h; rw [not_finite_iff_infinite] at h
  /-
    ι : Type u_5
    R : ι → Type u_4
    inst✝¹ : (i : ι) → CommSemiring (R i)
    inst✝ : ∀ (i : ι), Nontrivial (R i)
    h : Infinite ι
    ⊢ Not (Function.Surjective ⇑(PrimeSpectrum.toPiLocalization ((i : ι) → R i)))
  -/
  exact toPiLocalization_not_surjective_of_infinite _
  /-
    🎉 no goals
  -/


theorem finite_of_toPiLocalization_surjective
    (surj : Function.Surjective (toPiLocalization R)) :
    Finite (PrimeSpectrum R) := by
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    surj : Function.Surjective ⇑(PrimeSpectrum.toPiLocalization R)
    ⊢ Finite (PrimeSpectrum R)
  -/
  replace surj := (mapPiLocalization_bijective _ ⟨toPiLocalization_injective R, surj⟩).2.comp surj
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    surj : Function.Surjective (Function.comp ⇑(PrimeSpectrum.mapPiLocalization (P …
    ⊢ Finite (PrimeSpectrum R)
  -/
  rw [← RingHom.coe_comp, mapPiLocalization_naturality, RingHom.coe_comp] at surj
  /-
    R : Type u_1
    inst✝ : CommSemiring R
    surj : Function.Surjective (Function.comp ⇑(PrimeSpectrum.toPiLocalization (Pr …
    ⊢ Finite (PrimeSpectrum R)
  -/
  exact finite_of_toPiLocalization_pi_surjective surj.of_comp
  /-
    🎉 no goals
  -/


