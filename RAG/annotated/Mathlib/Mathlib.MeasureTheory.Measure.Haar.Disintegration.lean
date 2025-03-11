instance (T : Submodule 𝕜 E) : BorelSpace T := Subtype.borelSpace _

instance (T : Submodule 𝕜 E) : OpensMeasurableSpace T := Subtype.opensMeasurableSpace _


/-- The image of an additive Haar measure under a surjective linear map is proportional to a given
additive Haar measure. The proportionality factor will be infinite if the linear map has a
nontrivial kernel. -/
theorem LinearMap.exists_map_addHaar_eq_smul_addHaar' (h : Function.Surjective L) :
    ∃ (c : ℝ≥0∞), 0 < c ∧ c < ∞ ∧ μ.map L = (c * addHaar (univ : Set (LinearMap.ker L))) • ν := by
  /- This is true for the second projection in product spaces, as the projection of the Haar
  measure `μS.prod μT` is equal to the Haar measure `μT` multiplied by the total mass of `μS`. This
  is also true for linear equivalences, as they map Haar measure to Haar measure. The general case
  follows from these two and linear algebra, as `L` can be interpreted as the composition of the
  projection `P` on a complement `T` to its kernel `S`, together with a linear equivalence. -/
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  have : FiniteDimensional 𝕜 E := .of_locallyCompactSpace 𝕜
  have : ProperSpace F := by
    rcases subsingleton_or_nontrivial E with hE|hE
    · have : Subsingleton F := Function.Surjective.subsingleton h
      infer_instance
    · have : ProperSpace 𝕜 := .of_locallyCompact_module 𝕜 E
      have : FiniteDimensional 𝕜 F := Module.Finite.of_surjective L h
      exact FiniteDimensional.proper 𝕜 F
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  let S : Submodule 𝕜 E := LinearMap.ker L
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  obtain ⟨T, hT⟩ : ∃ T : Submodule 𝕜 E, IsCompl S T := Submodule.exists_isCompl S
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  let M : (S × T) ≃ₗ[𝕜] E := Submodule.prodEquivOfIsCompl S T hT
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    M : LinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Su …
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  have M_cont : Continuous M.symm := LinearMap.continuous_of_finiteDimensional _
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    M : LinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Su …
    M_cont : Continuous ⇑M.symm
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  let P : S × T →ₗ[𝕜] T := LinearMap.snd 𝕜 S T
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    M : LinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Su …
    M_cont : Continuous ⇑M.symm
    P : LinearMap (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Subt …
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  have P_cont : Continuous P := LinearMap.continuous_of_finiteDimensional _
  have I : Function.Bijective (LinearMap.domRestrict L T) :=
    ⟨LinearMap.injective_domRestrict_iff.2 (IsCompl.inf_eq_bot hT.symm),
    (LinearMap.surjective_domRestrict_iff h).2 hT.symm.sup_eq_top⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    M : LinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Su …
    M_cont : Continuous ⇑M.symm
    P : LinearMap (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Subt …
    P_cont : Continuous ⇑P
    I : Function.Bijective ⇑(L.domRestrict T)
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  let L' : T ≃ₗ[𝕜] F := LinearEquiv.ofBijective (LinearMap.domRestrict L T) I
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    M : LinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Su …
    M_cont : Continuous ⇑M.symm
    P : LinearMap (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Subt …
    P_cont : Continuous ⇑P
    I : Function.Bijective ⇑(L.domRestrict T)
    L' : LinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem T x) F := Lin …
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  have L'_cont : Continuous L' := LinearMap.continuous_of_finiteDimensional _
  have A : L = (L' : T →ₗ[𝕜] F).comp (P.comp (M.symm : E →ₗ[𝕜] (S × T))) := by
    ext x
    obtain ⟨y, z, hyz⟩ : ∃ (y : S) (z : T), M.symm x = (y, z) := ⟨_, _, rfl⟩
    have : x = M (y, z) := by
      rw [← hyz]; simp only [LinearEquiv.apply_symm_apply]
    simp [L', P, M, this]
  have I : μ.map L = ((μ.map M.symm).map P).map L' := by
    rw [Measure.map_map, Measure.map_map, A]
    · rfl
    · exact L'_cont.measurable.comp P_cont.measurable
    · exact M_cont.measurable
    · exact L'_cont.measurable
    · exact P_cont.measurable
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    M : LinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Su …
    M_cont : Continuous ⇑M.symm
    P : LinearMap (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Subt …
    P_cont : Continuous ⇑P
    I✝ : Function.Bijective ⇑(L.domRestrict T)
    L' : LinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem T x) F := Lin …
    L'_cont : Continuous ⇑L'
    A : Eq L ((↑L').comp (P.comp ↑M.symm))
    I : Eq (MeasureTheory.Measure.map (⇑L) μ) (MeasureTheory.Measure.map (⇑L') (Me …
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  let μS : Measure S := addHaar
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    M : LinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Su …
    M_cont : Continuous ⇑M.symm
    P : LinearMap (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Subt …
    P_cont : Continuous ⇑P
    I✝ : Function.Bijective ⇑(L.domRestrict T)
    L' : LinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem T x) F := Lin …
    L'_cont : Continuous ⇑L'
    A : Eq L ((↑L').comp (P.comp ↑M.symm))
    I : Eq (MeasureTheory.Measure.map (⇑L) μ) (MeasureTheory.Measure.map (⇑L') (Me …
    μS : MeasureTheory.Measure (Subtype fun x => Membership.mem S x) := MeasureThe …
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  let μT : Measure T := addHaar
  obtain ⟨c₀, c₀_pos, c₀_fin, h₀⟩ :
      ∃ c₀ : ℝ≥0∞, c₀ ≠ 0 ∧ c₀ ≠ ∞ ∧ μ.map M.symm = c₀ • μS.prod μT := by
    have : IsAddHaarMeasure (μ.map M.symm) :=
      M.toContinuousLinearEquiv.symm.isAddHaarMeasure_map μ
    refine ⟨addHaarScalarFactor (μ.map M.symm) (μS.prod μT), ?_, ENNReal.coe_ne_top,
      isAddLeftInvariant_eq_smul _ _⟩
    simpa only [ne_eq, ENNReal.coe_eq_zero] using
      (addHaarScalarFactor_pos_of_isAddHaarMeasure (μ.map M.symm) (μS.prod μT)).ne'
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    M : LinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Su …
    M_cont : Continuous ⇑M.symm
    P : LinearMap (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Subt …
    P_cont : Continuous ⇑P
    I✝ : Function.Bijective ⇑(L.domRestrict T)
    L' : LinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem T x) F := Lin …
    L'_cont : Continuous ⇑L'
    A : Eq L ((↑L').comp (P.comp ↑M.symm))
    I : Eq (MeasureTheory.Measure.map (⇑L) μ) (MeasureTheory.Measure.map (⇑L') (Me …
    μS : MeasureTheory.Measure (Subtype fun x => Membership.mem S x) := MeasureThe …
    μT : MeasureTheory.Measure (Subtype fun x => Membership.mem T x) := MeasureThe …
    c₀ : ENNReal
    c₀_pos : Ne c₀ 0
    c₀_fin : Ne c₀ Top.top
    h₀ : Eq (MeasureTheory.Measure.map (⇑M.symm) μ) (HSMul.hSMul c₀ (μS.prod μT))
    ⊢ Exists fun c => And (LT.lt 0 c) (And (LT.lt c Top.top) (Eq (MeasureTheory.Me …
  -/
  have J : (μS.prod μT).map P = (μS univ) • μT := map_snd_prod
  obtain ⟨c₁, c₁_pos, c₁_fin, h₁⟩ : ∃ c₁ : ℝ≥0∞, c₁ ≠ 0 ∧ c₁ ≠ ∞ ∧ μT.map L' = c₁ • ν := by
    have : IsAddHaarMeasure (μT.map L') :=
      L'.toContinuousLinearEquiv.isAddHaarMeasure_map μT
    refine ⟨addHaarScalarFactor (μT.map L') ν, ?_, ENNReal.coe_ne_top,
      isAddLeftInvariant_eq_smul _ _⟩
    simpa only [ne_eq, ENNReal.coe_eq_zero] using
      (addHaarScalarFactor_pos_of_isAddHaarMeasure (μT.map L') ν).ne'
  refine ⟨c₀ * c₁, by simp [pos_iff_ne_zero, c₀_pos, c₁_pos],
    ENNReal.mul_lt_top c₀_fin.lt_top c₁_fin.lt_top, ?_⟩
  /-
    case intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    M : LinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Su …
    M_cont : Continuous ⇑M.symm
    P : LinearMap (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Subt …
    P_cont : Continuous ⇑P
    I✝ : Function.Bijective ⇑(L.domRestrict T)
    L' : LinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem T x) F := Lin …
    L'_cont : Continuous ⇑L'
    A : Eq L ((↑L').comp (P.comp ↑M.symm))
    I : Eq (MeasureTheory.Measure.map (⇑L) μ) (MeasureTheory.Measure.map (⇑L') (Me …
    μS : MeasureTheory.Measure (Subtype fun x => Membership.mem S x) := MeasureThe …
    μT : MeasureTheory.Measure (Subtype fun x => Membership.mem T x) := MeasureThe …
    c₀ : ENNReal
    c₀_pos : Ne c₀ 0
    c₀_fin : Ne c₀ Top.top
    h₀ : Eq (MeasureTheory.Measure.map (⇑M.symm) μ) (HSMul.hSMul c₀ (μS.prod μT))
    J : Eq (MeasureTheory.Measure.map (⇑P) (μS.prod μT)) (HSMul.hSMul (μS Set.univ …
    c₁ : ENNReal
    c₁_pos : Ne c₁ 0
    c₁_fin : Ne c₁ Top.top
    h₁ : Eq (MeasureTheory.Measure.map (⇑L') μT) (HSMul.hSMul c₁ ν)
    ⊢ Eq (MeasureTheory.Measure.map (⇑L) μ) (HSMul.hSMul (HMul.hMul (HMul.hMul c₀  …
  -/
  simp only [I, h₀, Measure.map_smul, J, smul_smul, h₁]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    this✝ : FiniteDimensional 𝕜 E
    this : ProperSpace F
    S : Submodule 𝕜 E := LinearMap.ker L
    T : Submodule 𝕜 E
    hT : IsCompl S T
    M : LinearEquiv (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Su …
    M_cont : Continuous ⇑M.symm
    P : LinearMap (RingHom.id 𝕜) (Prod (Subtype fun x => Membership.mem S x) (Subt …
    P_cont : Continuous ⇑P
    I✝ : Function.Bijective ⇑(L.domRestrict T)
    L' : LinearEquiv (RingHom.id 𝕜) (Subtype fun x => Membership.mem T x) F := Lin …
    L'_cont : Continuous ⇑L'
    A : Eq L ((↑L').comp (P.comp ↑M.symm))
    I : Eq (MeasureTheory.Measure.map (⇑L) μ) (MeasureTheory.Measure.map (⇑L') (Me …
    μS : MeasureTheory.Measure (Subtype fun x => Membership.mem S x) := MeasureThe …
    μT : MeasureTheory.Measure (Subtype fun x => Membership.mem T x) := MeasureThe …
    c₀ : ENNReal
    c₀_pos : Ne c₀ 0
    c₀_fin : Ne c₀ Top.top
    h₀ : Eq (MeasureTheory.Measure.map (⇑M.symm) μ) (HSMul.hSMul c₀ (μS.prod μT))
    J : Eq (MeasureTheory.Measure.map (⇑P) (μS.prod μT)) (HSMul.hSMul (μS Set.univ …
    c₁ : ENNReal
    c₁_pos : Ne c₁ 0
    c₁_fin : Ne c₁ Top.top
    h₁ : Eq (MeasureTheory.Measure.map (⇑L') μT) (HSMul.hSMul c₁ ν)
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HMul.hMul c₀ (μS Set.univ)) c₁) ν) (HSMul.hSMul  …
  -/
  rw [mul_assoc, mul_comm _ c₁, ← mul_assoc]
  /-
    🎉 no goals
  -/


/-- The image of an additive Haar measure under a surjective linear map is proportional to a given
additive Haar measure, with a positive (but maybe infinite) factor. -/
theorem LinearMap.exists_map_addHaar_eq_smul_addHaar (h : Function.Surjective L) :
    ∃ (c : ℝ≥0∞), 0 < c ∧ μ.map L = c • ν := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    ⊢ Exists fun c => And (LT.lt 0 c) (Eq (MeasureTheory.Measure.map (⇑L) μ) (HSMu …
  -/
  rcases L.exists_map_addHaar_eq_smul_addHaar' μ ν h with ⟨c, c_pos, -, hc⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    c : ENNReal
    c_pos : LT.lt 0 c
    hc : Eq (MeasureTheory.Measure.map (⇑L) μ) (HSMul.hSMul (HMul.hMul c (MeasureT …
    ⊢ Exists fun c => And (LT.lt 0 c) (Eq (MeasureTheory.Measure.map (⇑L) μ) (HSMu …
  -/
  exact ⟨_, by simp [c_pos, NeZero.ne addHaar], hc⟩
  /-
    🎉 no goals
  -/


/-- Given a surjective linear map `L`, it is equivalent to require a property almost everywhere
in the source or the target spaces of `L`, with respect to additive Haar measures there. -/
lemma ae_comp_linearMap_mem_iff (h : Function.Surjective L) {s : Set F} (hs : MeasurableSet s) :
    (∀ᵐ x ∂μ, L x ∈ s) ↔ ∀ᵐ y ∂ν, y ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    s : Set F
    hs : MeasurableSet s
    ⊢ Iff (Filter.Eventually (fun x => Membership.mem s (L x)) (MeasureTheory.ae μ …
  -/
  have : FiniteDimensional 𝕜 E := .of_locallyCompactSpace 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    s : Set F
    hs : MeasurableSet s
    this : FiniteDimensional 𝕜 E
    ⊢ Iff (Filter.Eventually (fun x => Membership.mem s (L x)) (MeasureTheory.ae μ …
  -/
  have : AEMeasurable L μ := L.continuous_of_finiteDimensional.aemeasurable
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    s : Set F
    hs : MeasurableSet s
    this✝ : FiniteDimensional 𝕜 E
    this : AEMeasurable (⇑L) μ
    ⊢ Iff (Filter.Eventually (fun x => Membership.mem s (L x)) (MeasureTheory.ae μ …
  -/
  apply (ae_map_iff this hs).symm.trans
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    s : Set F
    hs : MeasurableSet s
    this✝ : FiniteDimensional 𝕜 E
    this : AEMeasurable (⇑L) μ
    ⊢ Iff (Filter.Eventually (fun y => s y) (MeasureTheory.ae (MeasureTheory.Measu …
  -/
  rcases L.exists_map_addHaar_eq_smul_addHaar μ ν h with ⟨c, c_pos, hc⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    s : Set F
    hs : MeasurableSet s
    this✝ : FiniteDimensional 𝕜 E
    this : AEMeasurable (⇑L) μ
    c : ENNReal
    c_pos : LT.lt 0 c
    hc : Eq (MeasureTheory.Measure.map (⇑L) μ) (HSMul.hSMul c ν)
    ⊢ Iff (Filter.Eventually (fun y => s y) (MeasureTheory.ae (MeasureTheory.Measu …
  -/
  rw [hc]
  /-
    case intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹² : NontriviallyNormedField 𝕜
    inst✝¹¹ : CompleteSpace 𝕜
    inst✝¹⁰ : NormedAddCommGroup E
    inst✝⁹ : MeasurableSpace E
    inst✝⁸ : BorelSpace E
    inst✝⁷ : NormedSpace 𝕜 E
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : MeasurableSpace F
    inst✝⁴ : BorelSpace F
    inst✝³ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝² : μ.IsAddHaarMeasure
    inst✝¹ : ν.IsAddHaarMeasure
    inst✝ : LocallyCompactSpace E
    h : Function.Surjective ⇑L
    s : Set F
    hs : MeasurableSet s
    this✝ : FiniteDimensional 𝕜 E
    this : AEMeasurable (⇑L) μ
    c : ENNReal
    c_pos : LT.lt 0 c
    hc : Eq (MeasureTheory.Measure.map (⇑L) μ) (HSMul.hSMul c ν)
    ⊢ Iff (Filter.Eventually (fun y => s y) (MeasureTheory.ae (HSMul.hSMul c ν)))  …
  -/
  exact ae_smul_measure_iff c_pos.ne'
  /-
    🎉 no goals
  -/


/-- Given a linear map `L : E → F`, a property holds almost everywhere in `F` if and only if,
almost everywhere in `F`, it holds almost everywhere along the subspace spanned by the
image of `L`. This is an instance of a disintegration argument for additive Haar measures. -/
lemma ae_ae_add_linearMap_mem_iff [LocallyCompactSpace F] {s : Set F} (hs : MeasurableSet s) :
    (∀ᵐ y ∂ν, ∀ᵐ x ∂μ, y + L x ∈ s) ↔ ∀ᵐ y ∂ν, y ∈ s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : CompleteSpace 𝕜
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : MeasurableSpace F
    inst✝⁵ : BorelSpace F
    inst✝⁴ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝³ : μ.IsAddHaarMeasure
    inst✝² : ν.IsAddHaarMeasure
    inst✝¹ : LocallyCompactSpace E
    inst✝ : LocallyCompactSpace F
    s : Set F
    hs : MeasurableSet s
    ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => Membership.mem  …
  -/
  have : FiniteDimensional 𝕜 E := .of_locallyCompactSpace 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : CompleteSpace 𝕜
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : MeasurableSpace F
    inst✝⁵ : BorelSpace F
    inst✝⁴ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝³ : μ.IsAddHaarMeasure
    inst✝² : ν.IsAddHaarMeasure
    inst✝¹ : LocallyCompactSpace E
    inst✝ : LocallyCompactSpace F
    s : Set F
    hs : MeasurableSet s
    this : FiniteDimensional 𝕜 E
    ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => Membership.mem  …
  -/
  have : FiniteDimensional 𝕜 F := .of_locallyCompactSpace 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : CompleteSpace 𝕜
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : MeasurableSpace F
    inst✝⁵ : BorelSpace F
    inst✝⁴ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝³ : μ.IsAddHaarMeasure
    inst✝² : ν.IsAddHaarMeasure
    inst✝¹ : LocallyCompactSpace E
    inst✝ : LocallyCompactSpace F
    s : Set F
    hs : MeasurableSet s
    this✝ : FiniteDimensional 𝕜 E
    this : FiniteDimensional 𝕜 F
    ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => Membership.mem  …
  -/
  have : ProperSpace E := .of_locallyCompactSpace 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : CompleteSpace 𝕜
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : MeasurableSpace F
    inst✝⁵ : BorelSpace F
    inst✝⁴ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝³ : μ.IsAddHaarMeasure
    inst✝² : ν.IsAddHaarMeasure
    inst✝¹ : LocallyCompactSpace E
    inst✝ : LocallyCompactSpace F
    s : Set F
    hs : MeasurableSet s
    this✝¹ : FiniteDimensional 𝕜 E
    this✝ : FiniteDimensional 𝕜 F
    this : ProperSpace E
    ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => Membership.mem  …
  -/
  have : ProperSpace F := .of_locallyCompactSpace 𝕜
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : CompleteSpace 𝕜
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : MeasurableSpace F
    inst✝⁵ : BorelSpace F
    inst✝⁴ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝³ : μ.IsAddHaarMeasure
    inst✝² : ν.IsAddHaarMeasure
    inst✝¹ : LocallyCompactSpace E
    inst✝ : LocallyCompactSpace F
    s : Set F
    hs : MeasurableSet s
    this✝² : FiniteDimensional 𝕜 E
    this✝¹ : FiniteDimensional 𝕜 F
    this✝ : ProperSpace E
    this : ProperSpace F
    ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => Membership.mem  …
  -/
  let M : F × E →ₗ[𝕜] F := LinearMap.id.coprod L
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : CompleteSpace 𝕜
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : MeasurableSpace F
    inst✝⁵ : BorelSpace F
    inst✝⁴ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝³ : μ.IsAddHaarMeasure
    inst✝² : ν.IsAddHaarMeasure
    inst✝¹ : LocallyCompactSpace E
    inst✝ : LocallyCompactSpace F
    s : Set F
    hs : MeasurableSet s
    this✝² : FiniteDimensional 𝕜 E
    this✝¹ : FiniteDimensional 𝕜 F
    this✝ : ProperSpace E
    this : ProperSpace F
    M : LinearMap (RingHom.id 𝕜) (Prod F E) F := LinearMap.id.coprod L
    ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => Membership.mem  …
  -/
  have M_cont : Continuous M := M.continuous_of_finiteDimensional
  -- Note: https://github.com/leanprover-community/mathlib4/pull/8386 had to change `range_eq_top` into
  -- `range_eq_top (f := _)`
  have hM : Function.Surjective M := by
    simp [M, ← LinearMap.range_eq_top (f := _), LinearMap.range_coprod]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : CompleteSpace 𝕜
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : MeasurableSpace F
    inst✝⁵ : BorelSpace F
    inst✝⁴ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝³ : μ.IsAddHaarMeasure
    inst✝² : ν.IsAddHaarMeasure
    inst✝¹ : LocallyCompactSpace E
    inst✝ : LocallyCompactSpace F
    s : Set F
    hs : MeasurableSet s
    this✝² : FiniteDimensional 𝕜 E
    this✝¹ : FiniteDimensional 𝕜 F
    this✝ : ProperSpace E
    this : ProperSpace F
    M : LinearMap (RingHom.id 𝕜) (Prod F E) F := LinearMap.id.coprod L
    M_cont : Continuous ⇑M
    hM : Function.Surjective ⇑M
    ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => Membership.mem  …
  -/
  have A : ∀ x, M x ∈ s ↔ x ∈ M ⁻¹' s := fun x ↦ Iff.rfl
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : CompleteSpace 𝕜
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : MeasurableSpace F
    inst✝⁵ : BorelSpace F
    inst✝⁴ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝³ : μ.IsAddHaarMeasure
    inst✝² : ν.IsAddHaarMeasure
    inst✝¹ : LocallyCompactSpace E
    inst✝ : LocallyCompactSpace F
    s : Set F
    hs : MeasurableSet s
    this✝² : FiniteDimensional 𝕜 E
    this✝¹ : FiniteDimensional 𝕜 F
    this✝ : ProperSpace E
    this : ProperSpace F
    M : LinearMap (RingHom.id 𝕜) (Prod F E) F := LinearMap.id.coprod L
    M_cont : Continuous ⇑M
    hM : Function.Surjective ⇑M
    A : ∀ (x : Prod F E), Iff (Membership.mem s (M x)) (Membership.mem (Set.preima …
    ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => Membership.mem  …
  -/
  simp_rw [← ae_comp_linearMap_mem_iff M (ν.prod μ) ν hM hs, A]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝¹³ : NontriviallyNormedField 𝕜
    inst✝¹² : CompleteSpace 𝕜
    inst✝¹¹ : NormedAddCommGroup E
    inst✝¹⁰ : MeasurableSpace E
    inst✝⁹ : BorelSpace E
    inst✝⁸ : NormedSpace 𝕜 E
    inst✝⁷ : NormedAddCommGroup F
    inst✝⁶ : MeasurableSpace F
    inst✝⁵ : BorelSpace F
    inst✝⁴ : NormedSpace 𝕜 F
    L : LinearMap (RingHom.id 𝕜) E F
    μ : MeasureTheory.Measure E
    ν : MeasureTheory.Measure F
    inst✝³ : μ.IsAddHaarMeasure
    inst✝² : ν.IsAddHaarMeasure
    inst✝¹ : LocallyCompactSpace E
    inst✝ : LocallyCompactSpace F
    s : Set F
    hs : MeasurableSet s
    this✝² : FiniteDimensional 𝕜 E
    this✝¹ : FiniteDimensional 𝕜 F
    this✝ : ProperSpace E
    this : ProperSpace F
    M : LinearMap (RingHom.id 𝕜) (Prod F E) F := LinearMap.id.coprod L
    M_cont : Continuous ⇑M
    hM : Function.Surjective ⇑M
    A : ∀ (x : Prod F E), Iff (Membership.mem s (M x)) (Membership.mem (Set.preima …
    ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => Membership.mem  …
  -/
  rw [Measure.ae_prod_mem_iff_ae_ae_mem]
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹³ : NontriviallyNormedField 𝕜
      inst✝¹² : CompleteSpace 𝕜
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : MeasurableSpace E
      inst✝⁹ : BorelSpace E
      inst✝⁸ : NormedSpace 𝕜 E
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : MeasurableSpace F
      inst✝⁵ : BorelSpace F
      inst✝⁴ : NormedSpace 𝕜 F
      L : LinearMap (RingHom.id 𝕜) E F
      μ : MeasureTheory.Measure E
      ν : MeasureTheory.Measure F
      inst✝³ : μ.IsAddHaarMeasure
      inst✝² : ν.IsAddHaarMeasure
      inst✝¹ : LocallyCompactSpace E
      inst✝ : LocallyCompactSpace F
      s : Set F
      hs : MeasurableSet s
      this✝² : FiniteDimensional 𝕜 E
      this✝¹ : FiniteDimensional 𝕜 F
      this✝ : ProperSpace E
      this : ProperSpace F
      M : LinearMap (RingHom.id 𝕜) (Prod F E) F := LinearMap.id.coprod L
      M_cont : Continuous ⇑M
      hM : Function.Surjective ⇑M
      A : ∀ (x : Prod F E), Iff (Membership.mem s (M x)) (Membership.mem (Set.preima …
      ⊢ Iff (Filter.Eventually (fun y => Filter.Eventually (fun x => Membership.mem  …
    -/
  · simp only [M, mem_preimage, LinearMap.coprod_apply, LinearMap.id_coe, id_eq]
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝¹³ : NontriviallyNormedField 𝕜
      inst✝¹² : CompleteSpace 𝕜
      inst✝¹¹ : NormedAddCommGroup E
      inst✝¹⁰ : MeasurableSpace E
      inst✝⁹ : BorelSpace E
      inst✝⁸ : NormedSpace 𝕜 E
      inst✝⁷ : NormedAddCommGroup F
      inst✝⁶ : MeasurableSpace F
      inst✝⁵ : BorelSpace F
      inst✝⁴ : NormedSpace 𝕜 F
      L : LinearMap (RingHom.id 𝕜) E F
      μ : MeasureTheory.Measure E
      ν : MeasureTheory.Measure F
      inst✝³ : μ.IsAddHaarMeasure
      inst✝² : ν.IsAddHaarMeasure
      inst✝¹ : LocallyCompactSpace E
      inst✝ : LocallyCompactSpace F
      s : Set F
      hs : MeasurableSet s
      this✝² : FiniteDimensional 𝕜 E
      this✝¹ : FiniteDimensional 𝕜 F
      this✝ : ProperSpace E
      this : ProperSpace F
      M : LinearMap (RingHom.id 𝕜) (Prod F E) F := LinearMap.id.coprod L
      M_cont : Continuous ⇑M
      hM : Function.Surjective ⇑M
      A : ∀ (x : Prod F E), Iff (Membership.mem s (M x)) (Membership.mem (Set.preima …
      ⊢ MeasurableSet (Set.preimage (⇑M) s)
    -/
  · exact M_cont.measurable hs
    /-
      🎉 no goals
    -/


/-- To check that a property holds almost everywhere with respect to an additive Haar measure, it
suffices to check it almost everywhere along all translates of a given vector subspace. This is an
instance of a disintegration argument for additive Haar measures. -/
lemma ae_mem_of_ae_add_linearMap_mem [LocallyCompactSpace F] {s : Set F} (hs : MeasurableSet s)
    (h : ∀ y, ∀ᵐ x ∂μ, y + L x ∈ s) : ∀ᵐ y ∂ν, y ∈ s :=
  (ae_ae_add_linearMap_mem_iff L μ ν hs).1 (Filter.Eventually.of_forall h)


