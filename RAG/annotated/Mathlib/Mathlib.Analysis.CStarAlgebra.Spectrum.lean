local notation "σ" => spectrum

local postfix:max "⋆" => star


theorem unitary.spectrum_subset_circle (u : unitary E) :
    spectrum 𝕜 (u : E) ⊆ Metric.sphere 0 1 := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedRing E
    inst✝³ : StarRing E
    inst✝² : CStarRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    inst✝ : CompleteSpace E
    u : Subtype fun x => Membership.mem (unitary E) x
    ⊢ HasSubset.Subset (spectrum 𝕜 ↑u) (Metric.sphere 0 1)
  -/
  nontriviality E
  /-
    𝕜 : Type u_1
    inst✝⁵ : NormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedRing E
    inst✝³ : StarRing E
    inst✝² : CStarRing E
    inst✝¹ : NormedAlgebra 𝕜 E
    inst✝ : CompleteSpace E
    u : Subtype fun x => Membership.mem (unitary E) x
    a✝ : Nontrivial E
    ⊢ HasSubset.Subset (spectrum 𝕜 ↑u) (Metric.sphere 0 1)
  -/
  refine fun k hk => mem_sphere_zero_iff_norm.mpr (le_antisymm ?_ ?_)
    /-
      case refine_1
      𝕜 : Type u_1
      inst✝⁵ : NormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedRing E
      inst✝³ : StarRing E
      inst✝² : CStarRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      inst✝ : CompleteSpace E
      u : Subtype fun x => Membership.mem (unitary E) x
      a✝ : Nontrivial E
      k : 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑u) k
      ⊢ LE.le (Norm.norm k) 1
    -/
  · simpa only [CStarRing.norm_coe_unitary u] using norm_le_norm_of_mem hk
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁵ : NormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedRing E
      inst✝³ : StarRing E
      inst✝² : CStarRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      inst✝ : CompleteSpace E
      u : Subtype fun x => Membership.mem (unitary E) x
      a✝ : Nontrivial E
      k : 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑u) k
      ⊢ LE.le 1 (Norm.norm k)
    -/
  · rw [← unitary.val_toUnits_apply u] at hk
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁵ : NormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedRing E
      inst✝³ : StarRing E
      inst✝² : CStarRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      inst✝ : CompleteSpace E
      u : Subtype fun x => Membership.mem (unitary E) x
      a✝ : Nontrivial E
      k : 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑(unitary.toUnits u)) k
      ⊢ LE.le 1 (Norm.norm k)
    -/
    have hnk := ne_zero_of_mem_of_unit hk
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁵ : NormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedRing E
      inst✝³ : StarRing E
      inst✝² : CStarRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      inst✝ : CompleteSpace E
      u : Subtype fun x => Membership.mem (unitary E) x
      a✝ : Nontrivial E
      k : 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑(unitary.toUnits u)) k
      hnk : Ne k 0
      ⊢ LE.le 1 (Norm.norm k)
    -/
    rw [← inv_inv (unitary.toUnits u), ← spectrum.map_inv, Set.mem_inv] at hk
    have : ‖k‖⁻¹ ≤ ‖(↑(unitary.toUnits u)⁻¹ : E)‖ := by
      simpa only [norm_inv] using norm_le_norm_of_mem hk
    /-
      case refine_2
      𝕜 : Type u_1
      inst✝⁵ : NormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedRing E
      inst✝³ : StarRing E
      inst✝² : CStarRing E
      inst✝¹ : NormedAlgebra 𝕜 E
      inst✝ : CompleteSpace E
      u : Subtype fun x => Membership.mem (unitary E) x
      a✝ : Nontrivial E
      k : 𝕜
      hk : Membership.mem (spectrum 𝕜 ↑(Inv.inv (unitary.toUnits u))) (Inv.inv k)
      hnk : Ne k 0
      this : LE.le (Inv.inv (Norm.norm k)) (Norm.norm ↑(Inv.inv (unitary.toUnits u)))
      ⊢ LE.le 1 (Norm.norm k)
    -/
    simpa using inv_le_of_inv_le₀ (norm_pos_iff.mpr hnk) this
    /-
      🎉 no goals
    -/


theorem spectrum.subset_circle_of_unitary {u : E} (h : u ∈ unitary E) :
    spectrum 𝕜 u ⊆ Metric.sphere 0 1 :=
  unitary.spectrum_subset_circle ⟨u, h⟩


open scoped NNReal in
lemma CStarAlgebra.le_nnnorm_of_mem_quasispectrum {A : Type*} [NonUnitalCStarAlgebra A]
    {a : A} {x : ℝ≥0} (hx : x ∈ quasispectrum ℝ≥0 a) : x ≤ ‖a‖₊ := by
  /-
    A : Type u_1
    inst✝ : NonUnitalCStarAlgebra A
    a : A
    x : NNReal
    hx : Membership.mem (quasispectrum NNReal a) x
    ⊢ LE.le x (NNNorm.nnnorm a)
  -/
  rw [Unitization.quasispectrum_eq_spectrum_inr' ℝ≥0 ℂ] at hx
  /-
    A : Type u_1
    inst✝ : NonUnitalCStarAlgebra A
    a : A
    x : NNReal
    hx : Membership.mem (spectrum NNReal ↑a) x
    ⊢ LE.le x (NNNorm.nnnorm a)
  -/
  simpa [Unitization.nnnorm_inr] using spectrum.le_nnnorm_of_mem hx
  /-
    🎉 no goals
  -/


local notation "↑ₐ" => algebraMap ℂ A


theorem IsSelfAdjoint.spectralRadius_eq_nnnorm {a : A} (ha : IsSelfAdjoint a) :
    spectralRadius ℂ a = ‖a‖₊ := by
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ⊢ Eq (spectralRadius Complex a) ↑(NNNorm.nnnorm a)
  -/
  have hconst : Tendsto (fun _n : ℕ => (‖a‖₊ : ℝ≥0∞)) atTop _ := tendsto_const_nhds
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    hconst : Filter.Tendsto (fun _n => ↑(NNNorm.nnnorm a)) Filter.atTop (nhds ↑(NN …
    ⊢ Eq (spectralRadius Complex a) ↑(NNNorm.nnnorm a)
  -/
  refine tendsto_nhds_unique ?_ hconst
  convert
    (spectrum.pow_nnnorm_pow_one_div_tendsto_nhds_spectralRadius (a : A)).comp
      (Nat.tendsto_pow_atTop_atTop_of_one_lt one_lt_two) using 1
  /-
    case h.e'_3
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    hconst : Filter.Tendsto (fun _n => ↑(NNNorm.nnnorm a)) Filter.atTop (nhds ↑(NN …
    ⊢ Eq (fun _n => ↑(NNNorm.nnnorm a)) (Function.comp (fun n => HPow.hPow (↑(NNNo …
  -/
  refine funext fun n => ?_
  /-
    case h.e'_3
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    hconst : Filter.Tendsto (fun _n => ↑(NNNorm.nnnorm a)) Filter.atTop (nhds ↑(NN …
    n : Nat
    ⊢ Eq (↑(NNNorm.nnnorm a)) (Function.comp (fun n => HPow.hPow (↑(NNNorm.nnnorm  …
  -/
  rw [Function.comp_apply, ha.nnnorm_pow_two_pow, ENNReal.coe_pow, ← rpow_natCast, ← rpow_mul]
  /-
    case h.e'_3
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    hconst : Filter.Tendsto (fun _n => ↑(NNNorm.nnnorm a)) Filter.atTop (nhds ↑(NN …
    n : Nat
    ⊢ Eq (↑(NNNorm.nnnorm a)) (HPow.hPow (↑(NNNorm.nnnorm a)) (HMul.hMul (↑(HPow.h …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- In a C⋆-algebra, the spectral radius of a self-adjoint element is equal to its norm.
See `IsSelfAdjoint.toReal_spectralRadius_eq_norm` for a version involving
`spectralRadius ℝ a`. -/
lemma IsSelfAdjoint.toReal_spectralRadius_complex_eq_norm {a : A} (ha : IsSelfAdjoint a) :
    (spectralRadius ℂ a).toReal = ‖a‖ := by
  /-
    A : Type u_1
    inst✝ : CStarAlgebra A
    a : A
    ha : IsSelfAdjoint a
    ⊢ Eq (spectralRadius Complex a).toReal (Norm.norm a)
  -/
  simp [ha.spectralRadius_eq_nnnorm]
  /-
    🎉 no goals
  -/


theorem IsStarNormal.spectralRadius_eq_nnnorm (a : A) [IsStarNormal a] :
    spectralRadius ℂ a = ‖a‖₊ := by
  /-
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    a : A
    inst✝ : IsStarNormal a
    ⊢ Eq (spectralRadius Complex a) ↑(NNNorm.nnnorm a)
  -/
  refine (ENNReal.pow_right_strictMono two_ne_zero).injective ?_
  have heq :
    (fun n : ℕ => (‖(a⋆ * a) ^ n‖₊ : ℝ≥0∞) ^ (1 / n : ℝ)) =
      (fun x => x ^ 2) ∘ fun n : ℕ => (‖a ^ n‖₊ : ℝ≥0∞) ^ (1 / n : ℝ) := by
    funext n
    rw [Function.comp_apply, ← rpow_natCast, ← rpow_mul, mul_comm, rpow_mul, rpow_natCast, ←
      coe_pow, sq, ← nnnorm_star_mul_self, Commute.mul_pow (star_comm_self' a), star_pow]
  have h₂ :=
    ((ENNReal.continuous_pow 2).tendsto (spectralRadius ℂ a)).comp
      (spectrum.pow_nnnorm_pow_one_div_tendsto_nhds_spectralRadius a)
  /-
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    a : A
    inst✝ : IsStarNormal a
    heq : Eq (fun n => HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow (HMul.hMul (Star.star …
    h₂ : Filter.Tendsto (Function.comp (fun a => HPow.hPow a 2) fun n => HPow.hPow …
    ⊢ Eq (HPow.hPow (spectralRadius Complex a) 2) (HPow.hPow (↑(NNNorm.nnnorm a)) 2)
  -/
  rw [← heq] at h₂
  /-
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    a : A
    inst✝ : IsStarNormal a
    heq : Eq (fun n => HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow (HMul.hMul (Star.star …
    h₂ : Filter.Tendsto (fun n => HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow (HMul.hMul …
    ⊢ Eq (HPow.hPow (spectralRadius Complex a) 2) (HPow.hPow (↑(NNNorm.nnnorm a)) 2)
  -/
  convert tendsto_nhds_unique h₂ (pow_nnnorm_pow_one_div_tendsto_nhds_spectralRadius (a⋆ * a))
  /-
    case h.e'_3
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    a : A
    inst✝ : IsStarNormal a
    heq : Eq (fun n => HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow (HMul.hMul (Star.star …
    h₂ : Filter.Tendsto (fun n => HPow.hPow (↑(NNNorm.nnnorm (HPow.hPow (HMul.hMul …
    ⊢ Eq (HPow.hPow (↑(NNNorm.nnnorm a)) 2) (spectralRadius Complex (HMul.hMul (St …
  -/
  rw [(IsSelfAdjoint.star_mul_self a).spectralRadius_eq_nnnorm, sq, nnnorm_star_mul_self, coe_mul]
  /-
    🎉 no goals
  -/


/-- Any element of the spectrum of a selfadjoint is real. -/
theorem IsSelfAdjoint.mem_spectrum_eq_re {a : A} (ha : IsSelfAdjoint a) {z : ℂ}
    (hz : z ∈ spectrum ℂ a) : z = z.re := by
  /-
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    inst✝ : StarModule Complex A
    a : A
    ha : IsSelfAdjoint a
    z : Complex
    hz : Membership.mem (spectrum Complex a) z
    ⊢ Eq z ↑z.re
  -/
  have hu := exp_mem_unitary_of_mem_skewAdjoint ℂ (ha.smul_mem_skewAdjoint conj_I)
  /-
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    inst✝ : StarModule Complex A
    a : A
    ha : IsSelfAdjoint a
    z : Complex
    hz : Membership.mem (spectrum Complex a) z
    hu : Membership.mem (unitary A) (NormedSpace.exp Complex (HSMul.hSMul Complex. …
    ⊢ Eq z ↑z.re
  -/
  let Iu := Units.mk0 I I_ne_zero
  have : NormedSpace.exp ℂ (I • z) ∈ spectrum ℂ (NormedSpace.exp ℂ (I • a)) := by
    simpa only [Units.smul_def, Units.val_mk0] using
      spectrum.exp_mem_exp (Iu • a) (smul_mem_smul_iff.mpr hz)
  exact Complex.ext (ofReal_re _) <| by
    simpa only [← Complex.exp_eq_exp_ℂ, mem_sphere_zero_iff_norm, norm_eq_abs, abs_exp,
      Real.exp_eq_one_iff, smul_eq_mul, I_mul, neg_eq_zero] using
      spectrum.subset_circle_of_unitary hu this


/-- Any element of the spectrum of a selfadjoint is real. -/
theorem selfAdjoint.mem_spectrum_eq_re (a : selfAdjoint A) {z : ℂ}
    (hz : z ∈ spectrum ℂ (a : A)) : z = z.re :=
  a.prop.mem_spectrum_eq_re hz


/-- Any element of the spectrum of a selfadjoint is real. -/
theorem IsSelfAdjoint.im_eq_zero_of_mem_spectrum {a : A} (ha : IsSelfAdjoint a)
    {z : ℂ} (hz : z ∈ spectrum ℂ a) : z.im = 0 := by
  /-
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    inst✝ : StarModule Complex A
    a : A
    ha : IsSelfAdjoint a
    z : Complex
    hz : Membership.mem (spectrum Complex a) z
    ⊢ Eq z.im 0
  -/
  rw [ha.mem_spectrum_eq_re hz, ofReal_im]
  /-
    🎉 no goals
  -/


/-- The spectrum of a selfadjoint is real -/
theorem IsSelfAdjoint.val_re_map_spectrum {a : A} (ha : IsSelfAdjoint a) :
    spectrum ℂ a = ((↑) ∘ re '' spectrum ℂ a : Set ℂ) :=
  le_antisymm (fun z hz => ⟨z, hz, (ha.mem_spectrum_eq_re hz).symm⟩) fun z => by
    /-
      A : Type u_1
      inst✝¹ : CStarAlgebra A
      inst✝ : StarModule Complex A
      a : A
      ha : IsSelfAdjoint a
      z : Complex
      ⊢ Membership.mem (Set.image (Function.comp Complex.ofReal Complex.re) (spectru …
    -/
    rintro ⟨z, hz, rfl⟩
    /-
      case intro.intro
      A : Type u_1
      inst✝¹ : CStarAlgebra A
      inst✝ : StarModule Complex A
      a : A
      ha : IsSelfAdjoint a
      z : Complex
      hz : Membership.mem (spectrum Complex a) z
      ⊢ Membership.mem (spectrum Complex a) (Function.comp Complex.ofReal Complex.re …
    -/
    simpa only [(ha.mem_spectrum_eq_re hz).symm, Function.comp_apply] using hz
    /-
      🎉 no goals
    -/


/-- The spectrum of a selfadjoint is real -/
theorem selfAdjoint.val_re_map_spectrum (a : selfAdjoint A) :
    spectrum ℂ (a : A) = ((↑) ∘ re '' spectrum ℂ (a : A) : Set ℂ) :=
  a.property.val_re_map_spectrum


/-- The complement of the spectrum of a selfadjoint element in a C⋆-algebra is connected. -/
lemma IsSelfAdjoint.isConnected_spectrum_compl {a : A} (ha : IsSelfAdjoint a) :
    IsConnected (σ ℂ a)ᶜ := by
  suffices IsConnected (((σ ℂ a)ᶜ ∩ {z | 0 ≤ z.im}) ∪ (σ ℂ a)ᶜ ∩ {z | z.im ≤ 0}) by
    rw [← Set.inter_union_distrib_left, ← Set.setOf_or] at this
    rw [← Set.inter_univ (σ ℂ a)ᶜ]
    convert this using 2
    exact Eq.symm <| Set.eq_univ_of_forall (fun z ↦ le_total 0 z.im)
  /-
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    inst✝ : StarModule Complex A
    a : A
    ha : IsSelfAdjoint a
    ⊢ IsConnected (Union.union (Inter.inter (HasCompl.compl (spectrum Complex a))  …
  -/
  refine IsConnected.union ?nonempty ?upper ?lower
  case nonempty =>
    have := Filter.NeBot.nonempty_of_mem inferInstance <| Filter.mem_map.mp <|
      Complex.isometry_ofReal.antilipschitz.tendsto_cobounded (spectrum.isBounded a |>.compl)
    exact this.image Complex.ofReal |>.mono <| by simp
  /-
    case upper
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    inst✝ : StarModule Complex A
    a : A
    ha : IsSelfAdjoint a
    ⊢ IsConnected (Inter.inter (HasCompl.compl (spectrum Complex a)) (setOf fun z  …
  -/
  case' upper => apply Complex.isConnected_of_upperHalfPlane ?_ <| Set.inter_subset_right
  /-
    case upper
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    inst✝ : StarModule Complex A
    a : A
    ha : IsSelfAdjoint a
    ⊢ HasSubset.Subset (setOf fun z => LT.lt 0 z.im) (Inter.inter (HasCompl.compl  …
  -/
  case' lower => apply Complex.isConnected_of_lowerHalfPlane ?_ <| Set.inter_subset_right
  all_goals
    refine Set.subset_inter (fun z hz hz' ↦ ?_) (fun _ ↦ by simpa using le_of_lt)
    rw [Set.mem_setOf_eq, ha.im_eq_zero_of_mem_spectrum hz'] at hz
    simp_all


/-- For a unital C⋆-subalgebra `S` of `A` and `x : S`, if `↑x : A` is invertible in `A`, then
`x` is invertible in `S`. -/
lemma coe_isUnit {a : S} : IsUnit (a : A) ↔ IsUnit a := by
  /-
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    inst✝ : StarModule Complex A
    S : StarSubalgebra Complex A
    hS : IsClosed ↑S
    a : Subtype fun x => Membership.mem S x
    ⊢ Iff (IsUnit ↑a) (IsUnit a)
  -/
  refine ⟨fun ha ↦ ?_, IsUnit.map S.subtype⟩
  /-
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    inst✝ : StarModule Complex A
    S : StarSubalgebra Complex A
    hS : IsClosed ↑S
    a : Subtype fun x => Membership.mem S x
    ha : IsUnit ↑a
    ⊢ IsUnit a
  -/
  have ha₁ := ha.star.mul ha
  /-
    A : Type u_1
    inst✝¹ : CStarAlgebra A
    inst✝ : StarModule Complex A
    S : StarSubalgebra Complex A
    hS : IsClosed ↑S
    a : Subtype fun x => Membership.mem S x
    ha : IsUnit ↑a
    ha₁ : IsUnit (HMul.hMul (Star.star ↑a) ↑a)
    ⊢ IsUnit a
  -/
  have ha₂ := ha.mul ha.star
  have spec_eq {x : S} (hx : IsSelfAdjoint x) : spectrum ℂ x = spectrum ℂ (x : A) :=
    Subalgebra.spectrum_eq_of_isPreconnected_compl S _ <|
      (hx.map S.subtype).isConnected_spectrum_compl.isPreconnected
  rw [← StarMemClass.coe_star, ← MulMemClass.coe_mul, ← spectrum.zero_not_mem_iff ℂ, ← spec_eq,
    spectrum.zero_not_mem_iff] at ha₁ ha₂
    /-
      A : Type u_1
      inst✝¹ : CStarAlgebra A
      inst✝ : StarModule Complex A
      S : StarSubalgebra Complex A
      hS : IsClosed ↑S
      a : Subtype fun x => Membership.mem S x
      ha : IsUnit ↑a
      ha₁ : IsUnit (HMul.hMul (Star.star a) a)
      ha₂ : IsUnit (HMul.hMul a (Star.star a))
      spec_eq : ∀ {x : Subtype fun x => Membership.mem S x}, IsSelfAdjoint x → Eq (s …
      ⊢ IsUnit a
    -/
  · have h₁ : ha₁.unit⁻¹ * star a * a = 1 := mul_assoc _ _ a ▸ ha₁.val_inv_mul
    /-
      A : Type u_1
      inst✝¹ : CStarAlgebra A
      inst✝ : StarModule Complex A
      S : StarSubalgebra Complex A
      hS : IsClosed ↑S
      a : Subtype fun x => Membership.mem S x
      ha : IsUnit ↑a
      ha₁ : IsUnit (HMul.hMul (Star.star a) a)
      ha₂ : IsUnit (HMul.hMul a (Star.star a))
      spec_eq : ∀ {x : Subtype fun x => Membership.mem S x}, IsSelfAdjoint x → Eq (s …
      h₁ : Eq (HMul.hMul (HMul.hMul (↑(Inv.inv ha₁.unit)) (Star.star a)) a) 1
      ⊢ IsUnit a
    -/
    have h₂ : a * (star a * ha₂.unit⁻¹) = 1 := (mul_assoc a _ _).symm ▸ ha₂.mul_val_inv
    /-
      A : Type u_1
      inst✝¹ : CStarAlgebra A
      inst✝ : StarModule Complex A
      S : StarSubalgebra Complex A
      hS : IsClosed ↑S
      a : Subtype fun x => Membership.mem S x
      ha : IsUnit ↑a
      ha₁ : IsUnit (HMul.hMul (Star.star a) a)
      ha₂ : IsUnit (HMul.hMul a (Star.star a))
      spec_eq : ∀ {x : Subtype fun x => Membership.mem S x}, IsSelfAdjoint x → Eq (s …
      h₁ : Eq (HMul.hMul (HMul.hMul (↑(Inv.inv ha₁.unit)) (Star.star a)) a) 1
      h₂ : Eq (HMul.hMul a (HMul.hMul (Star.star a) ↑(Inv.inv ha₂.unit))) 1
      ⊢ IsUnit a
    -/
    exact ⟨⟨a, ha₁.unit⁻¹ * star a, left_inv_eq_right_inv h₁ h₂ ▸ h₂, h₁⟩, rfl⟩
    /-
      🎉 no goals
    -/
    /-
      A : Type u_1
      inst✝¹ : CStarAlgebra A
      inst✝ : StarModule Complex A
      S : StarSubalgebra Complex A
      hS : IsClosed ↑S
      a : Subtype fun x => Membership.mem S x
      ha : IsUnit ↑a
      ha₁ : Not (Membership.mem (spectrum Complex (HMul.hMul (Star.star a) a)) 0)
      ha₂ : Not (Membership.mem (spectrum Complex ↑(HMul.hMul a (Star.star a))) 0)
      spec_eq : ∀ {x : Subtype fun x => Membership.mem S x}, IsSelfAdjoint x → Eq (s …
      ⊢ IsSelfAdjoint (HMul.hMul a (Star.star a))
    -/
  · exact IsSelfAdjoint.mul_star_self a
    /-
      🎉 no goals
    -/
    /-
      A : Type u_1
      inst✝¹ : CStarAlgebra A
      inst✝ : StarModule Complex A
      S : StarSubalgebra Complex A
      hS : IsClosed ↑S
      a : Subtype fun x => Membership.mem S x
      ha : IsUnit ↑a
      ha₁ : Not (Membership.mem (spectrum Complex ↑(HMul.hMul (Star.star a) a)) 0)
      ha₂ : Not (Membership.mem (spectrum Complex ↑(HMul.hMul a (Star.star a))) 0)
      spec_eq : ∀ {x : Subtype fun x => Membership.mem S x}, IsSelfAdjoint x → Eq (s …
      ⊢ IsSelfAdjoint (HMul.hMul (Star.star a) a)
    -/
  · exact IsSelfAdjoint.star_mul_self a
    /-
      🎉 no goals
    -/


lemma mem_spectrum_iff {a : S} {z : ℂ} : z ∈ spectrum ℂ a ↔ z ∈ spectrum ℂ (a : A) :=
  not_iff_not.mpr S.coe_isUnit.symm


/-- **Spectral permanence.** The spectrum of an element is invariant of the (closed)
`StarSubalgebra` in which it is contained. -/
lemma spectrum_eq {a : S} : spectrum ℂ a = spectrum ℂ (a : A) :=
  Set.ext fun _ ↦ S.mem_spectrum_iff


/-- A non-unital star algebra homomorphism of complex C⋆-algebras is norm contractive. -/
lemma nnnorm_apply_le (φ : F) (a : A) : ‖φ a‖₊ ≤ ‖a‖₊ := by
  have h (ψ : Unitization ℂ A →⋆ₐ[ℂ] Unitization ℂ B) (x : Unitization ℂ A) :
      ‖ψ x‖₊ ≤ ‖x‖₊ := by
    suffices ∀ {s}, IsSelfAdjoint s → ‖ψ s‖₊ ≤ ‖s‖₊ by
      refine nonneg_le_nonneg_of_sq_le_sq zero_le' ?_
      simp_rw [← nnnorm_star_mul_self, ← map_star, ← map_mul]
      exact this <| .star_mul_self x
    intro s hs
    suffices this : spectralRadius ℂ (ψ s) ≤ spectralRadius ℂ s by
      rwa [(hs.map ψ).spectralRadius_eq_nnnorm, hs.spectralRadius_eq_nnnorm, coe_le_coe]
        at this
    exact iSup_le_iSup_of_subset (AlgHom.spectrum_apply_subset ψ s)
  /-
    F : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁴ : NonUnitalCStarAlgebra A
    inst✝³ : NonUnitalCStarAlgebra B
    inst✝² : FunLike F A B
    inst✝¹ : NonUnitalAlgHomClass F Complex A B
    inst✝ : StarHomClass F A B
    φ : F
    a : A
    h : ∀ (ψ : StarAlgHom Complex (Unitization Complex A) (Unitization Complex B)) …
    ⊢ LE.le (NNNorm.nnnorm (φ a)) (NNNorm.nnnorm a)
  -/
  simpa [nnnorm_inr] using h (starLift (inrNonUnitalStarAlgHom ℂ B |>.comp (φ : A →⋆ₙₐ[ℂ] B))) a
  /-
    🎉 no goals
  -/


/-- A non-unital star algebra homomorphism of complex C⋆-algebras is norm contractive. -/
lemma norm_apply_le (φ : F) (a : A) : ‖φ a‖ ≤ ‖a‖ := by
  /-
    F : Type u_1
    A : Type u_2
    B : Type u_3
    inst✝⁴ : NonUnitalCStarAlgebra A
    inst✝³ : NonUnitalCStarAlgebra B
    inst✝² : FunLike F A B
    inst✝¹ : NonUnitalAlgHomClass F Complex A B
    inst✝ : StarHomClass F A B
    φ : F
    a : A
    ⊢ LE.le (Norm.norm (φ a)) (Norm.norm a)
  -/
  exact_mod_cast nnnorm_apply_le φ a
  /-
    🎉 no goals
  -/


/-- Non-unital star algebra homomorphisms between C⋆-algebras are continuous linear maps.
See note [lower instance priority] -/
lemma instContinuousLinearMapClassComplex : ContinuousLinearMapClass F ℂ A B :=
  { NonUnitalAlgHomClass.instLinearMapClass with
    map_continuous := fun φ =>
                                                    /-
                                                      F : Type u_1
                                                      A : Type u_2
                                                      B : Type u_3
                                                      inst✝⁴ : NonUnitalCStarAlgebra A
                                                      inst✝³ : NonUnitalCStarAlgebra B
                                                      inst✝² : FunLike F A B
                                                      inst✝¹ : NonUnitalAlgHomClass F Complex A B
                                                      inst✝ : StarHomClass F A B
                                                      φ : F
                                                      ⊢ ∀ (x : A), LE.le (Norm.norm (φ x)) (HMul.hMul 1 (Norm.norm x))
                                                    -/
      AddMonoidHomClass.continuous_of_bound φ 1 (by simpa only [one_mul] using nnnorm_apply_le φ) }
                                                    /-
                                                      🎉 no goals
                                                    -/


lemma nnnorm_map (φ : F) (a : A) : ‖φ a‖₊ = ‖a‖₊ :=
  le_antisymm (NonUnitalStarAlgHom.nnnorm_apply_le φ a) <| by
    /-
      F : Type u_1
      A : Type u_2
      B : Type u_3
      inst✝⁴ : NonUnitalCStarAlgebra A
      inst✝³ : NonUnitalCStarAlgebra B
      inst✝² : EquivLike F A B
      inst✝¹ : NonUnitalAlgEquivClass F Complex A B
      inst✝ : StarHomClass F A B
      φ : F
      a : A
      ⊢ LE.le (NNNorm.nnnorm a) (NNNorm.nnnorm (φ a))
    -/
    simpa using NonUnitalStarAlgHom.nnnorm_apply_le (symm (φ : A ≃⋆ₐ[ℂ] B)) ((φ : A ≃⋆ₐ[ℂ] B) a)
    /-
      🎉 no goals
    -/


lemma norm_map (φ : F) (a : A) : ‖φ a‖ = ‖a‖ :=
  congr_arg NNReal.toReal (nnnorm_map φ a)


lemma isometry (φ : F) : Isometry φ :=
  AddMonoidHomClass.isometry_of_norm φ (norm_map φ)


/-- This instance is provided instead of `StarHomClass` to avoid type class inference loops.
See note [lower instance priority] -/
noncomputable instance (priority := 100) Complex.instStarHomClass : StarHomClass F A ℂ where
  map_star φ a := by
    suffices hsa : ∀ s : selfAdjoint A, (φ s)⋆ = φ s by
      rw [← realPart_add_I_smul_imaginaryPart a]
      simp only [map_add, map_smul, star_add, star_smul, hsa, selfAdjoint.star_val_eq]
    /-
      F : Type u_1
      A : Type u_2
      inst✝¹ : CStarAlgebra A
      inst✝ : FunLike F A Complex
      hF : AlgHomClass F Complex A Complex
      φ : F
      a : A
      ⊢ ∀ (s : Subtype fun x => Membership.mem (selfAdjoint A) x), Eq (Star.star (φ  …
    -/
    intro s
    /-
      F : Type u_1
      A : Type u_2
      inst✝¹ : CStarAlgebra A
      inst✝ : FunLike F A Complex
      hF : AlgHomClass F Complex A Complex
      φ : F
      a : A
      s : Subtype fun x => Membership.mem (selfAdjoint A) x
      ⊢ Eq (Star.star (φ ↑s)) (φ ↑s)
    -/
    have := AlgHom.apply_mem_spectrum φ (s : A)
    /-
      F : Type u_1
      A : Type u_2
      inst✝¹ : CStarAlgebra A
      inst✝ : FunLike F A Complex
      hF : AlgHomClass F Complex A Complex
      φ : F
      a : A
      s : Subtype fun x => Membership.mem (selfAdjoint A) x
      this : Membership.mem (spectrum Complex ↑s) (φ ↑s)
      ⊢ Eq (Star.star (φ ↑s)) (φ ↑s)
    -/
    rw [selfAdjoint.val_re_map_spectrum s] at this
    /-
      F : Type u_1
      A : Type u_2
      inst✝¹ : CStarAlgebra A
      inst✝ : FunLike F A Complex
      hF : AlgHomClass F Complex A Complex
      φ : F
      a : A
      s : Subtype fun x => Membership.mem (selfAdjoint A) x
      this : Membership.mem (Set.image (Function.comp Complex.ofReal Complex.re) (sp …
      ⊢ Eq (Star.star (φ ↑s)) (φ ↑s)
    -/
    rcases this with ⟨⟨_, _⟩, _, heq⟩
    /-
      case intro.mk.intro
      F : Type u_1
      A : Type u_2
      inst✝¹ : CStarAlgebra A
      inst✝ : FunLike F A Complex
      hF : AlgHomClass F Complex A Complex
      φ : F
      a : A
      s : Subtype fun x => Membership.mem (selfAdjoint A) x
      re✝ im✝ : Real
      left✝ : Membership.mem (spectrum Complex ↑s) { re := re✝, im := im✝ }
      heq : Eq (Function.comp Complex.ofReal Complex.re { re := re✝, im := im✝ }) (φ …
      ⊢ Eq (Star.star (φ ↑s)) (φ ↑s)
    -/
    simp only [Function.comp_apply] at heq
    /-
      case intro.mk.intro
      F : Type u_1
      A : Type u_2
      inst✝¹ : CStarAlgebra A
      inst✝ : FunLike F A Complex
      hF : AlgHomClass F Complex A Complex
      φ : F
      a : A
      s : Subtype fun x => Membership.mem (selfAdjoint A) x
      re✝ im✝ : Real
      left✝ : Membership.mem (spectrum Complex ↑s) { re := re✝, im := im✝ }
      heq : Eq (↑re✝) (φ ↑s)
      ⊢ Eq (Star.star (φ ↑s)) (φ ↑s)
    -/
    rw [← heq, RCLike.star_def]
    /-
      case intro.mk.intro
      F : Type u_1
      A : Type u_2
      inst✝¹ : CStarAlgebra A
      inst✝ : FunLike F A Complex
      hF : AlgHomClass F Complex A Complex
      φ : F
      a : A
      s : Subtype fun x => Membership.mem (selfAdjoint A) x
      re✝ im✝ : Real
      left✝ : Membership.mem (spectrum Complex ↑s) { re := re✝, im := im✝ }
      heq : Eq (↑re✝) (φ ↑s)
      ⊢ Eq ((starRingEnd Complex) ↑re✝) ↑re✝
    -/
    exact RCLike.conj_ofReal _
    /-
      🎉 no goals
    -/


/-- This is not an instance to avoid type class inference loops. See
`WeakDual.Complex.instStarHomClass`. -/
lemma _root_.AlgHomClass.instStarHomClass : StarHomClass F A ℂ :=
  { WeakDual.Complex.instStarHomClass, hF with }


noncomputable instance instStarHomClass : StarHomClass (characterSpace ℂ A) A ℂ :=
  { AlgHomClass.instStarHomClass with }


