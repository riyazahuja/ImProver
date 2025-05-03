/-- The canonical embedding of a number field `K` of degree `n` into `ℂ^n`. -/
def _root_.NumberField.canonicalEmbedding : K →+* ((K →+* ℂ) → ℂ) := Pi.ringHom fun φ => φ


theorem _root_.NumberField.canonicalEmbedding_injective [NumberField K] :
    Function.Injective (NumberField.canonicalEmbedding K) := RingHom.injective _


@[simp]
theorem apply_at (φ : K →+* ℂ) (x : K) : (NumberField.canonicalEmbedding K x) φ = φ x := rfl


/-- The image of `canonicalEmbedding` lives in the `ℝ`-submodule of the `x ∈ ((K →+* ℂ) → ℂ)` such
that `conj x_φ = x_(conj φ)` for all `∀ φ : K →+* ℂ`. -/
theorem conj_apply {x : ((K →+* ℂ) → ℂ)} (φ : K →+* ℂ)
    (hx : x ∈ Submodule.span ℝ (Set.range (canonicalEmbedding K))) :
    conj (x φ) = x (ComplexEmbedding.conjugate φ) := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : RingHom K Complex → Complex
    φ : RingHom K Complex
    hx : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEmb …
    ⊢ Eq ((starRingEnd Complex) (x φ)) (x (NumberField.ComplexEmbedding.conjugate  …
  -/
  refine Submodule.span_induction ?_ ?_ (fun _ _ _ _ hx hy => ?_) (fun a _ _ hx => ?_) hx
    /-
      case refine_1
      K : Type u_1
      inst✝ : Field K
      x : RingHom K Complex → Complex
      φ : RingHom K Complex
      hx : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEmb …
      ⊢ ∀ (x : RingHom K Complex → Complex), Membership.mem (Set.range ⇑(NumberField …
    -/
  · rintro _ ⟨x, rfl⟩
    /-
      case refine_1.intro
      K : Type u_1
      inst✝ : Field K
      x✝ : RingHom K Complex → Complex
      φ : RingHom K Complex
      hx : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEmb …
      x : K
      ⊢ Eq ((starRingEnd Complex) ((NumberField.canonicalEmbedding K) x φ)) ((Number …
    -/
    rw [apply_at, apply_at, ComplexEmbedding.conjugate_coe_eq]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      K : Type u_1
      inst✝ : Field K
      x : RingHom K Complex → Complex
      φ : RingHom K Complex
      hx : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEmb …
      ⊢ Eq ((starRingEnd Complex) (0 φ)) (0 (NumberField.ComplexEmbedding.conjugate  …
    -/
  · rw [Pi.zero_apply, Pi.zero_apply, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      K : Type u_1
      inst✝ : Field K
      x : RingHom K Complex → Complex
      φ : RingHom K Complex
      hx✝ : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEm …
      x✝³ x✝² : RingHom K Complex → Complex
      x✝¹ : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEm …
      x✝ : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEmb …
      hx : Eq ((starRingEnd Complex) (x✝³ φ)) (x✝³ (NumberField.ComplexEmbedding.con …
      hy : Eq ((starRingEnd Complex) (x✝² φ)) (x✝² (NumberField.ComplexEmbedding.con …
      ⊢ Eq ((starRingEnd Complex) (HAdd.hAdd x✝³ x✝² φ)) (HAdd.hAdd x✝³ x✝² (NumberF …
    -/
  · rw [Pi.add_apply, Pi.add_apply, map_add, hx, hy]
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      K : Type u_1
      inst✝ : Field K
      x : RingHom K Complex → Complex
      φ : RingHom K Complex
      hx✝ : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEm …
      a : Real
      x✝¹ : RingHom K Complex → Complex
      x✝ : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEmb …
      hx : Eq ((starRingEnd Complex) (x✝¹ φ)) (x✝¹ (NumberField.ComplexEmbedding.con …
      ⊢ Eq ((starRingEnd Complex) (HSMul.hSMul a x✝¹ φ)) (HSMul.hSMul a x✝¹ (NumberF …
    -/
  · rw [Pi.smul_apply, Complex.real_smul, map_mul, Complex.conj_ofReal]
    /-
      case refine_4
      K : Type u_1
      inst✝ : Field K
      x : RingHom K Complex → Complex
      φ : RingHom K Complex
      hx✝ : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEm …
      a : Real
      x✝¹ : RingHom K Complex → Complex
      x✝ : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonicalEmb …
      hx : Eq ((starRingEnd Complex) (x✝¹ φ)) (x✝¹ (NumberField.ComplexEmbedding.con …
      ⊢ Eq (HMul.hMul (↑a) ((starRingEnd Complex) (x✝¹ φ))) (HSMul.hSMul a x✝¹ (Numb …
    -/
    exact congrArg ((a : ℂ) * ·) hx
    /-
      🎉 no goals
    -/


theorem nnnorm_eq [NumberField K] (x : K) :
    ‖canonicalEmbedding K x‖₊ = Finset.univ.sup (fun φ : K →+* ℂ => ‖φ x‖₊) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    ⊢ Eq (NNNorm.nnnorm ((NumberField.canonicalEmbedding K) x)) (Finset.univ.sup f …
  -/
  simp_rw [Pi.nnnorm_def, apply_at]
  /-
    🎉 no goals
  -/


theorem norm_le_iff [NumberField K] (x : K) (r : ℝ) :
    ‖canonicalEmbedding K x‖ ≤ r ↔ ∀ φ : K →+* ℂ, ‖φ x‖ ≤ r := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    r : Real
    ⊢ Iff (LE.le (Norm.norm ((NumberField.canonicalEmbedding K) x)) r) (∀ (φ : Rin …
  -/
  obtain hr | hr := lt_or_le r 0
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : K
      r : Real
      hr : LT.lt r 0
      ⊢ Iff (LE.le (Norm.norm ((NumberField.canonicalEmbedding K) x)) r) (∀ (φ : Rin …
    -/
  · obtain ⟨φ⟩ := (inferInstance : Nonempty (K →+* ℂ))
    /-
      case inl.intro
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : K
      r : Real
      hr : LT.lt r 0
      φ : RingHom K Complex
      ⊢ Iff (LE.le (Norm.norm ((NumberField.canonicalEmbedding K) x)) r) (∀ (φ : Rin …
    -/
    refine iff_of_false ?_ ?_
      /-
        case inl.intro.refine_1
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x : K
        r : Real
        hr : LT.lt r 0
        φ : RingHom K Complex
        ⊢ Not (LE.le (Norm.norm ((NumberField.canonicalEmbedding K) x)) r)
      -/
    · exact (hr.trans_le (norm_nonneg _)).not_le
      /-
        🎉 no goals
      -/
      /-
        case inl.intro.refine_2
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x : K
        r : Real
        hr : LT.lt r 0
        φ : RingHom K Complex
        ⊢ Not (∀ (φ : RingHom K Complex), LE.le (Norm.norm (φ x)) r)
      -/
    · exact fun h => hr.not_le (le_trans (norm_nonneg _) (h φ))
      /-
        🎉 no goals
      -/
    /-
      case inr
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : K
      r : Real
      hr : LE.le 0 r
      ⊢ Iff (LE.le (Norm.norm ((NumberField.canonicalEmbedding K) x)) r) (∀ (φ : Rin …
    -/
  · lift r to NNReal using hr
    simp_rw [← coe_nnnorm, nnnorm_eq, NNReal.coe_le_coe, Finset.sup_le_iff, Finset.mem_univ,
      forall_true_left]


/-- The image of `𝓞 K` as a subring of `ℂ^n`. -/
def integerLattice : Subring ((K →+* ℂ) → ℂ) :=
  (RingHom.range (algebraMap (𝓞 K) K)).map (canonicalEmbedding K)


theorem integerLattice.inter_ball_finite [NumberField K] (r : ℝ) :
    ((integerLattice K : Set ((K →+* ℂ) → ℂ)) ∩ Metric.closedBall 0 r).Finite := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    r : Real
    ⊢ (Inter.inter (↑(NumberField.canonicalEmbedding.integerLattice K)) (Metric.cl …
  -/
  obtain hr | _ := lt_or_le r 0
    /-
      case inl
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      hr : LT.lt r 0
      ⊢ (Inter.inter (↑(NumberField.canonicalEmbedding.integerLattice K)) (Metric.cl …
    -/
  · simp [Metric.closedBall_eq_empty.2 hr]
    /-
      🎉 no goals
    -/
  · have heq : ∀ x, canonicalEmbedding K x ∈ Metric.closedBall 0 r ↔
        ∀ φ : K →+* ℂ, ‖φ x‖ ≤ r := by
      intro x; rw [← norm_le_iff, mem_closedBall_zero_iff]
    /-
      case inr
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      h✝ : LE.le 0 r
      heq : ∀ (x : K), Iff (Membership.mem (Metric.closedBall 0 r) ((NumberField.can …
      ⊢ (Inter.inter (↑(NumberField.canonicalEmbedding.integerLattice K)) (Metric.cl …
    -/
    convert (Embeddings.finite_of_norm_le K ℂ r).image (canonicalEmbedding K)
    /-
      case h.e'_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      r : Real
      h✝ : LE.le 0 r
      heq : ∀ (x : K), Iff (Membership.mem (Metric.closedBall 0 r) ((NumberField.can …
      ⊢ Eq (Inter.inter (↑(NumberField.canonicalEmbedding.integerLattice K)) (Metric …
    -/
    ext; constructor
      /-
        case h.e'_2.h.mp
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        h✝ : LE.le 0 r
        heq : ∀ (x : K), Iff (Membership.mem (Metric.closedBall 0 r) ((NumberField.can …
        x✝ : RingHom K Complex → Complex
        ⊢ Membership.mem (Inter.inter (↑(NumberField.canonicalEmbedding.integerLattice …
      -/
    · rintro ⟨⟨_, ⟨x, rfl⟩, rfl⟩, hx⟩
      /-
        case h.e'_2.h.mp.intro.intro.intro.intro
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        h✝ : LE.le 0 r
        heq : ∀ (x : K), Iff (Membership.mem (Metric.closedBall 0 r) ((NumberField.can …
        x : NumberField.RingOfIntegers K
        hx : Membership.mem (Metric.closedBall 0 r) ((NumberField.canonicalEmbedding K …
        ⊢ Membership.mem (Set.image (⇑(NumberField.canonicalEmbedding K)) (setOf fun x …
      -/
      exact ⟨x, ⟨SetLike.coe_mem x, fun φ => (heq _).mp hx φ⟩, rfl⟩
      /-
        🎉 no goals
      -/
      /-
        case h.e'_2.h.mpr
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        h✝ : LE.le 0 r
        heq : ∀ (x : K), Iff (Membership.mem (Metric.closedBall 0 r) ((NumberField.can …
        x✝ : RingHom K Complex → Complex
        ⊢ Membership.mem (Set.image (⇑(NumberField.canonicalEmbedding K)) (setOf fun x …
      -/
    · rintro ⟨x, ⟨hx1, hx2⟩, rfl⟩
      /-
        case h.e'_2.h.mpr.intro.intro.intro
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        r : Real
        h✝ : LE.le 0 r
        heq : ∀ (x : K), Iff (Membership.mem (Metric.closedBall 0 r) ((NumberField.can …
        x : K
        hx1 : IsIntegral Int x
        hx2 : ∀ (φ : RingHom K Complex), LE.le (Norm.norm (φ x)) r
        ⊢ Membership.mem (Inter.inter (↑(NumberField.canonicalEmbedding.integerLattice …
      -/
      exact ⟨⟨x, ⟨⟨x, hx1⟩, rfl⟩, rfl⟩, (heq x).mpr hx2⟩
      /-
        🎉 no goals
      -/


/-- A `ℂ`-basis of `ℂ^n` that is also a `ℤ`-basis of the `integerLattice`. -/
noncomputable def latticeBasis [NumberField K] :
    Basis (Free.ChooseBasisIndex ℤ (𝓞 K)) ℂ ((K →+* ℂ) → ℂ) := by
  classical
  -- Let `B` be the canonical basis of `(K →+* ℂ) → ℂ`. We prove that the determinant of
  -- the image by `canonicalEmbedding` of the integral basis of `K` is nonzero. This
  -- will imply the result.
    let B := Pi.basisFun ℂ (K →+* ℂ)
    let e : (K →+* ℂ) ≃ Free.ChooseBasisIndex ℤ (𝓞 K) :=
      equivOfCardEq ((Embeddings.card K ℂ).trans (finrank_eq_card_basis (integralBasis K)))
    let M := B.toMatrix (fun i => canonicalEmbedding K (integralBasis K (e i)))
    suffices M.det ≠ 0 by
      rw [← isUnit_iff_ne_zero, ← Basis.det_apply, ← is_basis_iff_det] at this
      refine basisOfLinearIndependentOfCardEqFinrank
        ((linearIndependent_equiv e.symm).mpr this.1) ?_
      rw [← finrank_eq_card_chooseBasisIndex, RingOfIntegers.rank, finrank_fintype_fun_eq_card,
        Embeddings.card]
  -- In order to prove that the determinant is nonzero, we show that it is equal to the
  -- square of the discriminant of the integral basis and thus it is not zero
    let N := Algebra.embeddingsMatrixReindex ℚ ℂ (fun i => integralBasis K (e i))
      RingHom.equivRatAlgHom
    rw [show M = N.transpose by { ext : 2; rfl }]
    rw [Matrix.det_transpose, ← pow_ne_zero_iff two_ne_zero]
    convert (map_ne_zero_iff _ (algebraMap ℚ ℂ).injective).mpr
      (Algebra.discr_not_zero_of_basis ℚ (integralBasis K))
    rw [← Algebra.discr_reindex ℚ (integralBasis K) e.symm]
    exact (Algebra.discr_eq_det_embeddingsMatrixReindex_pow_two ℚ ℂ
      (fun i => integralBasis K (e i)) RingHom.equivRatAlgHom).symm


@[simp]
theorem latticeBasis_apply [NumberField K] (i : Free.ChooseBasisIndex ℤ (𝓞 K)) :
    latticeBasis K i = (canonicalEmbedding K) (integralBasis K i) := by
  simp only [latticeBasis, integralBasis_apply, coe_basisOfLinearIndependentOfCardEqFinrank,
    Function.comp_apply, Equiv.apply_symm_apply]


theorem mem_span_latticeBasis [NumberField K] {x : (K →+* ℂ) → ℂ} :
    x ∈ Submodule.span ℤ (Set.range (latticeBasis K)) ↔
      x ∈ ((canonicalEmbedding K).comp (algebraMap (𝓞 K) K)).range := by
  rw [show Set.range (latticeBasis K) =
      (canonicalEmbedding K).toIntAlgHom.toLinearMap '' (Set.range (integralBasis K)) by
    rw [← Set.range_comp]; exact congrArg Set.range (funext (fun i => latticeBasis_apply K i))]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : RingHom K Complex → Complex
    ⊢ Iff (Membership.mem (Submodule.span Int (Set.image (⇑(NumberField.canonicalE …
  -/
  rw [← Submodule.map_span, ← SetLike.mem_coe, Submodule.map_coe]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : RingHom K Complex → Complex
    ⊢ Iff (Membership.mem (Set.image ⇑(NumberField.canonicalEmbedding K).toIntAlgH …
  -/
  rw [← RingHom.map_range, Subring.mem_map, Set.mem_image]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : RingHom K Complex → Complex
    ⊢ Iff (Exists fun x_1 => And (Membership.mem (↑(Submodule.span Int (Set.range  …
  -/
  simp only [SetLike.mem_coe, mem_span_integralBasis K]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : RingHom K Complex → Complex
    ⊢ Iff (Exists fun x_1 => And (Membership.mem (algebraMap (NumberField.RingOfIn …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mem_rat_span_latticeBasis [NumberField K] (x : K) :
    canonicalEmbedding K x ∈ Submodule.span ℚ (Set.range (latticeBasis K)) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    ⊢ Membership.mem (Submodule.span Rat (Set.range ⇑(NumberField.canonicalEmbeddi …
  -/
  rw [← Basis.sum_repr (integralBasis K) x, map_sum]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    ⊢ Membership.mem (Submodule.span Rat (Set.range ⇑(NumberField.canonicalEmbeddi …
  -/
  simp_rw [map_rat_smul]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    ⊢ Membership.mem (Submodule.span Rat (Set.range ⇑(NumberField.canonicalEmbeddi …
  -/
  refine Submodule.sum_smul_mem _ _ (fun i _ ↦ Submodule.subset_span ?_)
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    i : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    x✝ : Membership.mem Finset.univ i
    ⊢ Membership.mem (Set.range ⇑(NumberField.canonicalEmbedding.latticeBasis K))  …
  -/
  rw [← latticeBasis_apply]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    i : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    x✝ : Membership.mem Finset.univ i
    ⊢ Membership.mem (Set.range ⇑(NumberField.canonicalEmbedding.latticeBasis K))  …
  -/
  exact Set.mem_range_self i
  /-
    🎉 no goals
  -/


theorem integralBasis_repr_apply [NumberField K] (x : K) (i : Free.ChooseBasisIndex ℤ (𝓞 K)) :
    (latticeBasis K).repr (canonicalEmbedding K x) i = (integralBasis K).repr x i := by
  rw [← Basis.restrictScalars_repr_apply ℚ _ ⟨_, mem_rat_span_latticeBasis K x⟩, eq_ratCast,
    Rat.cast_inj]
  let f := (canonicalEmbedding K).toRatAlgHom.toLinearMap.codRestrict _
    (fun x ↦ mem_rat_span_latticeBasis K x)
  suffices ((latticeBasis K).restrictScalars ℚ).repr.toLinearMap ∘ₗ f =
    (integralBasis K).repr.toLinearMap from DFunLike.congr_fun (LinearMap.congr_fun this x) i
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    i : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    f : LinearMap (RingHom.id Rat) K (Subtype fun x => Membership.mem (Submodule.s …
    ⊢ Eq ((↑(Basis.restrictScalars Rat (NumberField.canonicalEmbedding.latticeBasi …
  -/
  refine Basis.ext (integralBasis K) (fun i ↦ ?_)
  have : f (integralBasis K i) = ((latticeBasis K).restrictScalars ℚ) i := by
    apply Subtype.val_injective
    rw [LinearMap.codRestrict_apply, AlgHom.toLinearMap_apply, Basis.restrictScalars_apply,
      latticeBasis_apply]
    rfl
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    i✝ : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    f : LinearMap (RingHom.id Rat) K (Subtype fun x => Membership.mem (Submodule.s …
    i : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    this : Eq (f ((NumberField.integralBasis K) i)) ((Basis.restrictScalars Rat (N …
    ⊢ Eq (((↑(Basis.restrictScalars Rat (NumberField.canonicalEmbedding.latticeBas …
  -/
  simp_rw [LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply, this, Basis.repr_self]
  /-
    🎉 no goals
  -/


/-- The mixed space `ℝ^r₁ × ℂ^r₂` with `(r₁, r₂)` the signature of `K`. -/
abbrev mixedSpace :=
  ({w : InfinitePlace K // IsReal w} → ℝ) × ({w : InfinitePlace K // IsComplex w} → ℂ)


/-- The mixed embedding of a number field `K` into the mixed space of `K`. -/
noncomputable def _root_.NumberField.mixedEmbedding : K →+* (mixedSpace K) :=
  RingHom.prod (Pi.ringHom fun w => embedding_of_isReal w.prop)
    (Pi.ringHom fun w => w.val.embedding)


@[simp]
theorem mixedEmbedding_apply_ofIsReal (x : K) (w : {w // IsReal w}) :
    (mixedEmbedding K x).1 w = embedding_of_isReal w.prop x := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : K
    w : Subtype fun w => w.IsReal
    ⊢ Eq (((NumberField.mixedEmbedding K) x).1 w) ((NumberField.InfinitePlace.embe …
  -/
  simp_rw [mixedEmbedding, RingHom.prod_apply, Pi.ringHom_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem mixedEmbedding_apply_ofIsComplex (x : K) (w : {w // IsComplex w}) :
    (mixedEmbedding K x).2 w = w.val.embedding x := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : K
    w : Subtype fun w => w.IsComplex
    ⊢ Eq (((NumberField.mixedEmbedding K) x).2 w) ((↑w).embedding x)
  -/
  simp_rw [mixedEmbedding, RingHom.prod_apply, Pi.ringHom_apply]
  /-
    🎉 no goals
  -/


instance [NumberField K] : Nontrivial (mixedSpace K) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Nontrivial (NumberField.mixedEmbedding.mixedSpace K)
  -/
  obtain ⟨w⟩ := (inferInstance : Nonempty (InfinitePlace K))
  /-
    case intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w : NumberField.InfinitePlace K
    ⊢ Nontrivial (NumberField.mixedEmbedding.mixedSpace K)
  -/
  obtain hw | hw := w.isReal_or_isComplex
    /-
      case intro.inl
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : w.IsReal
      ⊢ Nontrivial (NumberField.mixedEmbedding.mixedSpace K)
    -/
  · have : Nonempty {w : InfinitePlace K // IsReal w} := ⟨⟨w, hw⟩⟩
    /-
      case intro.inl
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : w.IsReal
      this : Nonempty (Subtype fun w => w.IsReal)
      ⊢ Nontrivial (NumberField.mixedEmbedding.mixedSpace K)
    -/
    exact nontrivial_prod_left
    /-
      🎉 no goals
    -/
    /-
      case intro.inr
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : w.IsComplex
      ⊢ Nontrivial (NumberField.mixedEmbedding.mixedSpace K)
    -/
  · have : Nonempty {w : InfinitePlace K // IsComplex w} := ⟨⟨w, hw⟩⟩
    /-
      case intro.inr
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : w.IsComplex
      this : Nonempty (Subtype fun w => w.IsComplex)
      ⊢ Nontrivial (NumberField.mixedEmbedding.mixedSpace K)
    -/
    exact nontrivial_prod_right
    /-
      🎉 no goals
    -/


protected theorem finrank [NumberField K] : finrank ℝ (mixedSpace K) = finrank ℚ K := by
  classical
  rw [finrank_prod, finrank_pi, finrank_pi_fintype, Complex.finrank_real_complex, sum_const,
    card_univ, ← nrRealPlaces, ← nrComplexPlaces, ← card_real_embeddings, Algebra.id.smul_eq_mul,
    mul_comm, ← card_complex_embeddings, ← NumberField.Embeddings.card K ℂ,
    Fintype.card_subtype_compl, Nat.add_sub_of_le (Fintype.card_subtype_le _)]


theorem _root_.NumberField.mixedEmbedding_injective [NumberField K] :
    Function.Injective (NumberField.mixedEmbedding K) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Function.Injective ⇑(NumberField.mixedEmbedding K)
  -/
  exact RingHom.injective _
  /-
    🎉 no goals
  -/


open Classical in
instance : IsAddHaarMeasure (volume : Measure (mixedSpace K)) :=
  prod.instIsAddHaarMeasure volume volume


open Classical in
instance : NoAtoms (volume : Measure (mixedSpace K)) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ MeasureTheory.NoAtoms MeasureTheory.MeasureSpace.volume
  -/
  obtain ⟨w⟩ := (inferInstance : Nonempty (InfinitePlace K))
  /-
    case intro
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w : NumberField.InfinitePlace K
    ⊢ MeasureTheory.NoAtoms MeasureTheory.MeasureSpace.volume
  -/
  by_cases hw : IsReal w
    /-
      case pos
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : w.IsReal
      ⊢ MeasureTheory.NoAtoms MeasureTheory.MeasureSpace.volume
    -/
  · have : NoAtoms (volume : Measure ({w : InfinitePlace K // IsReal w} → ℝ)) := pi_noAtoms ⟨w, hw⟩
    /-
      case pos
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : w.IsReal
      this : MeasureTheory.NoAtoms MeasureTheory.MeasureSpace.volume
      ⊢ MeasureTheory.NoAtoms MeasureTheory.MeasureSpace.volume
    -/
    exact prod.instNoAtoms_fst
    /-
      🎉 no goals
    -/
  · have : NoAtoms (volume : Measure ({w : InfinitePlace K // IsComplex w} → ℂ)) :=
      pi_noAtoms ⟨w, not_isReal_iff_isComplex.mp hw⟩
    /-
      case neg
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : Not w.IsReal
      this : MeasureTheory.NoAtoms MeasureTheory.MeasureSpace.volume
      ⊢ MeasureTheory.NoAtoms MeasureTheory.MeasureSpace.volume
    -/
    exact prod.instNoAtoms_snd
    /-
      🎉 no goals
    -/


variable {K} in
open Classical in
/-- The set of points in the mixedSpace that are equal to `0` at a fixed (real) place has
volume zero. -/
theorem volume_eq_zero (w : {w // IsReal w}) :
    volume ({x : mixedSpace K | x.1 w = 0}) = 0 := by
  let A : AffineSubspace ℝ (mixedSpace K) :=
    Submodule.toAffineSubspace (Submodule.mk ⟨⟨{x | x.1 w = 0}, by aesop⟩, rfl⟩ (by aesop))
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w : Subtype fun w => w.IsReal
    A : AffineSubspace Real (NumberField.mixedEmbedding.mixedSpace K) := { carrier …
    ⊢ Eq (MeasureTheory.MeasureSpace.volume (setOf fun x => Eq (x.1 w) 0)) 0
  -/
  convert Measure.addHaar_affineSubspace volume A fun h ↦ ?_
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    w : Subtype fun w => w.IsReal
    A : AffineSubspace Real (NumberField.mixedEmbedding.mixedSpace K) := { carrier …
    h : Eq A Top.top
    ⊢ False
  -/
  simpa [A] using (h ▸ Set.mem_univ _ : 1 ∈ A)
  /-
    🎉 no goals
  -/


/-- The linear map that makes `canonicalEmbedding` and `mixedEmbedding` commute, see
`commMap_canonical_eq_mixed`. -/
noncomputable def commMap : ((K →+* ℂ) → ℂ) →ₗ[ℝ] (mixedSpace K) where
  toFun := fun x => ⟨fun w => (x w.val.embedding).re, fun w => x w.val.embedding⟩
  map_add' := by
    /-
      K : Type u_1
      inst✝ : Field K
      ⊢ ∀ (x y : RingHom K Complex → Complex), Eq ((fun x => { fst := fun w => (x (↑ …
    -/
    simp only [Pi.add_apply, Complex.add_re, Prod.mk_add_mk, Prod.mk.injEq]
    /-
      K : Type u_1
      inst✝ : Field K
      ⊢ ∀ (x y : RingHom K Complex → Complex), And (Eq (fun w => HAdd.hAdd (x (↑w).e …
    -/
    exact fun _ _ => ⟨rfl, rfl⟩
    /-
      🎉 no goals
    -/
  map_smul' := by
    simp only [Pi.smul_apply, Complex.real_smul, Complex.mul_re, Complex.ofReal_re,
      Complex.ofReal_im, zero_mul, sub_zero, RingHom.id_apply, Prod.smul_mk, Prod.mk.injEq]
    /-
      K : Type u_1
      inst✝ : Field K
      ⊢ ∀ (m : Real) (x : RingHom K Complex → Complex), And (Eq (fun w => HMul.hMul  …
    -/
    exact fun _ _ => ⟨rfl, rfl⟩
    /-
      🎉 no goals
    -/


theorem commMap_apply_of_isReal (x : (K →+* ℂ) → ℂ) {w : InfinitePlace K} (hw : IsReal w) :
    (commMap K x).1 ⟨w, hw⟩ = (x w.embedding).re := rfl


theorem commMap_apply_of_isComplex (x : (K →+* ℂ) → ℂ) {w : InfinitePlace K} (hw : IsComplex w) :
    (commMap K x).2 ⟨w, hw⟩ = x w.embedding := rfl


@[simp]
theorem commMap_canonical_eq_mixed (x : K) :
    commMap K (canonicalEmbedding K x) = mixedEmbedding K x := by
  simp only [canonicalEmbedding, commMap, LinearMap.coe_mk, AddHom.coe_mk, Pi.ringHom_apply,
    mixedEmbedding, RingHom.prod_apply, Prod.mk.injEq]
  /-
    K : Type u_1
    inst✝ : Field K
    x : K
    ⊢ And (Eq (fun w => ((↑w).embedding x).re) ((Pi.ringHom fun w => NumberField.I …
  -/
  exact ⟨rfl, rfl⟩
  /-
    🎉 no goals
  -/


/-- This is a technical result to ensure that the image of the `ℂ`-basis of `ℂ^n` defined in
`canonicalEmbedding.latticeBasis` is a `ℝ`-basis of the mixed space `ℝ^r₁ × ℂ^r₂`,
see `mixedEmbedding.latticeBasis`. -/
theorem disjoint_span_commMap_ker [NumberField K] :
    Disjoint (Submodule.span ℝ (Set.range (canonicalEmbedding.latticeBasis K)))
      (LinearMap.ker (commMap K)) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Disjoint (Submodule.span Real (Set.range ⇑(NumberField.canonicalEmbedding.la …
  -/
  refine LinearMap.disjoint_ker.mpr (fun x h_mem h_zero => ?_)
  replace h_mem : x ∈ Submodule.span ℝ (Set.range (canonicalEmbedding K)) := by
    refine (Submodule.span_mono ?_) h_mem
    rintro _ ⟨i, rfl⟩
    exact ⟨integralBasis K i, (canonicalEmbedding.latticeBasis_apply K i).symm⟩
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : RingHom K Complex → Complex
    h_zero : Eq ((NumberField.mixedEmbedding.commMap K) x) 0
    h_mem : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonical …
    ⊢ Eq x 0
  -/
  ext1 φ
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : RingHom K Complex → Complex
    h_zero : Eq ((NumberField.mixedEmbedding.commMap K) x) 0
    h_mem : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonical …
    φ : RingHom K Complex
    ⊢ Eq (x φ) (0 φ)
  -/
  rw [Pi.zero_apply]
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : RingHom K Complex → Complex
    h_zero : Eq ((NumberField.mixedEmbedding.commMap K) x) 0
    h_mem : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonical …
    φ : RingHom K Complex
    ⊢ Eq (x φ) 0
  -/
  by_cases hφ : ComplexEmbedding.IsReal φ
    /-
      case pos
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : RingHom K Complex → Complex
      h_zero : Eq ((NumberField.mixedEmbedding.commMap K) x) 0
      h_mem : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonical …
      φ : RingHom K Complex
      hφ : NumberField.ComplexEmbedding.IsReal φ
      ⊢ Eq (x φ) 0
    -/
  · apply Complex.ext
      /-
        case pos.a
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x : RingHom K Complex → Complex
        h_zero : Eq ((NumberField.mixedEmbedding.commMap K) x) 0
        h_mem : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonical …
        φ : RingHom K Complex
        hφ : NumberField.ComplexEmbedding.IsReal φ
        ⊢ Eq (x φ).re (Complex.re 0)
      -/
    · rw [← embedding_mk_eq_of_isReal hφ, ← commMap_apply_of_isReal K x ⟨φ, hφ, rfl⟩]
      /-
        case pos.a
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x : RingHom K Complex → Complex
        h_zero : Eq ((NumberField.mixedEmbedding.commMap K) x) 0
        h_mem : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonical …
        φ : RingHom K Complex
        hφ : NumberField.ComplexEmbedding.IsReal φ
        ⊢ Eq (((NumberField.mixedEmbedding.commMap K) x).1 ⟨NumberField.InfinitePlace. …
      -/
      exact congrFun (congrArg (fun x => x.1) h_zero) ⟨InfinitePlace.mk φ, _⟩
      /-
        🎉 no goals
      -/
    · rw [Complex.zero_im, ← Complex.conj_eq_iff_im, canonicalEmbedding.conj_apply _ h_mem,
        ComplexEmbedding.isReal_iff.mp hφ]
    /-
      case neg
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : RingHom K Complex → Complex
      h_zero : Eq ((NumberField.mixedEmbedding.commMap K) x) 0
      h_mem : Membership.mem (Submodule.span Real (Set.range ⇑(NumberField.canonical …
      φ : RingHom K Complex
      hφ : Not (NumberField.ComplexEmbedding.IsReal φ)
      ⊢ Eq (x φ) 0
    -/
  · have := congrFun (congrArg (fun x => x.2) h_zero) ⟨InfinitePlace.mk φ, ⟨φ, hφ, rfl⟩⟩
    cases embedding_mk_eq φ with
    | inl h => rwa [← h, ← commMap_apply_of_isComplex K x ⟨φ, hφ, rfl⟩]
    | inr h =>
        apply RingHom.injective (starRingEnd ℂ)
        rwa [canonicalEmbedding.conj_apply _ h_mem, ← h, map_zero,
          ← commMap_apply_of_isComplex K x ⟨φ, hφ, rfl⟩]


/-- The norm at the infinite place `w` of an element of the mixed space. --/
def normAtPlace (w : InfinitePlace K) : (mixedSpace K) →*₀ ℝ where
  toFun x := if hw : IsReal w then ‖x.1 ⟨w, hw⟩‖ else ‖x.2 ⟨w, not_isReal_iff_isComplex.mp hw⟩‖
                  /-
                    K : Type u_1
                    inst✝ : Field K
                    w : NumberField.InfinitePlace K
                    ⊢ Eq ((fun x => dite w.IsReal (fun hw => Norm.norm (x.1 ⟨w, hw⟩)) fun hw => No …
                  -/
  map_zero' := by simp
                  /-
                    🎉 no goals
                  -/
                 /-
                   K : Type u_1
                   inst✝ : Field K
                   w : NumberField.InfinitePlace K
                   ⊢ Eq ({ toFun := fun x => dite w.IsReal (fun hw => Norm.norm (x.1 ⟨w, hw⟩)) fu …
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
                     /-
                       K : Type u_1
                       inst✝ : Field K
                       w : NumberField.InfinitePlace K
                       x y : NumberField.mixedEmbedding.mixedSpace K
                       ⊢ Eq ({ toFun := fun x => dite w.IsReal (fun hw => Norm.norm (x.1 ⟨w, hw⟩)) fu …
                     -/
                                   /-
                                     🎉 no goals
                                   -/
  map_mul' x y := by split_ifs <;> simp
                                   /-
                                     🎉 no goals
                                   -/


theorem normAtPlace_nonneg (w : InfinitePlace K) (x : mixedSpace K) :
    0 ≤ normAtPlace w x := by
  /-
    K : Type u_1
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ LE.le 0 ((NumberField.mixedEmbedding.normAtPlace w) x)
  -/
  rw [normAtPlace, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk]
  /-
    K : Type u_1
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ LE.le 0 (dite w.IsReal (fun hw => Norm.norm (x.1 ⟨w, hw⟩)) fun hw => Norm.no …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> exact norm_nonneg _
                /-
                  🎉 no goals
                -/


theorem normAtPlace_neg (w : InfinitePlace K) (x : mixedSpace K)  :
    normAtPlace w (- x) = normAtPlace w x := by
  /-
    K : Type u_1
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Eq ((NumberField.mixedEmbedding.normAtPlace w) (Neg.neg x)) ((NumberField.mi …
  -/
  rw [normAtPlace, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk]
  /-
    K : Type u_1
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Eq (dite w.IsReal (fun hw => Norm.norm ((Neg.neg x).1 ⟨w, hw⟩)) fun hw => No …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> simp
                /-
                  🎉 no goals
                -/


theorem normAtPlace_add_le (w : InfinitePlace K) (x y : mixedSpace K) :
    normAtPlace w (x + y) ≤ normAtPlace w x + normAtPlace w y := by
  /-
    K : Type u_1
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    x y : NumberField.mixedEmbedding.mixedSpace K
    ⊢ LE.le ((NumberField.mixedEmbedding.normAtPlace w) (HAdd.hAdd x y)) (HAdd.hAd …
  -/
  rw [normAtPlace, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk]
  /-
    K : Type u_1
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    x y : NumberField.mixedEmbedding.mixedSpace K
    ⊢ LE.le (dite w.IsReal (fun hw => Norm.norm ((HAdd.hAdd x y).1 ⟨w, hw⟩)) fun h …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> exact norm_add_le _ _
                /-
                  🎉 no goals
                -/


theorem normAtPlace_smul (w : InfinitePlace K) (x : mixedSpace K) (c : ℝ) :
    normAtPlace w (c • x) = |c| * normAtPlace w x := by
  /-
    K : Type u_1
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    x : NumberField.mixedEmbedding.mixedSpace K
    c : Real
    ⊢ Eq ((NumberField.mixedEmbedding.normAtPlace w) (HSMul.hSMul c x)) (HMul.hMul …
  -/
  rw [normAtPlace, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk]
  /-
    K : Type u_1
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    x : NumberField.mixedEmbedding.mixedSpace K
    c : Real
    ⊢ Eq (dite w.IsReal (fun hw => Norm.norm ((HSMul.hSMul c x).1 ⟨w, hw⟩)) fun hw …
  -/
  split_ifs
    /-
      case pos
      K : Type u_1
      inst✝ : Field K
      w : NumberField.InfinitePlace K
      x : NumberField.mixedEmbedding.mixedSpace K
      c : Real
      h✝ : w.IsReal
      ⊢ Eq (Norm.norm ((HSMul.hSMul c x).1 ⟨w, h✝⟩)) (HMul.hMul (abs c) (Norm.norm ( …
    -/
  · rw [Prod.smul_fst, Pi.smul_apply, norm_smul, Real.norm_eq_abs]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_1
      inst✝ : Field K
      w : NumberField.InfinitePlace K
      x : NumberField.mixedEmbedding.mixedSpace K
      c : Real
      h✝ : Not w.IsReal
      ⊢ Eq (Norm.norm ((HSMul.hSMul c x).2 ⟨w, ⋯⟩)) (HMul.hMul (abs c) (Norm.norm (x …
    -/
  · rw [Prod.smul_snd, Pi.smul_apply, norm_smul, Real.norm_eq_abs, Complex.norm_eq_abs]
    /-
      🎉 no goals
    -/


theorem normAtPlace_real (w : InfinitePlace K) (c : ℝ) :
    normAtPlace w ((fun _ ↦ c, fun _ ↦ c) : (mixedSpace K)) = |c| := by
  rw [show ((fun _ ↦ c, fun _ ↦ c) : (mixedSpace K)) = c • 1 by ext <;> simp, normAtPlace_smul,
    map_one, mul_one]


theorem normAtPlace_apply_isReal {w : InfinitePlace K} (hw : IsReal w) (x : mixedSpace K) :
    normAtPlace w x = ‖x.1 ⟨w, hw⟩‖ := by
  /-
    K : Type u_1
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    hw : w.IsReal
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Eq ((NumberField.mixedEmbedding.normAtPlace w) x) (Norm.norm (x.1 ⟨w, hw⟩))
  -/
  rw [normAtPlace, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk, dif_pos]
  /-
    🎉 no goals
  -/


theorem normAtPlace_apply_isComplex {w : InfinitePlace K} (hw : IsComplex w) (x : mixedSpace K) :
    normAtPlace w x = ‖x.2 ⟨w, hw⟩‖ := by
  rw [normAtPlace, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk,
    dif_neg (not_isReal_iff_isComplex.mpr hw)]


@[simp]
theorem normAtPlace_apply (w : InfinitePlace K) (x : K) :
    normAtPlace w (mixedEmbedding K x) = w x := by
  simp_rw [normAtPlace, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk, mixedEmbedding,
    RingHom.prod_apply, Pi.ringHom_apply, norm_embedding_of_isReal, norm_embedding_eq, dite_eq_ite,
    ite_id]


theorem forall_normAtPlace_eq_zero_iff {x : mixedSpace K} :
    (∀ w, normAtPlace w x = 0) ↔ x = 0 := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (∀ (w : NumberField.InfinitePlace K), Eq ((NumberField.mixedEmbedding.no …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      K : Type u_1
      inst✝ : Field K
      x : NumberField.mixedEmbedding.mixedSpace K
      h : ∀ (w : NumberField.InfinitePlace K), Eq ((NumberField.mixedEmbedding.normA …
      ⊢ Eq x 0
    -/
  · ext w
      /-
        case refine_1.fst.h
        K : Type u_1
        inst✝ : Field K
        x : NumberField.mixedEmbedding.mixedSpace K
        h : ∀ (w : NumberField.InfinitePlace K), Eq ((NumberField.mixedEmbedding.normA …
        w : Subtype fun w => w.IsReal
        ⊢ Eq (x.1 w) (0.1 w)
      -/
    · exact norm_eq_zero.mp (normAtPlace_apply_isReal w.prop _ ▸ h w.1)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.snd.h
        K : Type u_1
        inst✝ : Field K
        x : NumberField.mixedEmbedding.mixedSpace K
        h : ∀ (w : NumberField.InfinitePlace K), Eq ((NumberField.mixedEmbedding.normA …
        w : Subtype fun w => w.IsComplex
        ⊢ Eq (x.2 w) (0.2 w)
      -/
    · exact norm_eq_zero.mp (normAtPlace_apply_isComplex w.prop _ ▸ h w.1)
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      K : Type u_1
      inst✝ : Field K
      x : NumberField.mixedEmbedding.mixedSpace K
      h : Eq x 0
      ⊢ ∀ (w : NumberField.InfinitePlace K), Eq ((NumberField.mixedEmbedding.normAtP …
    -/
  · simp_rw [h, map_zero, implies_true]
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-09-13")] alias normAtPlace_eq_zero := forall_normAtPlace_eq_zero_iff


@[simp]
theorem exists_normAtPlace_ne_zero_iff {x : mixedSpace K} :
    (∃ w, normAtPlace w x ≠ 0) ↔ x ≠ 0 := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Exists fun w => Ne ((NumberField.mixedEmbedding.normAtPlace w) x) 0) (N …
  -/
  rw [ne_eq, ← forall_normAtPlace_eq_zero_iff, not_forall]
  /-
    🎉 no goals
  -/


theorem nnnorm_eq_sup_normAtPlace (x : mixedSpace K) :
    ‖x‖₊ = univ.sup fun w ↦ ⟨normAtPlace w x, normAtPlace_nonneg w x⟩ := by
  have :
      (univ : Finset (InfinitePlace K)) =
      (univ.image (fun w : {w : InfinitePlace K // IsReal w} ↦ w.1)) ∪
      (univ.image (fun w : {w : InfinitePlace K // IsComplex w} ↦ w.1)) := by
    ext; simp [isReal_or_isComplex]
  rw [this, sup_union, univ.sup_image, univ.sup_image,
    Prod.nnnorm_def', Pi.nnnorm_def, Pi.nnnorm_def]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    this : Eq Finset.univ (Union.union (Finset.image (fun w => ↑w) Finset.univ) (F …
    ⊢ Eq (Max.max (Finset.univ.sup fun b => NNNorm.nnnorm (x.1 b)) (Finset.univ.su …
  -/
  congr
    /-
      case e_a.e_f
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.mixedEmbedding.mixedSpace K
      this : Eq Finset.univ (Union.union (Finset.image (fun w => ↑w) Finset.univ) (F …
      ⊢ Eq (fun b => NNNorm.nnnorm (x.1 b)) (Function.comp (fun w => ⟨(NumberField.m …
    -/
  · ext w
    /-
      case e_a.e_f.h.a
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.mixedEmbedding.mixedSpace K
      this : Eq Finset.univ (Union.union (Finset.image (fun w => ↑w) Finset.univ) (F …
      w : Subtype fun w => w.IsReal
      ⊢ Eq ↑(NNNorm.nnnorm (x.1 w)) ↑(Function.comp (fun w => ⟨(NumberField.mixedEmb …
    -/
    simp [normAtPlace_apply_isReal w.prop]
    /-
      🎉 no goals
    -/
    /-
      case e_a.e_f
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.mixedEmbedding.mixedSpace K
      this : Eq Finset.univ (Union.union (Finset.image (fun w => ↑w) Finset.univ) (F …
      ⊢ Eq (fun b => NNNorm.nnnorm (x.2 b)) (Function.comp (fun w => ⟨(NumberField.m …
    -/
  · ext w
    /-
      case e_a.e_f.h.a
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.mixedEmbedding.mixedSpace K
      this : Eq Finset.univ (Union.union (Finset.image (fun w => ↑w) Finset.univ) (F …
      w : Subtype fun w => w.IsComplex
      ⊢ Eq ↑(NNNorm.nnnorm (x.2 w)) ↑(Function.comp (fun w => ⟨(NumberField.mixedEmb …
    -/
    simp [normAtPlace_apply_isComplex w.prop]
    /-
      🎉 no goals
    -/


theorem norm_eq_sup'_normAtPlace (x : mixedSpace K) :
    ‖x‖ = univ.sup' univ_nonempty fun w ↦ normAtPlace w x := by
  rw [← coe_nnnorm, nnnorm_eq_sup_normAtPlace, ← sup'_eq_sup univ_nonempty, ← NNReal.val_eq_coe,
    ← OrderHom.Subtype.val_coe, map_finset_sup', OrderHom.Subtype.val_coe]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Eq (Finset.univ.sup' ⋯ (Function.comp Subtype.val fun w => ⟨(NumberField.mix …
  -/
  simp only [Function.comp_apply]
  /-
    🎉 no goals
  -/


/-- The norm of `x` is `∏ w, (normAtPlace x) ^ mult w`. It is defined such that the norm of
`mixedEmbedding K a` for `a : K` is equal to the absolute value of the norm of `a` over `ℚ`,
see `norm_eq_norm`. -/
protected def norm : (mixedSpace K) →*₀ ℝ where
  toFun x := ∏ w, (normAtPlace w x) ^ (mult w)
                 /-
                   K : Type u_1
                   inst✝¹ : Field K
                   inst✝ : NumberField K
                   ⊢ Eq ({ toFun := fun x => Finset.univ.prod fun w => HPow.hPow ((NumberField.mi …
                 -/
                  /-
                    K : Type u_1
                    inst✝¹ : Field K
                    inst✝ : NumberField K
                    ⊢ Eq ((fun x => Finset.univ.prod fun w => HPow.hPow ((NumberField.mixedEmbeddi …
                  -/
  map_one' := by simp only [map_one, one_pow, prod_const_one]
                  /-
                    🎉 no goals
                  -/
                 /-
                   🎉 no goals
                 -/
  map_zero' := by simp [mult]
                     /-
                       K : Type u_1
                       inst✝¹ : Field K
                       inst✝ : NumberField K
                       x✝¹ x✝ : NumberField.mixedEmbedding.mixedSpace K
                       ⊢ Eq ({ toFun := fun x => Finset.univ.prod fun w => HPow.hPow ((NumberField.mi …
                     -/
  map_mul' _ _ := by simp only [map_mul, mul_pow, prod_mul_distrib]
                     /-
                       🎉 no goals
                     -/


protected theorem norm_apply (x : mixedSpace K) :
    mixedEmbedding.norm x = ∏ w, (normAtPlace w x) ^ (mult w) := rfl


protected theorem norm_nonneg (x : mixedSpace K) :
    0 ≤ mixedEmbedding.norm x := univ.prod_nonneg fun _ _ ↦ pow_nonneg (normAtPlace_nonneg _ _) _


protected theorem norm_eq_zero_iff {x : mixedSpace K} :
    mixedEmbedding.norm x = 0 ↔ ∃ w, normAtPlace w x = 0 := by
  simp_rw [mixedEmbedding.norm, MonoidWithZeroHom.coe_mk, ZeroHom.coe_mk, prod_eq_zero_iff,
    mem_univ, true_and, pow_eq_zero_iff mult_ne_zero]


protected theorem norm_ne_zero_iff {x : mixedSpace K} :
    mixedEmbedding.norm x ≠ 0 ↔ ∀ w, normAtPlace w x ≠ 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Ne (NumberField.mixedEmbedding.norm x) 0) (∀ (w : NumberField.InfiniteP …
  -/
  rw [← not_iff_not]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Not (Ne (NumberField.mixedEmbedding.norm x) 0)) (Not (∀ (w : NumberFiel …
  -/
  simp_rw [ne_eq, mixedEmbedding.norm_eq_zero_iff, not_not, not_forall, not_not]
  /-
    🎉 no goals
  -/


theorem norm_eq_of_normAtPlace_eq {x y : mixedSpace K}
    (h : ∀ w, normAtPlace w x = normAtPlace w y) :
    mixedEmbedding.norm x = mixedEmbedding.norm y := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x y : NumberField.mixedEmbedding.mixedSpace K
    h : ∀ (w : NumberField.InfinitePlace K), Eq ((NumberField.mixedEmbedding.normA …
    ⊢ Eq (NumberField.mixedEmbedding.norm x) (NumberField.mixedEmbedding.norm y)
  -/
  simp_rw [mixedEmbedding.norm_apply, h]
  /-
    🎉 no goals
  -/


theorem norm_smul (c : ℝ) (x : mixedSpace K) :
    mixedEmbedding.norm (c • x) = |c| ^ finrank ℚ K * (mixedEmbedding.norm x) := by
  simp_rw [mixedEmbedding.norm_apply, normAtPlace_smul, mul_pow, prod_mul_distrib,
    prod_pow_eq_pow_sum, sum_mult_eq]


theorem norm_real (c : ℝ) :
    mixedEmbedding.norm ((fun _ ↦ c, fun _ ↦ c) : (mixedSpace K)) = |c| ^ finrank ℚ K := by
  rw [show ((fun _ ↦ c, fun _ ↦ c) : (mixedSpace K)) = c • 1 by ext <;> simp, norm_smul, map_one,
    mul_one]


@[simp]
theorem norm_eq_norm (x : K) :
    mixedEmbedding.norm (mixedEmbedding K x) = |Algebra.norm ℚ x| := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    ⊢ Eq (NumberField.mixedEmbedding.norm ((NumberField.mixedEmbedding K) x)) ↑(ab …
  -/
  simp_rw [mixedEmbedding.norm_apply, normAtPlace_apply, prod_eq_abs_norm]
  /-
    🎉 no goals
  -/


theorem norm_unit (u : (𝓞 K)ˣ) :
    mixedEmbedding.norm (mixedEmbedding K u) = 1 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    u : Units (NumberField.RingOfIntegers K)
    ⊢ Eq (NumberField.mixedEmbedding.norm ((NumberField.mixedEmbedding K) ((algebr …
  -/
  rw [norm_eq_norm, Units.norm, Rat.cast_one]
  /-
    🎉 no goals
  -/


theorem norm_eq_zero_iff' {x : mixedSpace K} (hx : x ∈ Set.range (mixedEmbedding K)) :
    mixedEmbedding.norm x = 0 ↔ x = 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : Membership.mem (Set.range ⇑(NumberField.mixedEmbedding K)) x
    ⊢ Iff (Eq (NumberField.mixedEmbedding.norm x) 0) (Eq x 0)
  -/
  obtain ⟨a, rfl⟩ := hx
  rw [norm_eq_norm, Rat.cast_abs, abs_eq_zero, Rat.cast_eq_zero, Algebra.norm_eq_zero_iff,
    map_eq_zero]


/-- The type indexing the basis `stdBasis`. -/
abbrev index := {w : InfinitePlace K // IsReal w} ⊕ ({w : InfinitePlace K // IsComplex w}) × (Fin 2)


/-- The `ℝ`-basis of the mixed space of `K` formed by the vector equal to `1` at `w` and `0`
elsewhere for `IsReal w` and by the couple of vectors equal to `1` (resp. `I`) at `w` and `0`
elsewhere for `IsComplex w`. -/
def stdBasis : Basis (index K) ℝ (mixedSpace K) :=
  Basis.prod (Pi.basisFun ℝ _)
    (Basis.reindex (Pi.basis fun _ => basisOneI) (Equiv.sigmaEquivProd _ _))


@[simp]
theorem stdBasis_apply_ofIsReal (x : mixedSpace K) (w : {w : InfinitePlace K // IsReal w}) :
    (stdBasis K).repr x (Sum.inl w) = x.1 w := rfl


@[simp]
theorem stdBasis_apply_ofIsComplex_fst (x : mixedSpace K)
    (w : {w : InfinitePlace K // IsComplex w}) :
    (stdBasis K).repr x (Sum.inr ⟨w, 0⟩) = (x.2 w).re := rfl


@[simp]
theorem stdBasis_apply_ofIsComplex_snd (x : mixedSpace K)
    (w : {w : InfinitePlace K // IsComplex w}) :
    (stdBasis K).repr x (Sum.inr ⟨w, 1⟩) = (x.2 w).im := rfl


theorem fundamentalDomain_stdBasis :
    fundamentalDomain (stdBasis K) =
        (Set.univ.pi fun _ => Set.Ico 0 1) ×ˢ
        (Set.univ.pi fun _ => Complex.measurableEquivPi⁻¹' (Set.univ.pi fun _ => Set.Ico 0 1)) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Eq (ZSpan.fundamentalDomain (NumberField.mixedEmbedding.stdBasis K)) (SProd. …
  -/
  ext
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x✝ : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Membership.mem (ZSpan.fundamentalDomain (NumberField.mixedEmbedding.std …
  -/
  simp [stdBasis, mem_fundamentalDomain, Complex.measurableEquivPi]
  /-
    🎉 no goals
  -/


theorem volume_fundamentalDomain_stdBasis :
    volume (fundamentalDomain (stdBasis K)) = 1 := by
  rw [fundamentalDomain_stdBasis, volume_eq_prod, prod_prod, volume_pi, volume_pi, pi_pi, pi_pi,
    Complex.volume_preserving_equiv_pi.measure_preimage ?_, volume_pi, pi_pi, Real.volume_Ico,
    sub_zero, ENNReal.ofReal_one, prod_const_one, prod_const_one, prod_const_one, one_mul]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ MeasureTheory.NullMeasurableSet (Set.univ.pi fun x => Set.Ico 0 1) MeasureTh …
  -/
  exact (MeasurableSet.pi Set.countable_univ (fun _ _ => measurableSet_Ico)).nullMeasurableSet
  /-
    🎉 no goals
  -/


/-- The `Equiv` between `index K` and `K →+* ℂ` defined by sending a real infinite place `w` to
the unique corresponding embedding `w.embedding`, and the pair `⟨w, 0⟩` (resp. `⟨w, 1⟩`) for a
complex infinite place `w` to `w.embedding` (resp. `conjugate w.embedding`). -/
def indexEquiv : (index K) ≃ (K →+* ℂ) := by
  refine Equiv.ofBijective (fun c => ?_)
    ((Fintype.bijective_iff_surjective_and_card _).mpr ⟨?_, ?_⟩)
  · cases c with
    | inl w => exact w.val.embedding
    | inr wj => rcases wj with ⟨w, j⟩
                exact if j = 0 then w.val.embedding else ComplexEmbedding.conjugate w.val.embedding
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ⊢ Function.Surjective fun c => Sum.casesOn (motive := fun t => Eq c t → RingHo …
    -/
  · intro φ
    /-
      case refine_2
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      φ : RingHom K Complex
      ⊢ Exists fun a => Eq ((fun c => Sum.casesOn (motive := fun t => Eq c t → RingH …
    -/
    by_cases hφ : ComplexEmbedding.IsReal φ
      /-
        case pos
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        φ : RingHom K Complex
        hφ : NumberField.ComplexEmbedding.IsReal φ
        ⊢ Exists fun a => Eq ((fun c => Sum.casesOn (motive := fun t => Eq c t → RingH …
      -/
    · exact ⟨Sum.inl (InfinitePlace.mkReal ⟨φ, hφ⟩), by simp [embedding_mk_eq_of_isReal hφ]⟩
      /-
        🎉 no goals
      -/
      /-
        case neg
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        φ : RingHom K Complex
        hφ : Not (NumberField.ComplexEmbedding.IsReal φ)
        ⊢ Exists fun a => Eq ((fun c => Sum.casesOn (motive := fun t => Eq c t → RingH …
      -/
    · by_cases hw : (InfinitePlace.mk φ).embedding = φ
        /-
          case pos
          K : Type u_1
          inst✝¹ : Field K
          inst✝ : NumberField K
          φ : RingHom K Complex
          hφ : Not (NumberField.ComplexEmbedding.IsReal φ)
          hw : Eq (NumberField.InfinitePlace.mk φ).embedding φ
          ⊢ Exists fun a => Eq ((fun c => Sum.casesOn (motive := fun t => Eq c t → RingH …
        -/
      · exact ⟨Sum.inr ⟨InfinitePlace.mkComplex ⟨φ, hφ⟩, 0⟩, by simp [hw]⟩
        /-
          🎉 no goals
        -/
      · exact ⟨Sum.inr ⟨InfinitePlace.mkComplex ⟨φ, hφ⟩, 1⟩,
          by simp [(embedding_mk_eq φ).resolve_left hw]⟩
  · rw [Embeddings.card, ← mixedEmbedding.finrank K,
      ← Module.finrank_eq_card_basis (stdBasis K)]


@[simp]
theorem indexEquiv_apply_ofIsReal (w : {w : InfinitePlace K // IsReal w}) :
    (indexEquiv K) (Sum.inl w) = w.val.embedding := rfl


@[simp]
theorem indexEquiv_apply_ofIsComplex_fst (w : {w : InfinitePlace K // IsComplex w}) :
    (indexEquiv K) (Sum.inr ⟨w, 0⟩) = w.val.embedding := rfl


@[simp]
theorem indexEquiv_apply_ofIsComplex_snd (w : {w : InfinitePlace K // IsComplex w}) :
    (indexEquiv K) (Sum.inr ⟨w, 1⟩) = ComplexEmbedding.conjugate w.val.embedding := rfl


/-- The matrix that gives the representation on `stdBasis` of the image by `commMap` of an
element `x` of `(K →+* ℂ) → ℂ` fixed by the map `x_φ ↦ conj x_(conjugate φ)`,
see `stdBasis_repr_eq_matrixToStdBasis_mul`. -/
def matrixToStdBasis : Matrix (index K) (index K) ℂ :=
  fromBlocks (diagonal fun _ => 1) 0 0 <| reindex (Equiv.prodComm _ _) (Equiv.prodComm _ _)
    (blockDiagonal (fun _ => (2 : ℂ)⁻¹ • !![1, 1; - I, I]))


theorem det_matrixToStdBasis :
    (matrixToStdBasis K).det = (2⁻¹ * I) ^ nrComplexPlaces K :=
  calc
  _ = ∏ _k : { w : InfinitePlace K // IsComplex w }, det ((2 : ℂ)⁻¹ • !![1, 1; -I, I]) := by
      rw [matrixToStdBasis, det_fromBlocks_zero₂₁, det_diagonal, prod_const_one, one_mul,
          det_reindex_self, det_blockDiagonal]
  _ = ∏ _k : { w : InfinitePlace K // IsComplex w }, (2⁻¹ * Complex.I) := by
      /-
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        ⊢ Eq (Finset.univ.prod fun _k => (HSMul.hSMul (Inv.inv 2) (Matrix.of (Matrix.v …
      -/
      refine prod_congr (Eq.refl _) (fun _ _ => ?_)
      /-
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        x✝¹ : Subtype fun w => w.IsComplex
        x✝ : Membership.mem Finset.univ x✝¹
        ⊢ Eq (HSMul.hSMul (Inv.inv 2) (Matrix.of (Matrix.vecCons (Matrix.vecCons 1 (Ma …
      -/
      field_simp; ring
                  /-
                    🎉 no goals
                  -/
  _ = (2⁻¹ * Complex.I) ^ Fintype.card {w : InfinitePlace K // IsComplex w} := by
      /-
        K : Type u_1
        inst✝¹ : Field K
        inst✝ : NumberField K
        ⊢ Eq (Finset.univ.prod fun _k => HMul.hMul (Inv.inv 2) Complex.I) (HPow.hPow ( …
      -/
      rw [prod_const, Fintype.card]
      /-
        🎉 no goals
      -/


/-- Let `x : (K →+* ℂ) → ℂ` such that `x_φ = conj x_(conj φ)` for all `φ : K →+* ℂ`, then the
representation of `commMap K x` on `stdBasis` is given (up to reindexing) by the product of
`matrixToStdBasis` by `x`. -/
theorem stdBasis_repr_eq_matrixToStdBasis_mul (x : (K →+* ℂ) → ℂ)
    (hx : ∀ φ, conj (x φ) = x (ComplexEmbedding.conjugate φ)) (c : index K) :
    ((stdBasis K).repr (commMap K x) c : ℂ) =
      (matrixToStdBasis K *ᵥ (x ∘ (indexEquiv K))) c := by
  simp_rw [commMap, matrixToStdBasis, LinearMap.coe_mk, AddHom.coe_mk,
    mulVec, dotProduct, Function.comp_apply, index, Fintype.sum_sum_type,
    diagonal_one, reindex_apply, ← univ_product_univ, sum_product,
    indexEquiv_apply_ofIsReal, Fin.sum_univ_two, indexEquiv_apply_ofIsComplex_fst,
    indexEquiv_apply_ofIsComplex_snd, smul_of, smul_cons, smul_eq_mul,
    mul_one, Matrix.smul_empty, Equiv.prodComm_symm, Equiv.coe_prodComm]
  cases c with
  | inl w =>
      simp_rw [stdBasis_apply_ofIsReal, fromBlocks_apply₁₁, fromBlocks_apply₁₂,
        one_apply, Matrix.zero_apply, ite_mul, one_mul, zero_mul, sum_ite_eq, mem_univ, ite_true,
        add_zero, sum_const_zero, add_zero, ← conj_eq_iff_re, hx (embedding w.val),
        conjugate_embedding_eq_of_isReal w.prop]
  | inr c =>
    rcases c with ⟨w, j⟩
    fin_cases j
    · simp only [Fin.zero_eta, Fin.isValue, id_eq, stdBasis_apply_ofIsComplex_fst, re_eq_add_conj,
        mul_neg, fromBlocks_apply₂₁, zero_apply, zero_mul, sum_const_zero, fromBlocks_apply₂₂,
        submatrix_apply, Prod.swap_prod_mk, blockDiagonal_apply, of_apply, cons_val', cons_val_zero,
        empty_val', cons_val_fin_one, ite_mul, cons_val_one, head_cons, sum_add_distrib, sum_ite_eq,
        mem_univ, ↓reduceIte, ← hx (embedding w), zero_add]
      field_simp
    · simp only [Fin.mk_one, Fin.isValue, id_eq, stdBasis_apply_ofIsComplex_snd, im_eq_sub_conj,
        mul_neg, fromBlocks_apply₂₁, zero_apply, zero_mul, sum_const_zero, fromBlocks_apply₂₂,
        submatrix_apply, Prod.swap_prod_mk, blockDiagonal_apply, of_apply, cons_val', cons_val_zero,
        empty_val', cons_val_fin_one, cons_val_one, head_fin_const, ite_mul, neg_mul, head_cons,
        sum_add_distrib, sum_ite_eq, mem_univ, ↓reduceIte, ← hx (embedding w), zero_add]
      ring_nf; field_simp


/-- The image of the ring of integers of `K` in the mixed space. -/
protected abbrev integerLattice : Submodule ℤ (mixedSpace K) :=
  LinearMap.range ((mixedEmbedding K).comp (algebraMap (𝓞 K) K)).toIntAlgHom.toLinearMap


/-- A `ℝ`-basis of the mixed space that is also a `ℤ`-basis of the image of `𝓞 K`. -/
def latticeBasis :
    Basis (ChooseBasisIndex ℤ (𝓞 K)) ℝ (mixedSpace K) := by
  classical
    -- We construct an `ℝ`-linear independent family from the image of
    -- `canonicalEmbedding.lattice_basis` by `commMap`
    have := LinearIndependent.map (LinearIndependent.restrict_scalars
      (by { simpa only [Complex.real_smul, mul_one] using Complex.ofReal_injective })
      (canonicalEmbedding.latticeBasis K).linearIndependent)
      (disjoint_span_commMap_ker K)
    -- and it's a basis since it has the right cardinality
    refine basisOfLinearIndependentOfCardEqFinrank this ?_
    rw [← finrank_eq_card_chooseBasisIndex, RingOfIntegers.rank, finrank_prod, finrank_pi,
      finrank_pi_fintype, Complex.finrank_real_complex, sum_const, card_univ, ← nrRealPlaces,
      ← nrComplexPlaces, ← card_real_embeddings, Algebra.id.smul_eq_mul, mul_comm,
      ← card_complex_embeddings, ← NumberField.Embeddings.card K ℂ, Fintype.card_subtype_compl,
      Nat.add_sub_of_le (Fintype.card_subtype_le _)]


@[simp]
theorem latticeBasis_apply (i : ChooseBasisIndex ℤ (𝓞 K)) :
    latticeBasis K i = (mixedEmbedding K) (integralBasis K i) := by
  simp only [latticeBasis, coe_basisOfLinearIndependentOfCardEqFinrank, Function.comp_apply,
    canonicalEmbedding.latticeBasis_apply, integralBasis_apply, commMap_canonical_eq_mixed]


theorem mem_span_latticeBasis {x : (mixedSpace K)} :
    x ∈ Submodule.span ℤ (Set.range (latticeBasis K)) ↔
      x ∈ mixedEmbedding.integerLattice K := by
  rw [show Set.range (latticeBasis K) =
      (mixedEmbedding K).toIntAlgHom.toLinearMap '' (Set.range (integralBasis K)) by
    rw [← Set.range_comp]; exact congrArg Set.range (funext (fun i => latticeBasis_apply K i))]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Membership.mem (Submodule.span Int (Set.image (⇑(NumberField.mixedEmbed …
  -/
  rw [← Submodule.map_span, ← SetLike.mem_coe, Submodule.map_coe]
  simp only [Set.mem_image, SetLike.mem_coe, mem_span_integralBasis K,
    RingHom.mem_range, exists_exists_eq_and]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Exists fun a => Eq ((NumberField.mixedEmbedding K).toIntAlgHom.toLinear …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem span_latticeBasis :
    Submodule.span ℤ (Set.range (latticeBasis K)) = mixedEmbedding.integerLattice K :=
  Submodule.ext_iff.mpr fun _ ↦ mem_span_latticeBasis K


instance : DiscreteTopology (mixedEmbedding.integerLattice K) := by
  classical
  rw [← span_latticeBasis]
  infer_instance


open Classical in
instance : IsZLattice ℝ (mixedEmbedding.integerLattice K) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ IsZLattice Real (NumberField.mixedEmbedding.integerLattice K)
  -/
  simp_rw [← span_latticeBasis]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ IsZLattice Real (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbedding. …
  -/
  exact ZSpan.isZLattice (latticeBasis K)
  /-
    🎉 no goals
  -/


open Classical in
theorem fundamentalDomain_integerLattice :
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ⊢ MeasureTheory.Measure (NumberField.mixedEmbedding.mixedSpace K)
    -/
    MeasureTheory.IsAddFundamentalDomain (mixedEmbedding.integerLattice K)
    /-
      🎉 no goals
    -/
      (ZSpan.fundamentalDomain (latticeBasis K)) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem (Numbe …
  -/
  rw [← span_latticeBasis]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem (Submo …
  -/
  exact ZSpan.isAddFundamentalDomain (latticeBasis K) _
  /-
    🎉 no goals
  -/


theorem mem_rat_span_latticeBasis (x : K) :
    mixedEmbedding K x ∈ Submodule.span ℚ (Set.range (latticeBasis K)) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    ⊢ Membership.mem (Submodule.span Rat (Set.range ⇑(NumberField.mixedEmbedding.l …
  -/
  rw [← Basis.sum_repr (integralBasis K) x, map_sum]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    ⊢ Membership.mem (Submodule.span Rat (Set.range ⇑(NumberField.mixedEmbedding.l …
  -/
  simp_rw [map_rat_smul]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    ⊢ Membership.mem (Submodule.span Rat (Set.range ⇑(NumberField.mixedEmbedding.l …
  -/
  refine Submodule.sum_smul_mem _ _ (fun i _ ↦ Submodule.subset_span ?_)
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    i : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    x✝ : Membership.mem Finset.univ i
    ⊢ Membership.mem (Set.range ⇑(NumberField.mixedEmbedding.latticeBasis K)) ((Nu …
  -/
  rw [← latticeBasis_apply]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    i : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    x✝ : Membership.mem Finset.univ i
    ⊢ Membership.mem (Set.range ⇑(NumberField.mixedEmbedding.latticeBasis K)) ((Nu …
  -/
  exact Set.mem_range_self i
  /-
    🎉 no goals
  -/


theorem latticeBasis_repr_apply (x : K) (i : ChooseBasisIndex ℤ (𝓞 K)) :
    (latticeBasis K).repr (mixedEmbedding K x) i = (integralBasis K).repr x i := by
  rw [← Basis.restrictScalars_repr_apply ℚ _ ⟨_, mem_rat_span_latticeBasis K x⟩, eq_ratCast,
    Rat.cast_inj]
  let f := (mixedEmbedding K).toRatAlgHom.toLinearMap.codRestrict _
    (fun x ↦ mem_rat_span_latticeBasis K x)
  suffices ((latticeBasis K).restrictScalars ℚ).repr.toLinearMap ∘ₗ f =
    (integralBasis K).repr.toLinearMap from DFunLike.congr_fun (LinearMap.congr_fun this x) i
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    i : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    f : LinearMap (RingHom.id Rat) K (Subtype fun x => Membership.mem (Submodule.s …
    ⊢ Eq ((↑(Basis.restrictScalars Rat (NumberField.mixedEmbedding.latticeBasis K) …
  -/
  refine Basis.ext (integralBasis K) (fun i ↦ ?_)
  have : f (integralBasis K i) = ((latticeBasis K).restrictScalars ℚ) i := by
    apply Subtype.val_injective
    rw [LinearMap.codRestrict_apply, AlgHom.toLinearMap_apply, Basis.restrictScalars_apply,
      latticeBasis_apply]
    rfl
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    i✝ : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    f : LinearMap (RingHom.id Rat) K (Subtype fun x => Membership.mem (Submodule.s …
    i : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    this : Eq (f ((NumberField.integralBasis K) i)) ((Basis.restrictScalars Rat (N …
    ⊢ Eq (((↑(Basis.restrictScalars Rat (NumberField.mixedEmbedding.latticeBasis K …
  -/
  simp_rw [LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply, this, Basis.repr_self]
  /-
    🎉 no goals
  -/


/-- The image of the fractional ideal `I` in the mixed space. -/
abbrev idealLattice : Submodule ℤ (mixedSpace K) := LinearMap.range <|
  (mixedEmbedding K).toIntAlgHom.toLinearMap ∘ₗ ((I : Submodule (𝓞 K) K).subtype.restrictScalars ℤ)


theorem mem_idealLattice {x : mixedSpace K} :
    x ∈ idealLattice K I ↔ ∃ y, y ∈ (I : Set K) ∧ mixedEmbedding K y = x := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Membership.mem (NumberField.mixedEmbedding.idealLattice K I) x) (Exists …
  -/
  simp [idealLattice]
  /-
    🎉 no goals
  -/


/-- The generalized index of the lattice generated by `I` in the lattice generated by
`𝓞 K` is equal to the norm of the ideal `I`. The result is stated in terms of base change
determinant and is the translation of `NumberField.det_basisOfFractionalIdeal_eq_absNorm` in
the mixed space. This is useful, in particular, to prove that the family obtained from
the `ℤ`-basis of `I` is actually an `ℝ`-basis of the mixed space, see
`fractionalIdealLatticeBasis`. -/
theorem det_basisOfFractionalIdeal_eq_norm
    (e : (ChooseBasisIndex ℤ (𝓞 K)) ≃ (ChooseBasisIndex ℤ I)) :
    |Basis.det (latticeBasis K) ((mixedEmbedding K ∘ (basisOfFractionalIdeal K I) ∘ e))| =
      FractionalIdeal.absNorm I.1 := by
  suffices Basis.det (latticeBasis K) ((mixedEmbedding K ∘ (basisOfFractionalIdeal K I) ∘ e)) =
      (algebraMap ℚ ℝ) ((Basis.det (integralBasis K)) ((basisOfFractionalIdeal K I) ∘ e)) by
    rw [this, eq_ratCast, ← Rat.cast_abs, ← Equiv.symm_symm e, ← Basis.coe_reindex,
      det_basisOfFractionalIdeal_eq_absNorm K I e]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    e : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (M …
    ⊢ Eq ((NumberField.mixedEmbedding.latticeBasis K).det (Function.comp (⇑(Number …
  -/
  rw [Basis.det_apply, Basis.det_apply, RingHom.map_det]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    e : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (M …
    ⊢ Eq ((NumberField.mixedEmbedding.latticeBasis K).toMatrix (Function.comp (⇑(N …
  -/
  congr
  /-
    case e_M
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    e : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (M …
    ⊢ Eq ((NumberField.mixedEmbedding.latticeBasis K).toMatrix (Function.comp (⇑(N …
  -/
  ext i j
  /-
    case e_M.a
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    e : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (M …
    i j : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    ⊢ Eq ((NumberField.mixedEmbedding.latticeBasis K).toMatrix (Function.comp (⇑(N …
  -/
  simp_rw [RingHom.mapMatrix_apply, Matrix.map_apply, Basis.toMatrix_apply, Function.comp_apply]
  /-
    case e_M.a
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    e : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (M …
    i j : Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)
    ⊢ Eq (((NumberField.mixedEmbedding.latticeBasis K).repr ((NumberField.mixedEmb …
  -/
  exact latticeBasis_repr_apply K _ i
  /-
    🎉 no goals
  -/


/-- A `ℝ`-basis of the mixed space of `K` that is also a `ℤ`-basis of the image of the fractional
ideal `I`. -/
def fractionalIdealLatticeBasis :
    Basis (ChooseBasisIndex ℤ I) ℝ (mixedSpace K) := by
  let e : (ChooseBasisIndex ℤ (𝓞 K)) ≃ (ChooseBasisIndex ℤ I) := by
    refine Fintype.equivOfCardEq ?_
    rw [← finrank_eq_card_chooseBasisIndex, ← finrank_eq_card_chooseBasisIndex,
      fractionalIdeal_rank]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    e : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (M …
    ⊢ Basis (Module.Free.ChooseBasisIndex Int (Subtype fun x => Membership.mem (↑↑ …
  -/
  refine Basis.reindex ?_ e
  suffices IsUnit ((latticeBasis K).det ((mixedEmbedding K) ∘ (basisOfFractionalIdeal K I) ∘ e)) by
    rw [← is_basis_iff_det] at this
    exact Basis.mk this.1 (by rw [this.2])
  rw [isUnit_iff_ne_zero, ne_eq, ← abs_eq_zero.not, det_basisOfFractionalIdeal_eq_norm,
    Rat.cast_eq_zero, FractionalIdeal.absNorm_eq_zero_iff]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    e : Equiv (Module.Free.ChooseBasisIndex Int (NumberField.RingOfIntegers K)) (M …
    ⊢ Not (Eq (↑I) 0)
  -/
  exact Units.ne_zero I
  /-
    🎉 no goals
  -/


@[simp]
theorem fractionalIdealLatticeBasis_apply (i : ChooseBasisIndex ℤ I) :
    fractionalIdealLatticeBasis K I i = (mixedEmbedding K) (basisOfFractionalIdeal K I i) := by
  simp only [fractionalIdealLatticeBasis, Basis.coe_reindex, Basis.coe_mk, Function.comp_apply,
    Equiv.apply_symm_apply]


theorem mem_span_fractionalIdealLatticeBasis {x : (mixedSpace K)} :
    x ∈ Submodule.span ℤ (Set.range (fractionalIdealLatticeBasis K I)) ↔
      x ∈ mixedEmbedding K '' I := by
  rw [show Set.range (fractionalIdealLatticeBasis K I) =
        (mixedEmbedding K).toIntAlgHom.toLinearMap '' (Set.range (basisOfFractionalIdeal K I)) by
      rw [← Set.range_comp]
      exact congr_arg Set.range (funext (fun i ↦ fractionalIdealLatticeBasis_apply K I i))]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Membership.mem (Submodule.span Int (Set.image (⇑(NumberField.mixedEmbed …
  -/
  rw [← Submodule.map_span, ← SetLike.mem_coe, Submodule.map_coe]
  rw [show Submodule.span ℤ (Set.range (basisOfFractionalIdeal K I)) = (I : Set K) by
        ext; erw [mem_span_basisOfFractionalIdeal]]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Membership.mem (Set.image ⇑(NumberField.mixedEmbedding K).toIntAlgHom.t …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem span_idealLatticeBasis :
    (Submodule.span ℤ (Set.range (fractionalIdealLatticeBasis K I))) =
      (mixedEmbedding.idealLattice K I) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    ⊢ Eq (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbedding.fractionalIde …
  -/
  ext x
  /-
    case h
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Membership.mem (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbedd …
  -/
  simp [mem_span_fractionalIdealLatticeBasis]
  /-
    🎉 no goals
  -/


instance : DiscreteTopology (mixedEmbedding.idealLattice K I) := by
  classical
  rw [← span_idealLatticeBasis]
  infer_instance


open Classical in
instance : IsZLattice ℝ (mixedEmbedding.idealLattice K I) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    ⊢ IsZLattice Real (NumberField.mixedEmbedding.idealLattice K I)
  -/
  simp_rw [← span_idealLatticeBasis]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    ⊢ IsZLattice Real (Submodule.span Int (Set.range ⇑(NumberField.mixedEmbedding. …
  -/
  exact ZSpan.isZLattice (fractionalIdealLatticeBasis K I)
  /-
    🎉 no goals
  -/


open Classical in
theorem fundamentalDomain_idealLattice :
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
      ⊢ MeasureTheory.Measure (NumberField.mixedEmbedding.mixedSpace K)
    -/
    MeasureTheory.IsAddFundamentalDomain (mixedEmbedding.idealLattice K I)
    /-
      🎉 no goals
    -/
      (ZSpan.fundamentalDomain (fractionalIdealLatticeBasis K I)) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    ⊢ MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem (Numbe …
  -/
  rw [← span_idealLatticeBasis]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    I : Units (FractionalIdeal (nonZeroDivisors (NumberField.RingOfIntegers K)) K)
    ⊢ MeasureTheory.IsAddFundamentalDomain (Subtype fun x => Membership.mem (Submo …
  -/
  exact ZSpan.isAddFundamentalDomain (fractionalIdealLatticeBasis K I) _
  /-
    🎉 no goals
  -/


/-- The mixed space `ℝ^r₁ × ℂ^r₂`, with `(r₁, r₂)` the signature of `K`, as an Euclidean space. -/
protected abbrev mixedSpace :=
    (WithLp 2 ((EuclideanSpace ℝ {w : InfinitePlace K // IsReal w}) ×
      (EuclideanSpace ℂ {w : InfinitePlace K // IsComplex w})))


instance : Ring (euclidean.mixedSpace K) :=
  have : Ring (EuclideanSpace ℝ {w : InfinitePlace K // IsReal w}) := Pi.ring
  have : Ring (EuclideanSpace ℂ {w : InfinitePlace K // IsComplex w}) := Pi.ring
  inferInstanceAs (Ring (_ × _))


instance : MeasurableSpace (euclidean.mixedSpace K) := borel _


instance : BorelSpace (euclidean.mixedSpace K) := ⟨rfl⟩


open Classical in
/-- The continuous linear equivalence between the euclidean mixed space and the mixed space. -/
def toMixed : (euclidean.mixedSpace K) ≃L[ℝ] (mixedSpace K) :=
  (WithLp.linearEquiv _ _ _).toContinuousLinearEquiv


instance : Nontrivial (euclidean.mixedSpace K) := (toMixed K).toEquiv.nontrivial


protected theorem finrank :
    finrank ℝ (euclidean.mixedSpace K) = finrank ℚ K := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Eq (Module.finrank Real (NumberField.mixedEmbedding.euclidean.mixedSpace K)) …
  -/
  rw [LinearEquiv.finrank_eq (toMixed K).toLinearEquiv, mixedEmbedding.finrank]
  /-
    🎉 no goals
  -/


open Classical in
/-- An orthonormal basis of the euclidean mixed space. -/
def stdOrthonormalBasis : OrthonormalBasis (index K) ℝ (euclidean.mixedSpace K) :=
  OrthonormalBasis.prod (EuclideanSpace.basisFun _ ℝ)
    ((Pi.orthonormalBasis fun _ ↦ Complex.orthonormalBasisOneI).reindex (Equiv.sigmaEquivProd _ _))


open Classical in
theorem stdOrthonormalBasis_map_eq :
    (euclidean.stdOrthonormalBasis K).toBasis.map (toMixed K).toLinearEquiv =
      mixedEmbedding.stdBasis K := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Eq ((NumberField.mixedEmbedding.euclidean.stdOrthonormalBasis K).toBasis.map …
  -/
          /-
            🎉 no goals
          -/
  ext <;> rfl
          /-
            🎉 no goals
          -/


open Classical in
theorem volumePreserving_toMixed :
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ⊢ MeasureTheory.Measure (NumberField.mixedEmbedding.euclidean.mixedSpace K)
    -/
    /-
      🎉 no goals
    -/
    MeasurePreserving (toMixed K) where
    /-
      🎉 no goals
    -/
  measurable := (toMixed K).continuous.measurable
  map_eq := by
    rw [← (OrthonormalBasis.addHaar_eq_volume (euclidean.stdOrthonormalBasis K)), Basis.map_addHaar,
      stdOrthonormalBasis_map_eq, Basis.addHaar_eq_iff, Basis.coe_parallelepiped,
      ← measure_congr (ZSpan.fundamentalDomain_ae_parallelepiped (stdBasis K) volume),
      volume_fundamentalDomain_stdBasis K]


open Classical in
theorem volumePreserving_toMixed_symm :
    /-
      K : Type u_1
      inst✝¹ : Field K
      inst✝ : NumberField K
      ⊢ MeasureTheory.Measure (NumberField.mixedEmbedding.mixedSpace K)
    -/
    /-
      🎉 no goals
    -/
    MeasurePreserving (toMixed K).symm := by
    /-
      🎉 no goals
    -/
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ MeasureTheory.MeasurePreserving (⇑(NumberField.mixedEmbedding.euclidean.toMi …
  -/
  have : MeasurePreserving (toMixed K).toHomeomorph.toMeasurableEquiv := volumePreserving_toMixed K
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    this : MeasureTheory.MeasurePreserving (⇑(NumberField.mixedEmbedding.euclidean …
    ⊢ MeasureTheory.MeasurePreserving (⇑(NumberField.mixedEmbedding.euclidean.toMi …
  -/
  exact this.symm
  /-
    🎉 no goals
  -/


open Classical in
/-- The image of ring of integers `𝓞 K` in the euclidean mixed space. -/
protected def integerLattice : Submodule ℤ (euclidean.mixedSpace K) :=
  ZLattice.comap ℝ (mixedEmbedding.integerLattice K) (toMixed K).toLinearMap


instance : DiscreteTopology (euclidean.integerLattice K) := by
  classical
  rw [euclidean.integerLattice]
  infer_instance


open Classical in
instance : IsZLattice ℝ (euclidean.integerLattice K) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ IsZLattice Real (NumberField.mixedEmbedding.euclidean.integerLattice K)
  -/
  simp_rw [euclidean.integerLattice]
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ IsZLattice Real (ZLattice.comap Real (NumberField.mixedEmbedding.integerLatt …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


open Classical in
/-- Let `s` be a set of real places, define the continuous linear equiv of the mixed space that
swaps sign at places in `s` and leaves the rest unchanged. -/
def negAt :
    (mixedSpace K) ≃L[ℝ] (mixedSpace K) :=
  (piCongrRight fun w ↦ if w ∈ s then neg ℝ else ContinuousLinearEquiv.refl ℝ ℝ).prod
    (ContinuousLinearEquiv.refl ℝ _)


@[simp]
theorem negAt_apply_of_isReal_and_mem (x : mixedSpace K) {w : {w // IsReal w}} (hw : w ∈ s) :
    (negAt s x).1 w = - x.1 w := by
  simp_rw [negAt, ContinuousLinearEquiv.prod_apply, piCongrRight_apply, if_pos hw,
    ContinuousLinearEquiv.neg_apply]


@[simp]
theorem negAt_apply_of_isReal_and_not_mem (x : mixedSpace K) {w : {w // IsReal w}} (hw : w ∉ s) :
    (negAt s x).1 w = x.1 w := by
  simp_rw [negAt, ContinuousLinearEquiv.prod_apply, piCongrRight_apply, if_neg hw,
    ContinuousLinearEquiv.refl_apply]


@[simp]
theorem negAt_apply_of_isComplex (x : mixedSpace K) (w : {w // IsComplex w}) :
    (negAt s x).2 w = x.2 w := rfl


@[simp]
theorem negAt_apply_snd (x : mixedSpace K) :
    (negAt s x).2 = x.2 := rfl


@[simp]
theorem negAt_apply_abs_of_isReal (x : mixedSpace K) (w : {w // IsReal w}) :
    |(negAt s x).1 w| = |x.1 w| := by
  /-
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    x : NumberField.mixedEmbedding.mixedSpace K
    w : Subtype fun w => w.IsReal
    ⊢ Eq (abs (((NumberField.mixedEmbedding.negAt s) x).1 w)) (abs (x.1 w))
  -/
                          /-
                            🎉 no goals
                          -/
  by_cases hw : w ∈ s <;> simp [hw]
                          /-
                            🎉 no goals
                          -/


open MeasureTheory Classical in
/-- `negAt` preserves the volume . -/
theorem volume_preserving_negAt [NumberField K] :
    /-
      K : Type u_1
      inst✝¹ : Field K
      s : Set (Subtype fun w => w.IsReal)
      inst✝ : NumberField K
      ⊢ MeasureTheory.Measure (NumberField.mixedEmbedding.mixedSpace K)
    -/
    /-
      🎉 no goals
    -/
    MeasurePreserving (negAt s) := by
    /-
      🎉 no goals
    -/
  /-
    K : Type u_1
    inst✝¹ : Field K
    s : Set (Subtype fun w => w.IsReal)
    inst✝ : NumberField K
    ⊢ MeasureTheory.MeasurePreserving (⇑(NumberField.mixedEmbedding.negAt s)) Meas …
  -/
  refine MeasurePreserving.prod (volume_preserving_pi fun w ↦ ?_) (MeasurePreserving.id _)
  /-
    K : Type u_1
    inst✝¹ : Field K
    s : Set (Subtype fun w => w.IsReal)
    inst✝ : NumberField K
    w : Subtype fun w => w.IsReal
    ⊢ MeasureTheory.MeasurePreserving (↑((fun w => ite (Membership.mem s w) (Conti …
  -/
  by_cases hw : w ∈ s
    /-
      case pos
      K : Type u_1
      inst✝¹ : Field K
      s : Set (Subtype fun w => w.IsReal)
      inst✝ : NumberField K
      w : Subtype fun w => w.IsReal
      hw : Membership.mem s w
      ⊢ MeasureTheory.MeasurePreserving (↑((fun w => ite (Membership.mem s w) (Conti …
    -/
  · simp_rw [if_pos hw]
    /-
      case pos
      K : Type u_1
      inst✝¹ : Field K
      s : Set (Subtype fun w => w.IsReal)
      inst✝ : NumberField K
      w : Subtype fun w => w.IsReal
      hw : Membership.mem s w
      ⊢ MeasureTheory.MeasurePreserving (↑(ContinuousLinearEquiv.neg Real).toLinearE …
    -/
    exact Measure.measurePreserving_neg _
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_1
      inst✝¹ : Field K
      s : Set (Subtype fun w => w.IsReal)
      inst✝ : NumberField K
      w : Subtype fun w => w.IsReal
      hw : Not (Membership.mem s w)
      ⊢ MeasureTheory.MeasurePreserving (↑((fun w => ite (Membership.mem s w) (Conti …
    -/
  · simp_rw [if_neg hw]
    /-
      case neg
      K : Type u_1
      inst✝¹ : Field K
      s : Set (Subtype fun w => w.IsReal)
      inst✝ : NumberField K
      w : Subtype fun w => w.IsReal
      hw : Not (Membership.mem s w)
      ⊢ MeasureTheory.MeasurePreserving (↑(ContinuousLinearEquiv.refl Real Real).toL …
    -/
    exact MeasurePreserving.id _
    /-
      🎉 no goals
    -/


variable (s) in
/-- `negAt` preserves `normAtPlace`. -/
@[simp]
theorem normAtPlace_negAt (x : mixedSpace K) (w : InfinitePlace K) :
    normAtPlace w (negAt s x) = normAtPlace w x := by
  /-
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    x : NumberField.mixedEmbedding.mixedSpace K
    w : NumberField.InfinitePlace K
    ⊢ Eq ((NumberField.mixedEmbedding.normAtPlace w) ((NumberField.mixedEmbedding. …
  -/
  obtain hw | hw := isReal_or_isComplex w
    /-
      case inl
      K : Type u_1
      inst✝ : Field K
      s : Set (Subtype fun w => w.IsReal)
      x : NumberField.mixedEmbedding.mixedSpace K
      w : NumberField.InfinitePlace K
      hw : w.IsReal
      ⊢ Eq ((NumberField.mixedEmbedding.normAtPlace w) ((NumberField.mixedEmbedding. …
    -/
  · simp_rw [normAtPlace_apply_isReal hw, Real.norm_eq_abs, negAt_apply_abs_of_isReal]
    /-
      🎉 no goals
    -/
    /-
      case inr
      K : Type u_1
      inst✝ : Field K
      s : Set (Subtype fun w => w.IsReal)
      x : NumberField.mixedEmbedding.mixedSpace K
      w : NumberField.InfinitePlace K
      hw : w.IsComplex
      ⊢ Eq ((NumberField.mixedEmbedding.normAtPlace w) ((NumberField.mixedEmbedding. …
    -/
  · simp_rw [normAtPlace_apply_isComplex hw, negAt_apply_of_isComplex]
    /-
      🎉 no goals
    -/


/-- `negAt` preserves the `norm`. -/
@[simp]
theorem norm_negAt [NumberField K] (x : mixedSpace K) :
    mixedEmbedding.norm (negAt s x) = mixedEmbedding.norm x :=
  norm_eq_of_normAtPlace_eq (fun w ↦ normAtPlace_negAt _ _ w)


/-- `negAt` is its own inverse. -/
@[simp]
theorem negAt_symm :
    (negAt s).symm = negAt s := by
  /-
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    ⊢ Eq (NumberField.mixedEmbedding.negAt s).symm (NumberField.mixedEmbedding.neg …
  -/
  ext x w
    /-
      case h.h.fst.h
      K : Type u_1
      inst✝ : Field K
      s : Set (Subtype fun w => w.IsReal)
      x : NumberField.mixedEmbedding.mixedSpace K
      w : Subtype fun w => w.IsReal
      ⊢ Eq (((NumberField.mixedEmbedding.negAt s).symm x).1 w) (((NumberField.mixedE …
    -/
  · by_cases hw : w ∈ s
    · simp_rw [negAt_apply_of_isReal_and_mem _ hw, negAt, prod_symm,
        ContinuousLinearEquiv.prod_apply, piCongrRight_symm_apply, if_pos hw, symm_neg, neg_apply]
    · simp_rw [negAt_apply_of_isReal_and_not_mem _ hw, negAt, prod_symm,
        ContinuousLinearEquiv.prod_apply, piCongrRight_symm_apply, if_neg hw, refl_symm, refl_apply]
    /-
      case h.h.snd.h
      K : Type u_1
      inst✝ : Field K
      s : Set (Subtype fun w => w.IsReal)
      x : NumberField.mixedEmbedding.mixedSpace K
      w : Subtype fun w => w.IsComplex
      ⊢ Eq (((NumberField.mixedEmbedding.negAt s).symm x).2 w) (((NumberField.mixedE …
    -/
  · rfl
    /-
      🎉 no goals
    -/


/-- For `x : mixedSpace K`, the set `signSet x` is the set of real places `w` s.t. `x w ≤ 0`. -/
def signSet (x : mixedSpace K) : Set {w : InfinitePlace K // IsReal w} := {w | x.1 w ≤ 0}


@[simp]
theorem negAt_signSet_apply_of_isReal (x : mixedSpace K) (w : {w // IsReal w}) :
    (negAt (signSet x) x).1 w = |x.1 w| := by
  /-
    K : Type u_1
    inst✝ : Field K
    x : NumberField.mixedEmbedding.mixedSpace K
    w : Subtype fun w => w.IsReal
    ⊢ Eq (((NumberField.mixedEmbedding.negAt (NumberField.mixedEmbedding.signSet x …
  -/
  by_cases hw : x.1 w ≤ 0
    /-
      case pos
      K : Type u_1
      inst✝ : Field K
      x : NumberField.mixedEmbedding.mixedSpace K
      w : Subtype fun w => w.IsReal
      hw : LE.le (x.1 w) 0
      ⊢ Eq (((NumberField.mixedEmbedding.negAt (NumberField.mixedEmbedding.signSet x …
    -/
  · rw [negAt_apply_of_isReal_and_mem _ hw, abs_of_nonpos hw]
    /-
      🎉 no goals
    -/
    /-
      case neg
      K : Type u_1
      inst✝ : Field K
      x : NumberField.mixedEmbedding.mixedSpace K
      w : Subtype fun w => w.IsReal
      hw : Not (LE.le (x.1 w) 0)
      ⊢ Eq (((NumberField.mixedEmbedding.negAt (NumberField.mixedEmbedding.signSet x …
    -/
  · rw [negAt_apply_of_isReal_and_not_mem _ hw, abs_of_pos (lt_of_not_ge hw)]
    /-
      🎉 no goals
    -/


@[simp]
theorem negAt_signSet_apply_of_isComplex (x : mixedSpace K) (w : {w // IsComplex w}) :
    (negAt (signSet x) x).2 w = x.2 w := rfl


variable (s) in
 /-- `negAt s A` is also equal to the preimage of `A` by `negAt s`. This fact is used to simplify
 some proofs. -/
 theorem negAt_preimage :
    negAt s ⁻¹' A = negAt s '' A := by
  /-
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    ⊢ Eq (Set.preimage (⇑(NumberField.mixedEmbedding.negAt s)) A) (Set.image (⇑(Nu …
  -/
  rw [ContinuousLinearEquiv.image_eq_preimage, negAt_symm]
  /-
    🎉 no goals
  -/


/-- The `plusPart` of a subset `A` of the `mixedSpace` is the set of points in `A` that are
positive at all real places. -/
abbrev plusPart : Set (mixedSpace K) := A ∩ {x | ∀ w, 0 < x.1 w}


theorem neg_of_mem_negA_plusPart (hx : x ∈ negAt s '' (plusPart A)) {w : {w // IsReal w}}
    (hw : w ∈ s) : x.1 w < 0 := by
  /-
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : Membership.mem (Set.image (⇑(NumberField.mixedEmbedding.negAt s)) (Number …
    w : Subtype fun w => w.IsReal
    hw : Membership.mem s w
    ⊢ LT.lt (x.1 w) 0
  -/
  obtain ⟨y, hy, rfl⟩ := hx
  /-
    case intro.intro
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    w : Subtype fun w => w.IsReal
    hw : Membership.mem s w
    y : NumberField.mixedEmbedding.mixedSpace K
    hy : Membership.mem (NumberField.mixedEmbedding.plusPart A) y
    ⊢ LT.lt (((NumberField.mixedEmbedding.negAt s) y).1 w) 0
  -/
  rw [negAt_apply_of_isReal_and_mem _ hw, neg_lt_zero]
  /-
    case intro.intro
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    w : Subtype fun w => w.IsReal
    hw : Membership.mem s w
    y : NumberField.mixedEmbedding.mixedSpace K
    hy : Membership.mem (NumberField.mixedEmbedding.plusPart A) y
    ⊢ LT.lt 0 (y.1 w)
  -/
  exact hy.2 w
  /-
    🎉 no goals
  -/

 
theorem pos_of_not_mem_negAt_plusPart (hx : x ∈ negAt s '' (plusPart A)) {w : {w // IsReal w}}
    (hw : w ∉ s) : 0 < x.1 w := by
  /-
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    x : NumberField.mixedEmbedding.mixedSpace K
    hx : Membership.mem (Set.image (⇑(NumberField.mixedEmbedding.negAt s)) (Number …
    w : Subtype fun w => w.IsReal
    hw : Not (Membership.mem s w)
    ⊢ LT.lt 0 (x.1 w)
  -/
  obtain ⟨y, hy, rfl⟩ := hx
  /-
    case intro.intro
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    w : Subtype fun w => w.IsReal
    hw : Not (Membership.mem s w)
    y : NumberField.mixedEmbedding.mixedSpace K
    hy : Membership.mem (NumberField.mixedEmbedding.plusPart A) y
    ⊢ LT.lt 0 (((NumberField.mixedEmbedding.negAt s) y).1 w)
  -/
  rw [negAt_apply_of_isReal_and_not_mem _ hw]
  /-
    case intro.intro
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    w : Subtype fun w => w.IsReal
    hw : Not (Membership.mem s w)
    y : NumberField.mixedEmbedding.mixedSpace K
    hy : Membership.mem (NumberField.mixedEmbedding.plusPart A) y
    ⊢ LT.lt 0 (y.1 w)
  -/
  exact hy.2 w
  /-
    🎉 no goals
  -/

 
/-- The images of `plusPart` by `negAt` are pairwise disjoint. -/
 theorem disjoint_negAt_plusPart : Pairwise (Disjoint on (fun s ↦ negAt s '' (plusPart A))) := by
  /-
    K : Type u_1
    inst✝ : Field K
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    ⊢ Pairwise (Function.onFun Disjoint fun s => Set.image (⇑(NumberField.mixedEmb …
  -/
  intro s t hst
  /-
    K : Type u_1
    inst✝ : Field K
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    s t : Set (Subtype fun w => w.IsReal)
    hst : Ne s t
    ⊢ Function.onFun Disjoint (fun s => Set.image (⇑(NumberField.mixedEmbedding.ne …
  -/
  refine Set.disjoint_left.mpr fun _ hx hx' ↦ ?_
  obtain ⟨w, hw | hw⟩ : ∃ w, (w ∈ s ∧ w ∉ t) ∨ (w ∈ t ∧ w ∉ s) := by
    exact Set.symmDiff_nonempty.mpr hst
  · exact lt_irrefl _ <|
      (neg_of_mem_negA_plusPart A hx hw.1).trans (pos_of_not_mem_negAt_plusPart A hx' hw.2)
  · exact lt_irrefl _ <|
      (neg_of_mem_negA_plusPart A hx' hw.1).trans (pos_of_not_mem_negAt_plusPart A hx hw.2)

-- We will assume from now that `A` is symmetric at real places

include hA in
theorem mem_negAt_plusPart_of_mem (hx₁ : x ∈ A) (hx₂ : ∀ w, x.1 w ≠ 0) :
    x ∈ negAt s '' (plusPart A) ↔ (∀ w, w ∈ s → x.1 w < 0) ∧ (∀ w, w ∉ s → x.1 w > 0) := by
  refine ⟨fun hx ↦ ⟨fun _ hw ↦ neg_of_mem_negA_plusPart A hx hw,
      fun _ hw ↦ pos_of_not_mem_negAt_plusPart A hx hw⟩,
      fun ⟨h₁, h₂⟩ ↦ ⟨(fun w ↦ |x.1 w|, x.2), ⟨(hA x).mp hx₁, fun w ↦ abs_pos.mpr (hx₂ w)⟩, ?_⟩⟩
  /-
    K : Type u_1
    inst✝ : Field K
    s : Set (Subtype fun w => w.IsReal)
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    x : NumberField.mixedEmbedding.mixedSpace K
    hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
    hx₁ : Membership.mem A x
    hx₂ : ∀ (w : Subtype fun w => w.IsReal), Ne (x.1 w) 0
    x✝ : And (∀ (w : Subtype fun w => w.IsReal), Membership.mem s w → LT.lt (x.1 w …
    h₁ : ∀ (w : Subtype fun w => w.IsReal), Membership.mem s w → LT.lt (x.1 w) 0
    h₂ : ∀ (w : Subtype fun w => w.IsReal), Not (Membership.mem s w) → GT.gt (x.1  …
    ⊢ Eq ((NumberField.mixedEmbedding.negAt s) { fst := fun w => abs (x.1 w), snd  …
  -/
  ext w
    /-
      case fst.h
      K : Type u_1
      inst✝ : Field K
      s : Set (Subtype fun w => w.IsReal)
      A : Set (NumberField.mixedEmbedding.mixedSpace K)
      x : NumberField.mixedEmbedding.mixedSpace K
      hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
      hx₁ : Membership.mem A x
      hx₂ : ∀ (w : Subtype fun w => w.IsReal), Ne (x.1 w) 0
      x✝ : And (∀ (w : Subtype fun w => w.IsReal), Membership.mem s w → LT.lt (x.1 w …
      h₁ : ∀ (w : Subtype fun w => w.IsReal), Membership.mem s w → LT.lt (x.1 w) 0
      h₂ : ∀ (w : Subtype fun w => w.IsReal), Not (Membership.mem s w) → GT.gt (x.1  …
      w : Subtype fun w => w.IsReal
      ⊢ Eq (((NumberField.mixedEmbedding.negAt s) { fst := fun w => abs (x.1 w), snd …
    -/
  · by_cases hw : w ∈ s
      /-
        case pos
        K : Type u_1
        inst✝ : Field K
        s : Set (Subtype fun w => w.IsReal)
        A : Set (NumberField.mixedEmbedding.mixedSpace K)
        x : NumberField.mixedEmbedding.mixedSpace K
        hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
        hx₁ : Membership.mem A x
        hx₂ : ∀ (w : Subtype fun w => w.IsReal), Ne (x.1 w) 0
        x✝ : And (∀ (w : Subtype fun w => w.IsReal), Membership.mem s w → LT.lt (x.1 w …
        h₁ : ∀ (w : Subtype fun w => w.IsReal), Membership.mem s w → LT.lt (x.1 w) 0
        h₂ : ∀ (w : Subtype fun w => w.IsReal), Not (Membership.mem s w) → GT.gt (x.1  …
        w : Subtype fun w => w.IsReal
        hw : Membership.mem s w
        ⊢ Eq (((NumberField.mixedEmbedding.negAt s) { fst := fun w => abs (x.1 w), snd …
      -/
    · simp only [negAt_apply_of_isReal_and_mem _ hw, abs_of_neg (h₁ w hw), neg_neg]
      /-
        🎉 no goals
      -/
      /-
        case neg
        K : Type u_1
        inst✝ : Field K
        s : Set (Subtype fun w => w.IsReal)
        A : Set (NumberField.mixedEmbedding.mixedSpace K)
        x : NumberField.mixedEmbedding.mixedSpace K
        hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
        hx₁ : Membership.mem A x
        hx₂ : ∀ (w : Subtype fun w => w.IsReal), Ne (x.1 w) 0
        x✝ : And (∀ (w : Subtype fun w => w.IsReal), Membership.mem s w → LT.lt (x.1 w …
        h₁ : ∀ (w : Subtype fun w => w.IsReal), Membership.mem s w → LT.lt (x.1 w) 0
        h₂ : ∀ (w : Subtype fun w => w.IsReal), Not (Membership.mem s w) → GT.gt (x.1  …
        w : Subtype fun w => w.IsReal
        hw : Not (Membership.mem s w)
        ⊢ Eq (((NumberField.mixedEmbedding.negAt s) { fst := fun w => abs (x.1 w), snd …
      -/
    · simp only [negAt_apply_of_isReal_and_not_mem _ hw, abs_of_pos (h₂ w hw)]
      /-
        🎉 no goals
      -/
    /-
      case snd.h
      K : Type u_1
      inst✝ : Field K
      s : Set (Subtype fun w => w.IsReal)
      A : Set (NumberField.mixedEmbedding.mixedSpace K)
      x : NumberField.mixedEmbedding.mixedSpace K
      hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
      hx₁ : Membership.mem A x
      hx₂ : ∀ (w : Subtype fun w => w.IsReal), Ne (x.1 w) 0
      x✝ : And (∀ (w : Subtype fun w => w.IsReal), Membership.mem s w → LT.lt (x.1 w …
      h₁ : ∀ (w : Subtype fun w => w.IsReal), Membership.mem s w → LT.lt (x.1 w) 0
      h₂ : ∀ (w : Subtype fun w => w.IsReal), Not (Membership.mem s w) → GT.gt (x.1  …
      w : Subtype fun w => w.IsComplex
      ⊢ Eq (((NumberField.mixedEmbedding.negAt s) { fst := fun w => abs (x.1 w), snd …
    -/
  · rfl
    /-
      🎉 no goals
    -/


include hA in
/-- Assume that `A`  is symmetric at real places then, the union of the images of `plusPart`
by `negAt` and of the set of elements of `A` that are zero at at least one real place
is equal to `A`. -/
theorem iUnion_negAt_plusPart_union :
    (⋃ s, negAt s '' (plusPart A)) ∪ (A ∩ (⋃ w, {x | x.1 w = 0})) = A := by
  /-
    K : Type u_1
    inst✝ : Field K
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
    ⊢ Eq (Union.union (Set.iUnion fun s => Set.image (⇑(NumberField.mixedEmbedding …
  -/
  ext x
  /-
    case h
    K : Type u_1
    inst✝ : Field K
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Membership.mem (Union.union (Set.iUnion fun s => Set.image (⇑(NumberFie …
  -/
  rw [Set.mem_union, Set.mem_inter_iff, Set.mem_iUnion, Set.mem_iUnion]
  /-
    case h
    K : Type u_1
    inst✝ : Field K
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
    x : NumberField.mixedEmbedding.mixedSpace K
    ⊢ Iff (Or (Exists fun i => Membership.mem (Set.image (⇑(NumberField.mixedEmbed …
  -/
  refine ⟨?_, fun h ↦ ?_⟩
    /-
      case h.refine_1
      K : Type u_1
      inst✝ : Field K
      A : Set (NumberField.mixedEmbedding.mixedSpace K)
      hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
      x : NumberField.mixedEmbedding.mixedSpace K
      ⊢ Or (Exists fun i => Membership.mem (Set.image (⇑(NumberField.mixedEmbedding. …
    -/
  · rintro (⟨s, ⟨x, ⟨hx, _⟩, rfl⟩⟩ | h)
      /-
        case h.refine_1.inl.intro.intro.intro.intro
        K : Type u_1
        inst✝ : Field K
        A : Set (NumberField.mixedEmbedding.mixedSpace K)
        hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
        s : Set (Subtype fun w => w.IsReal)
        x : NumberField.mixedEmbedding.mixedSpace K
        hx : Membership.mem A x
        right✝ : Membership.mem (setOf fun x => ∀ (w : Subtype fun w => w.IsReal), LT. …
        ⊢ Membership.mem A ((NumberField.mixedEmbedding.negAt s) x)
      -/
    · simp_rw (config := {singlePass := true}) [hA, negAt_apply_abs_of_isReal, negAt_apply_snd]
      /-
        case h.refine_1.inl.intro.intro.intro.intro
        K : Type u_1
        inst✝ : Field K
        A : Set (NumberField.mixedEmbedding.mixedSpace K)
        hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
        s : Set (Subtype fun w => w.IsReal)
        x : NumberField.mixedEmbedding.mixedSpace K
        hx : Membership.mem A x
        right✝ : Membership.mem (setOf fun x => ∀ (w : Subtype fun w => w.IsReal), LT. …
        ⊢ Membership.mem A { fst := fun w => abs (x.1 w), snd := x.2 }
      -/
      rwa [← hA]
      /-
        🎉 no goals
      -/
      /-
        case h.refine_1.inr
        K : Type u_1
        inst✝ : Field K
        A : Set (NumberField.mixedEmbedding.mixedSpace K)
        hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
        x : NumberField.mixedEmbedding.mixedSpace K
        h : And (Membership.mem A x) (Exists fun i => Membership.mem (setOf fun x => E …
        ⊢ Membership.mem A x
      -/
    · exact h.left
      /-
        🎉 no goals
      -/
    /-
      case h.refine_2
      K : Type u_1
      inst✝ : Field K
      A : Set (NumberField.mixedEmbedding.mixedSpace K)
      hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
      x : NumberField.mixedEmbedding.mixedSpace K
      h : Membership.mem A x
      ⊢ Or (Exists fun i => Membership.mem (Set.image (⇑(NumberField.mixedEmbedding. …
    -/
  · obtain hx | hx := exists_or_forall_not (fun w ↦ x.1 w = 0)
      /-
        case h.refine_2.inl
        K : Type u_1
        inst✝ : Field K
        A : Set (NumberField.mixedEmbedding.mixedSpace K)
        hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
        x : NumberField.mixedEmbedding.mixedSpace K
        h : Membership.mem A x
        hx : Exists fun a => Eq (x.1 a) 0
        ⊢ Or (Exists fun i => Membership.mem (Set.image (⇑(NumberField.mixedEmbedding. …
      -/
    · exact Or.inr ⟨h, hx⟩
      /-
        🎉 no goals
      -/
    · refine Or.inl ⟨signSet x,
        (mem_negAt_plusPart_of_mem A hA h hx).mpr ⟨fun w hw ↦ ?_, fun w hw ↦ ?_⟩⟩
        /-
          case h.refine_2.inr.refine_1
          K : Type u_1
          inst✝ : Field K
          A : Set (NumberField.mixedEmbedding.mixedSpace K)
          hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
          x : NumberField.mixedEmbedding.mixedSpace K
          h : Membership.mem A x
          hx : ∀ (a : Subtype fun w => w.IsReal), Not (Eq (x.1 a) 0)
          w : Subtype fun w => w.IsReal
          hw : Membership.mem (NumberField.mixedEmbedding.signSet x) w
          ⊢ LT.lt (x.1 w) 0
        -/
      · exact lt_of_le_of_ne hw (hx w)
        /-
          🎉 no goals
        -/
        /-
          case h.refine_2.inr.refine_2
          K : Type u_1
          inst✝ : Field K
          A : Set (NumberField.mixedEmbedding.mixedSpace K)
          hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
          x : NumberField.mixedEmbedding.mixedSpace K
          h : Membership.mem A x
          hx : ∀ (a : Subtype fun w => w.IsReal), Not (Eq (x.1 a) 0)
          w : Subtype fun w => w.IsReal
          hw : Not (Membership.mem (NumberField.mixedEmbedding.signSet x) w)
          ⊢ GT.gt (x.1 w) 0
        -/
      · exact lt_of_le_of_ne (lt_of_not_ge hw).le (Ne.symm (hx w))
        /-
          🎉 no goals
        -/


include hA in
open Classical in
theorem iUnion_negAt_plusPart_ae :
    ⋃ s, negAt s '' (plusPart A) =ᵐ[volume] A := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
    inst✝ : NumberField K
    ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (Set.iUnio …
  -/
  nth_rewrite 2 [← iUnion_negAt_plusPart_union A hA]
  /-
    K : Type u_1
    inst✝¹ : Field K
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    hA : ∀ (x : NumberField.mixedEmbedding.mixedSpace K), Iff (Membership.mem A x) …
    inst✝ : NumberField K
    ⊢ (MeasureTheory.ae MeasureTheory.MeasureSpace.volume).EventuallyEq (Set.iUnio …
  -/
  refine (MeasureTheory.union_ae_eq_left_of_ae_eq_empty (ae_eq_empty.mpr ?_)).symm
  exact measure_mono_null Set.inter_subset_right
    (measure_iUnion_null_iff.mpr fun _ ↦ volume_eq_zero _)


variable {A} in
theorem measurableSet_plusPart (hm : MeasurableSet A) :
    MeasurableSet (plusPart A) := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    A : Set (NumberField.mixedEmbedding.mixedSpace K)
    inst✝ : NumberField K
    hm : MeasurableSet A
    ⊢ MeasurableSet (NumberField.mixedEmbedding.plusPart A)
  -/
  convert_to MeasurableSet (A ∩ (⋂ w, {x | 0 < x.1 w}))
    /-
      case h.e'_3
      K : Type u_1
      inst✝¹ : Field K
      A : Set (NumberField.mixedEmbedding.mixedSpace K)
      inst✝ : NumberField K
      hm : MeasurableSet A
      ⊢ Eq (NumberField.mixedEmbedding.plusPart A) (Inter.inter A (Set.iInter fun w  …
    -/
  · ext; simp
         /-
           🎉 no goals
         -/
    /-
      K : Type u_1
      inst✝¹ : Field K
      A : Set (NumberField.mixedEmbedding.mixedSpace K)
      inst✝ : NumberField K
      hm : MeasurableSet A
      ⊢ MeasurableSet (Inter.inter A (Set.iInter fun w => setOf fun x => LT.lt 0 (x. …
    -/
  · refine hm.inter (MeasurableSet.iInter fun _ ↦ ?_)
    /-
      K : Type u_1
      inst✝¹ : Field K
      A : Set (NumberField.mixedEmbedding.mixedSpace K)
      inst✝ : NumberField K
      hm : MeasurableSet A
      x✝ : Subtype fun w => w.IsReal
      ⊢ MeasurableSet (setOf fun x => LT.lt 0 (x.1 x✝))
    -/
    exact measurableSet_lt measurable_const ((measurable_pi_apply _).comp' measurable_fst)
    /-
      🎉 no goals
    -/


variable (s) in
theorem measurableSet_negAt_plusPart (hm : MeasurableSet A) :
    MeasurableSet (negAt s '' (plusPart A)) :=
  negAt_preimage s _ ▸ (measurableSet_plusPart hm).preimage (negAt s).continuous.measurable


open Classical in
/-- The image of the `plusPart` of `A` by `negAt` have all the same volume as `plusPart A`. -/
theorem volume_negAt_plusPart (hm : MeasurableSet A) :
    volume (negAt s '' (plusPart A)) = volume (plusPart A) := by
  rw [← negAt_symm, ContinuousLinearEquiv.image_symm_eq_preimage,
    volume_preserving_negAt.measure_preimage (measurableSet_plusPart hm).nullMeasurableSet]


include hA in
open Classical in
/-- If a subset `A` of the `mixedSpace` is symmetric at real places, then its volume is
`2^ nrRealPlaces K` times the volume of its `plusPart`. -/
theorem volume_eq_two_pow_mul_volume_plusPart (hm : MeasurableSet A) :
    volume A = 2 ^ nrRealPlaces K * volume (plusPart A) := by
  simp only [← measure_congr (iUnion_negAt_plusPart_ae A hA),
    measure_iUnion (disjoint_negAt_plusPart A) (fun _ ↦ measurableSet_negAt_plusPart _ A hm),
    volume_negAt_plusPart _ hm, tsum_fintype, sum_const, card_univ, Fintype.card_set, nsmul_eq_mul,
    Nat.cast_pow, Nat.cast_ofNat, nrRealPlaces]


