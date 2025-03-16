theorem borel_eq_generateFrom_Ioo_rat :
    borel ℝ = .generateFrom (⋃ (a : ℚ) (b : ℚ) (_ : a < b), {Ioo (a : ℝ) (b : ℝ)}) :=
  isTopologicalBasis_Ioo_rat.borel_eq_generateFrom


theorem borel_eq_generateFrom_Iio_rat : borel ℝ = .generateFrom (⋃ a : ℚ, {Iio (a : ℝ)}) := by
  /-
    ⊢ Eq (borel Real) (MeasurableSpace.generateFrom (Set.iUnion fun a => Singleton …
  -/
  rw [borel_eq_generateFrom_Iio]
  refine le_antisymm
    (generateFrom_le ?_)
    (generateFrom_mono <| iUnion_subset fun q ↦ singleton_subset_iff.mpr <| mem_range_self _)
  /-
    ⊢ ∀ (t : Set Real), Membership.mem (Set.range Set.Iio) t → MeasurableSet t
  -/
  rintro _ ⟨a, rfl⟩
  have : IsLUB (range ((↑) : ℚ → ℝ) ∩ Iio a) a := by
    simp [isLUB_iff_le_iff, mem_upperBounds, ← le_iff_forall_rat_lt_imp_le]
  /-
    case intro
    a : Real
    this : IsLUB (Inter.inter (Set.range Rat.cast) (Set.Iio a)) a
    ⊢ MeasurableSet (Set.Iio a)
  -/
  rw [← this.biUnion_Iio_eq, ← image_univ, ← image_inter_preimage, univ_inter, biUnion_image]
  exact MeasurableSet.biUnion (to_countable _)
    fun b _ => GenerateMeasurable.basic (Iio (b : ℝ)) (by simp)


theorem borel_eq_generateFrom_Ioi_rat : borel ℝ = .generateFrom (⋃ a : ℚ, {Ioi (a : ℝ)}) := by
  /-
    ⊢ Eq (borel Real) (MeasurableSpace.generateFrom (Set.iUnion fun a => Singleton …
  -/
  rw [borel_eq_generateFrom_Ioi]
  refine le_antisymm
    (generateFrom_le ?_)
    (generateFrom_mono <| iUnion_subset fun q ↦ singleton_subset_iff.mpr <| mem_range_self _)
  /-
    ⊢ ∀ (t : Set Real), Membership.mem (Set.range Set.Ioi) t → MeasurableSet t
  -/
  rintro _ ⟨a, rfl⟩
  have : IsGLB (range ((↑) : ℚ → ℝ) ∩ Ioi a) a := by
    simp [isGLB_iff_le_iff, mem_lowerBounds, ← le_iff_forall_lt_rat_imp_le]
  /-
    case intro
    a : Real
    this : IsGLB (Inter.inter (Set.range Rat.cast) (Set.Ioi a)) a
    ⊢ MeasurableSet (Set.Ioi a)
  -/
  rw [← this.biUnion_Ioi_eq, ← image_univ, ← image_inter_preimage, univ_inter, biUnion_image]
  exact MeasurableSet.biUnion (to_countable _)
    fun b _ => GenerateMeasurable.basic (Ioi (b : ℝ)) (by simp)


theorem borel_eq_generateFrom_Iic_rat : borel ℝ = .generateFrom (⋃ a : ℚ, {Iic (a : ℝ)}) := by
  /-
    ⊢ Eq (borel Real) (MeasurableSpace.generateFrom (Set.iUnion fun a => Singleton …
  -/
  rw [borel_eq_generateFrom_Ioi_rat, iUnion_singleton_eq_range, iUnion_singleton_eq_range]
  refine le_antisymm (generateFrom_le ?_) (generateFrom_le ?_) <;>
  rintro _ ⟨q, rfl⟩ <;>
  dsimp only <;>
  [rw [← compl_Iic]; rw [← compl_Ioi]] <;>
  /-
    case refine_1.intro
    q : Rat
    ⊢ MeasurableSet (HasCompl.compl (Set.Iic ↑q))
  -/
  /-
    🎉 no goals
  -/
  exact MeasurableSet.compl (GenerateMeasurable.basic _ (mem_range_self q))
  /-
    🎉 no goals
  -/


theorem borel_eq_generateFrom_Ici_rat : borel ℝ = .generateFrom (⋃ a : ℚ, {Ici (a : ℝ)}) := by
  /-
    ⊢ Eq (borel Real) (MeasurableSpace.generateFrom (Set.iUnion fun a => Singleton …
  -/
  rw [borel_eq_generateFrom_Iio_rat, iUnion_singleton_eq_range, iUnion_singleton_eq_range]
  refine le_antisymm (generateFrom_le ?_) (generateFrom_le ?_) <;>
  rintro _ ⟨q, rfl⟩ <;>
  dsimp only <;>
  [rw [← compl_Ici]; rw [← compl_Iio]] <;>
  /-
    case refine_1.intro
    q : Rat
    ⊢ MeasurableSet (HasCompl.compl (Set.Ici ↑q))
  -/
  /-
    🎉 no goals
  -/
  exact MeasurableSet.compl (GenerateMeasurable.basic _ (mem_range_self q))
  /-
    🎉 no goals
  -/


theorem isPiSystem_Ioo_rat :
    IsPiSystem (⋃ (a : ℚ) (b : ℚ) (_ : a < b), {Ioo (a : ℝ) (b : ℝ)}) := by
  /-
    ⊢ IsPiSystem (Set.iUnion fun a => Set.iUnion fun b => Set.iUnion fun x => Sing …
  -/
  convert isPiSystem_Ioo ((↑) : ℚ → ℝ) ((↑) : ℚ → ℝ)
  /-
    case h.e'_2
    ⊢ Eq (Set.iUnion fun a => Set.iUnion fun b => Set.iUnion fun x => Singleton.si …
  -/
  ext x
  /-
    case h.e'_2.h
    x : Set Real
    ⊢ Iff (Membership.mem (Set.iUnion fun a => Set.iUnion fun b => Set.iUnion fun  …
  -/
  simp [eq_comm]
  /-
    🎉 no goals
  -/


theorem isPiSystem_Iio_rat : IsPiSystem (⋃ a : ℚ, {Iio (a : ℝ)}) := by
  /-
    ⊢ IsPiSystem (Set.iUnion fun a => Singleton.singleton (Set.Iio ↑a))
  -/
  convert isPiSystem_image_Iio (((↑) : ℚ → ℝ) '' univ)
  /-
    case h.e'_2
    ⊢ Eq (Set.iUnion fun a => Singleton.singleton (Set.Iio ↑a)) (Set.image Set.Iio …
  -/
  ext x
  /-
    case h.e'_2.h
    x : Set Real
    ⊢ Iff (Membership.mem (Set.iUnion fun a => Singleton.singleton (Set.Iio ↑a)) x …
  -/
  simp only [iUnion_singleton_eq_range, mem_range, image_univ, mem_image, exists_exists_eq_and]
  /-
    🎉 no goals
  -/


theorem isPiSystem_Ioi_rat : IsPiSystem (⋃ a : ℚ, {Ioi (a : ℝ)}) := by
  /-
    ⊢ IsPiSystem (Set.iUnion fun a => Singleton.singleton (Set.Ioi ↑a))
  -/
  convert isPiSystem_image_Ioi (((↑) : ℚ → ℝ) '' univ)
  /-
    case h.e'_2
    ⊢ Eq (Set.iUnion fun a => Singleton.singleton (Set.Ioi ↑a)) (Set.image Set.Ioi …
  -/
  ext x
  /-
    case h.e'_2.h
    x : Set Real
    ⊢ Iff (Membership.mem (Set.iUnion fun a => Singleton.singleton (Set.Ioi ↑a)) x …
  -/
  simp only [iUnion_singleton_eq_range, mem_range, image_univ, mem_image, exists_exists_eq_and]
  /-
    🎉 no goals
  -/


theorem isPiSystem_Iic_rat : IsPiSystem (⋃ a : ℚ, {Iic (a : ℝ)}) := by
  /-
    ⊢ IsPiSystem (Set.iUnion fun a => Singleton.singleton (Set.Iic ↑a))
  -/
  convert isPiSystem_image_Iic (((↑) : ℚ → ℝ) '' univ)
  /-
    case h.e'_2
    ⊢ Eq (Set.iUnion fun a => Singleton.singleton (Set.Iic ↑a)) (Set.image Set.Iic …
  -/
  ext x
  /-
    case h.e'_2.h
    x : Set Real
    ⊢ Iff (Membership.mem (Set.iUnion fun a => Singleton.singleton (Set.Iic ↑a)) x …
  -/
  simp only [iUnion_singleton_eq_range, mem_range, image_univ, mem_image, exists_exists_eq_and]
  /-
    🎉 no goals
  -/


theorem isPiSystem_Ici_rat : IsPiSystem (⋃ a : ℚ, {Ici (a : ℝ)}) := by
  /-
    ⊢ IsPiSystem (Set.iUnion fun a => Singleton.singleton (Set.Ici ↑a))
  -/
  convert isPiSystem_image_Ici (((↑) : ℚ → ℝ) '' univ)
  /-
    case h.e'_2
    ⊢ Eq (Set.iUnion fun a => Singleton.singleton (Set.Ici ↑a)) (Set.image Set.Ici …
  -/
  ext x
  /-
    case h.e'_2.h
    x : Set Real
    ⊢ Iff (Membership.mem (Set.iUnion fun a => Singleton.singleton (Set.Ici ↑a)) x …
  -/
  simp only [iUnion_singleton_eq_range, mem_range, image_univ, mem_image, exists_exists_eq_and]
  /-
    🎉 no goals
  -/


/-- The intervals `(-(n + 1), (n + 1))` form a finite spanning sets in the set of open intervals
with rational endpoints for a locally finite measure `μ` on `ℝ`. -/
def finiteSpanningSetsInIooRat (μ : Measure ℝ) [IsLocallyFiniteMeasure μ] :
    μ.FiniteSpanningSetsIn (⋃ (a : ℚ) (b : ℚ) (_ : a < b), {Ioo (a : ℝ) (b : ℝ)}) where
  set n := Ioo (-(n + 1)) (n + 1)
  set_mem n := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      n : Nat
      ⊢ Membership.mem (Set.iUnion fun a => Set.iUnion fun b => Set.iUnion fun x =>  …
    -/
    simp only [mem_iUnion, mem_singleton_iff]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      n : Nat
      ⊢ Exists fun i => Exists fun i_1 => Exists fun h => Eq (Set.Ioo (Neg.neg (HAdd …
    -/
    refine ⟨-(n + 1 : ℕ), n + 1, ?_, by simp⟩
    -- TODO: norm_cast fails here?
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      n : Nat
      ⊢ LT.lt (Neg.neg ↑(HAdd.hAdd n 1)) (HAdd.hAdd (↑n) 1)
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      n : Nat
      ⊢ LT.lt (Neg.neg (HAdd.hAdd (↑n) 1)) (HAdd.hAdd (↑n) 1)
    -/
    exact neg_lt_self n.cast_add_one_pos
    /-
      🎉 no goals
    -/
  finite _ := measure_Ioo_lt_top
  spanning :=
    iUnion_eq_univ_iff.2 fun x =>
      ⟨⌊|x|⌋₊, neg_lt.1 ((neg_le_abs x).trans_lt (Nat.lt_floor_add_one _)),
        (le_abs_self x).trans_lt (Nat.lt_floor_add_one _)⟩


theorem measure_ext_Ioo_rat {μ ν : Measure ℝ} [IsLocallyFiniteMeasure μ]
    (h : ∀ a b : ℚ, μ (Ioo a b) = ν (Ioo a b)) : μ = ν :=
  (finiteSpanningSetsInIooRat μ).ext borel_eq_generateFrom_Ioo_rat isPiSystem_Ioo_rat <| by
    /-
      μ ν : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      h : ∀ (a b : Rat), Eq (μ (Set.Ioo ↑a ↑b)) (ν (Set.Ioo ↑a ↑b))
      ⊢ ∀ (s : Set Real), Membership.mem (Set.iUnion fun a => Set.iUnion fun b => Se …
    -/
    simp only [mem_iUnion, mem_singleton_iff]
    /-
      μ ν : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      h : ∀ (a b : Rat), Eq (μ (Set.Ioo ↑a ↑b)) (ν (Set.Ioo ↑a ↑b))
      ⊢ ∀ (s : Set Real), (Exists fun i => Exists fun i_1 => Exists fun h => Eq s (S …
    -/
    rintro _ ⟨a, b, -, rfl⟩
    /-
      case intro.intro.intro
      μ ν : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsLocallyFiniteMeasure μ
      h : ∀ (a b : Rat), Eq (μ (Set.Ioo ↑a ↑b)) (ν (Set.Ioo ↑a ↑b))
      a b : Rat
      ⊢ Eq (μ (Set.Ioo ↑a ↑b)) (ν (Set.Ioo ↑a ↑b))
    -/
    apply h
    /-
      🎉 no goals
    -/


@[measurability, fun_prop]
theorem measurable_real_toNNReal : Measurable Real.toNNReal :=
  continuous_real_toNNReal.measurable


@[measurability, fun_prop]
theorem Measurable.real_toNNReal {f : α → ℝ} (hf : Measurable f) :
    Measurable fun x => Real.toNNReal (f x) :=
  measurable_real_toNNReal.comp hf


@[measurability, fun_prop]
theorem AEMeasurable.real_toNNReal {f : α → ℝ} {μ : Measure α} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x => Real.toNNReal (f x)) μ :=
  measurable_real_toNNReal.comp_aemeasurable hf


@[measurability]
theorem measurable_coe_nnreal_real : Measurable ((↑) : ℝ≥0 → ℝ) :=
  NNReal.continuous_coe.measurable


@[measurability, fun_prop]
theorem Measurable.coe_nnreal_real {f : α → ℝ≥0} (hf : Measurable f) :
    Measurable fun x => (f x : ℝ) :=
  measurable_coe_nnreal_real.comp hf


@[measurability, fun_prop]
theorem AEMeasurable.coe_nnreal_real {f : α → ℝ≥0} {μ : Measure α} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x => (f x : ℝ)) μ :=
  measurable_coe_nnreal_real.comp_aemeasurable hf


@[measurability]
theorem measurable_coe_nnreal_ennreal : Measurable ((↑) : ℝ≥0 → ℝ≥0∞) :=
  ENNReal.continuous_coe.measurable


@[measurability, fun_prop]
theorem Measurable.coe_nnreal_ennreal {f : α → ℝ≥0} (hf : Measurable f) :
    Measurable fun x => (f x : ℝ≥0∞) :=
  ENNReal.continuous_coe.measurable.comp hf


@[measurability, fun_prop]
theorem AEMeasurable.coe_nnreal_ennreal {f : α → ℝ≥0} {μ : Measure α} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x => (f x : ℝ≥0∞)) μ :=
  ENNReal.continuous_coe.measurable.comp_aemeasurable hf


@[measurability, fun_prop]
theorem Measurable.ennreal_ofReal {f : α → ℝ} (hf : Measurable f) :
    Measurable fun x => ENNReal.ofReal (f x) :=
  ENNReal.continuous_ofReal.measurable.comp hf


@[measurability, fun_prop]
lemma AEMeasurable.ennreal_ofReal {f : α → ℝ} {μ : Measure α} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x ↦ ENNReal.ofReal (f x)) μ :=
  ENNReal.continuous_ofReal.measurable.comp_aemeasurable hf


@[simp, norm_cast]
theorem measurable_coe_nnreal_real_iff {f : α → ℝ≥0} :
    Measurable (fun x => f x : α → ℝ) ↔ Measurable f :=
               /-
                 α : Type u_1
                 mα : MeasurableSpace α
                 f : α → NNReal
                 h : Measurable fun x => ↑(f x)
                 ⊢ Measurable f
               -/
  ⟨fun h => by simpa only [Real.toNNReal_coe] using h.real_toNNReal, Measurable.coe_nnreal_real⟩
               /-
                 🎉 no goals
               -/


@[simp, norm_cast]
theorem aemeasurable_coe_nnreal_real_iff {f : α → ℝ≥0} {μ : Measure α} :
    AEMeasurable (fun x => f x : α → ℝ) μ ↔ AEMeasurable f μ :=
              /-
                α : Type u_1
                mα : MeasurableSpace α
                f : α → NNReal
                μ : MeasureTheory.Measure α
                h : AEMeasurable (fun x => ↑(f x)) μ
                ⊢ AEMeasurable f μ
              -/
  ⟨fun h ↦ by simpa only [Real.toNNReal_coe] using h.real_toNNReal, AEMeasurable.coe_nnreal_real⟩
              /-
                🎉 no goals
              -/


@[deprecated (since := "2024-03-02")]
alias aEMeasurable_coe_nnreal_real_iff := aemeasurable_coe_nnreal_real_iff


/-- The set of finite `ℝ≥0∞` numbers is `MeasurableEquiv` to `ℝ≥0`. -/
def MeasurableEquiv.ennrealEquivNNReal : { r : ℝ≥0∞ | r ≠ ∞ } ≃ᵐ ℝ≥0 :=
  ENNReal.neTopHomeomorphNNReal.toMeasurableEquiv


theorem measurable_of_measurable_nnreal {f : ℝ≥0∞ → α} (h : Measurable fun p : ℝ≥0 => f p) :
    Measurable f :=
  measurable_of_measurable_on_compl_singleton ∞
    (MeasurableEquiv.ennrealEquivNNReal.symm.measurable_comp_iff.1 h)


/-- `ℝ≥0∞` is `MeasurableEquiv` to `ℝ≥0 ⊕ Unit`. -/
def ennrealEquivSum : ℝ≥0∞ ≃ᵐ ℝ≥0 ⊕ Unit :=
  { Equiv.optionEquivSumPUnit ℝ≥0 with
    measurable_toFun := measurable_of_measurable_nnreal measurable_inl
    measurable_invFun :=
      measurable_sum measurable_coe_nnreal_ennreal (@measurable_const ℝ≥0∞ Unit _ _ ∞) }


theorem measurable_of_measurable_nnreal_prod {_ : MeasurableSpace β} {_ : MeasurableSpace γ}
    {f : ℝ≥0∞ × β → γ} (H₁ : Measurable fun p : ℝ≥0 × β => f (p.1, p.2))
    (H₂ : Measurable fun x => f (∞, x)) : Measurable f :=
  let e : ℝ≥0∞ × β ≃ᵐ (ℝ≥0 × β) ⊕ (Unit × β) :=
    (ennrealEquivSum.prodCongr (MeasurableEquiv.refl β)).trans
      (MeasurableEquiv.sumProdDistrib _ _ _)
  e.symm.measurable_comp_iff.1 <| measurable_sum H₁ (H₂.comp measurable_id.snd)


theorem measurable_of_measurable_nnreal_nnreal {_ : MeasurableSpace β} {f : ℝ≥0∞ × ℝ≥0∞ → β}
    (h₁ : Measurable fun p : ℝ≥0 × ℝ≥0 => f (p.1, p.2)) (h₂ : Measurable fun r : ℝ≥0 => f (∞, r))
    (h₃ : Measurable fun r : ℝ≥0 => f (r, ∞)) : Measurable f :=
  measurable_of_measurable_nnreal_prod
    (measurable_swap_iff.1 <| measurable_of_measurable_nnreal_prod (h₁.comp measurable_swap) h₃)
    (measurable_of_measurable_nnreal h₂)


@[measurability]
theorem measurable_ofReal : Measurable ENNReal.ofReal :=
  ENNReal.continuous_ofReal.measurable


@[measurability]
theorem measurable_toReal : Measurable ENNReal.toReal :=
  ENNReal.measurable_of_measurable_nnreal measurable_coe_nnreal_real


@[measurability]
theorem measurable_toNNReal : Measurable ENNReal.toNNReal :=
  ENNReal.measurable_of_measurable_nnreal measurable_id


instance instMeasurableMul₂ : MeasurableMul₂ ℝ≥0∞ := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    δ : Type u_4
    ι : Sort y
    s t u : Set α
    mα : MeasurableSpace α
    ⊢ MeasurableMul₂ ENNReal
  -/
  refine ⟨measurable_of_measurable_nnreal_nnreal ?_ ?_ ?_⟩
    /-
      case refine_1
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      mα : MeasurableSpace α
      ⊢ Measurable fun p => HMul.hMul { fst := ↑p.1, snd := ↑p.2 }.1 { fst := ↑p.1,  …
    -/
  · simp only [← ENNReal.coe_mul, measurable_mul.coe_nnreal_ennreal]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      mα : MeasurableSpace α
      ⊢ Measurable fun r => HMul.hMul { fst := Top.top, snd := ↑r }.1 { fst := Top.t …
    -/
  · simp only [ENNReal.top_mul', ENNReal.coe_eq_zero]
    /-
      case refine_2
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      mα : MeasurableSpace α
      ⊢ Measurable fun r => ite (Eq r 0) 0 Top.top
    -/
    exact measurable_const.piecewise (measurableSet_singleton _) measurable_const
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      mα : MeasurableSpace α
      ⊢ Measurable fun r => HMul.hMul { fst := ↑r, snd := Top.top }.1 { fst := ↑r, s …
    -/
  · simp only [ENNReal.mul_top', ENNReal.coe_eq_zero]
    /-
      case refine_3
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      mα : MeasurableSpace α
      ⊢ Measurable fun r => ite (Eq r 0) 0 Top.top
    -/
    exact measurable_const.piecewise (measurableSet_singleton _) measurable_const
    /-
      🎉 no goals
    -/


instance instMeasurableSub₂ : MeasurableSub₂ ℝ≥0∞ :=
  ⟨by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      mα : MeasurableSpace α
      ⊢ Measurable fun p => HSub.hSub p.1 p.2
    -/
    apply measurable_of_measurable_nnreal_nnreal <;>
      /-
        case h₁
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        δ : Type u_4
        ι : Sort y
        s t u : Set α
        mα : MeasurableSpace α
        ⊢ Measurable fun p => HSub.hSub { fst := ↑p.1, snd := ↑p.2 }.1 { fst := ↑p.1,  …
      -/
      /-
        🎉 no goals
      -/
      simp [← WithTop.coe_sub, tsub_eq_zero_of_le];
      /-
        🎉 no goals
      -/
        /-
          case h₁
          α : Type u_1
          β : Type u_2
          γ : Type u_3
          δ : Type u_4
          ι : Sort y
          s t u : Set α
          mα : MeasurableSpace α
          ⊢ Measurable fun p => HSub.hSub ↑p.1 ↑p.2
        -/
        exact continuous_sub.measurable.coe_nnreal_ennreal⟩
        /-
          🎉 no goals
        -/


instance instMeasurableInv : MeasurableInv ℝ≥0∞ :=
  ⟨continuous_inv.measurable⟩


instance : MeasurableSMul ℝ≥0 ℝ≥0∞ where
  measurable_const_smul := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      mα : MeasurableSpace α
      ⊢ ∀ (c : NNReal), Measurable fun x => HSMul.hSMul c x
    -/
    simp_rw [ENNReal.smul_def]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      mα : MeasurableSpace α
      ⊢ ∀ (c : NNReal), Measurable fun x => HSMul.hSMul (↑c) x
    -/
    exact fun _ ↦ MeasurableSMul.measurable_const_smul _
    /-
      🎉 no goals
    -/
  measurable_smul_const := fun x ↦ by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      mα : MeasurableSpace α
      x : ENNReal
      ⊢ Measurable fun x_1 => HSMul.hSMul x_1 x
    -/
    simp_rw [ENNReal.smul_def]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α
      mα : MeasurableSpace α
      x : ENNReal
      ⊢ Measurable fun x_1 => HSMul.hSMul (↑x_1) x
    -/
    exact measurable_coe_nnreal_ennreal.mul_const _
    /-
      🎉 no goals
    -/


/-- A limit (over a general filter) of measurable `ℝ≥0∞` valued functions is measurable. -/
theorem measurable_of_tendsto' {ι : Type*} {f : ι → α → ℝ≥0∞} {g : α → ℝ≥0∞} (u : Filter ι)
    [NeBot u] [IsCountablyGenerated u] (hf : ∀ i, Measurable (f i)) (lim : Tendsto f u (𝓝 g)) :
    Measurable g := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → ENNReal
    g : α → ENNReal
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : Filter.Tendsto f u (nhds g)
    ⊢ Measurable g
  -/
  rcases u.exists_seq_tendsto with ⟨x, hx⟩
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → ENNReal
    g : α → ENNReal
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : Filter.Tendsto f u (nhds g)
    x : Nat → ι
    hx : Filter.Tendsto x Filter.atTop u
    ⊢ Measurable g
  -/
  rw [tendsto_pi_nhds] at lim
  have : (fun y => liminf (fun n => (f (x n) y : ℝ≥0∞)) atTop) = g := by
    ext1 y
    exact ((lim y).comp hx).liminf_eq
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → ENNReal
    g : α → ENNReal
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : ∀ (x : α), Filter.Tendsto (fun i => f i x) u (nhds (g x))
    x : Nat → ι
    hx : Filter.Tendsto x Filter.atTop u
    this : Eq (fun y => Filter.liminf (fun n => f (x n) y) Filter.atTop) g
    ⊢ Measurable g
  -/
  rw [← this]
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → ENNReal
    g : α → ENNReal
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : ∀ (x : α), Filter.Tendsto (fun i => f i x) u (nhds (g x))
    x : Nat → ι
    hx : Filter.Tendsto x Filter.atTop u
    this : Eq (fun y => Filter.liminf (fun n => f (x n) y) Filter.atTop) g
    ⊢ Measurable fun y => Filter.liminf (fun n => f (x n) y) Filter.atTop
  -/
  show Measurable fun y => liminf (fun n => (f (x n) y : ℝ≥0∞)) atTop
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → ENNReal
    g : α → ENNReal
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : ∀ (x : α), Filter.Tendsto (fun i => f i x) u (nhds (g x))
    x : Nat → ι
    hx : Filter.Tendsto x Filter.atTop u
    this : Eq (fun y => Filter.liminf (fun n => f (x n) y) Filter.atTop) g
    ⊢ Measurable fun y => Filter.liminf (fun n => f (x n) y) Filter.atTop
  -/
  exact .liminf fun n => hf (x n)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-09")] alias
_root_.measurable_of_tendsto_ennreal' := ENNReal.measurable_of_tendsto'


/-- A sequential limit of measurable `ℝ≥0∞` valued functions is measurable. -/
theorem measurable_of_tendsto {f : ℕ → α → ℝ≥0∞} {g : α → ℝ≥0∞} (hf : ∀ i, Measurable (f i))
    (lim : Tendsto f atTop (𝓝 g)) : Measurable g :=
  measurable_of_tendsto' atTop hf lim


@[deprecated (since := "2024-03-09")] alias
_root_.measurable_of_tendsto_ennreal := ENNReal.measurable_of_tendsto


/-- A limit (over a general filter) of a.e.-measurable `ℝ≥0∞` valued functions is
a.e.-measurable. -/
lemma aemeasurable_of_tendsto' {ι : Type*} {f : ι → α → ℝ≥0∞} {g : α → ℝ≥0∞}
    {μ : Measure α} (u : Filter ι) [NeBot u] [IsCountablyGenerated u]
    (hf : ∀ i, AEMeasurable (f i) μ) (hlim : ∀ᵐ a ∂μ, Tendsto (fun i ↦ f i a) u (𝓝 (g a))) :
    AEMeasurable g μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → ENNReal
    g : α → ENNReal
    μ : MeasureTheory.Measure α
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    hlim : Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) u (nhds (g  …
    ⊢ AEMeasurable g μ
  -/
  rcases u.exists_seq_tendsto with ⟨v, hv⟩
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → ENNReal
    g : α → ENNReal
    μ : MeasureTheory.Measure α
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    hlim : Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) u (nhds (g  …
    v : Nat → ι
    hv : Filter.Tendsto v Filter.atTop u
    ⊢ AEMeasurable g μ
  -/
  have h'f : ∀ n, AEMeasurable (f (v n)) μ := fun n ↦ hf (v n)
  /-
    case intro
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → ENNReal
    g : α → ENNReal
    μ : MeasureTheory.Measure α
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), AEMeasurable (f i) μ
    hlim : Filter.Eventually (fun a => Filter.Tendsto (fun i => f i a) u (nhds (g  …
    v : Nat → ι
    hv : Filter.Tendsto v Filter.atTop u
    h'f : ∀ (n : Nat), AEMeasurable (f (v n)) μ
    ⊢ AEMeasurable g μ
  -/
  set p : α → (ℕ → ℝ≥0∞) → Prop := fun x f' ↦ Tendsto f' atTop (𝓝 (g x))
  have hp : ∀ᵐ x ∂μ, p x fun n ↦ f (v n) x := by
    filter_upwards [hlim] with x hx using hx.comp hv
  classical
  set aeSeqLim := fun x ↦ ite (x ∈ aeSeqSet h'f p) (g x) (⟨f (v 0) x⟩ : Nonempty ℝ≥0∞).some
  refine ⟨aeSeqLim, measurable_of_tendsto' atTop (aeSeq.measurable h'f p)
    (tendsto_pi_nhds.mpr fun x ↦ ?_), ?_⟩
  · unfold aeSeqLim
    simp_rw [aeSeq]
    split_ifs with hx
    · simp_rw [aeSeq.mk_eq_fun_of_mem_aeSeqSet h'f hx]
      exact aeSeq.fun_prop_of_mem_aeSeqSet h'f hx
    · exact tendsto_const_nhds
  · exact (ite_ae_eq_of_measure_compl_zero g (fun x ↦ (⟨f (v 0) x⟩ : Nonempty ℝ≥0∞).some)
      (aeSeqSet h'f p) (aeSeq.measure_compl_aeSeqSet_eq_zero h'f hp)).symm


/-- A limit of a.e.-measurable `ℝ≥0∞` valued functions is a.e.-measurable. -/
lemma aemeasurable_of_tendsto {f : ℕ → α → ℝ≥0∞} {g : α → ℝ≥0∞} {μ : Measure α}
    (hf : ∀ i, AEMeasurable (f i) μ) (hlim : ∀ᵐ a ∂μ, Tendsto (fun i ↦ f i a) atTop (𝓝 (g a))) :
    AEMeasurable g μ :=
  aemeasurable_of_tendsto' atTop hf hlim


@[measurability, fun_prop]
theorem Measurable.ennreal_toNNReal {f : α → ℝ≥0∞} (hf : Measurable f) :
    Measurable fun x => (f x).toNNReal :=
  ENNReal.measurable_toNNReal.comp hf


@[measurability, fun_prop]
theorem AEMeasurable.ennreal_toNNReal {f : α → ℝ≥0∞} {μ : Measure α} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x => (f x).toNNReal) μ :=
  ENNReal.measurable_toNNReal.comp_aemeasurable hf


@[simp, norm_cast]
theorem measurable_coe_nnreal_ennreal_iff {f : α → ℝ≥0} :
    (Measurable fun x => (f x : ℝ≥0∞)) ↔ Measurable f :=
  ⟨fun h => h.ennreal_toNNReal, fun h => h.coe_nnreal_ennreal⟩


@[simp, norm_cast]
theorem aemeasurable_coe_nnreal_ennreal_iff {f : α → ℝ≥0} {μ : Measure α} :
    AEMeasurable (fun x => (f x : ℝ≥0∞)) μ ↔ AEMeasurable f μ :=
  ⟨fun h => h.ennreal_toNNReal, fun h => h.coe_nnreal_ennreal⟩


@[measurability, fun_prop]
theorem Measurable.ennreal_toReal {f : α → ℝ≥0∞} (hf : Measurable f) :
    Measurable fun x => ENNReal.toReal (f x) :=
  ENNReal.measurable_toReal.comp hf


@[measurability, fun_prop]
theorem AEMeasurable.ennreal_toReal {f : α → ℝ≥0∞} {μ : Measure α} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x => ENNReal.toReal (f x)) μ :=
  ENNReal.measurable_toReal.comp_aemeasurable hf


/-- note: `ℝ≥0∞` can probably be generalized in a future version of this lemma. -/
@[measurability, fun_prop]
theorem Measurable.ennreal_tsum {ι} [Countable ι] {f : ι → α → ℝ≥0∞} (h : ∀ i, Measurable (f i)) :
    Measurable fun x => ∑' i, f i x := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    inst✝ : Countable ι
    f : ι → α → ENNReal
    h : ∀ (i : ι), Measurable (f i)
    ⊢ Measurable fun x => tsum fun i => f i x
  -/
  simp_rw [ENNReal.tsum_eq_iSup_sum]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    inst✝ : Countable ι
    f : ι → α → ENNReal
    h : ∀ (i : ι), Measurable (f i)
    ⊢ Measurable fun x => iSup fun s => s.sum fun i => f i x
  -/
  exact .iSup fun s ↦ s.measurable_sum fun i _ => h i
  /-
    🎉 no goals
  -/


@[measurability, fun_prop]
theorem Measurable.ennreal_tsum' {ι} [Countable ι] {f : ι → α → ℝ≥0∞} (h : ∀ i, Measurable (f i)) :
    Measurable (∑' i, f i) := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    inst✝ : Countable ι
    f : ι → α → ENNReal
    h : ∀ (i : ι), Measurable (f i)
    ⊢ Measurable (tsum fun i => f i)
  -/
  convert Measurable.ennreal_tsum h with x
  /-
    case h.e'_5.h
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    inst✝ : Countable ι
    f : ι → α → ENNReal
    h : ∀ (i : ι), Measurable (f i)
    x : α
    ⊢ Eq (tsum (fun i => f i) x) (tsum fun i => f i x)
  -/
  exact tsum_apply (Pi.summable.2 fun _ => ENNReal.summable)
  /-
    🎉 no goals
  -/


@[measurability, fun_prop]
theorem Measurable.nnreal_tsum {ι} [Countable ι] {f : ι → α → ℝ≥0} (h : ∀ i, Measurable (f i)) :
    Measurable fun x => ∑' i, f i x := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    inst✝ : Countable ι
    f : ι → α → NNReal
    h : ∀ (i : ι), Measurable (f i)
    ⊢ Measurable fun x => tsum fun i => f i x
  -/
  simp_rw [NNReal.tsum_eq_toNNReal_tsum]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    inst✝ : Countable ι
    f : ι → α → NNReal
    h : ∀ (i : ι), Measurable (f i)
    ⊢ Measurable fun x => (tsum fun b => ↑(f b x)).toNNReal
  -/
  exact (Measurable.ennreal_tsum fun i => (h i).coe_nnreal_ennreal).ennreal_toNNReal
  /-
    🎉 no goals
  -/


@[measurability, fun_prop]
theorem AEMeasurable.ennreal_tsum {ι} [Countable ι] {f : ι → α → ℝ≥0∞} {μ : Measure α}
    (h : ∀ i, AEMeasurable (f i) μ) : AEMeasurable (fun x => ∑' i, f i x) μ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    inst✝ : Countable ι
    f : ι → α → ENNReal
    μ : MeasureTheory.Measure α
    h : ∀ (i : ι), AEMeasurable (f i) μ
    ⊢ AEMeasurable (fun x => tsum fun i => f i x) μ
  -/
  simp_rw [ENNReal.tsum_eq_iSup_sum]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    inst✝ : Countable ι
    f : ι → α → ENNReal
    μ : MeasureTheory.Measure α
    h : ∀ (i : ι), AEMeasurable (f i) μ
    ⊢ AEMeasurable (fun x => iSup fun s => s.sum fun i => f i x) μ
  -/
  exact .iSup fun s ↦ Finset.aemeasurable_sum s fun i _ => h i
  /-
    🎉 no goals
  -/


@[measurability, fun_prop]
theorem AEMeasurable.nnreal_tsum {α : Type*} {_ : MeasurableSpace α} {ι : Type*} [Countable ι]
    {f : ι → α → NNReal} {μ : Measure α} (h : ∀ i : ι, AEMeasurable (f i) μ) :
    AEMeasurable (fun x : α => ∑' i : ι, f i x) μ := by
  /-
    α : Type u_5
    x✝ : MeasurableSpace α
    ι : Type u_6
    inst✝ : Countable ι
    f : ι → α → NNReal
    μ : MeasureTheory.Measure α
    h : ∀ (i : ι), AEMeasurable (f i) μ
    ⊢ AEMeasurable (fun x => tsum fun i => f i x) μ
  -/
  simp_rw [NNReal.tsum_eq_toNNReal_tsum]
  /-
    α : Type u_5
    x✝ : MeasurableSpace α
    ι : Type u_6
    inst✝ : Countable ι
    f : ι → α → NNReal
    μ : MeasureTheory.Measure α
    h : ∀ (i : ι), AEMeasurable (f i) μ
    ⊢ AEMeasurable (fun x => (tsum fun b => ↑(f b x)).toNNReal) μ
  -/
  exact (AEMeasurable.ennreal_tsum fun i => (h i).coe_nnreal_ennreal).ennreal_toNNReal
  /-
    🎉 no goals
  -/


@[measurability, fun_prop]
theorem measurable_coe_real_ereal : Measurable ((↑) : ℝ → EReal) :=
  continuous_coe_real_ereal.measurable


@[measurability]
theorem Measurable.coe_real_ereal {f : α → ℝ} (hf : Measurable f) :
    Measurable fun x => (f x : EReal) :=
  measurable_coe_real_ereal.comp hf


@[measurability]
theorem AEMeasurable.coe_real_ereal {f : α → ℝ} {μ : Measure α} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x => (f x : EReal)) μ :=
  measurable_coe_real_ereal.comp_aemeasurable hf


/-- The set of finite `EReal` numbers is `MeasurableEquiv` to `ℝ`. -/
def MeasurableEquiv.erealEquivReal : ({⊥, ⊤}ᶜ : Set EReal) ≃ᵐ ℝ :=
  EReal.neBotTopHomeomorphReal.toMeasurableEquiv


theorem EReal.measurable_of_measurable_real {f : EReal → α} (h : Measurable fun p : ℝ => f p) :
    Measurable f :=
                                                      /-
                                                        α : Type u_1
                                                        mα : MeasurableSpace α
                                                        f : EReal → α
                                                        h : Measurable fun p => f ↑p
                                                        ⊢ (Insert.insert Bot.bot (Singleton.singleton Top.top)).Finite
                                                      -/
  measurable_of_measurable_on_compl_finite {⊥, ⊤} (by simp)
                                                      /-
                                                        🎉 no goals
                                                      -/
    (MeasurableEquiv.erealEquivReal.symm.measurable_comp_iff.1 h)


@[measurability]
theorem measurable_ereal_toReal : Measurable EReal.toReal :=
                                          /-
                                            ⊢ Measurable fun p => (↑p).toReal
                                          -/
  EReal.measurable_of_measurable_real (by simpa using measurable_id)
                                          /-
                                            🎉 no goals
                                          -/


@[measurability, fun_prop]
theorem Measurable.ereal_toReal {f : α → EReal} (hf : Measurable f) :
    Measurable fun x => (f x).toReal :=
  measurable_ereal_toReal.comp hf


@[measurability, fun_prop]
theorem AEMeasurable.ereal_toReal {f : α → EReal} {μ : Measure α} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x => (f x).toReal) μ :=
  measurable_ereal_toReal.comp_aemeasurable hf


@[measurability]
theorem measurable_coe_ennreal_ereal : Measurable ((↑) : ℝ≥0∞ → EReal) :=
  continuous_coe_ennreal_ereal.measurable


@[measurability, fun_prop]
theorem Measurable.coe_ereal_ennreal {f : α → ℝ≥0∞} (hf : Measurable f) :
    Measurable fun x => (f x : EReal) :=
  measurable_coe_ennreal_ereal.comp hf


@[measurability, fun_prop]
theorem AEMeasurable.coe_ereal_ennreal {f : α → ℝ≥0∞} {μ : Measure α} (hf : AEMeasurable f μ) :
    AEMeasurable (fun x => (f x : EReal)) μ :=
  measurable_coe_ennreal_ereal.comp_aemeasurable hf


instance : MeasurableSMul₂ ℝ≥0 ℝ≥0∞ where
                                                                               /-
                                                                                 α : Type u_1
                                                                                 β : Type u_2
                                                                                 γ : Type u_3
                                                                                 δ : Type u_4
                                                                                 ι : Sort y
                                                                                 s t u : Set α
                                                                                 mα : MeasurableSpace α
                                                                                 ⊢ Measurable fun r => HMul.hMul (↑r.1) r.2
                                                                               -/
  measurable_smul := show Measurable fun r : ℝ≥0 × ℝ≥0∞ ↦ (r.1 : ℝ≥0) * r.2 by fun_prop
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- A limit (over a general filter) of measurable `ℝ≥0` valued functions is measurable. -/
theorem measurable_of_tendsto' {ι} {f : ι → α → ℝ≥0} {g : α → ℝ≥0} (u : Filter ι) [NeBot u]
    [IsCountablyGenerated u] (hf : ∀ i, Measurable (f i)) (lim : Tendsto f u (𝓝 g)) :
    Measurable g := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → NNReal
    g : α → NNReal
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    hf : ∀ (i : ι), Measurable (f i)
    lim : Filter.Tendsto f u (nhds g)
    ⊢ Measurable g
  -/
  simp_rw [← measurable_coe_nnreal_ennreal_iff] at hf ⊢
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → NNReal
    g : α → NNReal
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    lim : Filter.Tendsto f u (nhds g)
    hf : ∀ (i : ι), Measurable fun x => ↑(f i x)
    ⊢ Measurable fun x => ↑(g x)
  -/
  refine ENNReal.measurable_of_tendsto' u hf ?_
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → NNReal
    g : α → NNReal
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    lim : Filter.Tendsto f u (nhds g)
    hf : ∀ (i : ι), Measurable fun x => ↑(f i x)
    ⊢ Filter.Tendsto (fun i x => ↑(f i x)) u (nhds fun x => ↑(g x))
  -/
  rw [tendsto_pi_nhds] at lim ⊢
  /-
    α : Type u_1
    mα : MeasurableSpace α
    ι : Type u_5
    f : ι → α → NNReal
    g : α → NNReal
    u : Filter ι
    inst✝¹ : u.NeBot
    inst✝ : u.IsCountablyGenerated
    lim : ∀ (x : α), Filter.Tendsto (fun i => f i x) u (nhds (g x))
    hf : ∀ (i : ι), Measurable fun x => ↑(f i x)
    ⊢ ∀ (x : α), Filter.Tendsto (fun i => ↑(f i x)) u (nhds ↑(g x))
  -/
  exact fun x => (ENNReal.continuous_coe.tendsto (g x)).comp (lim x)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-03-09")] alias
_root_.measurable_of_tendsto_nnreal' := NNReal.measurable_of_tendsto'


/-- A sequential limit of measurable `ℝ≥0` valued functions is measurable. -/
theorem measurable_of_tendsto {f : ℕ → α → ℝ≥0} {g : α → ℝ≥0} (hf : ∀ i, Measurable (f i))
    (lim : Tendsto f atTop (𝓝 g)) : Measurable g :=
  measurable_of_tendsto' atTop hf lim


@[deprecated (since := "2024-03-09")] alias
_root_.measurable_of_tendsto_nnreal := NNReal.measurable_of_tendsto


lemma measurableEmbedding_coe : MeasurableEmbedding Real.toEReal :=
  isOpenEmbedding_coe.measurableEmbedding


instance : MeasurableAdd₂ EReal := ⟨EReal.lowerSemicontinuous_add.measurable⟩


lemma measurable_of_real_prod {f : EReal × β → γ}
    (h_real : Measurable fun p : ℝ × β ↦ f (p.1, p.2))
    (h_bot : Measurable fun x ↦ f (⊥, x)) (h_top : Measurable fun x ↦ f (⊤, x)) : Measurable f :=
  .of_union₃_range_cover (measurableEmbedding_prod_mk_left _) (measurableEmbedding_prod_mk_left _)
                                              /-
                                                β : Type u_6
                                                γ : Type u_7
                                                mβ : MeasurableSpace β
                                                mγ : MeasurableSpace γ
                                                f : Prod EReal β → γ
                                                h_real : Measurable fun p => f { fst := ↑p.1, snd := p.2 }
                                                h_bot : Measurable fun x => f { fst := Bot.bot, snd := x }
                                                h_top : Measurable fun x => f { fst := Top.top, snd := x }
                                                ⊢ HasSubset.Subset Set.univ (Union.union (Union.union (Set.range (Prod.mk Bot. …
                                              -/
    (measurableEmbedding_coe.prodMap .id) (by simp [-univ_subset_iff, subset_def, EReal.forall])
                                              /-
                                                🎉 no goals
                                              -/
    h_bot h_top h_real


lemma measurable_of_real_real {f : EReal × EReal → β}
    (h_real : Measurable fun p : ℝ × ℝ ↦ f (p.1, p.2))
    (h_bot_left : Measurable fun r : ℝ ↦ f (⊥, r))
    (h_top_left : Measurable fun r : ℝ ↦ f (⊤, r))
    (h_bot_right : Measurable fun r : ℝ ↦ f (r, ⊥))
    (h_top_right : Measurable fun r : ℝ ↦ f (r, ⊤)) :
    Measurable f := by
  /-
    β : Type u_6
    mβ : MeasurableSpace β
    f : Prod EReal EReal → β
    h_real : Measurable fun p => f { fst := ↑p.1, snd := ↑p.2 }
    h_bot_left : Measurable fun r => f { fst := Bot.bot, snd := ↑r }
    h_top_left : Measurable fun r => f { fst := Top.top, snd := ↑r }
    h_bot_right : Measurable fun r => f { fst := ↑r, snd := Bot.bot }
    h_top_right : Measurable fun r => f { fst := ↑r, snd := Top.top }
    ⊢ Measurable f
  -/
  refine measurable_of_real_prod ?_ ?_ ?_
    /-
      case refine_1
      β : Type u_6
      mβ : MeasurableSpace β
      f : Prod EReal EReal → β
      h_real : Measurable fun p => f { fst := ↑p.1, snd := ↑p.2 }
      h_bot_left : Measurable fun r => f { fst := Bot.bot, snd := ↑r }
      h_top_left : Measurable fun r => f { fst := Top.top, snd := ↑r }
      h_bot_right : Measurable fun r => f { fst := ↑r, snd := Bot.bot }
      h_top_right : Measurable fun r => f { fst := ↑r, snd := Top.top }
      ⊢ Measurable fun p => f { fst := ↑p.1, snd := p.2 }
    -/
  · refine measurable_swap_iff.mp <| measurable_of_real_prod ?_ h_bot_right h_top_right
    /-
      case refine_1
      β : Type u_6
      mβ : MeasurableSpace β
      f : Prod EReal EReal → β
      h_real : Measurable fun p => f { fst := ↑p.1, snd := ↑p.2 }
      h_bot_left : Measurable fun r => f { fst := Bot.bot, snd := ↑r }
      h_top_left : Measurable fun r => f { fst := Top.top, snd := ↑r }
      h_bot_right : Measurable fun r => f { fst := ↑r, snd := Bot.bot }
      h_top_right : Measurable fun r => f { fst := ↑r, snd := Top.top }
      ⊢ Measurable fun p => Function.comp (fun p => f { fst := ↑p.1, snd := p.2 }) P …
    -/
    exact h_real.comp measurable_swap
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      β : Type u_6
      mβ : MeasurableSpace β
      f : Prod EReal EReal → β
      h_real : Measurable fun p => f { fst := ↑p.1, snd := ↑p.2 }
      h_bot_left : Measurable fun r => f { fst := Bot.bot, snd := ↑r }
      h_top_left : Measurable fun r => f { fst := Top.top, snd := ↑r }
      h_bot_right : Measurable fun r => f { fst := ↑r, snd := Bot.bot }
      h_top_right : Measurable fun r => f { fst := ↑r, snd := Top.top }
      ⊢ Measurable fun x => f { fst := Bot.bot, snd := x }
    -/
  · exact measurable_of_measurable_real h_bot_left
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      β : Type u_6
      mβ : MeasurableSpace β
      f : Prod EReal EReal → β
      h_real : Measurable fun p => f { fst := ↑p.1, snd := ↑p.2 }
      h_bot_left : Measurable fun r => f { fst := Bot.bot, snd := ↑r }
      h_top_left : Measurable fun r => f { fst := Top.top, snd := ↑r }
      h_bot_right : Measurable fun r => f { fst := ↑r, snd := Bot.bot }
      h_top_right : Measurable fun r => f { fst := ↑r, snd := Top.top }
      ⊢ Measurable fun x => f { fst := Top.top, snd := x }
    -/
  · exact measurable_of_measurable_real h_top_left
    /-
      🎉 no goals
    -/


private lemma measurable_const_mul (c : EReal) : Measurable fun (x : EReal) ↦ c * x := by
  /-
    c : EReal
    ⊢ Measurable fun x => HMul.hMul c x
  -/
  refine measurable_of_measurable_real ?_
  have h1 : (fun (p : ℝ) ↦ (⊥ : EReal) * p)
      = fun p ↦ if p = 0 then (0 : EReal) else (if p < 0 then ⊤ else ⊥) := by
    ext p
    split_ifs with h1 h2
    · simp [h1]
    · rw [bot_mul_coe_of_neg h2]
    · rw [bot_mul_coe_of_pos]
      exact lt_of_le_of_ne (not_lt.mp h2) (Ne.symm h1)
  have h2 : Measurable fun (p : ℝ) ↦ if p = 0 then (0 : EReal) else if p < 0 then ⊤ else ⊥ := by
    refine Measurable.piecewise (measurableSet_singleton _) measurable_const ?_
    exact Measurable.piecewise measurableSet_Iio measurable_const measurable_const
  induction c with
  | h_bot => rwa [h1]
  | h_real c => exact (measurable_id.const_mul _).coe_real_ereal
  | h_top =>
    simp_rw [← neg_bot, neg_mul]
    apply Measurable.neg
    rwa [h1]


instance : MeasurableMul₂ EReal := by
  /-
    α✝ : Type u_1
    β✝ : Type u_2
    γ✝ : Type u_3
    δ : Type u_4
    ι : Sort y
    s t u : Set α✝
    mα✝ : MeasurableSpace α✝
    α : Type u_5
    β : Type u_6
    γ : Type u_7
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    ⊢ MeasurableMul₂ EReal
  -/
  refine ⟨measurable_of_real_real ?_ ?_ ?_ ?_ ?_⟩
    /-
      case refine_1
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α✝
      mα✝ : MeasurableSpace α✝
      α : Type u_5
      β : Type u_6
      γ : Type u_7
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      ⊢ Measurable fun p => HMul.hMul { fst := ↑p.1, snd := ↑p.2 }.1 { fst := ↑p.1,  …
    -/
  · exact (measurable_fst.mul measurable_snd).coe_real_ereal
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α✝
      mα✝ : MeasurableSpace α✝
      α : Type u_5
      β : Type u_6
      γ : Type u_7
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      ⊢ Measurable fun r => HMul.hMul { fst := Bot.bot, snd := ↑r }.1 { fst := Bot.b …
    -/
  · exact (measurable_const_mul _).comp measurable_coe_real_ereal
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α✝
      mα✝ : MeasurableSpace α✝
      α : Type u_5
      β : Type u_6
      γ : Type u_7
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      ⊢ Measurable fun r => HMul.hMul { fst := Top.top, snd := ↑r }.1 { fst := Top.t …
    -/
  · exact (measurable_const_mul _).comp measurable_coe_real_ereal
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α✝
      mα✝ : MeasurableSpace α✝
      α : Type u_5
      β : Type u_6
      γ : Type u_7
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      ⊢ Measurable fun r => HMul.hMul { fst := ↑r, snd := Bot.bot }.1 { fst := ↑r, s …
    -/
  · simp_rw [mul_comm _ ⊥]
    /-
      case refine_4
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α✝
      mα✝ : MeasurableSpace α✝
      α : Type u_5
      β : Type u_6
      γ : Type u_7
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      ⊢ Measurable fun r => HMul.hMul Bot.bot ↑r
    -/
    exact (measurable_const_mul _).comp measurable_coe_real_ereal
    /-
      🎉 no goals
    -/
    /-
      case refine_5
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α✝
      mα✝ : MeasurableSpace α✝
      α : Type u_5
      β : Type u_6
      γ : Type u_7
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      ⊢ Measurable fun r => HMul.hMul { fst := ↑r, snd := Top.top }.1 { fst := ↑r, s …
    -/
  · simp_rw [mul_comm _ ⊤]
    /-
      case refine_5
      α✝ : Type u_1
      β✝ : Type u_2
      γ✝ : Type u_3
      δ : Type u_4
      ι : Sort y
      s t u : Set α✝
      mα✝ : MeasurableSpace α✝
      α : Type u_5
      β : Type u_6
      γ : Type u_7
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      ⊢ Measurable fun r => HMul.hMul Top.top ↑r
    -/
    exact (measurable_const_mul _).comp measurable_coe_real_ereal
    /-
      🎉 no goals
    -/


/-- If a function `f : α → ℝ≥0` is measurable and the measure is σ-finite, then there exists
spanning measurable sets with finite measure on which `f` is bounded.
See also `StronglyMeasurable.exists_spanning_measurableSet_norm_le` for functions into normed
groups. -/
theorem exists_spanning_measurableSet_le {f : α → ℝ≥0} (hf : Measurable f) (μ : Measure α)
    [SigmaFinite μ] :
    ∃ s : ℕ → Set α,
      (∀ n, MeasurableSet (s n) ∧ μ (s n) < ∞ ∧ ∀ x ∈ s n, f x ≤ n) ∧
      ⋃ i, s i = Set.univ := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    f : α → NNReal
    hf : Measurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    ⊢ Exists fun s => And (∀ (n : Nat), And (MeasurableSet (s n)) (And (LT.lt (μ ( …
  -/
  let sigma_finite_sets := spanningSets μ
  /-
    α : Type u_1
    mα : MeasurableSpace α
    f : α → NNReal
    hf : Measurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    sigma_finite_sets : Nat → Set α := MeasureTheory.spanningSets μ
    ⊢ Exists fun s => And (∀ (n : Nat), And (MeasurableSet (s n)) (And (LT.lt (μ ( …
  -/
  let norm_sets := fun n : ℕ => { x | f x ≤ n }
  have norm_sets_spanning : ⋃ n, norm_sets n = Set.univ := by
    ext1 x
    simp only [Set.mem_iUnion, Set.mem_setOf_eq, Set.mem_univ, iff_true]
    exact exists_nat_ge (f x)
  /-
    α : Type u_1
    mα : MeasurableSpace α
    f : α → NNReal
    hf : Measurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    sigma_finite_sets : Nat → Set α := MeasureTheory.spanningSets μ
    norm_sets : Nat → Set α := fun n => setOf fun x => LE.le (f x) ↑n
    norm_sets_spanning : Eq (Set.iUnion fun n => norm_sets n) Set.univ
    ⊢ Exists fun s => And (∀ (n : Nat), And (MeasurableSet (s n)) (And (LT.lt (μ ( …
  -/
  let sets n := sigma_finite_sets n ∩ norm_sets n
  have h_meas : ∀ n, MeasurableSet (sets n) := by
    refine fun n => MeasurableSet.inter ?_ ?_
    · exact measurableSet_spanningSets μ n
    · exact hf measurableSet_Iic
  have h_finite : ∀ n, μ (sets n) < ∞ := by
    refine fun n => (measure_mono Set.inter_subset_left).trans_lt ?_
    exact measure_spanningSets_lt_top μ n
  /-
    α : Type u_1
    mα : MeasurableSpace α
    f : α → NNReal
    hf : Measurable f
    μ : MeasureTheory.Measure α
    inst✝ : MeasureTheory.SigmaFinite μ
    sigma_finite_sets : Nat → Set α := MeasureTheory.spanningSets μ
    norm_sets : Nat → Set α := fun n => setOf fun x => LE.le (f x) ↑n
    norm_sets_spanning : Eq (Set.iUnion fun n => norm_sets n) Set.univ
    sets : Nat → Set α := fun n => Inter.inter (sigma_finite_sets n) (norm_sets n)
    h_meas : ∀ (n : Nat), MeasurableSet (sets n)
    h_finite : ∀ (n : Nat), LT.lt (μ (sets n)) Top.top
    ⊢ Exists fun s => And (∀ (n : Nat), And (MeasurableSet (s n)) (And (LT.lt (μ ( …
  -/
  refine ⟨sets, fun n => ⟨h_meas n, h_finite n, ?_⟩, ?_⟩
    /-
      case refine_1
      α : Type u_1
      mα : MeasurableSpace α
      f : α → NNReal
      hf : Measurable f
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      sigma_finite_sets : Nat → Set α := MeasureTheory.spanningSets μ
      norm_sets : Nat → Set α := fun n => setOf fun x => LE.le (f x) ↑n
      norm_sets_spanning : Eq (Set.iUnion fun n => norm_sets n) Set.univ
      sets : Nat → Set α := fun n => Inter.inter (sigma_finite_sets n) (norm_sets n)
      h_meas : ∀ (n : Nat), MeasurableSet (sets n)
      h_finite : ∀ (n : Nat), LT.lt (μ (sets n)) Top.top
      n : Nat
      ⊢ ∀ (x : α), Membership.mem (sets n) x → LE.le (f x) ↑n
    -/
  · exact fun x hx => hx.2
    /-
      🎉 no goals
    -/
  · have :
      ⋃ i, sigma_finite_sets i ∩ norm_sets i = (⋃ i, sigma_finite_sets i) ∩ ⋃ i, norm_sets i := by
      refine Set.iUnion_inter_of_monotone (monotone_spanningSets μ) fun i j hij x => ?_
      simp only [norm_sets, Set.mem_setOf_eq]
      refine fun hif => hif.trans ?_
      exact mod_cast hij
    /-
      case refine_2
      α : Type u_1
      mα : MeasurableSpace α
      f : α → NNReal
      hf : Measurable f
      μ : MeasureTheory.Measure α
      inst✝ : MeasureTheory.SigmaFinite μ
      sigma_finite_sets : Nat → Set α := MeasureTheory.spanningSets μ
      norm_sets : Nat → Set α := fun n => setOf fun x => LE.le (f x) ↑n
      norm_sets_spanning : Eq (Set.iUnion fun n => norm_sets n) Set.univ
      sets : Nat → Set α := fun n => Inter.inter (sigma_finite_sets n) (norm_sets n)
      h_meas : ∀ (n : Nat), MeasurableSet (sets n)
      h_finite : ∀ (n : Nat), LT.lt (μ (sets n)) Top.top
      this : Eq (Set.iUnion fun i => Inter.inter (sigma_finite_sets i) (norm_sets i) …
      ⊢ Eq (Set.iUnion fun i => sets i) Set.univ
    -/
    rw [this, norm_sets_spanning, iUnion_spanningSets μ, Set.inter_univ]
    /-
      🎉 no goals
    -/


lemma tendsto_measure_Icc_nhdsWithin_right' (b : ℝ) :
    Tendsto (fun δ ↦ μ (Icc (b - δ) (b + δ))) (𝓝[>] (0 : ℝ)) (𝓝 (μ {b})) := by
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    b : Real
    ⊢ Filter.Tendsto (fun δ => μ (Set.Icc (HSub.hSub b δ) (HAdd.hAdd b δ))) (nhdsW …
  -/
  rw [Real.singleton_eq_inter_Icc]
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    b : Real
    ⊢ Filter.Tendsto (fun δ => μ (Set.Icc (HSub.hSub b δ) (HAdd.hAdd b δ))) (nhdsW …
  -/
  apply tendsto_measure_biInter_gt (fun r hr ↦ nullMeasurableSet_Icc)
    /-
      case hm
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      b : Real
      ⊢ ∀ (i j : Real), LT.lt 0 i → LE.le i j → HasSubset.Subset (Set.Icc (HSub.hSub …
    -/
  · intro r s _rpos hrs
    /-
      case hm
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      b r s : Real
      _rpos : LT.lt 0 r
      hrs : LE.le r s
      ⊢ HasSubset.Subset (Set.Icc (HSub.hSub b r) (HAdd.hAdd b r)) (Set.Icc (HSub.hS …
    -/
    exact Icc_subset_Icc (by linarith) (by linarith)
    /-
      🎉 no goals
    -/
    /-
      case hf
      μ : MeasureTheory.Measure Real
      inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      b : Real
      ⊢ Exists fun r => And (GT.gt r 0) (Ne (μ (Set.Icc (HSub.hSub b r) (HAdd.hAdd b …
    -/
  · exact ⟨1, zero_lt_one, isCompact_Icc.measure_ne_top⟩
    /-
      🎉 no goals
    -/


lemma tendsto_measure_Icc_nhdsWithin_right (b : ℝ) :
    Tendsto (fun δ ↦ μ (Icc (b - δ) (b + δ))) (𝓝[≥] (0 : ℝ)) (𝓝 (μ {b})) := by
  simp only [← nhdsWithin_right_sup_nhds_singleton, nhdsWithin_singleton, tendsto_sup,
    tendsto_measure_Icc_nhdsWithin_right' μ b, true_and, tendsto_pure_left]
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    b : Real
    ⊢ ∀ (s : Set ENNReal), Membership.mem (nhds (μ (Singleton.singleton b))) s → M …
  -/
  intro s hs
  /-
    μ : MeasureTheory.Measure Real
    inst✝ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    b : Real
    s : Set ENNReal
    hs : Membership.mem (nhds (μ (Singleton.singleton b))) s
    ⊢ Membership.mem s (μ (Set.Icc (HSub.hSub b 0) (HAdd.hAdd b 0)))
  -/
  simpa using mem_of_mem_nhds hs
  /-
    🎉 no goals
  -/


lemma tendsto_measure_Icc [NoAtoms μ] (b : ℝ) :
    Tendsto (fun δ ↦ μ (Icc (b - δ) (b + δ))) (𝓝 (0 : ℝ)) (𝓝 0) := by
  /-
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : MeasureTheory.NoAtoms μ
    b : Real
    ⊢ Filter.Tendsto (fun δ => μ (Set.Icc (HSub.hSub b δ) (HAdd.hAdd b δ))) (nhds  …
  -/
  rw [← nhdsLT_sup_nhdsGE, tendsto_sup]
  /-
    μ : MeasureTheory.Measure Real
    inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
    inst✝ : MeasureTheory.NoAtoms μ
    b : Real
    ⊢ And (Filter.Tendsto (fun δ => μ (Set.Icc (HSub.hSub b δ) (HAdd.hAdd b δ))) ( …
  -/
  constructor
    /-
      case left
      μ : MeasureTheory.Measure Real
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝ : MeasureTheory.NoAtoms μ
      b : Real
      ⊢ Filter.Tendsto (fun δ => μ (Set.Icc (HSub.hSub b δ) (HAdd.hAdd b δ))) (nhdsW …
    -/
  · apply tendsto_const_nhds.congr'
    /-
      case left
      μ : MeasureTheory.Measure Real
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝ : MeasureTheory.NoAtoms μ
      b : Real
      ⊢ (nhdsWithin 0 (Set.Iio 0)).EventuallyEq (fun x => 0) fun δ => μ (Set.Icc (HS …
    -/
    filter_upwards [self_mem_nhdsWithin] with r (hr : r < 0)
    /-
      case h
      μ : MeasureTheory.Measure Real
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝ : MeasureTheory.NoAtoms μ
      b r : Real
      hr : LT.lt r 0
      ⊢ Eq 0 (μ (Set.Icc (HSub.hSub b r) (HAdd.hAdd b r)))
    -/
    rw [Icc_eq_empty (by linarith), measure_empty]
    /-
      🎉 no goals
    -/
    /-
      case right
      μ : MeasureTheory.Measure Real
      inst✝¹ : MeasureTheory.IsFiniteMeasureOnCompacts μ
      inst✝ : MeasureTheory.NoAtoms μ
      b : Real
      ⊢ Filter.Tendsto (fun δ => μ (Set.Icc (HSub.hSub b δ) (HAdd.hAdd b δ))) (nhdsW …
    -/
  · simpa only [measure_singleton] using tendsto_measure_Icc_nhdsWithin_right μ b
    /-
      🎉 no goals
    -/

