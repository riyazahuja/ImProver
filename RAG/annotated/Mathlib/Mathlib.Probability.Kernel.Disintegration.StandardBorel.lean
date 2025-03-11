lemma isRatCondKernelCDFAux_density_Iic (κ : Kernel α (γ × ℝ)) [IsFiniteKernel κ] :
    IsRatCondKernelCDFAux (fun (p : α × γ) q ↦ density κ (fst κ) p.1 p.2 (Iic q)) κ (fst κ) where
  measurable := measurable_pi_iff.mpr fun _ ↦ measurable_density κ (fst κ) measurableSet_Iic
  mono' a q r hqr :=
                                                                            /-
                                                                              α : Type u_1
                                                                              γ : Type u_3
                                                                              mα : MeasurableSpace α
                                                                              mγ : MeasurableSpace γ
                                                                              inst✝¹ : MeasurableSpace.CountablyGenerated γ
                                                                              κ : ProbabilityTheory.Kernel α (Prod γ Real)
                                                                              inst✝ : ProbabilityTheory.IsFiniteKernel κ
                                                                              a : α
                                                                              q r : Rat
                                                                              hqr : LE.le q r
                                                                              c : γ
                                                                              ⊢ LE.le ↑q ↑r
                                                                            -/
    ae_of_all _ fun c ↦ density_mono_set le_rfl a c (Iic_subset_Iic.mpr (by exact_mod_cast hqr))
                                                                            /-
                                                                              🎉 no goals
                                                                            -/
  nonneg' _ _ := ae_of_all _ fun _ ↦ density_nonneg le_rfl _ _ _
  le_one' _ _ := ae_of_all _ fun _ ↦ density_le_one le_rfl _ _ _
  tendsto_integral_of_antitone a s hs_anti hs_tendsto := by
    /-
      α : Type u_1
      γ : Type u_3
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ Real)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      s : Nat → Rat
      hs_anti : Antitone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
      ⊢ Filter.Tendsto (fun m => MeasureTheory.integral (κ.fst a) fun c => κ.density …
    -/
    let s' : ℕ → Set ℝ := fun n ↦ Iic (s n)
    /-
      α : Type u_1
      γ : Type u_3
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ Real)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      s : Nat → Rat
      hs_anti : Antitone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
      s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
      ⊢ Filter.Tendsto (fun m => MeasureTheory.integral (κ.fst a) fun c => κ.density …
    -/
    refine tendsto_integral_density_of_antitone le_rfl a s' ?_ ?_ (fun _ ↦ measurableSet_Iic)
      /-
        case refine_1
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_anti : Antitone s
        hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        ⊢ Antitone s'
      -/
    · refine fun i j hij ↦ Iic_subset_Iic.mpr ?_
      /-
        case refine_1
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_anti : Antitone s
        hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        i j : Nat
        hij : LE.le i j
        ⊢ LE.le ↑(s j) ↑(s i)
      -/
      exact mod_cast hs_anti hij
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_anti : Antitone s
        hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        ⊢ Eq (Set.iInter fun i => s' i) EmptyCollection.emptyCollection
      -/
    · ext x
      /-
        case refine_2.h
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_anti : Antitone s
        hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        ⊢ Iff (Membership.mem (Set.iInter fun i => s' i) x) (Membership.mem EmptyColle …
      -/
      simp only [mem_iInter, mem_Iic, mem_empty_iff_false, iff_false, not_forall, not_le, s']
      /-
        case refine_2.h
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_anti : Antitone s
        hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atBot
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        ⊢ Exists fun x_1 => LT.lt (↑(s x_1)) x
      -/
      rw [tendsto_atTop_atBot] at hs_tendsto
      /-
        case refine_2.h
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_anti : Antitone s
        hs_tendsto : ∀ (b : Rat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le (s a) b
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        ⊢ Exists fun x_1 => LT.lt (↑(s x_1)) x
      -/
      have ⟨q, hq⟩ := exists_rat_lt x
      /-
        case refine_2.h
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_anti : Antitone s
        hs_tendsto : ∀ (b : Rat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le (s a) b
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        q : Rat
        hq : LT.lt (↑q) x
        ⊢ Exists fun x_1 => LT.lt (↑(s x_1)) x
      -/
      obtain ⟨i, hi⟩ := hs_tendsto q
      /-
        case refine_2.h.intro
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_anti : Antitone s
        hs_tendsto : ∀ (b : Rat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le (s a) b
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        q : Rat
        hq : LT.lt (↑q) x
        i : Nat
        hi : ∀ (a : Nat), LE.le i a → LE.le (s a) q
        ⊢ Exists fun x_1 => LT.lt (↑(s x_1)) x
      -/
      refine ⟨i, lt_of_le_of_lt ?_ hq⟩
      /-
        case refine_2.h.intro
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_anti : Antitone s
        hs_tendsto : ∀ (b : Rat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le (s a) b
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        q : Rat
        hq : LT.lt (↑q) x
        i : Nat
        hi : ∀ (a : Nat), LE.le i a → LE.le (s a) q
        ⊢ LE.le ↑(s i) ↑q
      -/
      exact mod_cast hi i le_rfl
      /-
        🎉 no goals
      -/
  tendsto_integral_of_monotone a s hs_mono hs_tendsto := by
    /-
      α : Type u_1
      γ : Type u_3
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ Real)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      s : Nat → Rat
      hs_mono : Monotone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atTop
      ⊢ Filter.Tendsto (fun m => MeasureTheory.integral (κ.fst a) fun c => κ.density …
    -/
    rw [fst_apply' _ _ MeasurableSet.univ]
    /-
      α : Type u_1
      γ : Type u_3
      mα : MeasurableSpace α
      mγ : MeasurableSpace γ
      inst✝¹ : MeasurableSpace.CountablyGenerated γ
      κ : ProbabilityTheory.Kernel α (Prod γ Real)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      a : α
      s : Nat → Rat
      hs_mono : Monotone s
      hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atTop
      ⊢ Filter.Tendsto (fun m => MeasureTheory.integral (κ.fst a) fun c => κ.density …
    -/
    let s' : ℕ → Set ℝ := fun n ↦ Iic (s n)
    refine tendsto_integral_density_of_monotone (le_rfl : fst κ ≤ fst κ)
      a s' ?_ ?_ (fun _ ↦ measurableSet_Iic)
      /-
        case refine_1
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_mono : Monotone s
        hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atTop
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        ⊢ Monotone s'
      -/
    · exact fun i j hij ↦ Iic_subset_Iic.mpr (by exact mod_cast hs_mono hij)
      /-
        🎉 no goals
      -/
      /-
        case refine_2
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_mono : Monotone s
        hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atTop
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        ⊢ Eq (Set.iUnion fun i => s' i) Set.univ
      -/
    · ext x
      /-
        case refine_2.h
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_mono : Monotone s
        hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atTop
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        ⊢ Iff (Membership.mem (Set.iUnion fun i => s' i) x) (Membership.mem Set.univ x)
      -/
      simp only [mem_iUnion, mem_Iic, mem_univ, iff_true]
      /-
        case refine_2.h
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_mono : Monotone s
        hs_tendsto : Filter.Tendsto s Filter.atTop Filter.atTop
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        ⊢ Exists fun i => Membership.mem (s' i) x
      -/
      rw [tendsto_atTop_atTop] at hs_tendsto
      /-
        case refine_2.h
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_mono : Monotone s
        hs_tendsto : ∀ (b : Rat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le b (s a)
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        ⊢ Exists fun i => Membership.mem (s' i) x
      -/
      have ⟨q, hq⟩ := exists_rat_gt x
      /-
        case refine_2.h
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_mono : Monotone s
        hs_tendsto : ∀ (b : Rat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le b (s a)
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        q : Rat
        hq : LT.lt x ↑q
        ⊢ Exists fun i => Membership.mem (s' i) x
      -/
      obtain ⟨i, hi⟩ := hs_tendsto q
      /-
        case refine_2.h.intro
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_mono : Monotone s
        hs_tendsto : ∀ (b : Rat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le b (s a)
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        q : Rat
        hq : LT.lt x ↑q
        i : Nat
        hi : ∀ (a : Nat), LE.le i a → LE.le q (s a)
        ⊢ Exists fun i => Membership.mem (s' i) x
      -/
      refine ⟨i, hq.le.trans ?_⟩
      /-
        case refine_2.h.intro
        α : Type u_1
        γ : Type u_3
        mα : MeasurableSpace α
        mγ : MeasurableSpace γ
        inst✝¹ : MeasurableSpace.CountablyGenerated γ
        κ : ProbabilityTheory.Kernel α (Prod γ Real)
        inst✝ : ProbabilityTheory.IsFiniteKernel κ
        a : α
        s : Nat → Rat
        hs_mono : Monotone s
        hs_tendsto : ∀ (b : Rat), Exists fun i => ∀ (a : Nat), LE.le i a → LE.le b (s a)
        s' : Nat → Set Real := fun n => Set.Iic ↑(s n)
        x : Real
        q : Rat
        hq : LT.lt x ↑q
        i : Nat
        hi : ∀ (a : Nat), LE.le i a → LE.le q (s a)
        ⊢ LE.le ↑q ↑(s i)
      -/
      exact mod_cast hi i le_rfl
      /-
        🎉 no goals
      -/
  integrable a _ := integrable_density le_rfl a measurableSet_Iic
  setIntegral a _ hA _ := setIntegral_density le_rfl a measurableSet_Iic hA


/-- Taking the kernel density of intervals `Iic q` for `q : ℚ` gives a function with the property
`isRatCondKernelCDF`. -/
lemma isRatCondKernelCDF_density_Iic (κ : Kernel α (γ × ℝ)) [IsFiniteKernel κ] :
    IsRatCondKernelCDF (fun (p : α × γ) q ↦ density κ (fst κ) p.1 p.2 (Iic q)) κ (fst κ) :=
  (isRatCondKernelCDFAux_density_Iic κ).isRatCondKernelCDF


/-- The conditional kernel CDF of a kernel `κ : Kernel α (γ × ℝ)`, where `γ` is countably generated.
-/
noncomputable
def condKernelCDF (κ : Kernel α (γ × ℝ)) [IsFiniteKernel κ] : α × γ → StieltjesFunction :=
  stieltjesOfMeasurableRat (fun (p : α × γ) q ↦ density κ (fst κ) p.1 p.2 (Iic q))
    (isRatCondKernelCDF_density_Iic κ).measurable


lemma isCondKernelCDF_condKernelCDF (κ : Kernel α (γ × ℝ)) [IsFiniteKernel κ] :
    IsCondKernelCDF (condKernelCDF κ) κ (fst κ) :=
  isCondKernelCDF_stieltjesOfMeasurableRat (isRatCondKernelCDF_density_Iic κ)


/-- Auxiliary definition for `ProbabilityTheory.Kernel.condKernel`.
A conditional kernel for `κ : Kernel α (γ × ℝ)` where `γ` is countably generated. -/
noncomputable
def condKernelReal (κ : Kernel α (γ × ℝ)) [IsFiniteKernel κ] : Kernel (α × γ) ℝ :=
  (isCondKernelCDF_condKernelCDF κ).toKernel


instance instIsMarkovKernelCondKernelReal (κ : Kernel α (γ × ℝ)) [IsFiniteKernel κ] :
    IsMarkovKernel (condKernelReal κ) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod γ Real)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel κ.condKernelReal
  -/
  rw [condKernelReal]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod γ Real)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.IsCondKernelCDF.toKernel …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


lemma compProd_fst_condKernelReal (κ : Kernel α (γ × ℝ)) [IsFiniteKernel κ] :
    fst κ ⊗ₖ condKernelReal κ = κ := by
  /-
    α : Type u_1
    γ : Type u_3
    mα : MeasurableSpace α
    mγ : MeasurableSpace γ
    inst✝¹ : MeasurableSpace.CountablyGenerated γ
    κ : ProbabilityTheory.Kernel α (Prod γ Real)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ Eq (κ.fst.compProd κ.condKernelReal) κ
  -/
  rw [condKernelReal, compProd_toKernel]
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `MeasureTheory.Measure.condKernel` and
`ProbabilityTheory.Kernel.condKernel`.
A conditional kernel for `κ : Kernel Unit (α × ℝ)`. -/
noncomputable
def condKernelUnitReal (κ : Kernel Unit (α × ℝ)) [IsFiniteKernel κ] : Kernel (Unit × α) ℝ :=
  (isCondKernelCDF_condCDF (κ ())).toKernel


instance instIsMarkovKernelCondKernelUnitReal (κ : Kernel Unit (α × ℝ)) [IsFiniteKernel κ] :
    IsMarkovKernel (condKernelUnitReal κ) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    κ : ProbabilityTheory.Kernel Unit (Prod α Real)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel κ.condKernelUnitReal
  -/
  rw [condKernelUnitReal]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    κ : ProbabilityTheory.Kernel Unit (Prod α Real)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.IsCondKernelCDF.toKernel …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance condKernelUnitReal.instIsCondKernel (κ : Kernel Unit (α × ℝ)) [IsFiniteKernel κ] :
    κ.IsCondKernel κ.condKernelUnitReal where
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       Ω : Type u_4
                       mα : MeasurableSpace α
                       mβ : MeasurableSpace β
                       mγ : MeasurableSpace γ
                       inst✝⁴ : MeasurableSpace.CountablyGenerated γ
                       inst✝³ : MeasurableSpace Ω
                       inst✝² : StandardBorelSpace Ω
                       inst✝¹ : Nonempty Ω
                       κ : ProbabilityTheory.Kernel Unit (Prod α Real)
                       inst✝ : ProbabilityTheory.IsFiniteKernel κ
                       ⊢ Eq (κ.fst.compProd κ.condKernelUnitReal) κ
                     -/
  disintegrate := by rw [condKernelUnitReal, compProd_toKernel]; ext; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


@[deprecated disintegrate (since := "2024-07-26")]
lemma compProd_fst_condKernelUnitReal (κ : Kernel Unit (α × ℝ)) [IsFiniteKernel κ] :
    fst κ ⊗ₖ condKernelUnitReal κ = κ := disintegrate _ _


open Classical in
/-- Auxiliary definition for `ProbabilityTheory.Kernel.condKernel`.
A Borel space `Ω` embeds measurably into `ℝ` (with embedding `e`), hence we can get a `Kernel α Ω`
from a `Kernel α ℝ` by taking the comap by `e`.
Here we take the comap of a modification of `η : Kernel α ℝ`, useful when `η a` is a probability
measure with all its mass on `range e` almost everywhere with respect to some measure and we want to
ensure that the comap is a Markov kernel.
We thus take the comap by `e` of a kernel defined piecewise: `η` when
`η a (range (embeddingReal Ω))ᶜ = 0`, and an arbitrary deterministic kernel otherwise. -/
noncomputable
def borelMarkovFromReal (Ω : Type*) [Nonempty Ω] [MeasurableSpace Ω] [StandardBorelSpace Ω]
    (η : Kernel α ℝ) :
    Kernel α Ω :=
  have he := measurableEmbedding_embeddingReal Ω
  let x₀ := (range_nonempty (embeddingReal Ω)).choose
  comapRight
    (piecewise ((Kernel.measurable_coe η he.measurableSet_range.compl) (measurableSet_singleton 0) :
        MeasurableSet {a | η a (range (embeddingReal Ω))ᶜ = 0})
      η (deterministic (fun _ ↦ x₀) measurable_const)) he


lemma borelMarkovFromReal_apply (Ω : Type*) [Nonempty Ω] [MeasurableSpace Ω] [StandardBorelSpace Ω]
    (η : Kernel α ℝ) (a : α) :
    borelMarkovFromReal Ω η a
      = if η a (range (embeddingReal Ω))ᶜ = 0 then (η a).comap (embeddingReal Ω)
        else (Measure.dirac (range_nonempty (embeddingReal Ω)).choose).comap (embeddingReal Ω) := by
  classical
  rw [borelMarkovFromReal, comapRight_apply, piecewise_apply, deterministic_apply]
  simp only [mem_preimage, mem_singleton_iff]
  split_ifs <;> rfl


lemma borelMarkovFromReal_apply' (Ω : Type*) [Nonempty Ω] [MeasurableSpace Ω] [StandardBorelSpace Ω]
    (η : Kernel α ℝ) (a : α) {s : Set Ω} (hs : MeasurableSet s) :
    borelMarkovFromReal Ω η a s
      = if η a (range (embeddingReal Ω))ᶜ = 0 then η a (embeddingReal Ω '' s)
        else (embeddingReal Ω '' s).indicator 1 (range_nonempty (embeddingReal Ω)).choose := by
  /-
    α : Type u_1
    mα : MeasurableSpace α
    Ω : Type u_5
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace Ω
    inst✝ : StandardBorelSpace Ω
    η : ProbabilityTheory.Kernel α Real
    a : α
    s : Set Ω
    hs : MeasurableSet s
    ⊢ Eq (((ProbabilityTheory.Kernel.borelMarkovFromReal Ω η) a) s) (ite (Eq ((η a …
  -/
  have he := measurableEmbedding_embeddingReal Ω
  /-
    α : Type u_1
    mα : MeasurableSpace α
    Ω : Type u_5
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace Ω
    inst✝ : StandardBorelSpace Ω
    η : ProbabilityTheory.Kernel α Real
    a : α
    s : Set Ω
    hs : MeasurableSet s
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
    ⊢ Eq (((ProbabilityTheory.Kernel.borelMarkovFromReal Ω η) a) s) (ite (Eq ((η a …
  -/
  rw [borelMarkovFromReal_apply]
  /-
    α : Type u_1
    mα : MeasurableSpace α
    Ω : Type u_5
    inst✝² : Nonempty Ω
    inst✝¹ : MeasurableSpace Ω
    inst✝ : StandardBorelSpace Ω
    η : ProbabilityTheory.Kernel α Real
    a : α
    s : Set Ω
    hs : MeasurableSet s
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
    ⊢ Eq ((ite (Eq ((η a) (HasCompl.compl (Set.range (MeasureTheory.embeddingReal  …
  -/
  split_ifs with h
    /-
      case pos
      α : Type u_1
      mα : MeasurableSpace α
      Ω : Type u_5
      inst✝² : Nonempty Ω
      inst✝¹ : MeasurableSpace Ω
      inst✝ : StandardBorelSpace Ω
      η : ProbabilityTheory.Kernel α Real
      a : α
      s : Set Ω
      hs : MeasurableSet s
      he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
      h : Eq ((η a) (HasCompl.compl (Set.range (MeasureTheory.embeddingReal Ω)))) 0
      ⊢ Eq ((MeasureTheory.Measure.comap (MeasureTheory.embeddingReal Ω) (η a)) s) ( …
    -/
  · rw [Measure.comap_apply _ he.injective he.measurableSet_image' _ hs]
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      mα : MeasurableSpace α
      Ω : Type u_5
      inst✝² : Nonempty Ω
      inst✝¹ : MeasurableSpace Ω
      inst✝ : StandardBorelSpace Ω
      η : ProbabilityTheory.Kernel α Real
      a : α
      s : Set Ω
      hs : MeasurableSet s
      he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω)
      h : Not (Eq ((η a) (HasCompl.compl (Set.range (MeasureTheory.embeddingReal Ω)) …
      ⊢ Eq ((MeasureTheory.Measure.comap (MeasureTheory.embeddingReal Ω) (MeasureThe …
    -/
  · rw [Measure.comap_apply _ he.injective he.measurableSet_image' _ hs, Measure.dirac_apply]
    /-
      🎉 no goals
    -/


/-- When `η` is an s-finite kernel, `borelMarkovFromReal Ω η` is an s-finite kernel. -/
instance instIsSFiniteKernelBorelMarkovFromReal (η : Kernel α ℝ) [IsSFiniteKernel η] :
    IsSFiniteKernel (borelMarkovFromReal Ω η) :=
  IsSFiniteKernel.comapRight _ (measurableEmbedding_embeddingReal Ω)


/-- When `η` is a finite kernel, `borelMarkovFromReal Ω η` is a finite kernel. -/
instance instIsFiniteKernelBorelMarkovFromReal (η : Kernel α ℝ) [IsFiniteKernel η] :
    IsFiniteKernel (borelMarkovFromReal Ω η) :=
  IsFiniteKernel.comapRight _ (measurableEmbedding_embeddingReal Ω)


/-- When `η` is a Markov kernel, `borelMarkovFromReal Ω η` is a Markov kernel. -/
instance instIsMarkovKernelBorelMarkovFromReal (η : Kernel α ℝ) [IsMarkovKernel η] :
    IsMarkovKernel (borelMarkovFromReal Ω η) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    η : ProbabilityTheory.Kernel α Real
    inst✝ : ProbabilityTheory.IsMarkovKernel η
    ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.Kernel.borelMarkovFromRe …
  -/
  refine IsMarkovKernel.comapRight _ (measurableEmbedding_embeddingReal Ω) (fun a ↦ ?_)
  classical
  rw [piecewise_apply]
  split_ifs with h
  · rwa [← prob_compl_eq_zero_iff (measurableEmbedding_embeddingReal Ω).measurableSet_range]
  · rw [deterministic_apply]
    simp [(range_nonempty (embeddingReal Ω)).choose_spec]


/-- For `κ' := map κ (Prod.map (id : β → β) e)`, the hypothesis `hη` is `fst κ' ⊗ₖ η = κ'`.
The conclusion of the lemma is `fst κ ⊗ₖ borelMarkovFromReal Ω η = comapRight (fst κ' ⊗ₖ η) _`. -/
lemma compProd_fst_borelMarkovFromReal_eq_comapRight_compProd
    (κ : Kernel α (β × Ω)) [IsSFiniteKernel κ] (η : Kernel (α × β) ℝ) [IsSFiniteKernel η]
    (hη : (fst (map κ (Prod.map (id : β → β) (embeddingReal Ω)))) ⊗ₖ η
      = map κ (Prod.map (id : β → β) (embeddingReal Ω))) :
    fst κ ⊗ₖ borelMarkovFromReal Ω η
      = comapRight (fst (map κ (Prod.map (id : β → β) (embeddingReal Ω))) ⊗ₖ η)
        (MeasurableEmbedding.id.prodMap (measurableEmbedding_embeddingReal Ω)) := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) (((κ. …
  -/
  let e := embeddingReal Ω
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) (((κ. …
  -/
  let he := measurableEmbedding_embeddingReal Ω
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) (((κ. …
  -/
  let κ' := map κ (Prod.map (id : β → β) e)
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) (((κ. …
  -/
  have hη' : fst κ' ⊗ₖ η = κ' := hη
  have h_prod_embed : MeasurableEmbedding (Prod.map (id : β → β) e) :=
    MeasurableEmbedding.id.prodMap he
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) (((κ. …
  -/
  change fst κ ⊗ₖ borelMarkovFromReal Ω η = comapRight (fst κ' ⊗ₖ η) h_prod_embed
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) ((κ'. …
  -/
  rw [comapRight_compProd_id_prod _ _ he]
  have h_fst : fst κ' = fst κ := by
    ext a u
    unfold κ'
    rw [fst_apply, map_apply _ (by fun_prop),
      Measure.map_map measurable_fst h_prod_embed.measurable, fst_apply]
    congr
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    h_fst : Eq κ'.fst κ.fst
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) (κ'.f …
  -/
  rw [h_fst]
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    h_fst : Eq κ'.fst κ.fst
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) (κ.fs …
  -/
  ext a t ht : 2
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    h_fst : Eq κ'.fst κ.fst
    a : α
    t : Set (Prod β Ω)
    ht : MeasurableSet t
    ⊢ Eq (((κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) a)  …
  -/
  simp_rw [compProd_apply ht]
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    h_fst : Eq κ'.fst κ.fst
    a : α
    t : Set (Prod β Ω)
    ht : MeasurableSet t
    ⊢ Eq (MeasureTheory.lintegral (κ.fst a) fun b => ((ProbabilityTheory.Kernel.bo …
  -/
  refine lintegral_congr_ae ?_
  have h_ae : ∀ᵐ t ∂(fst κ a), (a, t) ∈ {p : α × β | η p (range e)ᶜ = 0} := by
    rw [← h_fst]
    have h_compProd : κ' a (univ ×ˢ range e)ᶜ = 0 := by
      unfold κ'
      rw [map_apply' _ (by fun_prop)]
      swap; · exact (MeasurableSet.univ.prod he.measurableSet_range).compl
      suffices Prod.map id e ⁻¹' (univ ×ˢ range e)ᶜ = ∅ by rw [this]; simp
      ext x
      simp
    rw [← hη', compProd_null] at h_compProd
    swap; · exact (MeasurableSet.univ.prod he.measurableSet_range).compl
    simp only [preimage_compl, mem_univ, mk_preimage_prod_right] at h_compProd
    exact h_compProd
  /-
    case h.h
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    h_fst : Eq κ'.fst κ.fst
    a : α
    t : Set (Prod β Ω)
    ht : MeasurableSet t
    h_ae : Filter.Eventually (fun t => Membership.mem (setOf fun p => Eq ((η p) (H …
    ⊢ (MeasureTheory.ae (κ.fst a)).EventuallyEq (fun b => ((ProbabilityTheory.Kern …
  -/
  filter_upwards [h_ae] with a ha
  /-
    case h
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    h_fst : Eq κ'.fst κ.fst
    a✝ : α
    t : Set (Prod β Ω)
    ht : MeasurableSet t
    h_ae : Filter.Eventually (fun t => Membership.mem (setOf fun p => Eq ((η p) (H …
    a : β
    ha : Eq ((η { fst := a✝, snd := a }) (HasCompl.compl (Set.range e))) 0
    ⊢ Eq (((ProbabilityTheory.Kernel.borelMarkovFromReal Ω η) { fst := a✝, snd :=  …
  -/
  rw [borelMarkovFromReal, comapRight_apply', comapRight_apply']
  /-
    case h
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    h_fst : Eq κ'.fst κ.fst
    a✝ : α
    t : Set (Prod β Ω)
    ht : MeasurableSet t
    h_ae : Filter.Eventually (fun t => Membership.mem (setOf fun p => Eq ((η p) (H …
    a : β
    ha : Eq ((η { fst := a✝, snd := a }) (HasCompl.compl (Set.range e))) 0
    ⊢ Eq (((ProbabilityTheory.Kernel.piecewise ⋯ η (ProbabilityTheory.Kernel.deter …
  -/
  rotate_left
    /-
      case h.ht
      α : Type u_1
      β : Type u_2
      Ω : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : StandardBorelSpace Ω
      inst✝² : Nonempty Ω
      κ : ProbabilityTheory.Kernel α (Prod β Ω)
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) Real
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
      e : Ω → Real := MeasureTheory.embeddingReal Ω
      he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
      κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
      hη' : Eq (κ'.fst.compProd η) κ'
      h_prod_embed : MeasurableEmbedding (Prod.map id e)
      h_fst : Eq κ'.fst κ.fst
      a✝ : α
      t : Set (Prod β Ω)
      ht : MeasurableSet t
      h_ae : Filter.Eventually (fun t => Membership.mem (setOf fun p => Eq ((η p) (H …
      a : β
      ha : Eq ((η { fst := a✝, snd := a }) (HasCompl.compl (Set.range e))) 0
      ⊢ MeasurableSet (setOf fun c => Membership.mem t { fst := a, snd := c })
    -/
  · exact measurable_prod_mk_left ht
    /-
      🎉 no goals
    -/
    /-
      case h.ht
      α : Type u_1
      β : Type u_2
      Ω : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : StandardBorelSpace Ω
      inst✝² : Nonempty Ω
      κ : ProbabilityTheory.Kernel α (Prod β Ω)
      inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
      η : ProbabilityTheory.Kernel (Prod α β) Real
      inst✝ : ProbabilityTheory.IsSFiniteKernel η
      hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
      e : Ω → Real := MeasureTheory.embeddingReal Ω
      he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
      κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
      hη' : Eq (κ'.fst.compProd η) κ'
      h_prod_embed : MeasurableEmbedding (Prod.map id e)
      h_fst : Eq κ'.fst κ.fst
      a✝ : α
      t : Set (Prod β Ω)
      ht : MeasurableSet t
      h_ae : Filter.Eventually (fun t => Membership.mem (setOf fun p => Eq ((η p) (H …
      a : β
      ha : Eq ((η { fst := a✝, snd := a }) (HasCompl.compl (Set.range e))) 0
      ⊢ MeasurableSet (setOf fun c => Membership.mem t { fst := a, snd := c })
    -/
  · exact measurable_prod_mk_left ht
    /-
      🎉 no goals
    -/
  classical
  rw [piecewise_apply, if_pos]
  exact ha


/-- For `κ' := map κ (Prod.map (id : β → β) e)`, the hypothesis `hη` is `fst κ' ⊗ₖ η = κ'`.
With that hypothesis, `fst κ ⊗ₖ borelMarkovFromReal κ η = κ`.-/
lemma compProd_fst_borelMarkovFromReal (κ : Kernel α (β × Ω)) [IsSFiniteKernel κ]
    (η : Kernel (α × β) ℝ) [IsSFiniteKernel η]
    (hη : (fst (map κ (Prod.map (id : β → β) (embeddingReal Ω)))) ⊗ₖ η
      = map κ (Prod.map (id : β → β) (embeddingReal Ω))) :
    fst κ ⊗ₖ borelMarkovFromReal Ω η = κ := by
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) κ
  -/
  let e := embeddingReal Ω
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) κ
  -/
  let he := measurableEmbedding_embeddingReal Ω
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) κ
  -/
  let κ' := map κ (Prod.map (id : β → β) e)
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) κ
  -/
  have hη' : fst κ' ⊗ₖ η = κ' := hη
  have h_prod_embed : MeasurableEmbedding (Prod.map (id : β → β) e) :=
    MeasurableEmbedding.id.prodMap he
  have : κ = comapRight κ' h_prod_embed := by
    ext c t : 2
    unfold κ'
    rw [comapRight_apply, map_apply _ (by fun_prop), h_prod_embed.comap_map]
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    this : Eq κ (κ'.comapRight h_prod_embed)
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) κ
  -/
  conv_rhs => rw [this, ← hη']
  /-
    α : Type u_1
    β : Type u_2
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝¹ : ProbabilityTheory.IsSFiniteKernel κ
    η : ProbabilityTheory.Kernel (Prod α β) Real
    inst✝ : ProbabilityTheory.IsSFiniteKernel η
    hη : Eq ((κ.map (Prod.map id (MeasureTheory.embeddingReal Ω))).fst.compProd η) …
    e : Ω → Real := MeasureTheory.embeddingReal Ω
    he : MeasurableEmbedding (MeasureTheory.embeddingReal Ω) := MeasureTheory.meas …
    κ' : ProbabilityTheory.Kernel α (Prod β Real) := κ.map (Prod.map id e)
    hη' : Eq (κ'.fst.compProd η) κ'
    h_prod_embed : MeasurableEmbedding (Prod.map id e)
    this : Eq κ (κ'.comapRight h_prod_embed)
    ⊢ Eq (κ.fst.compProd (ProbabilityTheory.Kernel.borelMarkovFromReal Ω η)) ((κ'. …
  -/
  exact compProd_fst_borelMarkovFromReal_eq_comapRight_compProd κ η hη
  /-
    🎉 no goals
  -/


/-- Auxiliary definition for `ProbabilityTheory.Kernel.condKernel`.
A conditional kernel for `κ : Kernel α (γ × Ω)` where `γ` is countably generated and `Ω` is
standard Borel. -/
noncomputable
def condKernelBorel (κ : Kernel α (γ × Ω)) [IsFiniteKernel κ] : Kernel (α × γ) Ω :=
  let κ' := map κ (Prod.map (id : γ → γ) (embeddingReal Ω))
  borelMarkovFromReal Ω (condKernelReal κ')


instance instIsMarkovKernelCondKernelBorel (κ : Kernel α (γ × Ω)) [IsFiniteKernel κ] :
    IsMarkovKernel (condKernelBorel κ) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod γ Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel κ.condKernelBorel
  -/
  rw [condKernelBorel]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    κ : ProbabilityTheory.Kernel α (Prod γ Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.Kernel.borelMarkovFromRe …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance condKernelBorel.instIsCondKernel (κ : Kernel α (γ × Ω)) [IsFiniteKernel κ] :
    κ.IsCondKernel κ.condKernelBorel where
  disintegrate := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      Ω : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝⁴ : MeasurableSpace.CountablyGenerated γ
      inst✝³ : MeasurableSpace Ω
      inst✝² : StandardBorelSpace Ω
      inst✝¹ : Nonempty Ω
      κ : ProbabilityTheory.Kernel α (Prod γ Ω)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      ⊢ Eq (κ.fst.compProd κ.condKernelBorel) κ
    -/
    rw [condKernelBorel, compProd_fst_borelMarkovFromReal _ _ (compProd_fst_condKernelReal _)]
    /-
      🎉 no goals
    -/


@[deprecated disintegrate (since := "2024-07-26")]
lemma compProd_fst_condKernelBorel (κ : Kernel α (γ × Ω)) [IsFiniteKernel κ] :
    fst κ ⊗ₖ condKernelBorel κ = κ := disintegrate _ _


/-- Auxiliary definition for `MeasureTheory.Measure.condKernel` and
`ProbabilityTheory.Kernel.condKernel`.
A conditional kernel for `κ : Kernel Unit (α × Ω)` where `Ω` is standard Borel. -/
noncomputable
def condKernelUnitBorel : Kernel (Unit × α) Ω :=
  let κ' := map κ (Prod.map (id : α → α) (embeddingReal Ω))
  borelMarkovFromReal Ω (condKernelUnitReal κ')


instance instIsMarkovKernelCondKernelUnitBorel : IsMarkovKernel κ.condKernelUnitBorel := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    κ : ProbabilityTheory.Kernel Unit (Prod α Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel κ.condKernelUnitBorel
  -/
  rw [condKernelUnitBorel]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    κ : ProbabilityTheory.Kernel Unit (Prod α Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel (ProbabilityTheory.Kernel.borelMarkovFromRe …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance condKernelUnitBorel.instIsCondKernel : κ.IsCondKernel κ.condKernelUnitBorel where
  disintegrate := by
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      Ω : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝⁴ : MeasurableSpace.CountablyGenerated γ
      inst✝³ : MeasurableSpace Ω
      inst✝² : StandardBorelSpace Ω
      inst✝¹ : Nonempty Ω
      κ : ProbabilityTheory.Kernel Unit (Prod α Ω)
      inst✝ : ProbabilityTheory.IsFiniteKernel κ
      ⊢ Eq (κ.fst.compProd κ.condKernelUnitBorel) κ
    -/
    rw [condKernelUnitBorel, compProd_fst_borelMarkovFromReal _ _ (disintegrate _ _)]
    /-
      🎉 no goals
    -/


@[deprecated disintegrate (since := "2024-07-26")]
lemma compProd_fst_condKernelUnitBorel (κ : Kernel Unit (α × Ω)) [IsFiniteKernel κ] :
    fst κ ⊗ₖ condKernelUnitBorel κ = κ := disintegrate _ _


/-- Conditional kernel of a measure on a product space: a Markov kernel such that
`ρ = ρ.fst ⊗ₘ ρ.condKernel` (see `MeasureTheory.Measure.compProd_fst_condKernel`). -/
noncomputable
irreducible_def _root_.MeasureTheory.Measure.condKernel (ρ : Measure (α × Ω)) [IsFiniteMeasure ρ] :
    Kernel α Ω :=
  comap (condKernelUnitBorel (const Unit ρ)) (fun a ↦ ((), a)) measurable_prod_mk_left


lemma _root_.MeasureTheory.Measure.condKernel_apply (ρ : Measure (α × Ω)) [IsFiniteMeasure ρ]
    (a : α) :
    ρ.condKernel a = condKernelUnitBorel (const Unit ρ) ((), a) := by
  /-
    α : Type u_1
    Ω : Type u_4
    mα : MeasurableSpace α
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    a : α
    ⊢ Eq (ρ.condKernel a) ((ProbabilityTheory.Kernel.const Unit ρ).condKernelUnitB …
  -/
  rw [Measure.condKernel]; rfl
                           /-
                             🎉 no goals
                           -/


instance _root_.MeasureTheory.Measure.condKernel.instIsCondKernel (ρ : Measure (α × Ω))
    [IsFiniteMeasure ρ] : ρ.IsCondKernel ρ.condKernel where
  disintegrate := by
    have h1 : const Unit (Measure.fst ρ) = fst (const Unit ρ) := by
      ext
      simp only [fst_apply, Measure.fst, const_apply]
    have h2 : prodMkLeft Unit (Measure.condKernel ρ) = condKernelUnitBorel (const Unit ρ) := by
      ext
      simp only [prodMkLeft_apply, Measure.condKernel_apply]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      Ω : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝⁵ : MeasurableSpace.CountablyGenerated γ
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : StandardBorelSpace Ω
      inst✝² : Nonempty Ω
      ρ✝ : MeasureTheory.Measure (Prod α Ω)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ✝
      ρ : MeasureTheory.Measure (Prod α Ω)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      h1 : Eq (ProbabilityTheory.Kernel.const Unit ρ.fst) (ProbabilityTheory.Kernel. …
      h2 : Eq (ProbabilityTheory.Kernel.prodMkLeft Unit ρ.condKernel) (ProbabilityTh …
      ⊢ Eq (ρ.fst.compProd ρ.condKernel) ρ
    -/
    rw [Measure.compProd, h1, h2, disintegrate]
    /-
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      Ω : Type u_4
      mα : MeasurableSpace α
      mβ : MeasurableSpace β
      mγ : MeasurableSpace γ
      inst✝⁵ : MeasurableSpace.CountablyGenerated γ
      inst✝⁴ : MeasurableSpace Ω
      inst✝³ : StandardBorelSpace Ω
      inst✝² : Nonempty Ω
      ρ✝ : MeasureTheory.Measure (Prod α Ω)
      inst✝¹ : MeasureTheory.IsFiniteMeasure ρ✝
      ρ : MeasureTheory.Measure (Prod α Ω)
      inst✝ : MeasureTheory.IsFiniteMeasure ρ
      h1 : Eq (ProbabilityTheory.Kernel.const Unit ρ.fst) (ProbabilityTheory.Kernel. …
      h2 : Eq (ProbabilityTheory.Kernel.prodMkLeft Unit ρ.condKernel) (ProbabilityTh …
      ⊢ Eq ((ProbabilityTheory.Kernel.const Unit ρ) Unit.unit) ρ
    -/
    simp
    /-
      🎉 no goals
    -/


instance _root_.MeasureTheory.Measure.instIsMarkovKernelCondKernel
    (ρ : Measure (α × Ω)) [IsFiniteMeasure ρ] : IsMarkovKernel ρ.condKernel := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁵ : MeasurableSpace.CountablyGenerated γ
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ✝ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ✝
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ ProbabilityTheory.IsMarkovKernel ρ.condKernel
  -/
  rw [Measure.condKernel]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁵ : MeasurableSpace.CountablyGenerated γ
    inst✝⁴ : MeasurableSpace Ω
    inst✝³ : StandardBorelSpace Ω
    inst✝² : Nonempty Ω
    ρ✝ : MeasureTheory.Measure (Prod α Ω)
    inst✝¹ : MeasureTheory.IsFiniteMeasure ρ✝
    ρ : MeasureTheory.Measure (Prod α Ω)
    inst✝ : MeasureTheory.IsFiniteMeasure ρ
    ⊢ ProbabilityTheory.IsMarkovKernel ((ProbabilityTheory.Kernel.const Unit ρ).co …
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- **Disintegration** of finite product measures on `α × Ω`, where `Ω` is standard Borel. Such a
measure can be written as the composition-product of `ρ.fst` (marginal measure over `α`) and
a Markov kernel from `α` to `Ω`. We call that Markov kernel `ρ.condKernel`. -/
@[deprecated Measure.disintegrate (since := "2024-07-24")]
lemma _root_.MeasureTheory.Measure.compProd_fst_condKernel
    (ρ : Measure (α × Ω)) [IsFiniteMeasure ρ] :
    ρ.fst ⊗ₘ ρ.condKernel = ρ := ρ.disintegrate ρ.condKernel


set_option linter.unusedVariables false in
/-- Auxiliary lemma for `condKernel_apply_of_ne_zero`. -/
@[deprecated Measure.IsCondKernel.apply_of_ne_zero (since := "2024-07-24"), nolint unusedArguments]
lemma _root_.MeasureTheory.Measure.condKernel_apply_of_ne_zero_of_measurableSet
    [MeasurableSingletonClass α] {x : α} (hx : ρ.fst {x} ≠ 0) {s : Set Ω} (hs : MeasurableSet s) :
    ρ.condKernel x s = (ρ.fst {x})⁻¹ * ρ ({x} ×ˢ s) :=
  Measure.IsCondKernel.apply_of_ne_zero _ _ hx _


/-- If the singleton `{x}` has non-zero mass for `ρ.fst`, then for all `s : Set Ω`,
`ρ.condKernel x s = (ρ.fst {x})⁻¹ * ρ ({x} ×ˢ s)` . -/
lemma _root_.MeasureTheory.Measure.condKernel_apply_of_ne_zero [MeasurableSingletonClass α]
    {x : α} (hx : ρ.fst {x} ≠ 0) (s : Set Ω) :
    ρ.condKernel x s = (ρ.fst {x})⁻¹ * ρ ({x} ×ˢ s) :=
  Measure.IsCondKernel.apply_of_ne_zero _ _ hx _


@[deprecated disintegrate (since := "2024-07-24")]
lemma compProd_fst_condKernelCountable (κ : Kernel α (β × Ω)) [IsFiniteKernel κ] :
    fst κ ⊗ₖ condKernelCountable (fun a ↦ (κ a).condKernel)
                      /-
                        α : Type u_1
                        β : Type u_2
                        γ : Type u_3
                        Ω : Type u_4
                        mα : MeasurableSpace α
                        mβ : MeasurableSpace β
                        mγ : MeasurableSpace γ
                        inst✝⁵ : MeasurableSpace.CountablyGenerated γ
                        inst✝⁴ : MeasurableSpace Ω
                        inst✝³ : StandardBorelSpace Ω
                        inst✝² : Nonempty Ω
                        inst✝¹ : Countable α
                        κ : ProbabilityTheory.Kernel α (Prod β Ω)
                        inst✝ : ProbabilityTheory.IsFiniteKernel κ
                        x y : α
                        h : Membership.mem (measurableAtom y) x
                        ⊢ Eq ((fun a => (κ a).condKernel) x) ((fun a => (κ a).condKernel) y)
                      -/
      (fun x y h ↦ by simp [apply_congr_of_mem_measurableAtom _ h]) = κ := disintegrate _ _
                      /-
                        🎉 no goals
                      -/


open Classical in

/-- Conditional kernel of a kernel `κ : Kernel α (β × Ω)`: a Markov kernel such that
`fst κ ⊗ₖ condKernel κ = κ` (see `MeasureTheory.Measure.compProd_fst_condKernel`).
It exists whenever `Ω` is standard Borel and either `α` is countable
or `β` is countably generated. -/
noncomputable
irreducible_def condKernel : Kernel (α × β) Ω :=
  if hα : Countable α then
    condKernelCountable (fun a ↦ (κ a).condKernel)
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       Ω : Type u_4
                       mα : MeasurableSpace α
                       mβ : MeasurableSpace β
                       mγ : MeasurableSpace γ
                       inst✝⁴ : MeasurableSpace.CountablyGenerated γ
                       inst✝³ : MeasurableSpace Ω
                       inst✝² : StandardBorelSpace Ω
                       inst✝¹ : Nonempty Ω
                       h✝ : MeasurableSpace.CountableOrCountablyGenerated α β
                       κ : ProbabilityTheory.Kernel α (Prod β Ω)
                       inst✝ : ProbabilityTheory.IsFiniteKernel κ
                       hα : Countable α
                       x y : α
                       h : Membership.mem (measurableAtom y) x
                       ⊢ Eq ((fun a => (κ a).condKernel) x) ((fun a => (κ a).condKernel) y)
                     -/
      fun x y h ↦ by simp [apply_congr_of_mem_measurableAtom _ h]
                     /-
                       🎉 no goals
                     -/
  else letI := h.countableOrCountablyGenerated.resolve_left hα; condKernelBorel κ


/-- `condKernel κ` is a Markov kernel. -/
instance instIsMarkovKernelCondKernel : IsMarkovKernel (condKernel κ) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    h : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel κ.condKernel
  -/
  rw [condKernel_def]
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    Ω : Type u_4
    mα : MeasurableSpace α
    mβ : MeasurableSpace β
    mγ : MeasurableSpace γ
    inst✝⁴ : MeasurableSpace.CountablyGenerated γ
    inst✝³ : MeasurableSpace Ω
    inst✝² : StandardBorelSpace Ω
    inst✝¹ : Nonempty Ω
    h : MeasurableSpace.CountableOrCountablyGenerated α β
    κ : ProbabilityTheory.Kernel α (Prod β Ω)
    inst✝ : ProbabilityTheory.IsFiniteKernel κ
    ⊢ ProbabilityTheory.IsMarkovKernel (dite (Countable α) (fun hα => ProbabilityT …
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> infer_instance
                /-
                  🎉 no goals
                -/


instance condKernel.instIsCondKernel : κ.IsCondKernel κ.condKernel where
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       Ω : Type u_4
                       mα : MeasurableSpace α
                       mβ : MeasurableSpace β
                       mγ : MeasurableSpace γ
                       inst✝⁴ : MeasurableSpace.CountablyGenerated γ
                       inst✝³ : MeasurableSpace Ω
                       inst✝² : StandardBorelSpace Ω
                       inst✝¹ : Nonempty Ω
                       h : MeasurableSpace.CountableOrCountablyGenerated α β
                       κ : ProbabilityTheory.Kernel α (Prod β Ω)
                       inst✝ : ProbabilityTheory.IsFiniteKernel κ
                       ⊢ Eq (κ.fst.compProd κ.condKernel) κ
                     -/
                                                                /-
                                                                  🎉 no goals
                                                                -/
  disintegrate := by rw [condKernel_def]; split_ifs with hα <;> exact disintegrate _ _
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- **Disintegration** of finite kernels.
The composition-product of `fst κ` and `condKernel κ` is equal to `κ`. -/
@[deprecated Kernel.disintegrate (since := "2024-07-26")]
lemma compProd_fst_condKernel : fst κ ⊗ₖ condKernel κ = κ := κ.disintegrate κ.condKernel


