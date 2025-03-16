/-- For two σ-algebras `𝓑 ≤ 𝓧` on a space `X`, a `𝓑, 𝓧`-kernel `π : X → Measure X` is proper if
`∫ x, g x * f x ∂(π x₀) = g x₀ * ∫ x, f x ∂(π x₀)` for all `x₀ : X`, `𝓧`-measurable function `f`
and `𝓑`-measurable function `g`.

By the standard machine, this is equivalent to having that, for all `B ∈ 𝓑`, `π` restricted to `B`
is the same as `π` times the indicator of `B`.

To avoid assuming `𝓑 ≤ 𝓧` in the definition, we replace `𝓑` by `𝓑 ⊓ 𝓧` in the restriction. -/
structure IsProper (π : Kernel[𝓑, 𝓧] X X) : Prop where
  restrict_eq_indicator_smul' :
    ∀ ⦃B : Set X⦄ (hB : MeasurableSet[𝓑 ⊓ 𝓧] B) (x : X),
      π.restrict (inf_le_right (b := 𝓧) _ hB) x = B.indicator (fun _ ↦ (1 : ℝ≥0∞)) x • π x


lemma isProper_iff_restrict_eq_indicator_smul (h𝓑𝓧 : 𝓑 ≤ 𝓧) :
    IsProper π ↔ ∀ ⦃B : Set X⦄ (hB : MeasurableSet[𝓑] B) (x : X),
      π.restrict (h𝓑𝓧 _ hB) x = B.indicator (fun _ ↦ (1 : ℝ≥0∞)) x • π x := by
  /-
    X : Type u_1
    𝓑 𝓧 : MeasurableSpace X
    π : ProbabilityTheory.Kernel X X
    h𝓑𝓧 : LE.le 𝓑 𝓧
    ⊢ Iff π.IsProper (∀ ⦃B : Set X⦄ (hB : MeasurableSet B) (x : X), Eq ((π.restric …
  -/
                                          /-
                                            🎉 no goals
                                          -/
  refine ⟨fun ⟨h⟩ ↦ ?_, fun h ↦ ⟨?_⟩⟩ <;> simpa only [inf_eq_left.2 h𝓑𝓧] using h
                                          /-
                                            🎉 no goals
                                          -/


lemma isProper_iff_inter_eq_indicator_mul (h𝓑𝓧 : 𝓑 ≤ 𝓧) :
    IsProper π ↔
      ∀ ⦃A : Set X⦄ (_hA : MeasurableSet[𝓧] A) ⦃B : Set X⦄ (_hB : MeasurableSet[𝓑] B) (x : X),
        π x (A ∩ B) = B.indicator 1 x * π x A := by
  calc
    _ ↔ ∀ ⦃A : Set X⦄ (_hA : MeasurableSet[𝓧] A) ⦃B : Set X⦄ (hB : MeasurableSet[𝓑] B) (x : X),
          π.restrict (h𝓑𝓧 _ hB) x A = B.indicator 1 x * π x A := by
      simp [isProper_iff_restrict_eq_indicator_smul h𝓑𝓧, Measure.ext_iff]; aesop
    _ ↔ _ := by congr! 5 with A hA B hB x; rw [restrict_apply, Measure.restrict_apply hA]


alias ⟨IsProper.restrict_eq_indicator_smul, IsProper.of_restrict_eq_indicator_smul⟩ :=
  isProper_iff_restrict_eq_indicator_smul


alias ⟨IsProper.inter_eq_indicator_mul, IsProper.of_inter_eq_indicator_mul⟩ :=
  isProper_iff_inter_eq_indicator_mul


lemma IsProper.setLIntegral_eq_bind (hπ : IsProper π) (h𝓑𝓧 : 𝓑 ≤ 𝓧) {μ : Measure[𝓧] X}
    (hA : MeasurableSet[𝓧] A) (hB : MeasurableSet[𝓑] B) :
    ∫⁻ a in B, π a A ∂μ = μ.bind π (A ∩ B) := by
  /-
    X : Type u_1
    𝓑 𝓧 : MeasurableSpace X
    π : ProbabilityTheory.Kernel X X
    A B : Set X
    hπ : π.IsProper
    h𝓑𝓧 : LE.le 𝓑 𝓧
    μ : MeasureTheory.Measure X
    hA : MeasurableSet A
    hB : MeasurableSet B
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict B) fun a => (π a) A) ((μ.bind ⇑π) (I …
  -/
  rw [Measure.bind_apply (by measurability) (π.measurable.mono h𝓑𝓧 le_rfl)]
  /-
    X : Type u_1
    𝓑 𝓧 : MeasurableSpace X
    π : ProbabilityTheory.Kernel X X
    A B : Set X
    hπ : π.IsProper
    h𝓑𝓧 : LE.le 𝓑 𝓧
    μ : MeasureTheory.Measure X
    hA : MeasurableSet A
    hB : MeasurableSet B
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict B) fun a => (π a) A) (MeasureTheory. …
  -/
  simp only [hπ.inter_eq_indicator_mul h𝓑𝓧 hA hB, ← indicator_mul_const, Pi.one_apply, one_mul]
  /-
    X : Type u_1
    𝓑 𝓧 : MeasurableSpace X
    π : ProbabilityTheory.Kernel X X
    A B : Set X
    hπ : π.IsProper
    h𝓑𝓧 : LE.le 𝓑 𝓧
    μ : MeasureTheory.Measure X
    hA : MeasurableSet A
    hB : MeasurableSet B
    ⊢ Eq (MeasureTheory.lintegral (μ.restrict B) fun a => (π a) A) (MeasureTheory. …
  -/
  rw [← lintegral_indicator (h𝓑𝓧 _ hB)]
  /-
    X : Type u_1
    𝓑 𝓧 : MeasurableSpace X
    π : ProbabilityTheory.Kernel X X
    A B : Set X
    hπ : π.IsProper
    h𝓑𝓧 : LE.le 𝓑 𝓧
    μ : MeasureTheory.Measure X
    hA : MeasurableSet A
    hB : MeasurableSet B
    ⊢ Eq (MeasureTheory.lintegral μ fun a => B.indicator (fun a => (π a) A) a) (Me …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- Auxiliary lemma for `IsProper.lintegral_mul` and
`IsProper.setLIntegral_eq_indicator_mul_lintegral`. -/
private lemma IsProper.lintegral_indicator_mul_indicator (hπ : IsProper π) (h𝓑𝓧 : 𝓑 ≤ 𝓧)
    (hA : MeasurableSet[𝓧] A) (hB : MeasurableSet[𝓑] B) :
    ∫⁻ x, B.indicator 1 x * A.indicator 1 x ∂(π x₀) =
      B.indicator 1 x₀ * ∫⁻ x, A.indicator 1 x ∂(π x₀) := by
  /-
    X : Type u_1
    𝓑 𝓧 : MeasurableSpace X
    π : ProbabilityTheory.Kernel X X
    A B : Set X
    x₀ : X
    hπ : π.IsProper
    h𝓑𝓧 : LE.le 𝓑 𝓧
    hA : MeasurableSet A
    hB : MeasurableSet B
    ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.indicator 1 x) (A.i …
  -/
  simp_rw [← inter_indicator_mul]
  /-
    X : Type u_1
    𝓑 𝓧 : MeasurableSpace X
    π : ProbabilityTheory.Kernel X X
    A B : Set X
    x₀ : X
    hπ : π.IsProper
    h𝓑𝓧 : LE.le 𝓑 𝓧
    hA : MeasurableSet A
    hB : MeasurableSet B
    ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => (Inter.inter B A).indicator (fun …
  -/
  rw [lintegral_indicator ((h𝓑𝓧 _ hB).inter hA), lintegral_indicator hA]
  simp only [MeasureTheory.lintegral_const, MeasurableSet.univ, Measure.restrict_apply, univ_inter,
    Pi.one_apply, one_mul]
  /-
    X : Type u_1
    𝓑 𝓧 : MeasurableSpace X
    π : ProbabilityTheory.Kernel X X
    A B : Set X
    x₀ : X
    hπ : π.IsProper
    h𝓑𝓧 : LE.le 𝓑 𝓧
    hA : MeasurableSet A
    hB : MeasurableSet B
    ⊢ Eq ((π x₀) (Inter.inter B A)) (HMul.hMul (B.indicator 1 x₀) ((π x₀) A))
  -/
  rw [← hπ.inter_eq_indicator_mul h𝓑𝓧 hA hB, inter_comm]
  /-
    🎉 no goals
  -/


set_option linter.style.multiGoal false in -- false positive
/-- Auxiliary lemma for `IsProper.lintegral_mul` and
`IsProper.setLIntegral_eq_indicator_mul_lintegral`. -/
private lemma IsProper.lintegral_indicator_mul (hπ : IsProper π) (h𝓑𝓧 : 𝓑 ≤ 𝓧)
    (hf : Measurable[𝓧] f) (hB : MeasurableSet[𝓑] B) :
    ∫⁻ x, B.indicator 1 x * f x ∂(π x₀) = B.indicator 1 x₀ * ∫⁻ x, f x ∂(π x₀) := by
  /-
    X : Type u_1
    𝓑 𝓧 : MeasurableSpace X
    π : ProbabilityTheory.Kernel X X
    B : Set X
    f : X → ENNReal
    x₀ : X
    hπ : π.IsProper
    h𝓑𝓧 : LE.le 𝓑 𝓧
    hf : Measurable f
    hB : MeasurableSet B
    ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.indicator 1 x) (f x …
  -/
  refine hf.ennreal_induction ?_ ?_ ?_
    /-
      case refine_1
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      B : Set X
      f : X → ENNReal
      x₀ : X
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hB : MeasurableSet B
      ⊢ ∀ (c : ENNReal) ⦃s : Set X⦄, MeasurableSet s → Eq (MeasureTheory.lintegral ( …
    -/
  · rintro c A hA
    /-
      case refine_1
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      B : Set X
      f : X → ENNReal
      x₀ : X
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hB : MeasurableSet B
      c : ENNReal
      A : Set X
      hA : MeasurableSet A
      ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.indicator 1 x) (A.i …
    -/
    simp_rw [← smul_indicator_one_apply, mul_smul_comm, smul_eq_mul]
    rw [lintegral_const_mul, lintegral_const_mul, hπ.lintegral_indicator_mul_indicator h𝓑𝓧 hA hB,
                         /-
                           case refine_1.hf
                           X : Type u_1
                           𝓑 𝓧 : MeasurableSpace X
                           π : ProbabilityTheory.Kernel X X
                           B : Set X
                           f : X → ENNReal
                           x₀ : X
                           hπ : π.IsProper
                           h𝓑𝓧 : LE.le 𝓑 𝓧
                           hf : Measurable f
                           hB : MeasurableSet B
                           c : ENNReal
                           A : Set X
                           hA : MeasurableSet A
                           ⊢ Measurable (A.indicator 1)
                         -/
                         /-
                           🎉 no goals
                         -/
      mul_left_comm] <;> measurability
                         /-
                           🎉 no goals
                         -/
    /-
      case refine_2
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      B : Set X
      f : X → ENNReal
      x₀ : X
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hB : MeasurableSet B
      ⊢ ∀ ⦃f g : X → ENNReal⦄, Disjoint (Function.support f) (Function.support g) →  …
    -/
  · rintro f₁ f₂ - _ _ hf₁ hf₂
    /-
      case refine_2
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      B : Set X
      f : X → ENNReal
      x₀ : X
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hB : MeasurableSet B
      f₁ f₂ : X → ENNReal
      a✝¹ : Measurable f₁
      a✝ : Measurable f₂
      hf₁ : Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.indicator 1 x)  …
      hf₂ : Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.indicator 1 x)  …
      ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.indicator 1 x) (HAd …
    -/
    simp only [Pi.add_apply, mul_add]
    /-
      case refine_2
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      B : Set X
      f : X → ENNReal
      x₀ : X
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hB : MeasurableSet B
      f₁ f₂ : X → ENNReal
      a✝¹ : Measurable f₁
      a✝ : Measurable f₂
      hf₁ : Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.indicator 1 x)  …
      hf₂ : Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.indicator 1 x)  …
      ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HAdd.hAdd (HMul.hMul (B.indicato …
    -/
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    rw [lintegral_add_right, lintegral_add_right, hf₁, hf₂, mul_add] <;> measurability
                                                                         /-
                                                                           🎉 no goals
                                                                         -/
    /-
      case refine_3
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      B : Set X
      f : X → ENNReal
      x₀ : X
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hB : MeasurableSet B
      ⊢ ∀ ⦃f : Nat → X → ENNReal⦄, (∀ (n : Nat), Measurable (f n)) → Monotone f → (∀ …
    -/
  · rintro f' hf'_meas hf'_mono hf'
    /-
      case refine_3
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      B : Set X
      f : X → ENNReal
      x₀ : X
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hB : MeasurableSet B
      f' : Nat → X → ENNReal
      hf'_meas : ∀ (n : Nat), Measurable (f' n)
      hf'_mono : Monotone f'
      hf' : ∀ (n : Nat), Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.in …
      ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.indicator 1 x) ((fu …
    -/
    simp_rw [ENNReal.mul_iSup]
    /-
      case refine_3
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      B : Set X
      f : X → ENNReal
      x₀ : X
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hB : MeasurableSet B
      f' : Nat → X → ENNReal
      hf'_meas : ∀ (n : Nat), Measurable (f' n)
      hf'_mono : Monotone f'
      hf' : ∀ (n : Nat), Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.in …
      ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => iSup fun i => HMul.hMul (B.indic …
    -/
    rw [lintegral_iSup (by measurability), lintegral_iSup hf'_meas hf'_mono, ENNReal.mul_iSup]
    /-
      case refine_3
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      B : Set X
      f : X → ENNReal
      x₀ : X
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hB : MeasurableSet B
      f' : Nat → X → ENNReal
      hf'_meas : ∀ (n : Nat), Measurable (f' n)
      hf'_mono : Monotone f'
      hf' : ∀ (n : Nat), Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.in …
      ⊢ Eq (iSup fun n => MeasureTheory.lintegral (π x₀) fun a => HMul.hMul (B.indic …
    -/
    simp_rw [hf']
      /-
        case refine_3
        X : Type u_1
        𝓑 𝓧 : MeasurableSpace X
        π : ProbabilityTheory.Kernel X X
        B : Set X
        f : X → ENNReal
        x₀ : X
        hπ : π.IsProper
        h𝓑𝓧 : LE.le 𝓑 𝓧
        hf : Measurable f
        hB : MeasurableSet B
        f' : Nat → X → ENNReal
        hf'_meas : ∀ (n : Nat), Measurable (f' n)
        hf'_mono : Monotone f'
        hf' : ∀ (n : Nat), Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (B.in …
        ⊢ Monotone fun i x => HMul.hMul (B.indicator 1 x) (f' i x)
      -/
    · exact hf'_mono.const_mul (zero_le _)
      /-
        🎉 no goals
      -/


lemma IsProper.setLIntegral_eq_indicator_mul_lintegral (hπ : IsProper π) (h𝓑𝓧 : 𝓑 ≤ 𝓧)
    (hf : Measurable[𝓧] f) (hB : MeasurableSet[𝓑] B) (x₀ : X) :
    ∫⁻ x in B, f x ∂(π x₀) = B.indicator 1 x₀ * ∫⁻ x, f x ∂(π x₀) := by
  simp [← hπ.lintegral_indicator_mul h𝓑𝓧 hf hB, ← indicator_mul_left,
    lintegral_indicator (h𝓑𝓧 _ hB)]


lemma IsProper.setLIntegral_inter_eq_indicator_mul_setLIntegral (hπ : IsProper π) (h𝓑𝓧 : 𝓑 ≤ 𝓧)
    (hf : Measurable[𝓧] f) (hA : MeasurableSet[𝓧] A) (hB : MeasurableSet[𝓑] B) (x₀ : X) :
    ∫⁻ x in A ∩ B, f x ∂(π x₀) = B.indicator 1 x₀ * ∫⁻ x in A, f x ∂(π x₀) := by
  rw [← lintegral_indicator hA, ← hπ.setLIntegral_eq_indicator_mul_lintegral h𝓑𝓧 _ hB,
                                /-
                                  case hs
                                  X : Type u_1
                                  𝓑 𝓧 : MeasurableSpace X
                                  π : ProbabilityTheory.Kernel X X
                                  A B : Set X
                                  f : X → ENNReal
                                  hπ : π.IsProper
                                  h𝓑𝓧 : LE.le 𝓑 𝓧
                                  hf : Measurable f
                                  hA : MeasurableSet A
                                  hB : MeasurableSet B
                                  x₀ : X
                                  ⊢ MeasurableSet A
                                -/
                                /-
                                  🎉 no goals
                                -/
    setLIntegral_indicator] <;> measurability
                                /-
                                  🎉 no goals
                                -/


lemma IsProper.lintegral_mul (hπ : IsProper π) (h𝓑𝓧 : 𝓑 ≤ 𝓧) (hf : Measurable[𝓧] f)
    (hg : Measurable[𝓑] g) (x₀ : X) :
    ∫⁻ x, g x * f x ∂(π x₀) = g x₀ * ∫⁻ x, f x ∂(π x₀) := by
  /-
    X : Type u_1
    𝓑 𝓧 : MeasurableSpace X
    π : ProbabilityTheory.Kernel X X
    f g : X → ENNReal
    hπ : π.IsProper
    h𝓑𝓧 : LE.le 𝓑 𝓧
    hf : Measurable f
    hg : Measurable g
    x₀ : X
    ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (g x) (f x)) (HMul.hMu …
  -/
  refine hg.ennreal_induction ?_ ?_ ?_
    /-
      case refine_1
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      f g : X → ENNReal
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hg : Measurable g
      x₀ : X
      ⊢ ∀ (c : ENNReal) ⦃s : Set X⦄, MeasurableSet s → Eq (MeasureTheory.lintegral ( …
    -/
  · rintro c A hA
    /-
      case refine_1
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      f g : X → ENNReal
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hg : Measurable g
      x₀ : X
      c : ENNReal
      A : Set X
      hA : MeasurableSet A
      ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (A.indicator (fun x => …
    -/
    simp_rw [← smul_indicator_one_apply, smul_mul_assoc, smul_eq_mul]
    /-
      case refine_1
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      f g : X → ENNReal
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hg : Measurable g
      x₀ : X
      c : ENNReal
      A : Set X
      hA : MeasurableSet A
      ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul c (HMul.hMul (A.indica …
    -/
    rw [lintegral_const_mul, hπ.lintegral_indicator_mul h𝓑𝓧 hf hA]
      /-
        case refine_1.hf
        X : Type u_1
        𝓑 𝓧 : MeasurableSpace X
        π : ProbabilityTheory.Kernel X X
        f g : X → ENNReal
        hπ : π.IsProper
        h𝓑𝓧 : LE.le 𝓑 𝓧
        hf : Measurable f
        hg : Measurable g
        x₀ : X
        c : ENNReal
        A : Set X
        hA : MeasurableSet A
        ⊢ Measurable fun x => HMul.hMul (A.indicator 1 x) (f x)
      -/
    · measurability
      /-
        🎉 no goals
      -/
    /-
      case refine_2
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      f g : X → ENNReal
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hg : Measurable g
      x₀ : X
      ⊢ ∀ ⦃f_1 g : X → ENNReal⦄, Disjoint (Function.support f_1) (Function.support g …
    -/
  · rintro g₁ g₂ - _ hg₂_meas hg₁ hg₂
    /-
      case refine_2
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      f g : X → ENNReal
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hg : Measurable g
      x₀ : X
      g₁ g₂ : X → ENNReal
      a✝ : Measurable g₁
      hg₂_meas : Measurable g₂
      hg₁ : Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (g₁ x) (f x)) (HMu …
      hg₂ : Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (g₂ x) (f x)) (HMu …
      ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (HAdd.hAdd g₁ g₂ x) (f …
    -/
    simp only [Pi.add_apply, mul_add, add_mul]
    /-
      case refine_2
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      f g : X → ENNReal
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hg : Measurable g
      x₀ : X
      g₁ g₂ : X → ENNReal
      a✝ : Measurable g₁
      hg₂_meas : Measurable g₂
      hg₁ : Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (g₁ x) (f x)) (HMu …
      hg₂ : Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (g₂ x) (f x)) (HMu …
      ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HAdd.hAdd (HMul.hMul (g₁ x) (f x …
    -/
    rw [lintegral_add_right, hg₁, hg₂]
      /-
        case refine_2.hg
        X : Type u_1
        𝓑 𝓧 : MeasurableSpace X
        π : ProbabilityTheory.Kernel X X
        f g : X → ENNReal
        hπ : π.IsProper
        h𝓑𝓧 : LE.le 𝓑 𝓧
        hf : Measurable f
        hg : Measurable g
        x₀ : X
        g₁ g₂ : X → ENNReal
        a✝ : Measurable g₁
        hg₂_meas : Measurable g₂
        hg₁ : Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (g₁ x) (f x)) (HMu …
        hg₂ : Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (g₂ x) (f x)) (HMu …
        ⊢ Measurable fun x => HMul.hMul (g₂ x) (f x)
      -/
    · exact (hg₂_meas.mono h𝓑𝓧 le_rfl).mul hf
      /-
        🎉 no goals
      -/
    /-
      case refine_3
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      f g : X → ENNReal
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hg : Measurable g
      x₀ : X
      ⊢ ∀ ⦃f_1 : Nat → X → ENNReal⦄, (∀ (n : Nat), Measurable (f_1 n)) → Monotone f_ …
    -/
  · rintro g' hg'_meas hg'_mono hg'
    /-
      case refine_3
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      f g : X → ENNReal
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hg : Measurable g
      x₀ : X
      g' : Nat → X → ENNReal
      hg'_meas : ∀ (n : Nat), Measurable (g' n)
      hg'_mono : Monotone g'
      hg' : ∀ (n : Nat), Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (g' n …
      ⊢ Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul ((fun x => iSup fun n  …
    -/
    simp_rw [ENNReal.iSup_mul]
    rw [lintegral_iSup (fun n ↦ ((hg'_meas _).mono h𝓑𝓧 le_rfl).mul hf)
      (hg'_mono.mul_const (zero_le _))]
    /-
      case refine_3
      X : Type u_1
      𝓑 𝓧 : MeasurableSpace X
      π : ProbabilityTheory.Kernel X X
      f g : X → ENNReal
      hπ : π.IsProper
      h𝓑𝓧 : LE.le 𝓑 𝓧
      hf : Measurable f
      hg : Measurable g
      x₀ : X
      g' : Nat → X → ENNReal
      hg'_meas : ∀ (n : Nat), Measurable (g' n)
      hg'_mono : Monotone g'
      hg' : ∀ (n : Nat), Eq (MeasureTheory.lintegral (π x₀) fun x => HMul.hMul (g' n …
      ⊢ Eq (iSup fun n => MeasureTheory.lintegral (π x₀) fun a => HMul.hMul (g' n a) …
    -/
    simp_rw [hg']
    /-
      🎉 no goals
    -/


