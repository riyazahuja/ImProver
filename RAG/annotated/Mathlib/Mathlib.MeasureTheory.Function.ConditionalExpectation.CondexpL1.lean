/-- Conditional expectation of the indicator of a measurable set with finite measure,
as a function in L1. -/
def condexpIndL1Fin (hm : m ≤ m0) [SigmaFinite (μ.trim hm)] (hs : MeasurableSet s) (hμs : μ s ≠ ∞)
    (x : G) : α →₁[μ] G :=
  (integrable_condexpIndSMul hm hs hμs x).toL1 _


theorem condexpIndL1Fin_ae_eq_condexpIndSMul (hm : m ≤ m0) [SigmaFinite (μ.trim hm)]
    (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : G) :
    condexpIndL1Fin hm hs hμs x =ᵐ[μ] condexpIndSMul hm hs hμs x :=
  (integrable_condexpIndSMul hm hs hμs x).coeFn_toL1


private theorem q {hs : MeasurableSet s} {hμs : μ s ≠ ∞} {x : G} :
    Memℒp (condexpIndSMul hm hs hμs x) 1 μ := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    ⊢ MeasureTheory.Memℒp (↑↑(MeasureTheory.condexpIndSMul hm hs hμs x)) 1 μ
  -/
  rw [memℒp_one_iff_integrable]; apply integrable_condexpIndSMul
                                 /-
                                   🎉 no goals
                                 -/


theorem condexpIndL1Fin_add (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x y : G) :
    condexpIndL1Fin hm hs hμs (x + y) =
    condexpIndL1Fin hm hs hμs x + condexpIndL1Fin hm hs hμs y := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x y : G
    ⊢ Eq (MeasureTheory.condexpIndL1Fin hm hs hμs (HAdd.hAdd x y)) (HAdd.hAdd (Mea …
  -/
  ext1
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x y : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm hs hμs …
  -/
  refine (Memℒp.coeFn_toLp q).trans ?_
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x y : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndSMul hm hs hμs  …
  -/
  refine EventuallyEq.trans ?_ (Lp.coeFn_add _ _).symm
  refine EventuallyEq.trans ?_
    (EventuallyEq.add (Memℒp.coeFn_toLp q).symm (Memℒp.coeFn_toLp q).symm)
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x y : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndSMul hm hs hμs  …
  -/
  rw [condexpIndSMul_add]
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x y : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(HAdd.hAdd (MeasureTheory.condexpIndSMul …
  -/
  refine (Lp.coeFn_add _ _).trans (Eventually.of_forall fun a => ?_)
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x y : G
    a : α
    ⊢ Eq (HAdd.hAdd (↑↑(MeasureTheory.condexpIndSMul hm hs hμs x)) (↑↑(MeasureTheo …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem condexpIndL1Fin_smul (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (c : ℝ) (x : G) :
    condexpIndL1Fin hm hs hμs (c • x) = c • condexpIndL1Fin hm hs hμs x := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : Real
    x : G
    ⊢ Eq (MeasureTheory.condexpIndL1Fin hm hs hμs (HSMul.hSMul c x)) (HSMul.hSMul  …
  -/
  ext1
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : Real
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm hs hμs …
  -/
  refine (Memℒp.coeFn_toLp q).trans ?_
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : Real
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndSMul hm hs hμs  …
  -/
  refine EventuallyEq.trans ?_ (Lp.coeFn_smul _ _).symm
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : Real
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.condexpIndSMul hm hs hμs …
  -/
  rw [condexpIndSMul_smul hs hμs c x]
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : Real
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(HSMul.hSMul c (MeasureTheory.condexpIn …
  -/
  refine (Lp.coeFn_smul _ _).trans ?_
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : Real
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul c ↑↑(MeasureTheory.condexpInd …
  -/
  refine (condexpIndL1Fin_ae_eq_condexpIndSMul hm hs hμs x).mono fun y hy => ?_
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : Real
    x : G
    y : α
    hy : Eq (↑↑(MeasureTheory.condexpIndL1Fin hm hs hμs x) y) (↑↑(MeasureTheory.co …
    ⊢ Eq (HSMul.hSMul c (↑↑(MeasureTheory.condexpIndSMul hm hs hμs x)) y) (HSMul.h …
  -/
  simp only [Pi.smul_apply, hy]
  /-
    🎉 no goals
  -/


theorem condexpIndL1Fin_smul' [NormedSpace ℝ F] [SMulCommClass ℝ 𝕜 F] (hs : MeasurableSet s)
    (hμs : μ s ≠ ∞) (c : 𝕜) (x : F) :
    condexpIndL1Fin hm hs hμs (c • x) = c • condexpIndL1Fin hm hs hμs x := by
  /-
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝¹ : NormedSpace Real F
    inst✝ : SMulCommClass Real 𝕜 F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : 𝕜
    x : F
    ⊢ Eq (MeasureTheory.condexpIndL1Fin hm hs hμs (HSMul.hSMul c x)) (HSMul.hSMul  …
  -/
  ext1
  /-
    case h
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝¹ : NormedSpace Real F
    inst✝ : SMulCommClass Real 𝕜 F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : 𝕜
    x : F
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm hs hμs …
  -/
  refine (Memℒp.coeFn_toLp q).trans ?_
  /-
    case h
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝¹ : NormedSpace Real F
    inst✝ : SMulCommClass Real 𝕜 F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : 𝕜
    x : F
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndSMul hm hs hμs  …
  -/
  refine EventuallyEq.trans ?_ (Lp.coeFn_smul _ _).symm
  /-
    case h
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝¹ : NormedSpace Real F
    inst✝ : SMulCommClass Real 𝕜 F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : 𝕜
    x : F
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.condexpIndSMul hm hs hμs …
  -/
  rw [condexpIndSMul_smul' hs hμs c x]
  /-
    case h
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝¹ : NormedSpace Real F
    inst✝ : SMulCommClass Real 𝕜 F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : 𝕜
    x : F
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(HSMul.hSMul c (MeasureTheory.condexpIn …
  -/
  refine (Lp.coeFn_smul _ _).trans ?_
  /-
    case h
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝¹ : NormedSpace Real F
    inst✝ : SMulCommClass Real 𝕜 F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : 𝕜
    x : F
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HSMul.hSMul c ↑↑(MeasureTheory.condexpInd …
  -/
  refine (condexpIndL1Fin_ae_eq_condexpIndSMul hm hs hμs x).mono fun y hy => ?_
  /-
    case h
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝¹ : NormedSpace Real F
    inst✝ : SMulCommClass Real 𝕜 F
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : 𝕜
    x : F
    y : α
    hy : Eq (↑↑(MeasureTheory.condexpIndL1Fin hm hs hμs x) y) (↑↑(MeasureTheory.co …
    ⊢ Eq (HSMul.hSMul c (↑↑(MeasureTheory.condexpIndSMul hm hs hμs x)) y) (HSMul.h …
  -/
  simp only [Pi.smul_apply, hy]
  /-
    🎉 no goals
  -/


theorem norm_condexpIndL1Fin_le (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : G) :
    ‖condexpIndL1Fin hm hs hμs x‖ ≤ (μ s).toReal * ‖x‖ := by
  rw [L1.norm_eq_integral_norm, ← ENNReal.toReal_ofReal (norm_nonneg x), ← ENNReal.toReal_mul,
    ← ENNReal.ofReal_le_iff_le_toReal (ENNReal.mul_ne_top hμs ENNReal.ofReal_ne_top),
    ofReal_integral_norm_eq_lintegral_nnnorm]
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    ⊢ LE.le (MeasureTheory.lintegral μ fun x_1 => ↑(NNNorm.nnnorm (↑↑(MeasureTheor …
  -/
  swap; · rw [← memℒp_one_iff_integrable]; exact Lp.memℒp _
                                           /-
                                             🎉 no goals
                                           -/
  have h_eq :
    ∫⁻ a, ‖condexpIndL1Fin hm hs hμs x a‖₊ ∂μ = ∫⁻ a, ‖condexpIndSMul hm hs hμs x a‖₊ ∂μ := by
    refine lintegral_congr_ae ?_
    refine (condexpIndL1Fin_ae_eq_condexpIndSMul hm hs hμs x).mono fun z hz => ?_
    dsimp only
    rw [hz]
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    h_eq : Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑(MeasureTheor …
    ⊢ LE.le (MeasureTheory.lintegral μ fun x_1 => ↑(NNNorm.nnnorm (↑↑(MeasureTheor …
  -/
  rw [h_eq, ofReal_norm_eq_coe_nnnorm]
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    h_eq : Eq (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑(MeasureTheor …
    ⊢ LE.le (MeasureTheory.lintegral μ fun a => ↑(NNNorm.nnnorm (↑↑(MeasureTheory. …
  -/
  exact lintegral_nnnorm_condexpIndSMul_le hm hs hμs x
  /-
    🎉 no goals
  -/


theorem condexpIndL1Fin_disjoint_union (hs : MeasurableSet s) (ht : MeasurableSet t) (hμs : μ s ≠ ∞)
    (hμt : μ t ≠ ∞) (hst : Disjoint s t) (x : G) :
    condexpIndL1Fin hm (hs.union ht) ((measure_union_le s t).trans_lt
      (lt_top_iff_ne_top.mpr (ENNReal.add_ne_top.mpr ⟨hμs, hμt⟩))).ne x =
    condexpIndL1Fin hm hs hμs x + condexpIndL1Fin hm ht hμt x := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    ⊢ Eq (MeasureTheory.condexpIndL1Fin hm ⋯ ⋯ x) (HAdd.hAdd (MeasureTheory.condex …
  -/
  ext1
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm ⋯ ⋯ x) …
  -/
  have hμst := measure_union_ne_top hμs hμt
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm ⋯ ⋯ x) …
  -/
  refine (condexpIndL1Fin_ae_eq_condexpIndSMul hm (hs.union ht) hμst x).trans ?_
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndSMul hm ⋯ hμst  …
  -/
  refine EventuallyEq.trans ?_ (Lp.coeFn_add _ _).symm
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.condexpIndSMul hm ⋯ hμst …
  -/
  have hs_eq := condexpIndL1Fin_ae_eq_condexpIndSMul hm hs hμs x
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    hs_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.condexpIndSMul hm ⋯ hμst …
  -/
  have ht_eq := condexpIndL1Fin_ae_eq_condexpIndSMul hm ht hμt x
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    hs_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ht_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.condexpIndSMul hm ⋯ hμst …
  -/
  refine EventuallyEq.trans ?_ (EventuallyEq.add hs_eq.symm ht_eq.symm)
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    hs_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ht_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndSMul hm ⋯ hμst  …
  -/
  rw [condexpIndSMul]
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    hs_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ht_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.compLpL 2 μ (Conti …
  -/
  rw [indicatorConstLp_disjoint_union hs ht hμs hμt hst (1 : ℝ)]
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    hs_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ht_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.compLpL 2 μ (Conti …
  -/
  rw [(condexpL2 ℝ ℝ hm).map_add]
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    hs_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ht_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.compLpL 2 μ (Conti …
  -/
  push_cast
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    hs_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ht_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((ContinuousLinearMap.compLpL 2 μ (Conti …
  -/
  rw [((toSpanSingleton ℝ x).compLpL 2 μ).map_add]
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    hs_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ht_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(HAdd.hAdd ((ContinuousLinearMap.compLpL …
  -/
  refine (Lp.coeFn_add _ _).trans ?_
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    hs_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ht_eq : (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndL1Fin hm  …
    ⊢ (MeasureTheory.ae μ).EventuallyEq (HAdd.hAdd ↑↑((ContinuousLinearMap.compLpL …
  -/
  filter_upwards with y using rfl
  /-
    🎉 no goals
  -/


/-- Conditional expectation of the indicator of a set, as a function in L1. Its value for sets
which are not both measurable and of finite measure is not used: we set it to 0. -/
def condexpIndL1 {m m0 : MeasurableSpace α} (hm : m ≤ m0) (μ : Measure α) (s : Set α)
    [SigmaFinite (μ.trim hm)] (x : G) : α →₁[μ] G :=
  if hs : MeasurableSet s ∧ μ s ≠ ∞ then condexpIndL1Fin hm hs.1 hs.2 x else 0


theorem condexpIndL1_of_measurableSet_of_measure_ne_top (hs : MeasurableSet s) (hμs : μ s ≠ ∞)
    (x : G) : condexpIndL1 hm μ s x = condexpIndL1Fin hm hs hμs x := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s x) (MeasureTheory.condexpIndL1Fin hm h …
  -/
  simp only [condexpIndL1, And.intro hs hμs, dif_pos, Ne, not_false_iff, and_self_iff]
  /-
    🎉 no goals
  -/


theorem condexpIndL1_of_measure_eq_top (hμs : μ s = ∞) (x : G) : condexpIndL1 hm μ s x = 0 := by
  simp only [condexpIndL1, hμs, eq_self_iff_true, not_true, Ne, dif_neg, not_false_iff,
    and_false]


theorem condexpIndL1_of_not_measurableSet (hs : ¬MeasurableSet s) (x : G) :
    condexpIndL1 hm μ s x = 0 := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : Not (MeasurableSet s)
    x : G
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s x) 0
  -/
  simp only [condexpIndL1, hs, dif_neg, not_false_iff, false_and]
  /-
    🎉 no goals
  -/


theorem condexpIndL1_add (x y : G) :
    condexpIndL1 hm μ s (x + y) = condexpIndL1 hm μ s x + condexpIndL1 hm μ s y := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x y : G
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HAdd.hAdd x y)) (HAdd.hAdd (MeasureTh …
  -/
  by_cases hs : MeasurableSet s
  /-
    case pos
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x y : G
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HAdd.hAdd x y)) (HAdd.hAdd (MeasureTh …
  -/
  swap; · simp_rw [condexpIndL1_of_not_measurableSet hs]; rw [zero_add]
                                                          /-
                                                            🎉 no goals
                                                          -/
  /-
    case pos
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x y : G
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HAdd.hAdd x y)) (HAdd.hAdd (MeasureTh …
  -/
  by_cases hμs : μ s = ∞
    /-
      case pos
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      x y : G
      hs : MeasurableSet s
      hμs : Eq (μ s) Top.top
      ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HAdd.hAdd x y)) (HAdd.hAdd (MeasureTh …
    -/
  · simp_rw [condexpIndL1_of_measure_eq_top hμs]; rw [zero_add]
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      case neg
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      x y : G
      hs : MeasurableSet s
      hμs : Not (Eq (μ s) Top.top)
      ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HAdd.hAdd x y)) (HAdd.hAdd (MeasureTh …
    -/
  · simp_rw [condexpIndL1_of_measurableSet_of_measure_ne_top hs hμs]
    /-
      case neg
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      x y : G
      hs : MeasurableSet s
      hμs : Not (Eq (μ s) Top.top)
      ⊢ Eq (MeasureTheory.condexpIndL1Fin hm hs hμs (HAdd.hAdd x y)) (HAdd.hAdd (Mea …
    -/
    exact condexpIndL1Fin_add hs hμs x y
    /-
      🎉 no goals
    -/


theorem condexpIndL1_smul (c : ℝ) (x : G) :
    condexpIndL1 hm μ s (c • x) = c • condexpIndL1 hm μ s x := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    c : Real
    x : G
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HSMul.hSMul c x)) (HSMul.hSMul c (Mea …
  -/
  by_cases hs : MeasurableSet s
  /-
    case pos
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    c : Real
    x : G
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HSMul.hSMul c x)) (HSMul.hSMul c (Mea …
  -/
  swap; · simp_rw [condexpIndL1_of_not_measurableSet hs]; rw [smul_zero]
                                                          /-
                                                            🎉 no goals
                                                          -/
  /-
    case pos
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    c : Real
    x : G
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HSMul.hSMul c x)) (HSMul.hSMul c (Mea …
  -/
  by_cases hμs : μ s = ∞
    /-
      case pos
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      c : Real
      x : G
      hs : MeasurableSet s
      hμs : Eq (μ s) Top.top
      ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HSMul.hSMul c x)) (HSMul.hSMul c (Mea …
    -/
  · simp_rw [condexpIndL1_of_measure_eq_top hμs]; rw [smul_zero]
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      case neg
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      c : Real
      x : G
      hs : MeasurableSet s
      hμs : Not (Eq (μ s) Top.top)
      ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HSMul.hSMul c x)) (HSMul.hSMul c (Mea …
    -/
  · simp_rw [condexpIndL1_of_measurableSet_of_measure_ne_top hs hμs]
    /-
      case neg
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      c : Real
      x : G
      hs : MeasurableSet s
      hμs : Not (Eq (μ s) Top.top)
      ⊢ Eq (MeasureTheory.condexpIndL1Fin hm hs hμs (HSMul.hSMul c x)) (HSMul.hSMul  …
    -/
    exact condexpIndL1Fin_smul hs hμs c x
    /-
      🎉 no goals
    -/


theorem condexpIndL1_smul' [NormedSpace ℝ F] [SMulCommClass ℝ 𝕜 F] (c : 𝕜) (x : F) :
    condexpIndL1 hm μ s (c • x) = c • condexpIndL1 hm μ s x := by
  /-
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝¹ : NormedSpace Real F
    inst✝ : SMulCommClass Real 𝕜 F
    c : 𝕜
    x : F
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HSMul.hSMul c x)) (HSMul.hSMul c (Mea …
  -/
  by_cases hs : MeasurableSet s
  /-
    case pos
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝¹ : NormedSpace Real F
    inst✝ : SMulCommClass Real 𝕜 F
    c : 𝕜
    x : F
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HSMul.hSMul c x)) (HSMul.hSMul c (Mea …
  -/
  swap; · simp_rw [condexpIndL1_of_not_measurableSet hs]; rw [smul_zero]
                                                          /-
                                                            🎉 no goals
                                                          -/
  /-
    case pos
    α : Type u_1
    F : Type u_2
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F
    inst✝³ : NormedSpace 𝕜 F
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
    inst✝¹ : NormedSpace Real F
    inst✝ : SMulCommClass Real 𝕜 F
    c : 𝕜
    x : F
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HSMul.hSMul c x)) (HSMul.hSMul c (Mea …
  -/
  by_cases hμs : μ s = ∞
    /-
      case pos
      α : Type u_1
      F : Type u_2
      𝕜 : Type u_6
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
      inst✝¹ : NormedSpace Real F
      inst✝ : SMulCommClass Real 𝕜 F
      c : 𝕜
      x : F
      hs : MeasurableSet s
      hμs : Eq (μ s) Top.top
      ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HSMul.hSMul c x)) (HSMul.hSMul c (Mea …
    -/
  · simp_rw [condexpIndL1_of_measure_eq_top hμs]; rw [smul_zero]
                                                  /-
                                                    🎉 no goals
                                                  -/
    /-
      case neg
      α : Type u_1
      F : Type u_2
      𝕜 : Type u_6
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
      inst✝¹ : NormedSpace Real F
      inst✝ : SMulCommClass Real 𝕜 F
      c : 𝕜
      x : F
      hs : MeasurableSet s
      hμs : Not (Eq (μ s) Top.top)
      ⊢ Eq (MeasureTheory.condexpIndL1 hm μ s (HSMul.hSMul c x)) (HSMul.hSMul c (Mea …
    -/
  · simp_rw [condexpIndL1_of_measurableSet_of_measure_ne_top hs hμs]
    /-
      case neg
      α : Type u_1
      F : Type u_2
      𝕜 : Type u_6
      inst✝⁵ : RCLike 𝕜
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      hm : LE.le m m0
      inst✝² : MeasureTheory.SigmaFinite (μ.trim hm)
      inst✝¹ : NormedSpace Real F
      inst✝ : SMulCommClass Real 𝕜 F
      c : 𝕜
      x : F
      hs : MeasurableSet s
      hμs : Not (Eq (μ s) Top.top)
      ⊢ Eq (MeasureTheory.condexpIndL1Fin hm hs hμs (HSMul.hSMul c x)) (HSMul.hSMul  …
    -/
    exact condexpIndL1Fin_smul' hs hμs c x
    /-
      🎉 no goals
    -/


theorem norm_condexpIndL1_le (x : G) : ‖condexpIndL1 hm μ s x‖ ≤ (μ s).toReal * ‖x‖ := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x : G
    ⊢ LE.le (Norm.norm (MeasureTheory.condexpIndL1 hm μ s x)) (HMul.hMul (μ s).toR …
  -/
  by_cases hs : MeasurableSet s
  /-
    case pos
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x : G
    hs : MeasurableSet s
    ⊢ LE.le (Norm.norm (MeasureTheory.condexpIndL1 hm μ s x)) (HMul.hMul (μ s).toR …
  -/
  swap
    /-
      case neg
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      x : G
      hs : Not (MeasurableSet s)
      ⊢ LE.le (Norm.norm (MeasureTheory.condexpIndL1 hm μ s x)) (HMul.hMul (μ s).toR …
    -/
  · simp_rw [condexpIndL1_of_not_measurableSet hs]; rw [Lp.norm_zero]
    /-
      case neg
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      x : G
      hs : Not (MeasurableSet s)
      ⊢ LE.le 0 (HMul.hMul (μ s).toReal (Norm.norm x))
    -/
    exact mul_nonneg ENNReal.toReal_nonneg (norm_nonneg _)
    /-
      🎉 no goals
    -/
  /-
    case pos
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x : G
    hs : MeasurableSet s
    ⊢ LE.le (Norm.norm (MeasureTheory.condexpIndL1 hm μ s x)) (HMul.hMul (μ s).toR …
  -/
  by_cases hμs : μ s = ∞
    /-
      case pos
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      x : G
      hs : MeasurableSet s
      hμs : Eq (μ s) Top.top
      ⊢ LE.le (Norm.norm (MeasureTheory.condexpIndL1 hm μ s x)) (HMul.hMul (μ s).toR …
    -/
  · rw [condexpIndL1_of_measure_eq_top hμs x, Lp.norm_zero]
    /-
      case pos
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      x : G
      hs : MeasurableSet s
      hμs : Eq (μ s) Top.top
      ⊢ LE.le 0 (HMul.hMul (μ s).toReal (Norm.norm x))
    -/
    exact mul_nonneg ENNReal.toReal_nonneg (norm_nonneg _)
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      x : G
      hs : MeasurableSet s
      hμs : Not (Eq (μ s) Top.top)
      ⊢ LE.le (Norm.norm (MeasureTheory.condexpIndL1 hm μ s x)) (HMul.hMul (μ s).toR …
    -/
  · rw [condexpIndL1_of_measurableSet_of_measure_ne_top hs hμs x]
    /-
      case neg
      α : Type u_1
      G : Type u_4
      inst✝² : NormedAddCommGroup G
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      s : Set α
      inst✝¹ : NormedSpace Real G
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      x : G
      hs : MeasurableSet s
      hμs : Not (Eq (μ s) Top.top)
      ⊢ LE.le (Norm.norm (MeasureTheory.condexpIndL1Fin hm hs hμs x)) (HMul.hMul (μ  …
    -/
    exact norm_condexpIndL1Fin_le hs hμs x
    /-
      🎉 no goals
    -/


theorem continuous_condexpIndL1 : Continuous fun x : G => condexpIndL1 hm μ s x :=
  continuous_of_linear_of_bound condexpIndL1_add condexpIndL1_smul norm_condexpIndL1_le


theorem condexpIndL1_disjoint_union (hs : MeasurableSet s) (ht : MeasurableSet t) (hμs : μ s ≠ ∞)
    (hμt : μ t ≠ ∞) (hst : Disjoint s t) (x : G) :
    condexpIndL1 hm μ (s ∪ t) x = condexpIndL1 hm μ s x + condexpIndL1 hm μ t x := by
  have hμst : μ (s ∪ t) ≠ ∞ :=
    ((measure_union_le s t).trans_lt (lt_top_iff_ne_top.mpr (ENNReal.add_ne_top.mpr ⟨hμs, hμt⟩))).ne
  rw [condexpIndL1_of_measurableSet_of_measure_ne_top hs hμs x,
    condexpIndL1_of_measurableSet_of_measure_ne_top ht hμt x,
    condexpIndL1_of_measurableSet_of_measure_ne_top (hs.union ht) hμst x]
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    x : G
    hμst : Ne (μ (Union.union s t)) Top.top
    ⊢ Eq (MeasureTheory.condexpIndL1Fin hm ⋯ hμst x) (HAdd.hAdd (MeasureTheory.con …
  -/
  exact condexpIndL1Fin_disjoint_union hs ht hμs hμt hst x
  /-
    🎉 no goals
  -/


/-- Conditional expectation of the indicator of a set, as a linear map from `G` to L1. -/
def condexpInd {m m0 : MeasurableSpace α} (hm : m ≤ m0) (μ : Measure α) [SigmaFinite (μ.trim hm)]
    (s : Set α) : G →L[ℝ] α →₁[μ] G where
  toFun := condexpIndL1 hm μ s
  map_add' := condexpIndL1_add
  map_smul' := condexpIndL1_smul
  cont := continuous_condexpIndL1


theorem condexpInd_ae_eq_condexpIndSMul (hm : m ≤ m0) [SigmaFinite (μ.trim hm)]
    (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : G) :
    condexpInd G hm μ s x =ᵐ[μ] condexpIndSMul hm hs hμs x := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((MeasureTheory.condexpInd G hm μ s) x)  …
  -/
  refine EventuallyEq.trans ?_ (condexpIndL1Fin_ae_eq_condexpIndSMul hm hs hμs x)
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((MeasureTheory.condexpInd G hm μ s) x)  …
  -/
  simp [condexpInd, condexpIndL1, hs, hμs]
  /-
    🎉 no goals
  -/


theorem aestronglyMeasurable'_condexpInd (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : G) :
    AEStronglyMeasurable' m (condexpInd G hm μ s x) μ :=
  AEStronglyMeasurable'.congr (aeStronglyMeasurable'_condexpIndSMul hm hs hμs x)
    (condexpInd_ae_eq_condexpIndSMul hm hs hμs x).symm


@[simp]
theorem condexpInd_empty : condexpInd G hm μ ∅ = (0 : G →L[ℝ] α →₁[μ] G) := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    ⊢ Eq (MeasureTheory.condexpInd G hm μ EmptyCollection.emptyCollection) 0
  -/
  ext1 x
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x : G
    ⊢ Eq ((MeasureTheory.condexpInd G hm μ EmptyCollection.emptyCollection) x) (0 x)
  -/
  ext1
  /-
    case h.h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((MeasureTheory.condexpInd G hm μ EmptyC …
  -/
  refine (condexpInd_ae_eq_condexpIndSMul hm MeasurableSet.empty (by simp) x).trans ?_
  /-
    case h.h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑(MeasureTheory.condexpIndSMul hm ⋯ ⋯ x)  …
  -/
  rw [condexpIndSMul_empty]
  /-
    case h.h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑0 ↑↑(0 x)
  -/
  refine (Lp.coeFn_zero G 2 μ).trans ?_
  /-
    case h.h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq 0 ↑↑(0 x)
  -/
  refine EventuallyEq.trans ?_ (Lp.coeFn_zero G 1 μ).symm
  /-
    case h.h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    x : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq 0 0
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem condexpInd_smul' [NormedSpace ℝ F] [SMulCommClass ℝ 𝕜 F] (c : 𝕜) (x : F) :
    condexpInd F hm μ s (c • x) = c • condexpInd F hm μ s x :=
  condexpIndL1_smul' c x


theorem norm_condexpInd_apply_le (x : G) : ‖condexpInd G hm μ s x‖ ≤ (μ s).toReal * ‖x‖ :=
  norm_condexpIndL1_le x


theorem norm_condexpInd_le : ‖(condexpInd G hm μ s : G →L[ℝ] α →₁[μ] G)‖ ≤ (μ s).toReal :=
  ContinuousLinearMap.opNorm_le_bound _ ENNReal.toReal_nonneg norm_condexpInd_apply_le


theorem condexpInd_disjoint_union_apply (hs : MeasurableSet s) (ht : MeasurableSet t)
    (hμs : μ s ≠ ∞) (hμt : μ t ≠ ∞) (hst : Disjoint s t) (x : G) :
    condexpInd G hm μ (s ∪ t) x = condexpInd G hm μ s x + condexpInd G hm μ t x :=
  condexpIndL1_disjoint_union hs ht hμs hμt hst x


theorem condexpInd_disjoint_union (hs : MeasurableSet s) (ht : MeasurableSet t) (hμs : μ s ≠ ∞)
    (hμt : μ t ≠ ∞) (hst : Disjoint s t) : (condexpInd G hm μ (s ∪ t) : G →L[ℝ] α →₁[μ] G) =
    condexpInd G hm μ s + condexpInd G hm μ t := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s t : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    ht : MeasurableSet t
    hμs : Ne (μ s) Top.top
    hμt : Ne (μ t) Top.top
    hst : Disjoint s t
    ⊢ Eq (MeasureTheory.condexpInd G hm μ (Union.union s t)) (HAdd.hAdd (MeasureTh …
  -/
  ext1 x; push_cast; exact condexpInd_disjoint_union_apply hs ht hμs hμt hst x
                     /-
                       🎉 no goals
                     -/


theorem dominatedFinMeasAdditive_condexpInd (hm : m ≤ m0) (μ : Measure α)
    [SigmaFinite (μ.trim hm)] :
    DominatedFinMeasAdditive μ (condexpInd G hm μ : Set α → G →L[ℝ] α →₁[μ] G) 1 :=
  ⟨fun _ _ => condexpInd_disjoint_union, fun _ _ _ => norm_condexpInd_le.trans (one_mul _).symm.le⟩


theorem setIntegral_condexpInd (hs : MeasurableSet[m] s) (ht : MeasurableSet t) (hμs : μ s ≠ ∞)
    (hμt : μ t ≠ ∞) (x : G') : ∫ a in s, condexpInd G' hm μ t x a ∂μ = (μ (t ∩ s)).toReal • x :=
  calc
    ∫ a in s, condexpInd G' hm μ t x a ∂μ = ∫ a in s, condexpIndSMul hm ht hμt x a ∂μ :=
      setIntegral_congr_ae (hm s hs)
        ((condexpInd_ae_eq_condexpIndSMul hm ht hμt x).mono fun _ hx _ => hx)
    _ = (μ (t ∩ s)).toReal • x := setIntegral_condexpIndSMul hs ht hμs hμt x


@[deprecated (since := "2024-04-17")]
alias set_integral_condexpInd := setIntegral_condexpInd


theorem condexpInd_of_measurable (hs : MeasurableSet[m] s) (hμs : μ s ≠ ∞) (c : G) :
    condexpInd G hm μ s c = indicatorConstLp 1 (hm s hs) hμs c := by
  /-
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : G
    ⊢ Eq ((MeasureTheory.condexpInd G hm μ s) c) (MeasureTheory.indicatorConstLp 1 …
  -/
  ext1
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((MeasureTheory.condexpInd G hm μ s) c)  …
  -/
  refine EventuallyEq.trans ?_ indicatorConstLp_coeFn.symm
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑((MeasureTheory.condexpInd G hm μ s) c) …
  -/
  refine (condexpInd_ae_eq_condexpIndSMul hm (hm s hs) hμs c).trans ?_
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.condexpIndSMul hm ⋯ hμs  …
  -/
  refine (condexpIndSMul_ae_eq_smul hm (hm s hs) hμs c).trans ?_
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => HSMul.hSMul (↑↑↑((MeasureTheory. …
  -/
  rw [lpMeas_coe, condexpL2_indicator_of_measurable hm hs hμs (1 : ℝ)]
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : G
    ⊢ (MeasureTheory.ae μ).EventuallyEq (fun a => HSMul.hSMul (↑↑(MeasureTheory.in …
  -/
  refine (@indicatorConstLp_coeFn α _ _ 2 μ _ s (hm s hs) hμs (1 : ℝ)).mono fun x hx => ?_
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : G
    x : α
    hx : Eq (↑↑(MeasureTheory.indicatorConstLp 2 ⋯ hμs 1) x) (s.indicator (fun x = …
    ⊢ Eq ((fun a => HSMul.hSMul (↑↑(MeasureTheory.indicatorConstLp 2 ⋯ hμs 1) a) c …
  -/
  dsimp only
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : G
    x : α
    hx : Eq (↑↑(MeasureTheory.indicatorConstLp 2 ⋯ hμs 1) x) (s.indicator (fun x = …
    ⊢ Eq (HSMul.hSMul (↑↑(MeasureTheory.indicatorConstLp 2 ⋯ hμs 1) x) c) (s.indic …
  -/
  rw [hx]
  /-
    case h
    α : Type u_1
    G : Type u_4
    inst✝² : NormedAddCommGroup G
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    inst✝¹ : NormedSpace Real G
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    c : G
    x : α
    hx : Eq (↑↑(MeasureTheory.indicatorConstLp 2 ⋯ hμs 1) x) (s.indicator (fun x = …
    ⊢ Eq (HSMul.hSMul (s.indicator (fun x => 1) x) c) (s.indicator (fun x => c) x)
  -/
                              /-
                                🎉 no goals
                              -/
  by_cases hx_mem : x ∈ s <;> simp [hx_mem]
                              /-
                                🎉 no goals
                              -/


theorem condexpInd_nonneg {E} [NormedLatticeAddCommGroup E] [NormedSpace ℝ E] [OrderedSMul ℝ E]
    (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : E) (hx : 0 ≤ x) : 0 ≤ condexpInd E hm μ s x := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝³ : MeasureTheory.SigmaFinite (μ.trim hm)
    E : Type u_7
    inst✝² : NormedLatticeAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    hx : LE.le 0 x
    ⊢ LE.le 0 ((MeasureTheory.condexpInd E hm μ s) x)
  -/
  rw [← coeFn_le]
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝³ : MeasureTheory.SigmaFinite (μ.trim hm)
    E : Type u_7
    inst✝² : NormedLatticeAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    hx : LE.le 0 x
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑↑0 ↑↑((MeasureTheory.condexpInd E hm μ s) …
  -/
  refine EventuallyLE.trans_eq ?_ (condexpInd_ae_eq_condexpIndSMul hm hs hμs x).symm
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    s : Set α
    hm : LE.le m m0
    inst✝³ : MeasureTheory.SigmaFinite (μ.trim hm)
    E : Type u_7
    inst✝² : NormedLatticeAddCommGroup E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : E
    hx : LE.le 0 x
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑↑0 ↑↑(MeasureTheory.condexpIndSMul hm hs  …
  -/
  exact (coeFn_zero E 1 μ).trans_le (condexpIndSMul_nonneg hs hμs x hx)
  /-
    🎉 no goals
  -/


/-- Conditional expectation of a function as a linear map from `α →₁[μ] F'` to itself. -/
def condexpL1CLM (hm : m ≤ m0) (μ : Measure α) [SigmaFinite (μ.trim hm)] :
    (α →₁[μ] F') →L[ℝ] α →₁[μ] F' :=
  L1.setToL1 (dominatedFinMeasAdditive_condexpInd F' hm μ)


theorem condexpL1CLM_smul (c : 𝕜) (f : α →₁[μ] F') :
    condexpL1CLM F' hm μ (c • f) = c • condexpL1CLM F' hm μ f := by
  /-
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
    ⊢ Eq ((MeasureTheory.condexpL1CLM F' hm μ) (HSMul.hSMul c f)) (HSMul.hSMul c ( …
  -/
  refine L1.setToL1_smul (dominatedFinMeasAdditive_condexpInd F' hm μ) ?_ c f
  /-
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    c : 𝕜
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
    ⊢ ∀ (c : 𝕜) (s : Set α) (x : F'), Eq ((MeasureTheory.condexpInd F' hm μ s) (HS …
  -/
  exact fun c s x => condexpInd_smul' c x
  /-
    🎉 no goals
  -/


theorem condexpL1CLM_indicatorConstLp (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : F') :
    (condexpL1CLM F' hm μ) (indicatorConstLp 1 hs hμs x) = condexpInd F' hm μ s x :=
  L1.setToL1_indicatorConstLp (dominatedFinMeasAdditive_condexpInd F' hm μ) hs hμs x


theorem condexpL1CLM_indicatorConst (hs : MeasurableSet s) (hμs : μ s ≠ ∞) (x : F') :
    (condexpL1CLM F' hm μ) ↑(simpleFunc.indicatorConst 1 hs hμs x) = condexpInd F' hm μ s x := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    s : Set α
    hs : MeasurableSet s
    hμs : Ne (μ s) Top.top
    x : F'
    ⊢ Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑(MeasureTheory.Lp.simpleFunc.indic …
  -/
  rw [Lp.simpleFunc.coe_indicatorConst]; exact condexpL1CLM_indicatorConstLp hs hμs x
                                         /-
                                           🎉 no goals
                                         -/


/-- Auxiliary lemma used in the proof of `setIntegral_condexpL1CLM`. -/
theorem setIntegral_condexpL1CLM_of_measure_ne_top (f : α →₁[μ] F') (hs : MeasurableSet[m] s)
    (hμs : μ s ≠ ∞) : ∫ x in s, condexpL1CLM F' hm μ f x ∂μ = ∫ x in s, f x ∂μ := by
  refine @Lp.induction _ _ _ _ _ _ _ ENNReal.one_ne_top
    (fun f : α →₁[μ] F' => ∫ x in s, condexpL1CLM F' hm μ f x ∂μ = ∫ x in s, f x ∂μ) ?_ ?_
    (isClosed_eq ?_ ?_) f
    /-
      case refine_1
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      s : Set α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ⊢ ∀ (c : F') {s_1 : Set α} (hs : MeasurableSet s_1) (hμs : LT.lt (μ s_1) Top.t …
    -/
  · intro x t ht hμt
    /-
      case refine_1
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      s : Set α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      x : F'
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt (μ t) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x_1 => ↑↑((MeasureTheory.conde …
    -/
    simp_rw [condexpL1CLM_indicatorConst ht hμt.ne x]
    /-
      case refine_1
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      s : Set α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      x : F'
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt (μ t) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x_1 => ↑↑((MeasureTheory.conde …
    -/
    rw [Lp.simpleFunc.coe_indicatorConst, setIntegral_indicatorConstLp (hm _ hs)]
    /-
      case refine_1
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      s : Set α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      x : F'
      t : Set α
      ht : MeasurableSet t
      hμt : LT.lt (μ t) Top.top
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x_1 => ↑↑((MeasureTheory.conde …
    -/
    exact setIntegral_condexpInd hs ht hμs hμt.ne x
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      s : Set α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ⊢ ∀ ⦃f g : α → F'⦄ (hf : MeasureTheory.Memℒp f 1 μ) (hg : MeasureTheory.Memℒp  …
    -/
  · intro f g hf_Lp hg_Lp _ hf hg
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      s : Set α
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f g : α → F'
      hf_Lp : MeasureTheory.Memℒp f 1 μ
      hg_Lp : MeasureTheory.Memℒp g 1 μ
      a✝ : Disjoint (Function.support f) (Function.support g)
      hf : Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.cond …
      hg : Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.cond …
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.condexp …
    -/
    simp_rw [(condexpL1CLM F' hm μ).map_add]
    rw [setIntegral_congr_ae (hm s hs) ((Lp.coeFn_add (condexpL1CLM F' hm μ (hf_Lp.toLp f))
      (condexpL1CLM F' hm μ (hg_Lp.toLp g))).mono fun x hx _ => hx)]
    rw [setIntegral_congr_ae (hm s hs)
      ((Lp.coeFn_add (hf_Lp.toLp f) (hg_Lp.toLp g)).mono fun x hx _ => hx)]
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      s : Set α
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      f g : α → F'
      hf_Lp : MeasureTheory.Memℒp f 1 μ
      hg_Lp : MeasureTheory.Memℒp g 1 μ
      a✝ : Disjoint (Function.support f) (Function.support g)
      hf : Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.cond …
      hg : Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.cond …
      ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => HAdd.hAdd (↑↑((MeasureThe …
    -/
    simp_rw [Pi.add_apply]
    rw [integral_add (L1.integrable_coeFn _).integrableOn (L1.integrable_coeFn _).integrableOn,
      integral_add (L1.integrable_coeFn _).integrableOn (L1.integrable_coeFn _).integrableOn, hf,
      hg]
    /-
      case refine_3
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      s : Set α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ⊢ Continuous fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑((Measu …
    -/
  · exact (continuous_setIntegral s).comp (condexpL1CLM F' hm μ).continuous
    /-
      🎉 no goals
    -/
    /-
      case refine_4
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      s : Set α
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      hs : MeasurableSet s
      hμs : Ne (μ s) Top.top
      ⊢ Continuous fun f => MeasureTheory.integral (μ.restrict s) fun x => ↑↑f x
    -/
  · exact continuous_setIntegral s
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias set_integral_condexpL1CLM_of_measure_ne_top :=
  setIntegral_condexpL1CLM_of_measure_ne_top


/-- The integral of the conditional expectation `condexpL1CLM` over an `m`-measurable set is equal
to the integral of `f` on that set. See also `setIntegral_condexp`, the similar statement for
`condexp`. -/
theorem setIntegral_condexpL1CLM (f : α →₁[μ] F') (hs : MeasurableSet[m] s) :
    ∫ x in s, condexpL1CLM F' hm μ f x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    s : Set α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.condexp …
  -/
  let S := spanningSets (μ.trim hm)
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    s : Set α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
    hs : MeasurableSet s
    S : Nat → Set α := MeasureTheory.spanningSets (μ.trim hm)
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.condexp …
  -/
  have hS_meas : ∀ i, MeasurableSet[m] (S i) := measurableSet_spanningSets (μ.trim hm)
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    s : Set α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
    hs : MeasurableSet s
    S : Nat → Set α := MeasureTheory.spanningSets (μ.trim hm)
    hS_meas : ∀ (i : Nat), MeasurableSet (S i)
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.condexp …
  -/
  have hS_meas0 : ∀ i, MeasurableSet (S i) := fun i => hm _ (hS_meas i)
  have hs_eq : s = ⋃ i, S i ∩ s := by
    simp_rw [Set.inter_comm]
    rw [← Set.inter_iUnion, iUnion_spanningSets (μ.trim hm), Set.inter_univ]
  have hS_finite : ∀ i, μ (S i ∩ s) < ∞ := by
    refine fun i => (measure_mono Set.inter_subset_left).trans_lt ?_
    have hS_finite_trim := measure_spanningSets_lt_top (μ.trim hm) i
    rwa [trim_measurableSet_eq hm (hS_meas i)] at hS_finite_trim
  have h_mono : Monotone fun i => S i ∩ s := by
    intro i j hij x
    simp_rw [Set.mem_inter_iff]
    exact fun h => ⟨monotone_spanningSets (μ.trim hm) hij h.1, h.2⟩
  have h_eq_forall :
    (fun i => ∫ x in S i ∩ s, condexpL1CLM F' hm μ f x ∂μ) = fun i => ∫ x in S i ∩ s, f x ∂μ :=
    funext fun i =>
      setIntegral_condexpL1CLM_of_measure_ne_top f (@MeasurableSet.inter α m _ _ (hS_meas i) hs)
        (hS_finite i).ne
  have h_right : Tendsto (fun i => ∫ x in S i ∩ s, f x ∂μ) atTop (𝓝 (∫ x in s, f x ∂μ)) := by
    have h :=
      tendsto_setIntegral_of_monotone (fun i => (hS_meas0 i).inter (hm s hs)) h_mono
        (L1.integrable_coeFn f).integrableOn
    rwa [← hs_eq] at h
  have h_left : Tendsto (fun i => ∫ x in S i ∩ s, condexpL1CLM F' hm μ f x ∂μ) atTop
      (𝓝 (∫ x in s, condexpL1CLM F' hm μ f x ∂μ)) := by
    have h := tendsto_setIntegral_of_monotone (fun i => (hS_meas0 i).inter (hm s hs)) h_mono
      (L1.integrable_coeFn (condexpL1CLM F' hm μ f)).integrableOn
    rwa [← hs_eq] at h
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    s : Set α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
    hs : MeasurableSet s
    S : Nat → Set α := MeasureTheory.spanningSets (μ.trim hm)
    hS_meas : ∀ (i : Nat), MeasurableSet (S i)
    hS_meas0 : ∀ (i : Nat), MeasurableSet (S i)
    hs_eq : Eq s (Set.iUnion fun i => Inter.inter (S i) s)
    hS_finite : ∀ (i : Nat), LT.lt (μ (Inter.inter (S i) s)) Top.top
    h_mono : Monotone fun i => Inter.inter (S i) s
    h_eq_forall : Eq (fun i => MeasureTheory.integral (μ.restrict (Inter.inter (S  …
    h_right : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (Inter.i …
    h_left : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (Inter.in …
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.condexp …
  -/
  rw [h_eq_forall] at h_left
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    s : Set α
    f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
    hs : MeasurableSet s
    S : Nat → Set α := MeasureTheory.spanningSets (μ.trim hm)
    hS_meas : ∀ (i : Nat), MeasurableSet (S i)
    hS_meas0 : ∀ (i : Nat), MeasurableSet (S i)
    hs_eq : Eq s (Set.iUnion fun i => Inter.inter (S i) s)
    hS_finite : ∀ (i : Nat), LT.lt (μ (Inter.inter (S i) s)) Top.top
    h_mono : Monotone fun i => Inter.inter (S i) s
    h_eq_forall : Eq (fun i => MeasureTheory.integral (μ.restrict (Inter.inter (S  …
    h_right : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (Inter.i …
    h_left : Filter.Tendsto (fun i => MeasureTheory.integral (μ.restrict (Inter.in …
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.condexp …
  -/
  exact tendsto_nhds_unique h_left h_right
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_condexpL1CLM := setIntegral_condexpL1CLM


theorem aestronglyMeasurable'_condexpL1CLM (f : α →₁[μ] F') :
    AEStronglyMeasurable' m (condexpL1CLM F' hm μ f) μ := by
  refine @Lp.induction _ _ _ _ _ _ _ ENNReal.one_ne_top
    (fun f : α →₁[μ] F' => AEStronglyMeasurable' m (condexpL1CLM F' hm μ f) μ) ?_ ?_ ?_ f
    /-
      case refine_1
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      ⊢ ∀ (c : F') {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt (μ s) Top.top), ( …
    -/
  · intro c s hs hμs
    /-
      case refine_1
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      c : F'
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTheory.condexpL1CLM F' hm  …
    -/
    rw [condexpL1CLM_indicatorConst hs hμs.ne c]
    /-
      case refine_1
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      c : F'
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt (μ s) Top.top
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTheory.condexpInd F' hm μ  …
    -/
    exact aestronglyMeasurable'_condexpInd hs hμs.ne c
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      ⊢ ∀ ⦃f g : α → F'⦄ (hf : MeasureTheory.Memℒp f 1 μ) (hg : MeasureTheory.Memℒp  …
    -/
  · intro f g hf hg _ hfm hgm
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      f g : α → F'
      hf : MeasureTheory.Memℒp f 1 μ
      hg : MeasureTheory.Memℒp g 1 μ
      a✝ : Disjoint (Function.support f) (Function.support g)
      hfm : MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTheory.condexpL1CLM F' …
      hgm : MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTheory.condexpL1CLM F' …
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTheory.condexpL1CLM F' hm  …
    -/
    rw [(condexpL1CLM F' hm μ).map_add]
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      f g : α → F'
      hf : MeasureTheory.Memℒp f 1 μ
      hg : MeasureTheory.Memℒp g 1 μ
      a✝ : Disjoint (Function.support f) (Function.support g)
      hfm : MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTheory.condexpL1CLM F' …
      hgm : MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTheory.condexpL1CLM F' …
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(HAdd.hAdd ((MeasureTheory.condexpL …
    -/
    refine AEStronglyMeasurable'.congr ?_ (coeFn_add _ _).symm
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      f g : α → F'
      hf : MeasureTheory.Memℒp f 1 μ
      hg : MeasureTheory.Memℒp g 1 μ
      a✝ : Disjoint (Function.support f) (Function.support g)
      hfm : MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTheory.condexpL1CLM F' …
      hgm : MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTheory.condexpL1CLM F' …
      ⊢ MeasureTheory.AEStronglyMeasurable' m (HAdd.hAdd ↑↑((MeasureTheory.condexpL1 …
    -/
    exact AEStronglyMeasurable'.add hfm hgm
    /-
      🎉 no goals
    -/
  · have : {f : Lp F' 1 μ | AEStronglyMeasurable' m (condexpL1CLM F' hm μ f) μ} =
        condexpL1CLM F' hm μ ⁻¹' {f | AEStronglyMeasurable' m f μ} := rfl
    /-
      case refine_3
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      this : Eq (setOf fun f => MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTh …
      ⊢ IsClosed (setOf fun f => (fun f => MeasureTheory.AEStronglyMeasurable' m (↑↑ …
    -/
    rw [this]
    /-
      case refine_3
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      this : Eq (setOf fun f => MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTh …
      ⊢ IsClosed (Set.preimage (⇑(MeasureTheory.condexpL1CLM F' hm μ)) (setOf fun f  …
    -/
    refine IsClosed.preimage (condexpL1CLM F' hm μ).continuous ?_
    /-
      case refine_3
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 μ) x
      this : Eq (setOf fun f => MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTh …
      ⊢ IsClosed (setOf fun f => MeasureTheory.AEStronglyMeasurable' m (↑↑f) μ)
    -/
    exact isClosed_aeStronglyMeasurable' hm
    /-
      🎉 no goals
    -/


theorem condexpL1CLM_lpMeas (f : lpMeas F' ℝ m 1 μ) :
    condexpL1CLM F' hm μ (f : α →₁[μ] F') = ↑f := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
    ⊢ Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑f) ↑f
  -/
  let g := lpMeasToLpTrimLie F' ℝ 1 μ hm f
  have hfg : f = (lpMeasToLpTrimLie F' ℝ 1 μ hm).symm g := by
    simp only [g, LinearIsometryEquiv.symm_apply_apply]
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
    g : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x := ( …
    hfg : Eq f ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g)
    ⊢ Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑f) ↑f
  -/
  rw [hfg]
  refine @Lp.induction α F' m _ 1 (μ.trim hm) _ ENNReal.coe_ne_top (fun g : α →₁[μ.trim hm] F' =>
    condexpL1CLM F' hm μ ((lpMeasToLpTrimLie F' ℝ 1 μ hm).symm g : α →₁[μ] F') =
    ↑((lpMeasToLpTrimLie F' ℝ 1 μ hm).symm g)) ?_ ?_ ?_ g
    /-
      case refine_1
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x := ( …
      hfg : Eq f ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g)
      ⊢ ∀ (c : F') {s : Set α} (hs : MeasurableSet s) (hμs : LT.lt ((μ.trim hm) s) T …
    -/
  · intro c s hs hμs
    rw [@Lp.simpleFunc.coe_indicatorConst _ _ m, lpMeasToLpTrimLie_symm_indicator hs hμs.ne c,
      condexpL1CLM_indicatorConstLp]
    /-
      case refine_1
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x := ( …
      hfg : Eq f ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g)
      c : F'
      s : Set α
      hs : MeasurableSet s
      hμs : LT.lt ((μ.trim hm) s) Top.top
      ⊢ Eq ((MeasureTheory.condexpInd F' hm μ s) c) (MeasureTheory.indicatorConstLp  …
    -/
    exact condexpInd_of_measurable hs ((le_trim hm).trans_lt hμs).ne c
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x := ( …
      hfg : Eq f ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g)
      ⊢ ∀ ⦃f g : α → F'⦄ (hf : MeasureTheory.Memℒp f 1 (μ.trim hm)) (hg : MeasureThe …
    -/
  · intro f g hf hg _ hf_eq hg_eq
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
      g✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x :=  …
      hfg : Eq f✝ ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g✝)
      f g : α → F'
      hf : MeasureTheory.Memℒp f 1 (μ.trim hm)
      hg : MeasureTheory.Memℒp g 1 (μ.trim hm)
      a✝ : Disjoint (Function.support f) (Function.support g)
      hf_eq : Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑((MeasureTheory.lpMeasToLpTr …
      hg_eq : Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑((MeasureTheory.lpMeasToLpTr …
      ⊢ Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑((MeasureTheory.lpMeasToLpTrimLie  …
    -/
    rw [LinearIsometryEquiv.map_add]
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
      g✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x :=  …
      hfg : Eq f✝ ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g✝)
      f g : α → F'
      hf : MeasureTheory.Memℒp f 1 (μ.trim hm)
      hg : MeasureTheory.Memℒp g 1 (μ.trim hm)
      a✝ : Disjoint (Function.support f) (Function.support g)
      hf_eq : Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑((MeasureTheory.lpMeasToLpTr …
      hg_eq : Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑((MeasureTheory.lpMeasToLpTr …
      ⊢ Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑(HAdd.hAdd ((MeasureTheory.lpMeasT …
    -/
    push_cast
    /-
      case refine_2
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f✝ : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
      g✝ : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x :=  …
      hfg : Eq f✝ ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g✝)
      f g : α → F'
      hf : MeasureTheory.Memℒp f 1 (μ.trim hm)
      hg : MeasureTheory.Memℒp g 1 (μ.trim hm)
      a✝ : Disjoint (Function.support f) (Function.support g)
      hf_eq : Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑((MeasureTheory.lpMeasToLpTr …
      hg_eq : Eq ((MeasureTheory.condexpL1CLM F' hm μ) ↑((MeasureTheory.lpMeasToLpTr …
      ⊢ Eq ((MeasureTheory.condexpL1CLM F' hm μ) (HAdd.hAdd ↑((MeasureTheory.lpMeasT …
    -/
    rw [map_add, hf_eq, hg_eq]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
      g : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x := ( …
      hfg : Eq f ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g)
      ⊢ IsClosed (setOf fun f => (fun g => Eq ((MeasureTheory.condexpL1CLM F' hm μ)  …
    -/
  · refine isClosed_eq ?_ ?_
      /-
        case refine_3.refine_1
        α : Type u_1
        F' : Type u_3
        inst✝³ : NormedAddCommGroup F'
        inst✝² : NormedSpace Real F'
        inst✝¹ : CompleteSpace F'
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        hm : LE.le m m0
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
        f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
        g : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x := ( …
        hfg : Eq f ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g)
        ⊢ Continuous fun f => (MeasureTheory.condexpL1CLM F' hm μ) ↑((MeasureTheory.lp …
      -/
    · refine (condexpL1CLM F' hm μ).continuous.comp (continuous_induced_dom.comp ?_)
      /-
        case refine_3.refine_1
        α : Type u_1
        F' : Type u_3
        inst✝³ : NormedAddCommGroup F'
        inst✝² : NormedSpace Real F'
        inst✝¹ : CompleteSpace F'
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        hm : LE.le m m0
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
        f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
        g : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x := ( …
        hfg : Eq f ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g)
        ⊢ Continuous ⇑(MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm
      -/
      exact LinearIsometryEquiv.continuous _
      /-
        🎉 no goals
      -/
      /-
        case refine_3.refine_2
        α : Type u_1
        F' : Type u_3
        inst✝³ : NormedAddCommGroup F'
        inst✝² : NormedSpace Real F'
        inst✝¹ : CompleteSpace F'
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        hm : LE.le m m0
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
        f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
        g : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x := ( …
        hfg : Eq f ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g)
        ⊢ Continuous fun f => ↑((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm f)
      -/
    · refine continuous_induced_dom.comp ?_
      /-
        case refine_3.refine_2
        α : Type u_1
        F' : Type u_3
        inst✝³ : NormedAddCommGroup F'
        inst✝² : NormedSpace Real F'
        inst✝¹ : CompleteSpace F'
        m m0 : MeasurableSpace α
        μ : MeasureTheory.Measure α
        hm : LE.le m m0
        inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
        f : Subtype fun x => Membership.mem (MeasureTheory.lpMeas F' Real m 1 μ) x
        g : Subtype fun x => Membership.mem (MeasureTheory.Lp F' 1 (μ.trim hm)) x := ( …
        hfg : Eq f ((MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm g)
        ⊢ Continuous ⇑(MeasureTheory.lpMeasToLpTrimLie F' Real 1 μ hm).symm
      -/
      exact LinearIsometryEquiv.continuous _
      /-
        🎉 no goals
      -/


theorem condexpL1CLM_of_aestronglyMeasurable' (f : α →₁[μ] F') (hfm : AEStronglyMeasurable' m f μ) :
    condexpL1CLM F' hm μ f = f :=
  condexpL1CLM_lpMeas (⟨f, hfm⟩ : lpMeas F' ℝ m 1 μ)


/-- Conditional expectation of a function, in L1. Its value is 0 if the function is not
integrable. The function-valued `condexp` should be used instead in most cases. -/
def condexpL1 (hm : m ≤ m0) (μ : Measure α) [SigmaFinite (μ.trim hm)] (f : α → F') : α →₁[μ] F' :=
  setToFun μ (condexpInd F' hm μ) (dominatedFinMeasAdditive_condexpInd F' hm μ) f


theorem condexpL1_undef (hf : ¬Integrable f μ) : condexpL1 hm μ f = 0 :=
  setToFun_undef (dominatedFinMeasAdditive_condexpInd F' hm μ) hf


theorem condexpL1_eq (hf : Integrable f μ) : condexpL1 hm μ f = condexpL1CLM F' hm μ (hf.toL1 f) :=
  setToFun_eq (dominatedFinMeasAdditive_condexpInd F' hm μ) hf


@[simp]
theorem condexpL1_zero : condexpL1 hm μ (0 : α → F') = 0 :=
  setToFun_zero _


@[simp]
theorem condexpL1_measure_zero (hm : m ≤ m0) : condexpL1 hm (0 : Measure α) f = 0 :=
  setToFun_measure_zero _ rfl


theorem aestronglyMeasurable'_condexpL1 {f : α → F'} :
    AEStronglyMeasurable' m (condexpL1 hm μ f) μ := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(MeasureTheory.condexpL1 hm μ f)) μ
  -/
  by_cases hf : Integrable f μ
    /-
      case pos
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : α → F'
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(MeasureTheory.condexpL1 hm μ f)) μ
    -/
  · rw [condexpL1_eq hf]
    /-
      case pos
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : α → F'
      hf : MeasureTheory.Integrable f μ
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑((MeasureTheory.condexpL1CLM F' hm  …
    -/
    exact aestronglyMeasurable'_condexpL1CLM _
    /-
      🎉 no goals
    -/
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : α → F'
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(MeasureTheory.condexpL1 hm μ f)) μ
    -/
  · rw [condexpL1_undef hf]
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : α → F'
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑0) μ
    -/
    refine AEStronglyMeasurable'.congr ?_ (coeFn_zero _ _ _).symm
    /-
      case neg
      α : Type u_1
      F' : Type u_3
      inst✝³ : NormedAddCommGroup F'
      inst✝² : NormedSpace Real F'
      inst✝¹ : CompleteSpace F'
      m m0 : MeasurableSpace α
      μ : MeasureTheory.Measure α
      hm : LE.le m m0
      inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
      f : α → F'
      hf : Not (MeasureTheory.Integrable f μ)
      ⊢ MeasureTheory.AEStronglyMeasurable' m 0 μ
    -/
    exact StronglyMeasurable.aeStronglyMeasurable' (@stronglyMeasurable_zero _ _ m _ _)
    /-
      🎉 no goals
    -/


theorem condexpL1_congr_ae (hm : m ≤ m0) [SigmaFinite (μ.trim hm)] (h : f =ᵐ[μ] g) :
    condexpL1 hm μ f = condexpL1 hm μ g :=
  setToFun_congr_ae _ h


theorem integrable_condexpL1 (f : α → F') : Integrable (condexpL1 hm μ f) μ :=
  L1.integrable_coeFn _


/-- The integral of the conditional expectation `condexpL1` over an `m`-measurable set is equal to
the integral of `f` on that set. See also `setIntegral_condexp`, the similar statement for
`condexp`. -/
theorem setIntegral_condexpL1 (hf : Integrable f μ) (hs : MeasurableSet[m] s) :
    ∫ x in s, condexpL1 hm μ f x ∂μ = ∫ x in s, f x ∂μ := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    s : Set α
    hf : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑(MeasureTheory.condexpL …
  -/
  simp_rw [condexpL1_eq hf]
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    s : Set α
    hf : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑((MeasureTheory.condexp …
  -/
  rw [setIntegral_condexpL1CLM (hf.toL1 f) hs]
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    s : Set α
    hf : MeasureTheory.Integrable f μ
    hs : MeasurableSet s
    ⊢ Eq (MeasureTheory.integral (μ.restrict s) fun x => ↑↑(MeasureTheory.Integrab …
  -/
  exact setIntegral_congr_ae (hm s hs) (hf.coeFn_toL1.mono fun x hx _ => hx)
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias set_integral_condexpL1 := setIntegral_condexpL1


theorem condexpL1_add (hf : Integrable f μ) (hg : Integrable g μ) :
    condexpL1 hm μ (f + g) = condexpL1 hm μ f + condexpL1 hm μ g :=
  setToFun_add _ hf hg


theorem condexpL1_neg (f : α → F') : condexpL1 hm μ (-f) = -condexpL1 hm μ f :=
  setToFun_neg _ f


theorem condexpL1_smul (c : 𝕜) (f : α → F') : condexpL1 hm μ (c • f) = c • condexpL1 hm μ f := by
  /-
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    c : 𝕜
    f : α → F'
    ⊢ Eq (MeasureTheory.condexpL1 hm μ (HSMul.hSMul c f)) (HSMul.hSMul c (MeasureT …
  -/
  refine setToFun_smul _ ?_ c f
  /-
    α : Type u_1
    F' : Type u_3
    𝕜 : Type u_6
    inst✝⁵ : RCLike 𝕜
    inst✝⁴ : NormedAddCommGroup F'
    inst✝³ : NormedSpace 𝕜 F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    c : 𝕜
    f : α → F'
    ⊢ ∀ (c : 𝕜) (s : Set α) (x : F'), Eq ((MeasureTheory.condexpInd F' hm μ s) (HS …
  -/
  exact fun c _ x => condexpInd_smul' c x
  /-
    🎉 no goals
  -/


theorem condexpL1_sub (hf : Integrable f μ) (hg : Integrable g μ) :
    condexpL1 hm μ (f - g) = condexpL1 hm μ f - condexpL1 hm μ g :=
  setToFun_sub _ hf hg


theorem condexpL1_of_aestronglyMeasurable' (hfm : AEStronglyMeasurable' m f μ)
    (hfi : Integrable f μ) : condexpL1 hm μ f =ᵐ[μ] f := by
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hfi : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑(MeasureTheory.condexpL1 hm μ f)) f
  -/
  rw [condexpL1_eq hfi]
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hfi : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq (↑↑((MeasureTheory.condexpL1CLM F' hm μ) ( …
  -/
  refine EventuallyEq.trans ?_ (Integrable.coeFn_toL1 hfi)
  /-
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hfi : MeasureTheory.Integrable f μ
    ⊢ (MeasureTheory.ae μ).EventuallyEq ↑↑((MeasureTheory.condexpL1CLM F' hm μ) (M …
  -/
  rw [condexpL1CLM_of_aestronglyMeasurable']
  /-
    case hfm
    α : Type u_1
    F' : Type u_3
    inst✝³ : NormedAddCommGroup F'
    inst✝² : NormedSpace Real F'
    inst✝¹ : CompleteSpace F'
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝ : MeasureTheory.SigmaFinite (μ.trim hm)
    f : α → F'
    hfm : MeasureTheory.AEStronglyMeasurable' m f μ
    hfi : MeasureTheory.Integrable f μ
    ⊢ MeasureTheory.AEStronglyMeasurable' m (↑↑(MeasureTheory.Integrable.toL1 f hf …
  -/
  exact AEStronglyMeasurable'.congr hfm (Integrable.coeFn_toL1 hfi).symm
  /-
    🎉 no goals
  -/


theorem condexpL1_mono {E} [NormedLatticeAddCommGroup E] [CompleteSpace E] [NormedSpace ℝ E]
    [OrderedSMul ℝ E] {f g : α → E} (hf : Integrable f μ) (hg : Integrable g μ) (hfg : f ≤ᵐ[μ] g) :
    condexpL1 hm μ f ≤ᵐ[μ] condexpL1 hm μ g := by
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝⁴ : MeasureTheory.SigmaFinite (μ.trim hm)
    E : Type u_7
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    ⊢ (MeasureTheory.ae μ).EventuallyLE ↑↑(MeasureTheory.condexpL1 hm μ f) ↑↑(Meas …
  -/
  rw [coeFn_le]
  have h_nonneg : ∀ s, MeasurableSet s → μ s < ∞ → ∀ x : E, 0 ≤ x → 0 ≤ condexpInd E hm μ s x :=
    fun s hs hμs x hx => condexpInd_nonneg hs hμs.ne x hx
  /-
    α : Type u_1
    m m0 : MeasurableSpace α
    μ : MeasureTheory.Measure α
    hm : LE.le m m0
    inst✝⁴ : MeasureTheory.SigmaFinite (μ.trim hm)
    E : Type u_7
    inst✝³ : NormedLatticeAddCommGroup E
    inst✝² : CompleteSpace E
    inst✝¹ : NormedSpace Real E
    inst✝ : OrderedSMul Real E
    f g : α → E
    hf : MeasureTheory.Integrable f μ
    hg : MeasureTheory.Integrable g μ
    hfg : (MeasureTheory.ae μ).EventuallyLE f g
    h_nonneg : ∀ (s : Set α), MeasurableSet s → LT.lt (μ s) Top.top → ∀ (x : E), L …
    ⊢ LE.le (MeasureTheory.condexpL1 hm μ f) (MeasureTheory.condexpL1 hm μ g)
  -/
  exact setToFun_mono (dominatedFinMeasAdditive_condexpInd E hm μ) h_nonneg hf hg hfg
  /-
    🎉 no goals
  -/


