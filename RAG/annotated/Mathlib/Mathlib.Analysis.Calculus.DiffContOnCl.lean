/-- A predicate saying that a function is differentiable on a set and is continuous on its
closure. This is a common assumption in complex analysis. -/
structure DiffContOnCl (f : E → F) (s : Set E) : Prop where
  protected differentiableOn : DifferentiableOn 𝕜 f s
  protected continuousOn : ContinuousOn f (closure s)


theorem DifferentiableOn.diffContOnCl (h : DifferentiableOn 𝕜 f (closure s)) : DiffContOnCl 𝕜 f s :=
  ⟨h.mono subset_closure, h.continuousOn⟩


theorem Differentiable.diffContOnCl (h : Differentiable 𝕜 f) : DiffContOnCl 𝕜 f s :=
  ⟨h.differentiableOn, h.continuous.continuousOn⟩


theorem IsClosed.diffContOnCl_iff (hs : IsClosed s) : DiffContOnCl 𝕜 f s ↔ DifferentiableOn 𝕜 f s :=
  ⟨fun h => h.differentiableOn, fun h => ⟨h, hs.closure_eq.symm ▸ h.continuousOn⟩⟩


theorem diffContOnCl_univ : DiffContOnCl 𝕜 f univ ↔ Differentiable 𝕜 f :=
  isClosed_univ.diffContOnCl_iff.trans differentiableOn_univ


theorem diffContOnCl_const {c : F} : DiffContOnCl 𝕜 (fun _ : E => c) s :=
  ⟨differentiableOn_const c, continuousOn_const⟩


theorem comp {g : G → E} {t : Set G} (hf : DiffContOnCl 𝕜 f s) (hg : DiffContOnCl 𝕜 g t)
    (h : MapsTo g t s) : DiffContOnCl 𝕜 (f ∘ g) t :=
  ⟨hf.1.comp hg.1 h, hf.2.comp hg.2 <| h.closure_of_continuousOn hg.2⟩


theorem continuousOn_ball [NormedSpace ℝ E] {x : E} {r : ℝ} (h : DiffContOnCl 𝕜 f (ball x r)) :
    ContinuousOn f (closedBall x r) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁵ : NontriviallyNormedField 𝕜
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    inst✝ : NormedSpace Real E
    x : E
    r : Real
    h : DiffContOnCl 𝕜 f (Metric.ball x r)
    ⊢ ContinuousOn f (Metric.closedBall x r)
  -/
  rcases eq_or_ne r 0 with (rfl | hr)
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      inst✝ : NormedSpace Real E
      x : E
      h : DiffContOnCl 𝕜 f (Metric.ball x 0)
      ⊢ ContinuousOn f (Metric.closedBall x 0)
    -/
  · rw [closedBall_zero]
    /-
      case inl
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      inst✝ : NormedSpace Real E
      x : E
      h : DiffContOnCl 𝕜 f (Metric.ball x 0)
      ⊢ ContinuousOn f (Singleton.singleton x)
    -/
    exact continuousOn_singleton f x
    /-
      🎉 no goals
    -/
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      inst✝ : NormedSpace Real E
      x : E
      r : Real
      h : DiffContOnCl 𝕜 f (Metric.ball x r)
      hr : Ne r 0
      ⊢ ContinuousOn f (Metric.closedBall x r)
    -/
  · rw [← closure_ball x hr]
    /-
      case inr
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      inst✝ : NormedSpace Real E
      x : E
      r : Real
      h : DiffContOnCl 𝕜 f (Metric.ball x r)
      hr : Ne r 0
      ⊢ ContinuousOn f (closure (Metric.ball x r))
    -/
    exact h.continuousOn
    /-
      🎉 no goals
    -/


theorem mk_ball {x : E} {r : ℝ} (hd : DifferentiableOn 𝕜 f (ball x r))
    (hc : ContinuousOn f (closedBall x r)) : DiffContOnCl 𝕜 f (ball x r) :=
  ⟨hd, hc.mono <| closure_ball_subset_closedBall⟩


protected theorem differentiableAt (h : DiffContOnCl 𝕜 f s) (hs : IsOpen s) (hx : x ∈ s) :
    DifferentiableAt 𝕜 f x :=
  h.differentiableOn.differentiableAt <| hs.mem_nhds hx


theorem differentiableAt' (h : DiffContOnCl 𝕜 f s) (hx : s ∈ 𝓝 x) : DifferentiableAt 𝕜 f x :=
  h.differentiableOn.differentiableAt hx


protected theorem mono (h : DiffContOnCl 𝕜 f s) (ht : t ⊆ s) : DiffContOnCl 𝕜 f t :=
  ⟨h.differentiableOn.mono ht, h.continuousOn.mono (closure_mono ht)⟩


theorem add (hf : DiffContOnCl 𝕜 f s) (hg : DiffContOnCl 𝕜 g s) : DiffContOnCl 𝕜 (f + g) s :=
  ⟨hf.1.add hg.1, hf.2.add hg.2⟩


theorem add_const (hf : DiffContOnCl 𝕜 f s) (c : F) : DiffContOnCl 𝕜 (fun x => f x + c) s :=
  hf.add diffContOnCl_const


theorem const_add (hf : DiffContOnCl 𝕜 f s) (c : F) : DiffContOnCl 𝕜 (fun x => c + f x) s :=
  diffContOnCl_const.add hf


theorem neg (hf : DiffContOnCl 𝕜 f s) : DiffContOnCl 𝕜 (-f) s :=
  ⟨hf.1.neg, hf.2.neg⟩


theorem sub (hf : DiffContOnCl 𝕜 f s) (hg : DiffContOnCl 𝕜 g s) : DiffContOnCl 𝕜 (f - g) s :=
  ⟨hf.1.sub hg.1, hf.2.sub hg.2⟩


theorem sub_const (hf : DiffContOnCl 𝕜 f s) (c : F) : DiffContOnCl 𝕜 (fun x => f x - c) s :=
  hf.sub diffContOnCl_const


theorem const_sub (hf : DiffContOnCl 𝕜 f s) (c : F) : DiffContOnCl 𝕜 (fun x => c - f x) s :=
  diffContOnCl_const.sub hf


theorem const_smul {R : Type*} [Semiring R] [Module R F] [SMulCommClass 𝕜 R F]
    [ContinuousConstSMul R F] (hf : DiffContOnCl 𝕜 f s) (c : R) : DiffContOnCl 𝕜 (c • f) s :=
  ⟨hf.1.const_smul c, hf.2.const_smul c⟩


theorem smul {𝕜' : Type*} [NontriviallyNormedField 𝕜'] [NormedAlgebra 𝕜 𝕜'] [NormedSpace 𝕜' F]
    [IsScalarTower 𝕜 𝕜' F] {c : E → 𝕜'} {f : E → F} {s : Set E} (hc : DiffContOnCl 𝕜 c s)
    (hf : DiffContOnCl 𝕜 f s) : DiffContOnCl 𝕜 (fun x => c x • f x) s :=
  ⟨hc.1.smul hf.1, hc.2.smul hf.2⟩


theorem smul_const {𝕜' : Type*} [NontriviallyNormedField 𝕜'] [NormedAlgebra 𝕜 𝕜']
    [NormedSpace 𝕜' F] [IsScalarTower 𝕜 𝕜' F] {c : E → 𝕜'} {s : Set E} (hc : DiffContOnCl 𝕜 c s)
    (y : F) : DiffContOnCl 𝕜 (fun x => c x • y) s :=
  hc.smul diffContOnCl_const


theorem inv {f : E → 𝕜} (hf : DiffContOnCl 𝕜 f s) (h₀ : ∀ x ∈ closure s, f x ≠ 0) :
    DiffContOnCl 𝕜 f⁻¹ s :=
  ⟨differentiableOn_inv.comp hf.1 fun _ hx => h₀ _ (subset_closure hx), hf.2.inv₀ h₀⟩


theorem Differentiable.comp_diffContOnCl {g : G → E} {t : Set G} (hf : Differentiable 𝕜 f)
    (hg : DiffContOnCl 𝕜 g t) : DiffContOnCl 𝕜 (f ∘ g) t :=
  hf.diffContOnCl.comp hg (mapsTo_image _ _)


theorem DifferentiableOn.diffContOnCl_ball {U : Set E} {c : E} {R : ℝ} (hf : DifferentiableOn 𝕜 f U)
    (hc : closedBall c R ⊆ U) : DiffContOnCl 𝕜 f (ball c R) :=
  DiffContOnCl.mk_ball (hf.mono (ball_subset_closedBall.trans hc)) (hf.continuousOn.mono hc)

