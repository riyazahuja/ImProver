/-- A polynomial function is infinitely differentiable. -/
theorem CPolynomialOn.contDiffOn (h : CPolynomialOn 𝕜 f s) {n : WithTop ℕ∞} :
    ContDiffOn 𝕜 n f s := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : CPolynomialOn 𝕜 f s
    n : WithTop ENat
    ⊢ ContDiffOn 𝕜 n f s
  -/
  let t := { x | CPolynomialAt 𝕜 f x }
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : CPolynomialOn 𝕜 f s
    n : WithTop ENat
    t : Set E := setOf fun x => CPolynomialAt 𝕜 f x
    ⊢ ContDiffOn 𝕜 n f s
  -/
  suffices ContDiffOn 𝕜 n f t from this.mono h
  suffices AnalyticOnNhd 𝕜 f t by
    have t_open : IsOpen t := isOpen_cPolynomialAt 𝕜 f
    exact AnalyticOnNhd.contDiffOn this t_open.uniqueDiffOn
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : CPolynomialOn 𝕜 f s
    n : WithTop ENat
    t : Set E := setOf fun x => CPolynomialAt 𝕜 f x
    ⊢ AnalyticOnNhd 𝕜 f t
  -/
  have H : CPolynomialOn 𝕜 f t := fun _x hx ↦ hx
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type v
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    h : CPolynomialOn 𝕜 f s
    n : WithTop ENat
    t : Set E := setOf fun x => CPolynomialAt 𝕜 f x
    H : CPolynomialOn 𝕜 f t
    ⊢ AnalyticOnNhd 𝕜 f t
  -/
  exact H.analyticOnNhd
  /-
    🎉 no goals
  -/


theorem CPolynomialAt.contDiffAt (h : CPolynomialAt 𝕜 f x) {n : WithTop ℕ∞} :
    ContDiffAt 𝕜 n f x :=
  let ⟨_, hs, hf⟩ := h.exists_mem_nhds_cPolynomialOn
  hf.contDiffOn.contDiffAt hs


lemma contDiffAt : ContDiffAt 𝕜 n f x := f.cpolynomialAt.contDiffAt


lemma contDiff : ContDiff 𝕜 n f := contDiff_iff_contDiffAt.mpr (fun _ ↦ f.contDiffAt)


