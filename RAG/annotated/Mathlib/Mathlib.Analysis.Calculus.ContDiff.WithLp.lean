theorem contDiffWithinAt_piLp :
    ContDiffWithinAt 𝕜 n f t y ↔ ∀ i, ContDiffWithinAt 𝕜 n (fun x => f x i) t y := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    n : WithTop ENat
    f : H → PiLp p E
    t : Set H
    y : H
    ⊢ Iff (ContDiffWithinAt 𝕜 n f t y) (∀ (i : ι), ContDiffWithinAt 𝕜 n (fun x =>  …
  -/
  rw [← (PiLp.continuousLinearEquiv p 𝕜 E).comp_contDiffWithinAt_iff, contDiffWithinAt_pi]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    n : WithTop ENat
    f : H → PiLp p E
    t : Set H
    y : H
    ⊢ Iff (∀ (i : ι), ContDiffWithinAt 𝕜 n (fun x => Function.comp (⇑(PiLp.continu …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem contDiffAt_piLp :
    ContDiffAt 𝕜 n f y ↔ ∀ i, ContDiffAt 𝕜 n (fun x => f x i) y := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    n : WithTop ENat
    f : H → PiLp p E
    y : H
    ⊢ Iff (ContDiffAt 𝕜 n f y) (∀ (i : ι), ContDiffAt 𝕜 n (fun x => f x i) y)
  -/
  rw [← (PiLp.continuousLinearEquiv p 𝕜 E).comp_contDiffAt_iff, contDiffAt_pi]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    n : WithTop ENat
    f : H → PiLp p E
    y : H
    ⊢ Iff (∀ (i : ι), ContDiffAt 𝕜 n (fun x => Function.comp (⇑(PiLp.continuousLin …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem contDiffOn_piLp :
    ContDiffOn 𝕜 n f t ↔ ∀ i, ContDiffOn 𝕜 n (fun x => f x i) t := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    n : WithTop ENat
    f : H → PiLp p E
    t : Set H
    ⊢ Iff (ContDiffOn 𝕜 n f t) (∀ (i : ι), ContDiffOn 𝕜 n (fun x => f x i) t)
  -/
  rw [← (PiLp.continuousLinearEquiv p 𝕜 E).comp_contDiffOn_iff, contDiffOn_pi]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    n : WithTop ENat
    f : H → PiLp p E
    t : Set H
    ⊢ Iff (∀ (i : ι), ContDiffOn 𝕜 n (fun x => Function.comp (⇑(PiLp.continuousLin …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem contDiff_piLp : ContDiff 𝕜 n f ↔ ∀ i, ContDiff 𝕜 n fun x => f x i := by
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    n : WithTop ENat
    f : H → PiLp p E
    ⊢ Iff (ContDiff 𝕜 n f) (∀ (i : ι), ContDiff 𝕜 n fun x => f x i)
  -/
  rw [← (PiLp.continuousLinearEquiv p 𝕜 E).comp_contDiff_iff, contDiff_pi]
  /-
    𝕜 : Type u_1
    ι : Type u_2
    E : ι → Type u_3
    H : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup H
    inst✝⁴ : (i : ι) → NormedAddCommGroup (E i)
    inst✝³ : (i : ι) → NormedSpace 𝕜 (E i)
    inst✝² : NormedSpace 𝕜 H
    inst✝¹ : Fintype ι
    p : ENNReal
    inst✝ : Fact (LE.le 1 p)
    n : WithTop ENat
    f : H → PiLp p E
    ⊢ Iff (∀ (i : ι), ContDiff 𝕜 n fun x => Function.comp (⇑(PiLp.continuousLinear …
  -/
  rfl
  /-
    🎉 no goals
  -/


