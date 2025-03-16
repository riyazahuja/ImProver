/-- A family of continuous linear maps is `C^n` on `s` if all its applications are. -/
theorem contDiffOn_clm_apply {f : D → E →L[𝕜] F} {s : Set D} [FiniteDimensional 𝕜 E] :
    ContDiffOn 𝕜 n f s ↔ ∀ y, ContDiffOn 𝕜 n (fun x => f x y) s := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    n : WithTop ENat
    inst✝¹ : CompleteSpace 𝕜
    f : D → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set D
    inst✝ : FiniteDimensional 𝕜 E
    ⊢ Iff (ContDiffOn 𝕜 n f s) (∀ (y : E), ContDiffOn 𝕜 n (fun x => (f x) y) s)
  -/
  refine ⟨fun h y => h.clm_apply contDiffOn_const, fun h => ?_⟩
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    n : WithTop ENat
    inst✝¹ : CompleteSpace 𝕜
    f : D → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set D
    inst✝ : FiniteDimensional 𝕜 E
    h : ∀ (y : E), ContDiffOn 𝕜 n (fun x => (f x) y) s
    ⊢ ContDiffOn 𝕜 n f s
  -/
  let d := finrank 𝕜 E
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    n : WithTop ENat
    inst✝¹ : CompleteSpace 𝕜
    f : D → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set D
    inst✝ : FiniteDimensional 𝕜 E
    h : ∀ (y : E), ContDiffOn 𝕜 n (fun x => (f x) y) s
    d : Nat := Module.finrank 𝕜 E
    ⊢ ContDiffOn 𝕜 n f s
  -/
  have hd : d = finrank 𝕜 (Fin d → 𝕜) := (finrank_fin_fun 𝕜).symm
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    n : WithTop ENat
    inst✝¹ : CompleteSpace 𝕜
    f : D → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set D
    inst✝ : FiniteDimensional 𝕜 E
    h : ∀ (y : E), ContDiffOn 𝕜 n (fun x => (f x) y) s
    d : Nat := Module.finrank 𝕜 E
    hd : Eq d (Module.finrank 𝕜 (Fin d → 𝕜))
    ⊢ ContDiffOn 𝕜 n f s
  -/
  let e₁ := ContinuousLinearEquiv.ofFinrankEq hd
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    n : WithTop ENat
    inst✝¹ : CompleteSpace 𝕜
    f : D → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set D
    inst✝ : FiniteDimensional 𝕜 E
    h : ∀ (y : E), ContDiffOn 𝕜 n (fun x => (f x) y) s
    d : Nat := Module.finrank 𝕜 E
    hd : Eq d (Module.finrank 𝕜 (Fin d → 𝕜))
    e₁ : ContinuousLinearEquiv (RingHom.id 𝕜) E (Fin d → 𝕜) := ContinuousLinearEqu …
    ⊢ ContDiffOn 𝕜 n f s
  -/
  let e₂ := (e₁.arrowCongr (1 : F ≃L[𝕜] F)).trans (ContinuousLinearEquiv.piRing (Fin d))
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    n : WithTop ENat
    inst✝¹ : CompleteSpace 𝕜
    f : D → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set D
    inst✝ : FiniteDimensional 𝕜 E
    h : ∀ (y : E), ContDiffOn 𝕜 n (fun x => (f x) y) s
    d : Nat := Module.finrank 𝕜 E
    hd : Eq d (Module.finrank 𝕜 (Fin d → 𝕜))
    e₁ : ContinuousLinearEquiv (RingHom.id 𝕜) E (Fin d → 𝕜) := ContinuousLinearEqu …
    e₂ : ContinuousLinearEquiv (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜)  …
    ⊢ ContDiffOn 𝕜 n f s
  -/
  rw [← id_comp f, ← e₂.symm_comp_self]
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    n : WithTop ENat
    inst✝¹ : CompleteSpace 𝕜
    f : D → ContinuousLinearMap (RingHom.id 𝕜) E F
    s : Set D
    inst✝ : FiniteDimensional 𝕜 E
    h : ∀ (y : E), ContDiffOn 𝕜 n (fun x => (f x) y) s
    d : Nat := Module.finrank 𝕜 E
    hd : Eq d (Module.finrank 𝕜 (Fin d → 𝕜))
    e₁ : ContinuousLinearEquiv (RingHom.id 𝕜) E (Fin d → 𝕜) := ContinuousLinearEqu …
    e₂ : ContinuousLinearEquiv (RingHom.id 𝕜) (ContinuousLinearMap (RingHom.id 𝕜)  …
    ⊢ ContDiffOn 𝕜 n (Function.comp (Function.comp ⇑e₂.symm ⇑e₂) f) s
  -/
  exact e₂.symm.contDiff.comp_contDiffOn (contDiffOn_pi.mpr fun i => h _)
  /-
    🎉 no goals
  -/


theorem contDiff_clm_apply_iff {f : D → E →L[𝕜] F} [FiniteDimensional 𝕜 E] :
    ContDiff 𝕜 n f ↔ ∀ y, ContDiff 𝕜 n fun x => f x y := by
  /-
    𝕜 : Type u_1
    inst✝⁸ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁷ : NormedAddCommGroup D
    inst✝⁶ : NormedSpace 𝕜 D
    E : Type uE
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type uF
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    n : WithTop ENat
    inst✝¹ : CompleteSpace 𝕜
    f : D → ContinuousLinearMap (RingHom.id 𝕜) E F
    inst✝ : FiniteDimensional 𝕜 E
    ⊢ Iff (ContDiff 𝕜 n f) (∀ (y : E), ContDiff 𝕜 n fun x => (f x) y)
  -/
  simp_rw [← contDiffOn_univ, contDiffOn_clm_apply]
  /-
    🎉 no goals
  -/


/-- This is a useful lemma to prove that a certain operation preserves functions being `C^n`.
When you do induction on `n`, this gives a useful characterization of a function being `C^(n+1)`,
assuming you have already computed the derivative. The advantage of this version over
`contDiff_succ_iff_fderiv` is that both occurrences of `ContDiff` are for functions with the same
domain and codomain (`D` and `E`). This is not the case for `contDiff_succ_iff_fderiv`, which
often requires an inconvenient need to generalize `F`, which results in universe issues
(see the discussion in the section of `ContDiff.comp`).

This lemma avoids these universe issues, but only applies for finite dimensional `D`. -/
theorem contDiff_succ_iff_fderiv_apply [FiniteDimensional 𝕜 D] :
    ContDiff 𝕜 (n + 1) f ↔ Differentiable 𝕜 f ∧
      (n = ω → AnalyticOnNhd 𝕜 f Set.univ) ∧ ∀ y, ContDiff 𝕜 n fun x => fderiv 𝕜 f x y := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁵ : NormedAddCommGroup D
    inst✝⁴ : NormedSpace 𝕜 D
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    n : WithTop ENat
    f : D → E
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 D
    ⊢ Iff (ContDiff 𝕜 (HAdd.hAdd n 1) f) (And (Differentiable 𝕜 f) (And (Eq n Top. …
  -/
  rw [contDiff_succ_iff_fderiv, contDiff_clm_apply_iff]
  /-
    🎉 no goals
  -/


theorem contDiffOn_succ_of_fderiv_apply [FiniteDimensional 𝕜 D]
    (hf : DifferentiableOn 𝕜 f s) (h'f : n = ω → AnalyticOn 𝕜 f s)
    (h : ∀ y, ContDiffOn 𝕜 n (fun x => fderivWithin 𝕜 f s x y) s) :
    ContDiffOn 𝕜 (n + 1) f s :=
  contDiffOn_succ_of_fderivWithin hf h'f <| contDiffOn_clm_apply.mpr h


theorem contDiffOn_succ_iff_fderiv_apply [FiniteDimensional 𝕜 D] (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 (n + 1) f s ↔
      DifferentiableOn 𝕜 f s ∧ (n = ω → AnalyticOn 𝕜 f s) ∧
      ∀ y, ContDiffOn 𝕜 n (fun x => fderivWithin 𝕜 f s x y) s := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    D : Type uD
    inst✝⁵ : NormedAddCommGroup D
    inst✝⁴ : NormedSpace 𝕜 D
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    n : WithTop ENat
    f : D → E
    s : Set D
    inst✝¹ : CompleteSpace 𝕜
    inst✝ : FiniteDimensional 𝕜 D
    hs : UniqueDiffOn 𝕜 s
    ⊢ Iff (ContDiffOn 𝕜 (HAdd.hAdd n 1) f s) (And (DifferentiableOn 𝕜 f s) (And (E …
  -/
  rw [contDiffOn_succ_iff_fderivWithin hs, contDiffOn_clm_apply]
  /-
    🎉 no goals
  -/


