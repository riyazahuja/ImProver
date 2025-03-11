/-- The `n`-th iterated derivative of a function from `𝕜` to `F`, as a function from `𝕜` to `F`. -/
def iteratedDeriv (n : ℕ) (f : 𝕜 → F) (x : 𝕜) : F :=
  (iteratedFDeriv 𝕜 n f x : (Fin n → 𝕜) → F) fun _ : Fin n => 1


/-- The `n`-th iterated derivative of a function from `𝕜` to `F` within a set `s`, as a function
from `𝕜` to `F`. -/
def iteratedDerivWithin (n : ℕ) (f : 𝕜 → F) (s : Set 𝕜) (x : 𝕜) : F :=
  (iteratedFDerivWithin 𝕜 n f s x : (Fin n → 𝕜) → F) fun _ : Fin n => 1


theorem iteratedDerivWithin_univ : iteratedDerivWithin n f univ = iteratedDeriv n f := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    ⊢ Eq (iteratedDerivWithin n f Set.univ) (iteratedDeriv n f)
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    x : 𝕜
    ⊢ Eq (iteratedDerivWithin n f Set.univ x) (iteratedDeriv n f x)
  -/
  rw [iteratedDerivWithin, iteratedDeriv, iteratedFDerivWithin_univ]
  /-
    🎉 no goals
  -/


theorem iteratedDerivWithin_eq_iteratedFDerivWithin : iteratedDerivWithin n f s x =
    (iteratedFDerivWithin 𝕜 n f s x : (Fin n → 𝕜) → F) fun _ : Fin n => 1 :=
  rfl


/-- Write the iterated derivative as the composition of a continuous linear equiv and the iterated
Fréchet derivative -/
theorem iteratedDerivWithin_eq_equiv_comp : iteratedDerivWithin n f s =
    (ContinuousMultilinearMap.piFieldEquiv 𝕜 (Fin n) F).symm ∘ iteratedFDerivWithin 𝕜 n f s := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    s : Set 𝕜
    ⊢ Eq (iteratedDerivWithin n f s) (Function.comp (⇑(ContinuousMultilinearMap.pi …
  -/
  ext x; rfl
         /-
           🎉 no goals
         -/


/-- Write the iterated Fréchet derivative as the composition of a continuous linear equiv and the
iterated derivative. -/
theorem iteratedFDerivWithin_eq_equiv_comp :
    iteratedFDerivWithin 𝕜 n f s =
      ContinuousMultilinearMap.piFieldEquiv 𝕜 (Fin n) F ∘ iteratedDerivWithin n f s := by
  rw [iteratedDerivWithin_eq_equiv_comp, ← Function.comp_assoc, LinearIsometryEquiv.self_comp_symm,
    Function.id_comp]


/-- The `n`-th Fréchet derivative applied to a vector `(m 0, ..., m (n-1))` is the derivative
multiplied by the product of the `m i`s. -/
theorem iteratedFDerivWithin_apply_eq_iteratedDerivWithin_mul_prod {m : Fin n → 𝕜} :
    (iteratedFDerivWithin 𝕜 n f s x : (Fin n → 𝕜) → F) m =
      (∏ i, m i) • iteratedDerivWithin n f s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    s : Set 𝕜
    x : 𝕜
    m : Fin n → 𝕜
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) m) (HSMul.hSMul (Finset.univ.prod fun i …
  -/
  rw [iteratedDerivWithin_eq_iteratedFDerivWithin, ← ContinuousMultilinearMap.map_smul_univ]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    s : Set 𝕜
    x : 𝕜
    m : Fin n → 𝕜
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) m) ((iteratedFDerivWithin 𝕜 n f s x) fu …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem norm_iteratedFDerivWithin_eq_norm_iteratedDerivWithin :
    ‖iteratedFDerivWithin 𝕜 n f s x‖ = ‖iteratedDerivWithin n f s x‖ := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    s : Set 𝕜
    x : 𝕜
    ⊢ Eq (Norm.norm (iteratedFDerivWithin 𝕜 n f s x)) (Norm.norm (iteratedDerivWit …
  -/
  rw [iteratedDerivWithin_eq_equiv_comp, Function.comp_apply, LinearIsometryEquiv.norm_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem iteratedDerivWithin_zero : iteratedDerivWithin 0 f s = f := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s : Set 𝕜
    ⊢ Eq (iteratedDerivWithin 0 f s) f
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s : Set 𝕜
    x : 𝕜
    ⊢ Eq (iteratedDerivWithin 0 f s x) (f x)
  -/
  simp [iteratedDerivWithin]
  /-
    🎉 no goals
  -/


@[simp]
theorem iteratedDerivWithin_one {x : 𝕜} (h : UniqueDiffWithinAt 𝕜 s x) :
    iteratedDerivWithin 1 f s x = derivWithin f s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s : Set 𝕜
    x : 𝕜
    h : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq (iteratedDerivWithin 1 f s x) (derivWithin f s x)
  -/
  simp only [iteratedDerivWithin, iteratedFDerivWithin_one_apply h]; rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- If the first `n` derivatives within a set of a function are continuous, and its first `n-1`
derivatives are differentiable, then the function is `C^n`. This is not an equivalence in general,
but this is an equivalence when the set has unique derivatives, see
`contDiffOn_iff_continuousOn_differentiableOn_deriv`. -/
theorem contDiffOn_of_continuousOn_differentiableOn_deriv {n : ℕ∞}
    (Hcont : ∀ m : ℕ, (m : ℕ∞) ≤ n → ContinuousOn (fun x => iteratedDerivWithin m f s x) s)
    (Hdiff : ∀ m : ℕ, (m : ℕ∞) < n → DifferentiableOn 𝕜 (fun x => iteratedDerivWithin m f s x) s) :
    ContDiffOn 𝕜 n f s := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s : Set 𝕜
    n : ENat
    Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedDerivWithin …
    Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedDeriv …
    ⊢ ContDiffOn 𝕜 (↑n) f s
  -/
  apply contDiffOn_of_continuousOn_differentiableOn
    /-
      case Hcont
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s : Set 𝕜
      n : ENat
      Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedDerivWithin …
      Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedDeriv …
      ⊢ ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedFDerivWithin 𝕜 m  …
    -/
  · simpa only [iteratedFDerivWithin_eq_equiv_comp, LinearIsometryEquiv.comp_continuousOn_iff]
    /-
      🎉 no goals
    -/
    /-
      case Hdiff
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      s : Set 𝕜
      n : ENat
      Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedDerivWithin …
      Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedDeriv …
      ⊢ ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedFDerivWithi …
    -/
  · simpa only [iteratedFDerivWithin_eq_equiv_comp, LinearIsometryEquiv.comp_differentiableOn_iff]
    /-
      🎉 no goals
    -/


/-- To check that a function is `n` times continuously differentiable, it suffices to check that its
first `n` derivatives are differentiable. This is slightly too strong as the condition we
require on the `n`-th derivative is differentiability instead of continuity, but it has the
advantage of avoiding the discussion of continuity in the proof (and for `n = ∞` this is optimal).
-/
theorem contDiffOn_of_differentiableOn_deriv {n : ℕ∞}
    (h : ∀ m : ℕ, (m : ℕ∞) ≤ n → DifferentiableOn 𝕜 (iteratedDerivWithin m f s) s) :
    ContDiffOn 𝕜 n f s := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s : Set 𝕜
    n : ENat
    h : ∀ (m : Nat), LE.le (↑m) n → DifferentiableOn 𝕜 (iteratedDerivWithin m f s) s
    ⊢ ContDiffOn 𝕜 (↑n) f s
  -/
  apply contDiffOn_of_differentiableOn
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s : Set 𝕜
    n : ENat
    h : ∀ (m : Nat), LE.le (↑m) n → DifferentiableOn 𝕜 (iteratedDerivWithin m f s) s
    ⊢ ∀ (m : Nat), LE.le (↑m) n → DifferentiableOn 𝕜 (iteratedFDerivWithin 𝕜 m f s …
  -/
  simpa only [iteratedFDerivWithin_eq_equiv_comp, LinearIsometryEquiv.comp_differentiableOn_iff]
  /-
    🎉 no goals
  -/


/-- On a set with unique derivatives, a `C^n` function has derivatives up to `n` which are
continuous. -/
theorem ContDiffOn.continuousOn_iteratedDerivWithin
    {n : WithTop ℕ∞} {m : ℕ} (h : ContDiffOn 𝕜 n f s)
    (hmn : m ≤ n) (hs : UniqueDiffOn 𝕜 s) : ContinuousOn (iteratedDerivWithin m f s) s := by
  simpa only [iteratedDerivWithin_eq_equiv_comp, LinearIsometryEquiv.comp_continuousOn_iff] using
    h.continuousOn_iteratedFDerivWithin hmn hs


theorem ContDiffWithinAt.differentiableWithinAt_iteratedDerivWithin {n : WithTop ℕ∞} {m : ℕ}
    (h : ContDiffWithinAt 𝕜 n f s x) (hmn : m < n) (hs : UniqueDiffOn 𝕜 (insert x s)) :
    DifferentiableWithinAt 𝕜 (iteratedDerivWithin m f s) s x := by
  simpa only [iteratedDerivWithin_eq_equiv_comp,
    LinearIsometryEquiv.comp_differentiableWithinAt_iff] using
    h.differentiableWithinAt_iteratedFDerivWithin hmn hs


/-- On a set with unique derivatives, a `C^n` function has derivatives less than `n` which are
differentiable. -/
theorem ContDiffOn.differentiableOn_iteratedDerivWithin {n : WithTop ℕ∞} {m : ℕ}
    (h : ContDiffOn 𝕜 n f s) (hmn : m < n) (hs : UniqueDiffOn 𝕜 s) :
    DifferentiableOn 𝕜 (iteratedDerivWithin m f s) s := fun x hx =>
                                                                /-
                                                                  𝕜 : Type u_1
                                                                  inst✝² : NontriviallyNormedField 𝕜
                                                                  F : Type u_2
                                                                  inst✝¹ : NormedAddCommGroup F
                                                                  inst✝ : NormedSpace 𝕜 F
                                                                  f : 𝕜 → F
                                                                  s : Set 𝕜
                                                                  n : WithTop ENat
                                                                  m : Nat
                                                                  h : ContDiffOn 𝕜 n f s
                                                                  hmn : LT.lt (↑m) n
                                                                  hs : UniqueDiffOn 𝕜 s
                                                                  x : 𝕜
                                                                  hx : Membership.mem s x
                                                                  ⊢ UniqueDiffOn 𝕜 (Insert.insert x s)
                                                                -/
  (h x hx).differentiableWithinAt_iteratedDerivWithin hmn <| by rwa [insert_eq_of_mem hx]
                                                                /-
                                                                  🎉 no goals
                                                                -/


/-- The property of being `C^n`, initially defined in terms of the Fréchet derivative, can be
reformulated in terms of the one-dimensional derivative on sets with unique derivatives. -/
theorem contDiffOn_iff_continuousOn_differentiableOn_deriv {n : ℕ∞} (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 n f s ↔ (∀ m : ℕ, (m : ℕ∞) ≤ n → ContinuousOn (iteratedDerivWithin m f s) s) ∧
      ∀ m : ℕ, (m : ℕ∞) < n → DifferentiableOn 𝕜 (iteratedDerivWithin m f s) s := by
  simp only [contDiffOn_iff_continuousOn_differentiableOn hs, iteratedFDerivWithin_eq_equiv_comp,
    LinearIsometryEquiv.comp_continuousOn_iff, LinearIsometryEquiv.comp_differentiableOn_iff]


/-- The property of being `C^n`, initially defined in terms of the Fréchet derivative, can be
reformulated in terms of the one-dimensional derivative on sets with unique derivatives. -/
theorem contDiffOn_nat_iff_continuousOn_differentiableOn_deriv {n : ℕ} (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 n f s ↔ (∀ m : ℕ, m ≤ n → ContinuousOn (iteratedDerivWithin m f s) s) ∧
      ∀ m : ℕ, m < n → DifferentiableOn 𝕜 (iteratedDerivWithin m f s) s := by
  rw [show n = ((n : ℕ∞) : WithTop ℕ∞) from rfl,
    contDiffOn_iff_continuousOn_differentiableOn_deriv hs]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    s : Set 𝕜
    n : Nat
    hs : UniqueDiffOn 𝕜 s
    ⊢ Iff (And (∀ (m : Nat), LE.le ↑m ↑n → ContinuousOn (iteratedDerivWithin m f s …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The `n+1`-th iterated derivative within a set with unique derivatives can be obtained by
differentiating the `n`-th iterated derivative. -/
theorem iteratedDerivWithin_succ {x : 𝕜} (hxs : UniqueDiffWithinAt 𝕜 s x) :
    iteratedDerivWithin (n + 1) f s x = derivWithin (iteratedDerivWithin n f s) s x := by
  rw [iteratedDerivWithin_eq_iteratedFDerivWithin, iteratedFDerivWithin_succ_apply_left,
    iteratedFDerivWithin_eq_equiv_comp, LinearIsometryEquiv.comp_fderivWithin _ hxs, derivWithin]
  change ((ContinuousMultilinearMap.mkPiRing 𝕜 (Fin n) ((fderivWithin 𝕜
    (iteratedDerivWithin n f s) s x : 𝕜 → F) 1) : (Fin n → 𝕜) → F) fun _ : Fin n => 1) =
    (fderivWithin 𝕜 (iteratedDerivWithin n f s) s x : 𝕜 → F) 1
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    s : Set 𝕜
    x : 𝕜
    hxs : UniqueDiffWithinAt 𝕜 s x
    ⊢ Eq ((ContinuousMultilinearMap.mkPiRing 𝕜 (Fin n) ((fderivWithin 𝕜 (iteratedD …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The `n`-th iterated derivative within a set with unique derivatives can be obtained by
iterating `n` times the differentiation operation. -/
theorem iteratedDerivWithin_eq_iterate {x : 𝕜} (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    iteratedDerivWithin n f s x = (fun g : 𝕜 → F => derivWithin g s)^[n] f x := by
  induction n generalizing x with
  | zero => simp
  | succ n IH =>
    rw [iteratedDerivWithin_succ (hs x hx), Function.iterate_succ']
    exact derivWithin_congr (fun y hy => IH hy) (IH hx)


/-- The `n+1`-th iterated derivative within a set with unique derivatives can be obtained by
taking the `n`-th derivative of the derivative. -/
theorem iteratedDerivWithin_succ' {x : 𝕜} (hxs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    iteratedDerivWithin (n + 1) f s x = (iteratedDerivWithin n (derivWithin f s) s) x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    s : Set 𝕜
    x : 𝕜
    hxs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    ⊢ Eq (iteratedDerivWithin (HAdd.hAdd n 1) f s x) (iteratedDerivWithin n (deriv …
  -/
  rw [iteratedDerivWithin_eq_iterate hxs hx, iteratedDerivWithin_eq_iterate hxs hx]; rfl
                                                                                     /-
                                                                                       🎉 no goals
                                                                                     -/


theorem iteratedDeriv_eq_iteratedFDeriv :
    iteratedDeriv n f x = (iteratedFDeriv 𝕜 n f x : (Fin n → 𝕜) → F) fun _ : Fin n => 1 :=
  rfl


/-- Write the iterated derivative as the composition of a continuous linear equiv and the iterated
Fréchet derivative -/
theorem iteratedDeriv_eq_equiv_comp : iteratedDeriv n f =
    (ContinuousMultilinearMap.piFieldEquiv 𝕜 (Fin n) F).symm ∘ iteratedFDeriv 𝕜 n f := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    ⊢ Eq (iteratedDeriv n f) (Function.comp (⇑(ContinuousMultilinearMap.piFieldEqu …
  -/
  ext x; rfl
         /-
           🎉 no goals
         -/


/-- Write the iterated Fréchet derivative as the composition of a continuous linear equiv and the
iterated derivative. -/
theorem iteratedFDeriv_eq_equiv_comp : iteratedFDeriv 𝕜 n f =
    ContinuousMultilinearMap.piFieldEquiv 𝕜 (Fin n) F ∘ iteratedDeriv n f := by
  rw [iteratedDeriv_eq_equiv_comp, ← Function.comp_assoc, LinearIsometryEquiv.self_comp_symm,
    Function.id_comp]


/-- The `n`-th Fréchet derivative applied to a vector `(m 0, ..., m (n-1))` is the derivative
multiplied by the product of the `m i`s. -/
theorem iteratedFDeriv_apply_eq_iteratedDeriv_mul_prod {m : Fin n → 𝕜} :
    (iteratedFDeriv 𝕜 n f x : (Fin n → 𝕜) → F) m = (∏ i, m i) • iteratedDeriv n f x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    x : 𝕜
    m : Fin n → 𝕜
    ⊢ Eq ((iteratedFDeriv 𝕜 n f x) m) (HSMul.hSMul (Finset.univ.prod fun i => m i) …
  -/
  rw [iteratedDeriv_eq_iteratedFDeriv, ← ContinuousMultilinearMap.map_smul_univ]; simp
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


theorem norm_iteratedFDeriv_eq_norm_iteratedDeriv :
    ‖iteratedFDeriv 𝕜 n f x‖ = ‖iteratedDeriv n f x‖ := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    x : 𝕜
    ⊢ Eq (Norm.norm (iteratedFDeriv 𝕜 n f x)) (Norm.norm (iteratedDeriv n f x))
  -/
  rw [iteratedDeriv_eq_equiv_comp, Function.comp_apply, LinearIsometryEquiv.norm_map]
  /-
    🎉 no goals
  -/


@[simp]
                                                         /-
                                                           𝕜 : Type u_1
                                                           inst✝² : NontriviallyNormedField 𝕜
                                                           F : Type u_2
                                                           inst✝¹ : NormedAddCommGroup F
                                                           inst✝ : NormedSpace 𝕜 F
                                                           f : 𝕜 → F
                                                           ⊢ Eq (iteratedDeriv 0 f) f
                                                         -/
theorem iteratedDeriv_zero : iteratedDeriv 0 f = f := by ext x; simp [iteratedDeriv]
                                                                /-
                                                                  🎉 no goals
                                                                -/


@[simp]
                                                              /-
                                                                𝕜 : Type u_1
                                                                inst✝² : NontriviallyNormedField 𝕜
                                                                F : Type u_2
                                                                inst✝¹ : NormedAddCommGroup F
                                                                inst✝ : NormedSpace 𝕜 F
                                                                f : 𝕜 → F
                                                                ⊢ Eq (iteratedDeriv 1 f) (deriv f)
                                                              -/
theorem iteratedDeriv_one : iteratedDeriv 1 f = deriv f := by ext x; simp [iteratedDeriv]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


/-- The property of being `C^n`, initially defined in terms of the Fréchet derivative, can be
reformulated in terms of the one-dimensional derivative. -/
theorem contDiff_iff_iteratedDeriv {n : ℕ∞} : ContDiff 𝕜 n f ↔
    (∀ m : ℕ, (m : ℕ∞) ≤ n → Continuous (iteratedDeriv m f)) ∧
      ∀ m : ℕ, (m : ℕ∞) < n → Differentiable 𝕜 (iteratedDeriv m f) := by
  simp only [contDiff_iff_continuous_differentiable, iteratedFDeriv_eq_equiv_comp,
    LinearIsometryEquiv.comp_continuous_iff, LinearIsometryEquiv.comp_differentiable_iff]


/-- The property of being `C^n`, initially defined in terms of the Fréchet derivative, can be
reformulated in terms of the one-dimensional derivative. -/
theorem contDiff_nat_iff_iteratedDeriv {n : ℕ} : ContDiff 𝕜 n f ↔
    (∀ m : ℕ, m ≤ n → Continuous (iteratedDeriv m f)) ∧
      ∀ m : ℕ, m < n → Differentiable 𝕜 (iteratedDeriv m f) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    n : Nat
    ⊢ Iff (ContDiff 𝕜 (↑n) f) (And (∀ (m : Nat), LE.le m n → Continuous (iteratedD …
  -/
  rw [show n = ((n : ℕ∞) : WithTop ℕ∞) from rfl, contDiff_iff_iteratedDeriv]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : 𝕜 → F
    n : Nat
    ⊢ Iff (And (∀ (m : Nat), LE.le ↑m ↑n → Continuous (iteratedDeriv m f)) (∀ (m : …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- To check that a function is `n` times continuously differentiable, it suffices to check that its
first `n` derivatives are differentiable. This is slightly too strong as the condition we
require on the `n`-th derivative is differentiability instead of continuity, but it has the
advantage of avoiding the discussion of continuity in the proof (and for `n = ∞` this is optimal).
-/
theorem contDiff_of_differentiable_iteratedDeriv {n : ℕ∞}
    (h : ∀ m : ℕ, (m : ℕ∞) ≤ n → Differentiable 𝕜 (iteratedDeriv m f)) : ContDiff 𝕜 n f :=
  contDiff_iff_iteratedDeriv.2 ⟨fun m hm => (h m hm).continuous, fun m hm => h m (le_of_lt hm)⟩


theorem ContDiff.continuous_iteratedDeriv {n : WithTop ℕ∞} (m : ℕ) (h : ContDiff 𝕜 n f)
    (hmn : m ≤ n) : Continuous (iteratedDeriv m f) :=
  (contDiff_iff_iteratedDeriv.1 (h.of_le hmn)).1 m le_rfl


theorem ContDiff.differentiable_iteratedDeriv {n : WithTop ℕ∞} (m : ℕ) (h : ContDiff 𝕜 n f)
    (hmn : m < n) : Differentiable 𝕜 (iteratedDeriv m f) :=
  (contDiff_iff_iteratedDeriv.1 (h.of_le (ENat.add_one_natCast_le_withTop_of_lt hmn))).2 m
    (mod_cast (lt_add_one m))


/-- The `n+1`-th iterated derivative can be obtained by differentiating the `n`-th
iterated derivative. -/
theorem iteratedDeriv_succ : iteratedDeriv (n + 1) f = deriv (iteratedDeriv n f) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    ⊢ Eq (iteratedDeriv (HAdd.hAdd n 1) f) (deriv (iteratedDeriv n f))
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    x : 𝕜
    ⊢ Eq (iteratedDeriv (HAdd.hAdd n 1) f x) (deriv (iteratedDeriv n f) x)
  -/
  rw [← iteratedDerivWithin_univ, ← iteratedDerivWithin_univ, ← derivWithin_univ]
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    x : 𝕜
    ⊢ Eq (iteratedDerivWithin (HAdd.hAdd n 1) f Set.univ x) (derivWithin (iterated …
  -/
  exact iteratedDerivWithin_succ uniqueDiffWithinAt_univ
  /-
    🎉 no goals
  -/


/-- The `n`-th iterated derivative can be obtained by iterating `n` times the
differentiation operation. -/
theorem iteratedDeriv_eq_iterate : iteratedDeriv n f = deriv^[n] f := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    ⊢ Eq (iteratedDeriv n f) (Nat.iterate deriv n f)
  -/
  ext x
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    x : 𝕜
    ⊢ Eq (iteratedDeriv n f x) (Nat.iterate deriv n f x)
  -/
  rw [← iteratedDerivWithin_univ]
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    x : 𝕜
    ⊢ Eq (iteratedDerivWithin n f Set.univ x) (Nat.iterate deriv n f x)
  -/
  convert iteratedDerivWithin_eq_iterate uniqueDiffOn_univ (F := F) (mem_univ x)
  /-
    case h.e'_3.h.e'_1.h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    x : 𝕜
    x✝ : 𝕜 → F
    ⊢ Eq (deriv x✝) (derivWithin x✝ Set.univ)
  -/
  simp [derivWithin_univ]
  /-
    🎉 no goals
  -/


/-- The `n+1`-th iterated derivative can be obtained by taking the `n`-th derivative of the
derivative. -/
theorem iteratedDeriv_succ' : iteratedDeriv (n + 1) f = iteratedDeriv n (deriv f) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    ⊢ Eq (iteratedDeriv (HAdd.hAdd n 1) f) (iteratedDeriv n (deriv f))
  -/
  rw [iteratedDeriv_eq_iterate, iteratedDeriv_eq_iterate]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/

