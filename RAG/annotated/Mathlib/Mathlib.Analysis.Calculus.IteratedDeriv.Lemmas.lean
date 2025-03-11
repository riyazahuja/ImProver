theorem iteratedDerivWithin_congr (hfg : Set.EqOn f g s) :
    Set.EqOn (iteratedDerivWithin n f s) (iteratedDerivWithin n g s) s := by
  induction n generalizing f g with
  | zero => rwa [iteratedDerivWithin_zero]
  | succ n IH =>
    intro y hy
    have : UniqueDiffWithinAt 𝕜 s y := h.uniqueDiffWithinAt hy
    rw [iteratedDerivWithin_succ this, iteratedDerivWithin_succ this]
    exact derivWithin_congr (IH hfg) (IH hfg hy)


theorem iteratedDerivWithin_add (hf : ContDiffOn 𝕜 n f s) (hg : ContDiffOn 𝕜 n g s) :
    iteratedDerivWithin n (f + g) s x =
      iteratedDerivWithin n f s x + iteratedDerivWithin n g s x := by
  simp_rw [iteratedDerivWithin, iteratedFDerivWithin_add_apply hf hg h hx,
    ContinuousMultilinearMap.add_apply]


theorem iteratedDerivWithin_const_add (hn : 0 < n) (c : F) :
    iteratedDerivWithin n (fun z => c + f z) s x = iteratedDerivWithin n f s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    hn : LT.lt 0 n
    c : F
    ⊢ Eq (iteratedDerivWithin n (fun z => HAdd.hAdd c (f z)) s x) (iteratedDerivWi …
  -/
  obtain ⟨n, rfl⟩ := n.exists_eq_succ_of_ne_zero hn.ne'
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : F
    n : Nat
    hn : LT.lt 0 n.succ
    ⊢ Eq (iteratedDerivWithin n.succ (fun z => HAdd.hAdd c (f z)) s x) (iteratedDe …
  -/
  rw [iteratedDerivWithin_succ' h hx, iteratedDerivWithin_succ' h hx]
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : F
    n : Nat
    hn : LT.lt 0 n.succ
    ⊢ Eq (iteratedDerivWithin n (derivWithin (fun z => HAdd.hAdd c (f z)) s) s x)  …
  -/
  refine iteratedDerivWithin_congr h ?_ hx
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : F
    n : Nat
    hn : LT.lt 0 n.succ
    ⊢ Set.EqOn (derivWithin (fun z => HAdd.hAdd c (f z)) s) (derivWithin f s) s
  -/
  intro y hy
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : F
    n : Nat
    hn : LT.lt 0 n.succ
    y : 𝕜
    hy : Membership.mem s y
    ⊢ Eq (derivWithin (fun z => HAdd.hAdd c (f z)) s y) (derivWithin f s y)
  -/
  exact derivWithin_const_add (h.uniqueDiffWithinAt hy) _
  /-
    🎉 no goals
  -/


theorem iteratedDerivWithin_const_sub (hn : 0 < n) (c : F) :
    iteratedDerivWithin n (fun z => c - f z) s x = iteratedDerivWithin n (fun z => -f z) s x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    hn : LT.lt 0 n
    c : F
    ⊢ Eq (iteratedDerivWithin n (fun z => HSub.hSub c (f z)) s x) (iteratedDerivWi …
  -/
  obtain ⟨n, rfl⟩ := n.exists_eq_succ_of_ne_zero hn.ne'
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : F
    n : Nat
    hn : LT.lt 0 n.succ
    ⊢ Eq (iteratedDerivWithin n.succ (fun z => HSub.hSub c (f z)) s x) (iteratedDe …
  -/
  rw [iteratedDerivWithin_succ' h hx, iteratedDerivWithin_succ' h hx]
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : F
    n : Nat
    hn : LT.lt 0 n.succ
    ⊢ Eq (iteratedDerivWithin n (derivWithin (fun z => HSub.hSub c (f z)) s) s x)  …
  -/
  refine iteratedDerivWithin_congr h ?_ hx
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : F
    n : Nat
    hn : LT.lt 0 n.succ
    ⊢ Set.EqOn (derivWithin (fun z => HSub.hSub c (f z)) s) (derivWithin (fun z => …
  -/
  intro y hy
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : F
    n : Nat
    hn : LT.lt 0 n.succ
    y : 𝕜
    hy : Membership.mem s y
    ⊢ Eq (derivWithin (fun z => HSub.hSub c (f z)) s y) (derivWithin (fun z => Neg …
  -/
  have : UniqueDiffWithinAt 𝕜 s y := h.uniqueDiffWithinAt hy
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : F
    n : Nat
    hn : LT.lt 0 n.succ
    y : 𝕜
    hy : Membership.mem s y
    this : UniqueDiffWithinAt 𝕜 s y
    ⊢ Eq (derivWithin (fun z => HSub.hSub c (f z)) s y) (derivWithin (fun z => Neg …
  -/
  rw [derivWithin.neg this]
  /-
    case intro
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : F
    n : Nat
    hn : LT.lt 0 n.succ
    y : 𝕜
    hy : Membership.mem s y
    this : UniqueDiffWithinAt 𝕜 s y
    ⊢ Eq (derivWithin (fun z => HSub.hSub c (f z)) s y) (Neg.neg (derivWithin f s  …
  -/
  exact derivWithin_const_sub this _
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-10")]
alias iteratedDerivWithin_const_neg := iteratedDerivWithin_const_sub


theorem iteratedDerivWithin_const_smul (c : R) (hf : ContDiffOn 𝕜 n f s) :
    iteratedDerivWithin n (c • f) s x = c • iteratedDerivWithin n f s x := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    R : Type u_3
    inst✝³ : Semiring R
    inst✝² : Module R F
    inst✝¹ : SMulCommClass 𝕜 R F
    inst✝ : ContinuousConstSMul R F
    n : Nat
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : R
    hf : ContDiffOn 𝕜 (↑n) f s
    ⊢ Eq (iteratedDerivWithin n (HSMul.hSMul c f) s x) (HSMul.hSMul c (iteratedDer …
  -/
  simp_rw [iteratedDerivWithin]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    R : Type u_3
    inst✝³ : Semiring R
    inst✝² : Module R F
    inst✝¹ : SMulCommClass 𝕜 R F
    inst✝ : ContinuousConstSMul R F
    n : Nat
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : R
    hf : ContDiffOn 𝕜 (↑n) f s
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n (HSMul.hSMul c f) s x) fun x => 1) (HSMul.hSMu …
  -/
  rw [iteratedFDerivWithin_const_smul_apply hf h hx]
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝⁵ : NormedAddCommGroup F
    inst✝⁴ : NormedSpace 𝕜 F
    R : Type u_3
    inst✝³ : Semiring R
    inst✝² : Module R F
    inst✝¹ : SMulCommClass 𝕜 R F
    inst✝ : ContinuousConstSMul R F
    n : Nat
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    f : 𝕜 → F
    c : R
    hf : ContDiffOn 𝕜 (↑n) f s
    ⊢ Eq ((HSMul.hSMul c (iteratedFDerivWithin 𝕜 n f s x)) fun x => 1) (HSMul.hSMu …
  -/
  simp only [ContinuousMultilinearMap.smul_apply]
  /-
    🎉 no goals
  -/


theorem iteratedDerivWithin_const_mul (c : 𝕜) {f : 𝕜 → 𝕜} (hf : ContDiffOn 𝕜 n f s) :
    iteratedDerivWithin n (fun z => c * f z) s x = c * iteratedDerivWithin n f s x := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    n : Nat
    x : 𝕜
    s : Set 𝕜
    hx : Membership.mem s x
    h : UniqueDiffOn 𝕜 s
    c : 𝕜
    f : 𝕜 → 𝕜
    hf : ContDiffOn 𝕜 (↑n) f s
    ⊢ Eq (iteratedDerivWithin n (fun z => HMul.hMul c (f z)) s x) (HMul.hMul c (it …
  -/
  simpa using iteratedDerivWithin_const_smul (F := 𝕜) hx h c hf
  /-
    🎉 no goals
  -/


variable (f) in
theorem iteratedDerivWithin_neg :
    iteratedDerivWithin n (-f) s x = -iteratedDerivWithin n f s x := by
  rw [iteratedDerivWithin, iteratedDerivWithin, iteratedFDerivWithin_neg_apply h hx,
    ContinuousMultilinearMap.neg_apply]


variable (f) in
theorem iteratedDerivWithin_neg' :
    iteratedDerivWithin n (fun z => -f z) s x = -iteratedDerivWithin n f s x :=
  iteratedDerivWithin_neg hx h f


theorem iteratedDerivWithin_sub (hf : ContDiffOn 𝕜 n f s) (hg : ContDiffOn 𝕜 n g s) :
    iteratedDerivWithin n (f - g) s x =
      iteratedDerivWithin n f s x - iteratedDerivWithin n g s x := by
  rw [sub_eq_add_neg, sub_eq_add_neg, Pi.neg_def, iteratedDerivWithin_add hx h hf hg.neg,
    iteratedDerivWithin_neg' hx h]


theorem iteratedDerivWithin_comp_const_smul (hf : ContDiffOn 𝕜 n f s) (c : 𝕜)
    (hs : Set.MapsTo (c * ·) s s) :
    iteratedDerivWithin n (fun x => f (c * x)) s x = c ^ n • iteratedDerivWithin n f s (c * x) := by
  induction n generalizing x with
  | zero => simp
  | succ n ih =>
    have hcx : c * x ∈ s := hs hx
    have h₀ : s.EqOn
        (iteratedDerivWithin n (fun x ↦ f (c * x)) s)
        (fun x => c ^ n • iteratedDerivWithin n f s (c * x)) :=
      fun x hx => ih hx hf.of_succ
    have h₁ : DifferentiableWithinAt 𝕜 (iteratedDerivWithin n f s) s (c * x) :=
      hf.differentiableOn_iteratedDerivWithin (Nat.cast_lt.mpr n.lt_succ_self) h _ hcx
    have h₂ : DifferentiableWithinAt 𝕜 (fun x => iteratedDerivWithin n f s (c * x)) s x := by
      rw [← Function.comp_def]
      apply DifferentiableWithinAt.comp
      · exact hf.differentiableOn_iteratedDerivWithin (Nat.cast_lt.mpr n.lt_succ_self) h _ hcx
      · exact differentiableWithinAt_id'.const_mul _
      · exact hs
    rw [iteratedDerivWithin_succ (h _ hx), derivWithin_congr h₀ (ih hx hf.of_succ),
      derivWithin_const_smul (h _ hx) (c ^ n) h₂, iteratedDerivWithin_succ (h _ hcx),
      ← Function.comp_def,
      derivWithin.scomp x h₁ (differentiableWithinAt_id'.const_mul _) hs (h _ hx),
      derivWithin_const_mul (h _ hx) _ differentiableWithinAt_id', derivWithin_id' _ _ (h _ hx),
      smul_smul, mul_one, pow_succ]


lemma iteratedDeriv_add (hf : ContDiff 𝕜 n f) (hg : ContDiff 𝕜 n g) :
    iteratedDeriv n (f + g) x = iteratedDeriv n f x + iteratedDeriv n g x := by
  simpa only [iteratedDerivWithin_univ] using
    iteratedDerivWithin_add (Set.mem_univ _) uniqueDiffOn_univ
      (contDiffOn_univ.mpr hf) (contDiffOn_univ.mpr hg)


theorem iteratedDeriv_const_add (hn : 0 < n) (c : F) :
    iteratedDeriv n (fun z => c + f z) x = iteratedDeriv n f x := by
  simpa only [iteratedDerivWithin_univ] using
    iteratedDerivWithin_const_add (Set.mem_univ _) uniqueDiffOn_univ hn c


theorem iteratedDeriv_const_sub (hn : 0 < n) (c : F) :
    iteratedDeriv n (fun z => c - f z) x = iteratedDeriv n (-f) x := by
  simpa only [iteratedDerivWithin_univ] using
    iteratedDerivWithin_const_sub (Set.mem_univ _) uniqueDiffOn_univ hn c


lemma iteratedDeriv_neg (n : ℕ) (f : 𝕜 → F) (a : 𝕜) :
    iteratedDeriv n (fun x ↦ -(f x)) a = -(iteratedDeriv n f a) := by
  simpa only [iteratedDerivWithin_univ] using
    iteratedDerivWithin_neg (Set.mem_univ a) uniqueDiffOn_univ f


lemma iteratedDeriv_sub (hf : ContDiff 𝕜 n f) (hg : ContDiff 𝕜 n g) :
    iteratedDeriv n (f - g) x = iteratedDeriv n f x - iteratedDeriv n g x := by
  simpa only [iteratedDerivWithin_univ] using
    iteratedDerivWithin_sub (Set.mem_univ _) uniqueDiffOn_univ
      (contDiffOn_univ.mpr hf) (contDiffOn_univ.mpr hg)


theorem iteratedDeriv_comp_const_smul {n : ℕ} {f : 𝕜 → F} (h : ContDiff 𝕜 n f) (c : 𝕜) :
    iteratedDeriv n (fun x => f (c * x)) = fun x => c ^ n • iteratedDeriv n f (c * x) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    h : ContDiff 𝕜 (↑n) f
    c : 𝕜
    ⊢ Eq (iteratedDeriv n fun x => f (HMul.hMul c x)) fun x => HSMul.hSMul (HPow.h …
  -/
  funext x
  simpa only [iteratedDerivWithin_univ] using
    iteratedDerivWithin_comp_const_smul (Set.mem_univ x) uniqueDiffOn_univ (contDiffOn_univ.mpr h)
      c (Set.mapsTo_univ _ _)


@[deprecated (since := "2024-12-20")]
alias iteratedDeriv_const_smul := iteratedDeriv_comp_const_smul


theorem iteratedDeriv_comp_const_mul {n : ℕ} {f : 𝕜 → 𝕜} (h : ContDiff 𝕜 n f) (c : 𝕜) :
    iteratedDeriv n (fun x => f (c * x)) = fun x => c ^ n * iteratedDeriv n f (c * x) := by
  /-
    𝕜 : Type u_1
    inst✝ : NontriviallyNormedField 𝕜
    n : Nat
    f : 𝕜 → 𝕜
    h : ContDiff 𝕜 (↑n) f
    c : 𝕜
    ⊢ Eq (iteratedDeriv n fun x => f (HMul.hMul c x)) fun x => HMul.hMul (HPow.hPo …
  -/
  simpa only [smul_eq_mul] using iteratedDeriv_comp_const_smul h c
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-12-20")]
alias iteratedDeriv_const_mul := iteratedDeriv_comp_const_mul


lemma iteratedDeriv_comp_neg (n : ℕ) (f : 𝕜 → F) (a : 𝕜) :
    iteratedDeriv n (fun x ↦ f (-x)) a = (-1 : 𝕜) ^ n • iteratedDeriv n f (-a) := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : 𝕜 → F
    a : 𝕜
    ⊢ Eq (iteratedDeriv n (fun x => f (Neg.neg x)) a) (HSMul.hSMul (HPow.hPow (-1) …
  -/
  induction' n with n ih generalizing a
    /-
      case zero
      𝕜 : Type u_1
      inst✝² : NontriviallyNormedField 𝕜
      F : Type u_2
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : 𝕜 → F
      a : 𝕜
      ⊢ Eq (iteratedDeriv 0 (fun x => f (Neg.neg x)) a) (HSMul.hSMul (HPow.hPow (-1) …
    -/
  · simp only [iteratedDeriv_zero, pow_zero, one_smul]
    /-
      🎉 no goals
    -/
  · have ih' : iteratedDeriv n (fun x ↦ f (-x)) = fun x ↦ (-1 : 𝕜) ^ n • iteratedDeriv n f (-x) :=
      funext ih
    rw [iteratedDeriv_succ, iteratedDeriv_succ, ih', pow_succ', neg_mul, one_mul,
      deriv_comp_neg (f := fun x ↦ (-1 : 𝕜) ^ n • iteratedDeriv n f x), deriv_const_smul',
      neg_smul]


open Topology in
lemma Filter.EventuallyEq.iteratedDeriv_eq (n : ℕ) {f g : 𝕜 → F} {x : 𝕜} (hfg : f =ᶠ[𝓝 x] g) :
    iteratedDeriv n f x = iteratedDeriv n g x := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f g : 𝕜 → F
    x : 𝕜
    hfg : (nhds x).EventuallyEq f g
    ⊢ Eq (iteratedDeriv n f x) (iteratedDeriv n g x)
  -/
  simp only [← iteratedDerivWithin_univ, iteratedDerivWithin_eq_iteratedFDerivWithin]
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f g : 𝕜 → F
    x : 𝕜
    hfg : (nhds x).EventuallyEq f g
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f Set.univ x) fun x => 1) ((iteratedFDerivWith …
  -/
  rw [(hfg.filter_mono nhdsWithin_le_nhds).iteratedFDerivWithin_eq hfg.eq_of_nhds n]
  /-
    🎉 no goals
  -/


lemma Set.EqOn.iteratedDeriv_of_isOpen (hfg : Set.EqOn f g s) (hs : IsOpen s) (n : ℕ) :
    Set.EqOn (iteratedDeriv n f) (iteratedDeriv n g) s := by
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set 𝕜
    f g : 𝕜 → F
    hfg : Set.EqOn f g s
    hs : IsOpen s
    n : Nat
    ⊢ Set.EqOn (iteratedDeriv n f) (iteratedDeriv n g) s
  -/
  refine fun x hx ↦ Filter.EventuallyEq.iteratedDeriv_eq n ?_
  /-
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set 𝕜
    f g : 𝕜 → F
    hfg : Set.EqOn f g s
    hs : IsOpen s
    n : Nat
    x : 𝕜
    hx : Membership.mem s x
    ⊢ (nhds x).EventuallyEq f g
  -/
  filter_upwards [IsOpen.mem_nhds hs hx] with a ha
  /-
    case h
    𝕜 : Type u_1
    inst✝² : NontriviallyNormedField 𝕜
    F : Type u_2
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set 𝕜
    f g : 𝕜 → F
    hfg : Set.EqOn f g s
    hs : IsOpen s
    n : Nat
    x : 𝕜
    hx : Membership.mem s x
    a : 𝕜
    ha : Membership.mem s a
    ⊢ Eq (f a) (g a)
  -/
  exact hfg ha
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with shifting the function by a constant on the left. -/
lemma iteratedDeriv_comp_const_add (n : ℕ) (f : 𝕜 → F) (s : 𝕜) :
    iteratedDeriv n (fun z ↦ f (s + z)) = fun t ↦ iteratedDeriv n f (s + t) := by
  induction n with
  | zero => simp only [iteratedDeriv_zero]
  | succ n IH =>
    simpa only [iteratedDeriv_succ, IH] using funext <| deriv_comp_const_add _ s


/-- The iterated derivative commutes with shifting the function by a constant on the right. -/
lemma iteratedDeriv_comp_add_const (n : ℕ) (f : 𝕜 → F) (s : 𝕜) :
    iteratedDeriv n (fun z ↦ f (z + s)) = fun t ↦ iteratedDeriv n f (t + s) := by
  induction n with
  | zero => simp only [iteratedDeriv_zero]
  | succ n IH =>
    simpa only [iteratedDeriv_succ, IH] using funext <| deriv_comp_add_const _ s


