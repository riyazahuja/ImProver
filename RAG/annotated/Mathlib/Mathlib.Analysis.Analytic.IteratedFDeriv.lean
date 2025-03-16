/-- Formal multilinear series associated to the iterated derivative, defined by iterating
`p ↦ p.derivSeries` and currying suitably. It is defined so that, if a function has `p` as a power
series, then its iterated derivative of order `k` has `p.iteratedFDerivSeries k` as a power
series. -/
noncomputable def FormalMultilinearSeries.iteratedFDerivSeries
    (p : FormalMultilinearSeries 𝕜 E F) (k : ℕ) :
    FormalMultilinearSeries 𝕜 E (E [×k]→L[𝕜] F) :=
  match k with
  | 0 => (continuousMultilinearCurryFin0 𝕜 E F).symm
      |>.toContinuousLinearEquiv.toContinuousLinearMap.compFormalMultilinearSeries p
  | (k + 1) => (continuousMultilinearCurryLeftEquiv 𝕜 (fun _ : Fin (k + 1) ↦ E) F).symm
      |>.toContinuousLinearEquiv.toContinuousLinearMap.compFormalMultilinearSeries
      (p.iteratedFDerivSeries k).derivSeries


/-- If a function has a power series on a ball, then so do its iterated derivatives. -/
protected theorem HasFPowerSeriesWithinOnBall.iteratedFDerivWithin
    (h : HasFPowerSeriesWithinOnBall f p s x r) (h' : AnalyticOn 𝕜 f s)
    (k : ℕ) (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    HasFPowerSeriesWithinOnBall (iteratedFDerivWithin 𝕜 k f s)
      (p.iteratedFDerivSeries k) s x r := by
  induction k with
  | zero =>
    exact (continuousMultilinearCurryFin0 𝕜 E F).symm
      |>.toContinuousLinearEquiv.toContinuousLinearMap.comp_hasFPowerSeriesWithinOnBall h
  | succ k ih =>
    rw [iteratedFDerivWithin_succ_eq_comp_left]
    apply (continuousMultilinearCurryLeftEquiv 𝕜 (fun _ : Fin (k + 1) ↦ E) F).symm
      |>.toContinuousLinearEquiv.toContinuousLinearMap.comp_hasFPowerSeriesWithinOnBall
        (ih.fderivWithin_of_mem_of_analyticOn (h'.iteratedFDerivWithin hs _) hs hx)


lemma FormalMultilinearSeries.iteratedFDerivSeries_eq_zero {k n : ℕ}
    (h : p (n + k) = 0) : p.iteratedFDerivSeries k n = 0 := by
  induction k generalizing n with
  | zero =>
    ext
    have : p n = 0 := p.congr_zero rfl h
    simp [FormalMultilinearSeries.iteratedFDerivSeries, this]
  | succ k ih =>
    ext
    simp only [iteratedFDerivSeries, Nat.succ_eq_add_one,
      ContinuousLinearMap.compFormalMultilinearSeries_apply,
      ContinuousLinearMap.compContinuousMultilinearMap_coe, ContinuousLinearEquiv.coe_coe,
      LinearIsometryEquiv.coe_toContinuousLinearEquiv, Function.comp_apply,
      continuousMultilinearCurryLeftEquiv_symm_apply, ContinuousMultilinearMap.zero_apply,
      ContinuousLinearMap.zero_apply,
      derivSeries_eq_zero _ (ih (p.congr_zero (Nat.succ_add_eq_add_succ _ _).symm h))]


/-- If the `n`-th term in a power series is zero, then the `n`-th derivative of the corresponding
function vanishes. -/
lemma HasFPowerSeriesWithinOnBall.iteratedFDerivWithin_eq_zero
    (h : HasFPowerSeriesWithinOnBall f p s x r) (h' : AnalyticOn 𝕜 f s)
    (hu : UniqueDiffOn 𝕜 s) (hx : x ∈ s) {n : ℕ} (hn : p n = 0) :
    iteratedFDerivWithin 𝕜 n f s x = 0 := by
  have : iteratedFDerivWithin 𝕜 n f s x = p.iteratedFDerivSeries n 0 (fun _ ↦ 0) :=
    ((h.iteratedFDerivWithin h' n hu hx).coeff_zero _).symm
  rw [this, p.iteratedFDerivSeries_eq_zero (p.congr_zero (Nat.zero_add n).symm hn),
    ContinuousMultilinearMap.zero_apply]


lemma ContinuousMultilinearMap.iteratedFDeriv_comp_diagonal
    {n : ℕ} (f : E [×n]→L[𝕜] F) (x : E) (v : Fin n → E) :
    iteratedFDeriv 𝕜 n (fun x ↦ f (fun _ ↦ x)) x v = ∑ σ : Perm (Fin n), f (fun i ↦ v (σ i)) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => E) F
    x : E
    v : Fin n → E
    ⊢ Eq ((iteratedFDeriv 𝕜 n (fun x => f fun x_1 => x) x) v) (Finset.univ.sum fun …
  -/
  rw [← sum_comp (Equiv.inv (Perm (Fin n)))]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => E) F
    x : E
    v : Fin n → E
    ⊢ Eq ((iteratedFDeriv 𝕜 n (fun x => f fun x_1 => x) x) v) (Finset.univ.sum fun …
  -/
  let g : E →L[𝕜] (Fin n → E) := ContinuousLinearMap.pi (fun i ↦ ContinuousLinearMap.id 𝕜 E)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => E) F
    x : E
    v : Fin n → E
    g : ContinuousLinearMap (RingHom.id 𝕜) E (Fin n → E) := ContinuousLinearMap.pi …
    ⊢ Eq ((iteratedFDeriv 𝕜 n (fun x => f fun x_1 => x) x) v) (Finset.univ.sum fun …
  -/
  change iteratedFDeriv 𝕜 n (f ∘ g) x v = _
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => E) F
    x : E
    v : Fin n → E
    g : ContinuousLinearMap (RingHom.id 𝕜) E (Fin n → E) := ContinuousLinearMap.pi …
    ⊢ Eq ((iteratedFDeriv 𝕜 n (Function.comp ⇑f ⇑g) x) v) (Finset.univ.sum fun i = …
  -/
  rw [ContinuousLinearMap.iteratedFDeriv_comp_right _ f.contDiff _ le_rfl, f.iteratedFDeriv_eq]
  simp only [ContinuousMultilinearMap.iteratedFDeriv,
    ContinuousMultilinearMap.compContinuousLinearMap_apply, ContinuousMultilinearMap.sum_apply,
    ContinuousMultilinearMap.iteratedFDerivComponent_apply, Set.mem_range, Pi.compRightL_apply]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => E) F
    x : E
    v : Fin n → E
    g : ContinuousLinearMap (RingHom.id 𝕜) E (Fin n → E) := ContinuousLinearMap.pi …
    ⊢ Eq (Finset.univ.sum fun x_1 => f fun j => dite (Exists fun y => Eq (x_1 y) j …
  -/
  rw [← sum_comp (Equiv.embeddingEquivOfFinite (Fin n))]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => E) F
    x : E
    v : Fin n → E
    g : ContinuousLinearMap (RingHom.id 𝕜) E (Fin n → E) := ContinuousLinearMap.pi …
    ⊢ Eq (Finset.univ.sum fun x_1 => f fun j => dite (Exists fun y => Eq (x_1 y) j …
  -/
  congr with σ
  /-
    case e_f.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => E) F
    x : E
    v : Fin n → E
    g : ContinuousLinearMap (RingHom.id 𝕜) E (Fin n → E) := ContinuousLinearMap.pi …
    σ : Function.Embedding (Fin n) (Fin n)
    ⊢ Eq (f fun j => dite (Exists fun y => Eq (σ y) j) (fun h => g (v (σ.toEquivRa …
  -/
  congr with i
  have A : ∃ y, σ y = i := by
    have : Function.Bijective σ := (Fintype.bijective_iff_injective_and_card _).2 ⟨σ.injective, rfl⟩
    exact this.surjective i
  /-
    case e_f.h.h.e_6.h.h
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => E) F
    x : E
    v : Fin n → E
    g : ContinuousLinearMap (RingHom.id 𝕜) E (Fin n → E) := ContinuousLinearMap.pi …
    σ : Function.Embedding (Fin n) (Fin n)
    i : Fin n
    A : Exists fun y => Eq (σ y) i
    ⊢ Eq (dite (Exists fun y => Eq (σ y) i) (fun h => g (v (σ.toEquivRange.symm ⟨i …
  -/
  rcases A with ⟨y, rfl⟩
  simp only [EmbeddingLike.apply_eq_iff_eq, exists_eq, ↓reduceDIte,
    Function.Embedding.toEquivRange_symm_apply_self, ContinuousLinearMap.coe_pi',
    ContinuousLinearMap.coe_id', id_eq, g]
  /-
    case e_f.h.h.e_6.h.h.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => E) F
    x : E
    v : Fin n → E
    g : ContinuousLinearMap (RingHom.id 𝕜) E (Fin n → E) := ContinuousLinearMap.pi …
    σ : Function.Embedding (Fin n) (Fin n)
    y : Fin n
    ⊢ Eq (v y) (v (((Equiv.inv (Equiv.Perm (Fin n))) ((Equiv.embeddingEquivOfFinit …
  -/
  congr 1
  /-
    case e_f.h.h.e_6.h.h.intro.e_a
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    n : Nat
    f : ContinuousMultilinearMap 𝕜 (fun i => E) F
    x : E
    v : Fin n → E
    g : ContinuousLinearMap (RingHom.id 𝕜) E (Fin n → E) := ContinuousLinearMap.pi …
    σ : Function.Embedding (Fin n) (Fin n)
    y : Fin n
    ⊢ Eq y (((Equiv.inv (Equiv.Perm (Fin n))) ((Equiv.embeddingEquivOfFinite (Fin  …
  -/
  symm
  simp [coe_fn_mk, inv_apply, Perm.inv_def,
    ofBijective_symm_apply_apply, Function.Embedding.equivOfFiniteSelfEmbedding]


private lemma HasFPowerSeriesWithinOnBall.iteratedFDerivWithin_eq_sum_of_subset
    (h : HasFPowerSeriesWithinOnBall f p s x r) (h' : AnalyticOn 𝕜 f s)
    (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s)
    {n : ℕ} (v : Fin n → E) (h's : s ⊆ EMetric.ball x r) :
    iteratedFDerivWithin 𝕜 n f s x v = ∑ σ : Perm (Fin n), p n (fun i ↦ v (σ i)) := by
  have I : insert x s ∩ EMetric.ball x r = s := by
    rw [Set.insert_eq_of_mem hx]
    exact Set.inter_eq_left.2 h's
  have fcont : ContDiffOn 𝕜 (↑n) f s := by
    apply AnalyticOn.contDiffOn _ hs
    simpa [I] using h'
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    h : HasFPowerSeriesWithinOnBall f p s x r
    h' : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    h's : HasSubset.Subset s (EMetric.ball x r)
    I : Eq (Inter.inter (Insert.insert x s) (EMetric.ball x r)) s
    fcont : ContDiffOn 𝕜 (↑n) f s
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) v) (Finset.univ.sum fun σ => (p n) fun  …
  -/
  let g : E → F := fun z ↦ p n (fun _ ↦ z - x)
  have gcont : ContDiff 𝕜 ω g := by
    apply (p n).contDiff.comp
    exact contDiff_pi.2 (fun i ↦ contDiff_id.sub contDiff_const)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    h : HasFPowerSeriesWithinOnBall f p s x r
    h' : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    h's : HasSubset.Subset s (EMetric.ball x r)
    I : Eq (Inter.inter (Insert.insert x s) (EMetric.ball x r)) s
    fcont : ContDiffOn 𝕜 (↑n) f s
    g : E → F := fun z => (p n) fun x_1 => HSub.hSub z x
    gcont : ContDiff 𝕜 Top.top g
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) v) (Finset.univ.sum fun σ => (p n) fun  …
  -/
  let q : FormalMultilinearSeries 𝕜 E F := fun k ↦ if h : n = k then (h ▸ p n) else 0
  have A : HasFiniteFPowerSeriesOnBall g q x (n + 1) r := by
    apply HasFiniteFPowerSeriesOnBall.mk' _ h.r_pos
    · intro y hy
      rw [Finset.sum_eq_single_of_mem n]
      · simp [q, g]
      · simp
      · intro i hi h'i
        simp [q, h'i.symm]
    · intro m hm
      have : n ≠ m := by omega
      simp [q, this]
  have B : HasFPowerSeriesWithinOnBall g q s x r :=
    A.toHasFPowerSeriesOnBall.hasFPowerSeriesWithinOnBall
  have J1 : iteratedFDerivWithin 𝕜 n f s x =
      iteratedFDerivWithin 𝕜 n g s x + iteratedFDerivWithin 𝕜 n (f - g) s x := by
    have : f = g + (f - g) := by abel
    nth_rewrite 1 [this]
    rw [iteratedFDerivWithin_add_apply (gcont.of_le le_top).contDiffOn
      (by exact fcont.sub (gcont.of_le le_top).contDiffOn) hs hx]
  have J2 : iteratedFDerivWithin 𝕜 n (f - g) s x = 0 := by
    apply (h.sub B).iteratedFDerivWithin_eq_zero (h'.sub ?_) hs hx
    · simp [q]
    · apply gcont.contDiffOn.analyticOn
  have J3 : iteratedFDerivWithin 𝕜 n g s x = iteratedFDeriv 𝕜 n g x :=
    iteratedFDerivWithin_eq_iteratedFDeriv hs (gcont.of_le le_top).contDiffAt hx
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    h : HasFPowerSeriesWithinOnBall f p s x r
    h' : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    h's : HasSubset.Subset s (EMetric.ball x r)
    I : Eq (Inter.inter (Insert.insert x s) (EMetric.ball x r)) s
    fcont : ContDiffOn 𝕜 (↑n) f s
    g : E → F := fun z => (p n) fun x_1 => HSub.hSub z x
    gcont : ContDiff 𝕜 Top.top g
    q : FormalMultilinearSeries 𝕜 E F := fun k => dite (Eq n k) (fun h => Eq.rec ( …
    A : HasFiniteFPowerSeriesOnBall g q x (HAdd.hAdd n 1) r
    B : HasFPowerSeriesWithinOnBall g q s x r
    J1 : Eq (iteratedFDerivWithin 𝕜 n f s x) (HAdd.hAdd (iteratedFDerivWithin 𝕜 n  …
    J2 : Eq (iteratedFDerivWithin 𝕜 n (HSub.hSub f g) s x) 0
    J3 : Eq (iteratedFDerivWithin 𝕜 n g s x) (iteratedFDeriv 𝕜 n g x)
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) v) (Finset.univ.sum fun σ => (p n) fun  …
  -/
  simp only [J1, J3, J2, add_zero]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    h : HasFPowerSeriesWithinOnBall f p s x r
    h' : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    h's : HasSubset.Subset s (EMetric.ball x r)
    I : Eq (Inter.inter (Insert.insert x s) (EMetric.ball x r)) s
    fcont : ContDiffOn 𝕜 (↑n) f s
    g : E → F := fun z => (p n) fun x_1 => HSub.hSub z x
    gcont : ContDiff 𝕜 Top.top g
    q : FormalMultilinearSeries 𝕜 E F := fun k => dite (Eq n k) (fun h => Eq.rec ( …
    A : HasFiniteFPowerSeriesOnBall g q x (HAdd.hAdd n 1) r
    B : HasFPowerSeriesWithinOnBall g q s x r
    J1 : Eq (iteratedFDerivWithin 𝕜 n f s x) (HAdd.hAdd (iteratedFDerivWithin 𝕜 n  …
    J2 : Eq (iteratedFDerivWithin 𝕜 n (HSub.hSub f g) s x) 0
    J3 : Eq (iteratedFDerivWithin 𝕜 n g s x) (iteratedFDeriv 𝕜 n g x)
    ⊢ Eq ((iteratedFDeriv 𝕜 n g x) v) (Finset.univ.sum fun σ => (p n) fun i => v ( …
  -/
  let g' : E → F := fun z ↦ p n (fun _ ↦ z)
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    h : HasFPowerSeriesWithinOnBall f p s x r
    h' : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    h's : HasSubset.Subset s (EMetric.ball x r)
    I : Eq (Inter.inter (Insert.insert x s) (EMetric.ball x r)) s
    fcont : ContDiffOn 𝕜 (↑n) f s
    g : E → F := fun z => (p n) fun x_1 => HSub.hSub z x
    gcont : ContDiff 𝕜 Top.top g
    q : FormalMultilinearSeries 𝕜 E F := fun k => dite (Eq n k) (fun h => Eq.rec ( …
    A : HasFiniteFPowerSeriesOnBall g q x (HAdd.hAdd n 1) r
    B : HasFPowerSeriesWithinOnBall g q s x r
    J1 : Eq (iteratedFDerivWithin 𝕜 n f s x) (HAdd.hAdd (iteratedFDerivWithin 𝕜 n  …
    J2 : Eq (iteratedFDerivWithin 𝕜 n (HSub.hSub f g) s x) 0
    J3 : Eq (iteratedFDerivWithin 𝕜 n g s x) (iteratedFDeriv 𝕜 n g x)
    g' : E → F := fun z => (p n) fun x => z
    ⊢ Eq ((iteratedFDeriv 𝕜 n g x) v) (Finset.univ.sum fun σ => (p n) fun i => v ( …
  -/
  have : g = fun z ↦ g' (z - x) := rfl
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    h : HasFPowerSeriesWithinOnBall f p s x r
    h' : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    h's : HasSubset.Subset s (EMetric.ball x r)
    I : Eq (Inter.inter (Insert.insert x s) (EMetric.ball x r)) s
    fcont : ContDiffOn 𝕜 (↑n) f s
    g : E → F := fun z => (p n) fun x_1 => HSub.hSub z x
    gcont : ContDiff 𝕜 Top.top g
    q : FormalMultilinearSeries 𝕜 E F := fun k => dite (Eq n k) (fun h => Eq.rec ( …
    A : HasFiniteFPowerSeriesOnBall g q x (HAdd.hAdd n 1) r
    B : HasFPowerSeriesWithinOnBall g q s x r
    J1 : Eq (iteratedFDerivWithin 𝕜 n f s x) (HAdd.hAdd (iteratedFDerivWithin 𝕜 n  …
    J2 : Eq (iteratedFDerivWithin 𝕜 n (HSub.hSub f g) s x) 0
    J3 : Eq (iteratedFDerivWithin 𝕜 n g s x) (iteratedFDeriv 𝕜 n g x)
    g' : E → F := fun z => (p n) fun x => z
    this : Eq g fun z => g' (HSub.hSub z x)
    ⊢ Eq ((iteratedFDeriv 𝕜 n g x) v) (Finset.univ.sum fun σ => (p n) fun i => v ( …
  -/
  rw [this, iteratedFDeriv_comp_sub]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    h : HasFPowerSeriesWithinOnBall f p s x r
    h' : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    h's : HasSubset.Subset s (EMetric.ball x r)
    I : Eq (Inter.inter (Insert.insert x s) (EMetric.ball x r)) s
    fcont : ContDiffOn 𝕜 (↑n) f s
    g : E → F := fun z => (p n) fun x_1 => HSub.hSub z x
    gcont : ContDiff 𝕜 Top.top g
    q : FormalMultilinearSeries 𝕜 E F := fun k => dite (Eq n k) (fun h => Eq.rec ( …
    A : HasFiniteFPowerSeriesOnBall g q x (HAdd.hAdd n 1) r
    B : HasFPowerSeriesWithinOnBall g q s x r
    J1 : Eq (iteratedFDerivWithin 𝕜 n f s x) (HAdd.hAdd (iteratedFDerivWithin 𝕜 n  …
    J2 : Eq (iteratedFDerivWithin 𝕜 n (HSub.hSub f g) s x) 0
    J3 : Eq (iteratedFDerivWithin 𝕜 n g s x) (iteratedFDeriv 𝕜 n g x)
    g' : E → F := fun z => (p n) fun x => z
    this : Eq g fun z => g' (HSub.hSub z x)
    ⊢ Eq ((iteratedFDeriv 𝕜 n g' (HSub.hSub x x)) v) (Finset.univ.sum fun σ => (p  …
  -/
  exact (p n).iteratedFDeriv_comp_diagonal _ v
  /-
    🎉 no goals
  -/


/-- If a function has a power series in a ball, then its `n`-th iterated derivative is given by
`(v₁, ..., vₙ) ↦ ∑ pₙ (v_{σ (1)}, ..., v_{σ (n)})` where the sum is over all
permutations of `{1, ..., n}`.-/
theorem HasFPowerSeriesWithinOnBall.iteratedFDerivWithin_eq_sum
    (h : HasFPowerSeriesWithinOnBall f p s x r) (h' : AnalyticOn 𝕜 f s)
    (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) {n : ℕ} (v : Fin n → E) :
    iteratedFDerivWithin 𝕜 n f s x v = ∑ σ : Perm (Fin n), p n (fun i ↦ v (σ i)) := by
  have : iteratedFDerivWithin 𝕜 n f s x
      = iteratedFDerivWithin 𝕜 n f (s ∩ EMetric.ball x r) x :=
    (iteratedFDerivWithin_inter_open EMetric.isOpen_ball (EMetric.mem_ball_self h.r_pos)).symm
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    h : HasFPowerSeriesWithinOnBall f p s x r
    h' : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) v) (Finset.univ.sum fun σ => (p n) fun  …
  -/
  rw [this]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    h : HasFPowerSeriesWithinOnBall f p s x r
    h' : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f (Inter.inter s (EMetric.ball x r)) x) v) (Fi …
  -/
  apply HasFPowerSeriesWithinOnBall.iteratedFDerivWithin_eq_sum_of_subset
    /-
      case h
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      h : HasFPowerSeriesWithinOnBall f p s x r
      h' : AnalyticOn 𝕜 f s
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ HasFPowerSeriesWithinOnBall f p (Inter.inter s (EMetric.ball x r)) x ?r
    -/
  · exact h.mono inter_subset_left
    /-
      🎉 no goals
    -/
    /-
      case h'
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      h : HasFPowerSeriesWithinOnBall f p s x r
      h' : AnalyticOn 𝕜 f s
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ AnalyticOn 𝕜 f (Inter.inter s (EMetric.ball x r))
    -/
  · exact h'.mono inter_subset_left
    /-
      🎉 no goals
    -/
    /-
      case hs
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      h : HasFPowerSeriesWithinOnBall f p s x r
      h' : AnalyticOn 𝕜 f s
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ UniqueDiffOn 𝕜 (Inter.inter s (EMetric.ball x r))
    -/
  · exact hs.inter EMetric.isOpen_ball
    /-
      🎉 no goals
    -/
    /-
      case hx
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      h : HasFPowerSeriesWithinOnBall f p s x r
      h' : AnalyticOn 𝕜 f s
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ Membership.mem (Inter.inter s (EMetric.ball x r)) x
    -/
  · exact ⟨hx, EMetric.mem_ball_self h.r_pos⟩
    /-
      🎉 no goals
    -/
    /-
      case h's
      𝕜 : Type u_1
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type u_3
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      h : HasFPowerSeriesWithinOnBall f p s x r
      h' : AnalyticOn 𝕜 f s
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ HasSubset.Subset (Inter.inter s (EMetric.ball x r)) (EMetric.ball x r)
    -/
  · exact inter_subset_right
    /-
      🎉 no goals
    -/


/-- If a function has a power series in a ball, then its `n`-th iterated derivative is given by
`(v₁, ..., vₙ) ↦ ∑ pₙ (v_{σ (1)}, ..., v_{σ (n)})` where the sum is over all
permutations of `{1, ..., n}`.-/
theorem HasFPowerSeriesOnBall.iteratedFDeriv_eq_sum
    (h : HasFPowerSeriesOnBall f p x r) (h' : AnalyticOn 𝕜 f univ) {n : ℕ} (v : Fin n → E) :
    iteratedFDeriv 𝕜 n f x v = ∑ σ : Perm (Fin n), p n (fun i ↦ v (σ i)) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    h : HasFPowerSeriesOnBall f p x r
    h' : AnalyticOn 𝕜 f Set.univ
    n : Nat
    v : Fin n → E
    ⊢ Eq ((iteratedFDeriv 𝕜 n f x) v) (Finset.univ.sum fun σ => (p n) fun i => v ( …
  -/
  simp only [← iteratedFDerivWithin_univ, ← hasFPowerSeriesWithinOnBall_univ] at h ⊢
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    h' : AnalyticOn 𝕜 f Set.univ
    n : Nat
    v : Fin n → E
    h : HasFPowerSeriesWithinOnBall f p Set.univ x r
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f Set.univ x) v) (Finset.univ.sum fun σ => (p  …
  -/
  exact h.iteratedFDerivWithin_eq_sum h' uniqueDiffOn_univ (mem_univ x) v
  /-
    🎉 no goals
  -/


/-- If a function has a power series in a ball, then its `n`-th iterated derivative is given by
`(v₁, ..., vₙ) ↦ ∑ pₙ (v_{σ (1)}, ..., v_{σ (n)})` where the sum is over all
permutations of `{1, ..., n}`.-/
theorem HasFPowerSeriesWithinOnBall.iteratedFDerivWithin_eq_sum_of_completeSpace [CompleteSpace F]
    (h : HasFPowerSeriesWithinOnBall f p s x r)
    (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) {n : ℕ} (v : Fin n → E) :
    iteratedFDerivWithin 𝕜 n f s x v = ∑ σ : Perm (Fin n), p n (fun i ↦ v (σ i)) := by
  have : iteratedFDerivWithin 𝕜 n f s x
      = iteratedFDerivWithin 𝕜 n f (s ∩ EMetric.ball x r) x :=
    (iteratedFDerivWithin_inter_open EMetric.isOpen_ball (EMetric.mem_ball_self h.r_pos)).symm
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesWithinOnBall f p s x r
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) v) (Finset.univ.sum fun σ => (p n) fun  …
  -/
  rw [this]
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    s : Set E
    x : E
    r : ENNReal
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesWithinOnBall f p s x r
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f (Inter.inter s (EMetric.ball x r)) x) v) (Fi …
  -/
  apply HasFPowerSeriesWithinOnBall.iteratedFDerivWithin_eq_sum_of_subset
    /-
      case h
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ HasFPowerSeriesWithinOnBall f p (Inter.inter s (EMetric.ball x r)) x ?r
    -/
  · exact h.mono inter_subset_left
    /-
      🎉 no goals
    -/
    /-
      case h'
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ AnalyticOn 𝕜 f (Inter.inter s (EMetric.ball x r))
    -/
  · apply h.analyticOn.mono
    /-
      case h'
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ HasSubset.Subset (Inter.inter s (EMetric.ball x r)) (Inter.inter (Insert.ins …
    -/
    rw [insert_eq_of_mem hx]
    /-
      🎉 no goals
    -/
    /-
      case hs
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ UniqueDiffOn 𝕜 (Inter.inter s (EMetric.ball x r))
    -/
  · exact hs.inter EMetric.isOpen_ball
    /-
      🎉 no goals
    -/
    /-
      case hx
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ Membership.mem (Inter.inter s (EMetric.ball x r)) x
    -/
  · exact ⟨hx, EMetric.mem_ball_self h.r_pos⟩
    /-
      🎉 no goals
    -/
    /-
      case h's
      𝕜 : Type u_1
      inst✝⁵ : NontriviallyNormedField 𝕜
      E : Type u_2
      inst✝⁴ : NormedAddCommGroup E
      inst✝³ : NormedSpace 𝕜 E
      F : Type u_3
      inst✝² : NormedAddCommGroup F
      inst✝¹ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      s : Set E
      x : E
      r : ENNReal
      inst✝ : CompleteSpace F
      h : HasFPowerSeriesWithinOnBall f p s x r
      hs : UniqueDiffOn 𝕜 s
      hx : Membership.mem s x
      n : Nat
      v : Fin n → E
      this : Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f (Inter. …
      ⊢ HasSubset.Subset (Inter.inter s (EMetric.ball x r)) (EMetric.ball x r)
    -/
  · exact inter_subset_right
    /-
      🎉 no goals
    -/


/-- If a function has a power series in a ball, then its `n`-th iterated derivative is given by
`(v₁, ..., vₙ) ↦ ∑ pₙ (v_{σ (1)}, ..., v_{σ (n)})` where the sum is over all
permutations of `{1, ..., n}`.-/
theorem HasFPowerSeriesOnBall.iteratedFDeriv_eq_sum_of_completeSpace [CompleteSpace F]
    (h : HasFPowerSeriesOnBall f p x r) {n : ℕ} (v : Fin n → E) :
    iteratedFDeriv 𝕜 n f x v = ∑ σ : Perm (Fin n), p n (fun i ↦ v (σ i)) := by
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    inst✝ : CompleteSpace F
    h : HasFPowerSeriesOnBall f p x r
    n : Nat
    v : Fin n → E
    ⊢ Eq ((iteratedFDeriv 𝕜 n f x) v) (Finset.univ.sum fun σ => (p n) fun i => v ( …
  -/
  simp only [← iteratedFDerivWithin_univ, ← hasFPowerSeriesWithinOnBall_univ] at h ⊢
  /-
    𝕜 : Type u_1
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    inst✝ : CompleteSpace F
    n : Nat
    v : Fin n → E
    h : HasFPowerSeriesWithinOnBall f p Set.univ x r
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f Set.univ x) v) (Finset.univ.sum fun σ => (p  …
  -/
  exact h.iteratedFDerivWithin_eq_sum_of_completeSpace uniqueDiffOn_univ (mem_univ _) v
  /-
    🎉 no goals
  -/


/-- The `n`-th iterated derivative of an analytic function on a set is symmetric. -/
theorem AnalyticOn.iteratedFDerivWithin_comp_perm
    (h : AnalyticOn 𝕜 f s) (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) {n : ℕ} (v : Fin n → E)
    (σ : Perm (Fin n)) :
    iteratedFDerivWithin 𝕜 n f s x (v ∘ σ) = iteratedFDerivWithin 𝕜 n f s x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    x : E
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) (Function.comp v ⇑σ)) ((iteratedFDerivW …
  -/
  rcases h x hx with ⟨p, r, hp⟩
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    x : E
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hp : HasFPowerSeriesWithinOnBall f p s x r
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) (Function.comp v ⇑σ)) ((iteratedFDerivW …
  -/
  rw [hp.iteratedFDerivWithin_eq_sum h hs hx, hp.iteratedFDerivWithin_eq_sum h hs hx]
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    x : E
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hp : HasFPowerSeriesWithinOnBall f p s x r
    ⊢ Eq (Finset.univ.sum fun σ_1 => (p n) fun i => Function.comp v (⇑σ) (σ_1 i))  …
  -/
  conv_rhs => rw [← Equiv.sum_comp (Equiv.mulLeft σ)]
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    x : E
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    hp : HasFPowerSeriesWithinOnBall f p s x r
    ⊢ Eq (Finset.univ.sum fun σ_1 => (p n) fun i => Function.comp v (⇑σ) (σ_1 i))  …
  -/
  simp only [coe_mulLeft, Perm.coe_mul, Function.comp_apply]
  /-
    🎉 no goals
  -/


/-- The `n`-th iterated derivative of an analytic function on a set is symmetric. -/
theorem ContDiffWithinAt.iteratedFDerivWithin_comp_perm
    (h : ContDiffWithinAt 𝕜 ω f s x) (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) {n : ℕ} (v : Fin n → E)
    (σ : Perm (Fin n)) :
    iteratedFDerivWithin 𝕜 n f s x (v ∘ σ) = iteratedFDerivWithin 𝕜 n f s x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    x : E
    h : ContDiffWithinAt 𝕜 Top.top f s x
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) (Function.comp v ⇑σ)) ((iteratedFDerivW …
  -/
  rcases h.contDiffOn' le_rfl (by simp) with ⟨u, u_open, xu, hu⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    x : E
    h : ContDiffWithinAt 𝕜 Top.top f s x
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 Top.top f (Inter.inter (Insert.insert x s) u)
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) (Function.comp v ⇑σ)) ((iteratedFDerivW …
  -/
  rw [insert_eq_of_mem hx] at hu
  have : iteratedFDerivWithin 𝕜 n f (s ∩ u) x = iteratedFDerivWithin 𝕜 n f s x :=
    iteratedFDerivWithin_inter_open u_open xu
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    x : E
    h : ContDiffWithinAt 𝕜 Top.top f s x
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 Top.top f (Inter.inter s u)
    this : Eq (iteratedFDerivWithin 𝕜 n f (Inter.inter s u) x) (iteratedFDerivWith …
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f s x) (Function.comp v ⇑σ)) ((iteratedFDerivW …
  -/
  rw [← this]
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    s : Set E
    x : E
    h : ContDiffWithinAt 𝕜 Top.top f s x
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 Top.top f (Inter.inter s u)
    this : Eq (iteratedFDerivWithin 𝕜 n f (Inter.inter s u) x) (iteratedFDerivWith …
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f (Inter.inter s u) x) (Function.comp v ⇑σ)) ( …
  -/
  exact AnalyticOn.iteratedFDerivWithin_comp_perm hu.analyticOn (hs.inter u_open) ⟨hx, xu⟩ _ _
  /-
    🎉 no goals
  -/


/-- The `n`-th iterated derivative of an analytic function is symmetric. -/
theorem AnalyticOn.iteratedFDeriv_comp_perm
    (h : AnalyticOn 𝕜 f univ) {n : ℕ} (v : Fin n → E) (σ : Perm (Fin n)) :
    iteratedFDeriv 𝕜 n f x (v ∘ σ) = iteratedFDeriv 𝕜 n f x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : AnalyticOn 𝕜 f Set.univ
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    ⊢ Eq ((iteratedFDeriv 𝕜 n f x) (Function.comp v ⇑σ)) ((iteratedFDeriv 𝕜 n f x) …
  -/
  rw [← iteratedFDerivWithin_univ]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : AnalyticOn 𝕜 f Set.univ
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f Set.univ x) (Function.comp v ⇑σ)) ((iterated …
  -/
  exact h.iteratedFDerivWithin_comp_perm uniqueDiffOn_univ (mem_univ x) _ _
  /-
    🎉 no goals
  -/


/-- The `n`-th iterated derivative of an analytic function is symmetric. -/
theorem ContDiffAt.iteratedFDeriv_comp_perm
    (h : ContDiffAt 𝕜 ω f x) {n : ℕ} (v : Fin n → E) (σ : Perm (Fin n)) :
    iteratedFDeriv 𝕜 n f x (v ∘ σ) = iteratedFDeriv 𝕜 n f x v := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : ContDiffAt 𝕜 Top.top f x
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    ⊢ Eq ((iteratedFDeriv 𝕜 n f x) (Function.comp v ⇑σ)) ((iteratedFDeriv 𝕜 n f x) …
  -/
  rw [← iteratedFDerivWithin_univ]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    h : ContDiffAt 𝕜 Top.top f x
    n : Nat
    v : Fin n → E
    σ : Equiv.Perm (Fin n)
    ⊢ Eq ((iteratedFDerivWithin 𝕜 n f Set.univ x) (Function.comp v ⇑σ)) ((iterated …
  -/
  exact h.iteratedFDerivWithin_comp_perm uniqueDiffOn_univ (mem_univ x) _ _
  /-
    🎉 no goals
  -/

