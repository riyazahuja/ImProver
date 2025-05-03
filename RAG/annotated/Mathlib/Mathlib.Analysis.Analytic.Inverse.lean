/-- The left inverse of a formal multilinear series, where the `n`-th term is defined inductively
in terms of the previous ones to make sure that `(leftInv p i) ∘ p = id`. For this, the linear term
`p₁` in `p` should be invertible. In the definition, `i` is a linear isomorphism that should
coincide with `p₁`, so that one can use its inverse in the construction. The definition does not
use that `i = p₁`, but proofs that the definition is well-behaved do.

The `n`-th term in `q ∘ p` is `∑ qₖ (p_{j₁}, ..., p_{jₖ})` over `j₁ + ... + jₖ = n`. In this
expression, `qₙ` appears only once, in `qₙ (p₁, ..., p₁)`. We adjust the definition so that this
term compensates the rest of the sum, using `i⁻¹` as an inverse to `p₁`.

These formulas only make sense when the constant term `p₀` vanishes. The definition we give is
general, but it ignores the value of `p₀`.
-/
noncomputable def leftInv (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E) :
    FormalMultilinearSeries 𝕜 F E
  | 0 => ContinuousMultilinearMap.uncurry0 𝕜 _ x
  | 1 => (continuousMultilinearCurryFin1 𝕜 F E).symm i.symm
  | n + 2 =>
    -∑ c : { c : Composition (n + 2) // c.length < n + 2 },
        (leftInv p i x (c : Composition (n + 2)).length).compAlongComposition
          (p.compContinuousLinearMap i.symm) c


@[simp]
theorem leftInv_coeff_zero (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E) :
                                                                    /-
                                                                      𝕜 : Type u_1
                                                                      inst✝⁴ : NontriviallyNormedField 𝕜
                                                                      E : Type u_2
                                                                      inst✝³ : NormedAddCommGroup E
                                                                      inst✝² : NormedSpace 𝕜 E
                                                                      F : Type u_3
                                                                      inst✝¹ : NormedAddCommGroup F
                                                                      inst✝ : NormedSpace 𝕜 F
                                                                      p : FormalMultilinearSeries 𝕜 E F
                                                                      i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
                                                                      x : E
                                                                      ⊢ Eq (p.leftInv i x 0) (ContinuousMultilinearMap.uncurry0 𝕜 F x)
                                                                    -/
    p.leftInv i x 0 = ContinuousMultilinearMap.uncurry0 𝕜 _ x := by rw [leftInv]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


@[simp]
theorem leftInv_coeff_one (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E) :
                                                                               /-
                                                                                 𝕜 : Type u_1
                                                                                 inst✝⁴ : NontriviallyNormedField 𝕜
                                                                                 E : Type u_2
                                                                                 inst✝³ : NormedAddCommGroup E
                                                                                 inst✝² : NormedSpace 𝕜 E
                                                                                 F : Type u_3
                                                                                 inst✝¹ : NormedAddCommGroup F
                                                                                 inst✝ : NormedSpace 𝕜 F
                                                                                 p : FormalMultilinearSeries 𝕜 E F
                                                                                 i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
                                                                                 x : E
                                                                                 ⊢ Eq (p.leftInv i x 1) ((continuousMultilinearCurryFin1 𝕜 F E).symm ↑i.symm)
                                                                               -/
    p.leftInv i x 1 = (continuousMultilinearCurryFin1 𝕜 F E).symm i.symm := by rw [leftInv]
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- The left inverse does not depend on the zeroth coefficient of a formal multilinear
series. -/
theorem leftInv_removeZero (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E) :
    p.removeZero.leftInv i x = p.leftInv i x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    ⊢ Eq (p.removeZero.leftInv i x) (p.leftInv i x)
  -/
  ext1 n
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
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    n : Nat
    ⊢ Eq (p.removeZero.leftInv i x n) (p.leftInv i x n)
  -/
  induction' n using Nat.strongRec' with n IH
  match n with
  | 0 => simp -- if one replaces `simp` with `refl`, the proof times out in the kernel.
  | 1 => simp -- TODO: why?
  | n + 2 =>
    simp only [leftInv, neg_inj]
    refine Finset.sum_congr rfl fun c cuniv => ?_
    rcases c with ⟨c, hc⟩
    ext v
    dsimp
    simp [IH _ hc]


/-- The left inverse to a formal multilinear series is indeed a left inverse, provided its linear
term is invertible. -/
theorem leftInv_comp (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E)
    (h : p 1 = (continuousMultilinearCurryFin1 𝕜 E F).symm i) :
    (leftInv p i x).comp p = id 𝕜 E x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    h : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E F).symm ↑i)
    ⊢ Eq ((p.leftInv i x).comp p) (FormalMultilinearSeries.id 𝕜 E x)
  -/
  ext n v
  classical
  match n with
  | 0 =>
    simp only [comp_coeff_zero', leftInv_coeff_zero, ContinuousMultilinearMap.uncurry0_apply,
      id_apply_zero]
  | 1 =>
    simp only [leftInv_coeff_one, comp_coeff_one, h, id_apply_one, ContinuousLinearEquiv.coe_apply,
      ContinuousLinearEquiv.symm_apply_apply, continuousMultilinearCurryFin1_symm_apply]
  | n + 2 =>
    have A :
      (Finset.univ : Finset (Composition (n + 2))) =
        {c | Composition.length c < n + 2}.toFinset ∪ {Composition.ones (n + 2)} := by
      refine Subset.antisymm (fun c _ => ?_) (subset_univ _)
      by_cases h : c.length < n + 2
      · simp [h, Set.mem_toFinset (s := {c | Composition.length c < n + 2})]
      · simp [Composition.eq_ones_iff_le_length.2 (not_lt.1 h)]
    have B :
      Disjoint ({c | Composition.length c < n + 2} : Set (Composition (n + 2))).toFinset
        {Composition.ones (n + 2)} := by
      simp [Set.mem_toFinset (s := {c | Composition.length c < n + 2})]
    have C :
      ((p.leftInv i x (Composition.ones (n + 2)).length)
          fun j : Fin (Composition.ones n.succ.succ).length =>
          p 1 fun _ => v ((Fin.castLE (Composition.length_le _)) j)) =
        p.leftInv i x (n + 2) fun j : Fin (n + 2) => p 1 fun _ => v j := by
      apply FormalMultilinearSeries.congr _ (Composition.ones_length _) fun j hj1 hj2 => ?_
      exact FormalMultilinearSeries.congr _ rfl fun k _ _ => by congr
    have D :
      (p.leftInv i x (n + 2) fun j : Fin (n + 2) => p 1 fun _ => v j) =
        -∑ c ∈ {c : Composition (n + 2) | c.length < n + 2}.toFinset,
            (p.leftInv i x c.length) (p.applyComposition c v) := by
      simp only [leftInv, ContinuousMultilinearMap.neg_apply, neg_inj,
        ContinuousMultilinearMap.sum_apply]
      convert
        (sum_toFinset_eq_subtype
          (fun c : Composition (n + 2) => c.length < n + 2)
          (fun c : Composition (n + 2) =>
          (ContinuousMultilinearMap.compAlongComposition
            (p.compContinuousLinearMap (i.symm : F →L[𝕜] E)) c (p.leftInv i x c.length))
            fun j : Fin (n + 2) => p 1 fun _ : Fin 1 => v j)).symm.trans
          _
      simp only [compContinuousLinearMap_applyComposition,
        ContinuousMultilinearMap.compAlongComposition_apply]
      congr
      ext c
      congr
      ext k
      simp [h, Function.comp_def]
    simp [FormalMultilinearSeries.comp, show n + 2 ≠ 1 by omega, A, Finset.sum_union B,
      applyComposition_ones, C, D, -Set.toFinset_setOf]


/-- The right inverse of a formal multilinear series, where the `n`-th term is defined inductively
in terms of the previous ones to make sure that `p ∘ (rightInv p i) = id`. For this, the linear
term `p₁` in `p` should be invertible. In the definition, `i` is a linear isomorphism that should
coincide with `p₁`, so that one can use its inverse in the construction. The definition does not
use that `i = p₁`, but proofs that the definition is well-behaved do.

The `n`-th term in `p ∘ q` is `∑ pₖ (q_{j₁}, ..., q_{jₖ})` over `j₁ + ... + jₖ = n`. In this
expression, `qₙ` appears only once, in `p₁ (qₙ)`. We adjust the definition of `qₙ` so that this
term compensates the rest of the sum, using `i⁻¹` as an inverse to `p₁`.

These formulas only make sense when the constant term `p₀` vanishes. The definition we give is
general, but it ignores the value of `p₀`.
-/
noncomputable def rightInv (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E) :
    FormalMultilinearSeries 𝕜 F E
  | 0 => ContinuousMultilinearMap.uncurry0 𝕜 _ x
  | 1 => (continuousMultilinearCurryFin1 𝕜 F E).symm i.symm
  | n + 2 =>
    let q : FormalMultilinearSeries 𝕜 F E := fun k => if k < n + 2 then rightInv p i x k else 0;
    -(i.symm : F →L[𝕜] E).compContinuousMultilinearMap ((p.comp q) (n + 2))


@[simp]
theorem rightInv_coeff_zero (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E) :
                                                                     /-
                                                                       𝕜 : Type u_1
                                                                       inst✝⁴ : NontriviallyNormedField 𝕜
                                                                       E : Type u_2
                                                                       inst✝³ : NormedAddCommGroup E
                                                                       inst✝² : NormedSpace 𝕜 E
                                                                       F : Type u_3
                                                                       inst✝¹ : NormedAddCommGroup F
                                                                       inst✝ : NormedSpace 𝕜 F
                                                                       p : FormalMultilinearSeries 𝕜 E F
                                                                       i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
                                                                       x : E
                                                                       ⊢ Eq (p.rightInv i x 0) (ContinuousMultilinearMap.uncurry0 𝕜 F x)
                                                                     -/
    p.rightInv i x 0 = ContinuousMultilinearMap.uncurry0 𝕜 _ x := by rw [rightInv]
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


@[simp]
theorem rightInv_coeff_one (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E) :
                                                                                /-
                                                                                  𝕜 : Type u_1
                                                                                  inst✝⁴ : NontriviallyNormedField 𝕜
                                                                                  E : Type u_2
                                                                                  inst✝³ : NormedAddCommGroup E
                                                                                  inst✝² : NormedSpace 𝕜 E
                                                                                  F : Type u_3
                                                                                  inst✝¹ : NormedAddCommGroup F
                                                                                  inst✝ : NormedSpace 𝕜 F
                                                                                  p : FormalMultilinearSeries 𝕜 E F
                                                                                  i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
                                                                                  x : E
                                                                                  ⊢ Eq (p.rightInv i x 1) ((continuousMultilinearCurryFin1 𝕜 F E).symm ↑i.symm)
                                                                                -/
    p.rightInv i x 1 = (continuousMultilinearCurryFin1 𝕜 F E).symm i.symm := by rw [rightInv]
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


/-- The right inverse does not depend on the zeroth coefficient of a formal multilinear
series. -/
theorem rightInv_removeZero (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E) :
    p.removeZero.rightInv i x = p.rightInv i x := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    ⊢ Eq (p.removeZero.rightInv i x) (p.rightInv i x)
  -/
  ext1 n
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
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    n : Nat
    ⊢ Eq (p.removeZero.rightInv i x n) (p.rightInv i x n)
  -/
  induction' n using Nat.strongRec' with n IH
  match n with
  | 0 => simp only [rightInv_coeff_zero]
  | 1 => simp only [rightInv_coeff_one]
  | n + 2 =>
    simp only [rightInv, neg_inj]
    rw [removeZero_comp_of_pos _ _ (add_pos_of_nonneg_of_pos n.zero_le zero_lt_two)]
    congr (config := { closePost := false }) 2 with k
    by_cases hk : k < n + 2 <;> simp [hk, IH]


theorem comp_rightInv_aux1 {n : ℕ} (hn : 0 < n) (p : FormalMultilinearSeries 𝕜 E F)
    (q : FormalMultilinearSeries 𝕜 F E) (v : Fin n → F) :
    p.comp q n v =
      ∑ c ∈ {c : Composition n | 1 < c.length}.toFinset,
          p c.length (q.applyComposition c v) + p 1 fun _ => q n v := by
  classical
  have A :
    (Finset.univ : Finset (Composition n)) =
      {c | 1 < Composition.length c}.toFinset ∪ {Composition.single n hn} := by
    refine Subset.antisymm (fun c _ => ?_) (subset_univ _)
    by_cases h : 1 < c.length
    · simp [h, Set.mem_toFinset (s := {c | 1 < Composition.length c})]
    · have : c.length = 1 := by
        refine (eq_iff_le_not_lt.2 ⟨?_, h⟩).symm; exact c.length_pos_of_pos hn
      rw [← Composition.eq_single_iff_length hn] at this
      simp [this]
  have B :
    Disjoint ({c | 1 < Composition.length c} : Set (Composition n)).toFinset
      {Composition.single n hn} := by
    simp [Set.mem_toFinset (s := {c | 1 < Composition.length c})]
  have C :
    p (Composition.single n hn).length (q.applyComposition (Composition.single n hn) v) =
      p 1 fun _ : Fin 1 => q n v := by
    apply p.congr (Composition.single_length hn) fun j hj1 _ => ?_
    simp [applyComposition_single]
  simp [FormalMultilinearSeries.comp, A, Finset.sum_union B, C, -Set.toFinset_setOf,
    -add_right_inj, -Composition.single_length]


theorem comp_rightInv_aux2 (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E) (n : ℕ)
    (v : Fin (n + 2) → F) :
    ∑ c ∈ {c : Composition (n + 2) | 1 < c.length}.toFinset,
        p c.length (applyComposition (fun k : ℕ => ite (k < n + 2) (p.rightInv i x k) 0) c v) =
      ∑ c ∈ {c : Composition (n + 2) | 1 < c.length}.toFinset,
        p c.length ((p.rightInv i x).applyComposition c v) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    n : Nat
    v : Fin (HAdd.hAdd n 2) → F
    ⊢ Eq ((setOf fun c => LT.lt 1 c.length).toFinset.sum fun c => (p c.length) (Fo …
  -/
  have N : 0 < n + 2 := by norm_num
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    n : Nat
    v : Fin (HAdd.hAdd n 2) → F
    N : LT.lt 0 (HAdd.hAdd n 2)
    ⊢ Eq ((setOf fun c => LT.lt 1 c.length).toFinset.sum fun c => (p c.length) (Fo …
  -/
  refine sum_congr rfl fun c hc => p.congr rfl fun j hj1 hj2 => ?_
  have : ∀ k, c.blocksFun k < n + 2 := by
    simp only [Set.mem_toFinset (s := {c : Composition (n + 2) | 1 < c.length}),
      Set.mem_setOf_eq] at hc
    simp [← Composition.ne_single_iff N, Composition.eq_single_iff_length, ne_of_gt hc]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    n : Nat
    v : Fin (HAdd.hAdd n 2) → F
    N : LT.lt 0 (HAdd.hAdd n 2)
    c : Composition (HAdd.hAdd n 2)
    hc : Membership.mem (setOf fun c => LT.lt 1 c.length).toFinset c
    j : Nat
    hj1 hj2 : LT.lt j c.length
    this : ∀ (k : Fin c.length), LT.lt (c.blocksFun k) (HAdd.hAdd n 2)
    ⊢ Eq (FormalMultilinearSeries.applyComposition (fun k => ite (LT.lt k (HAdd.hA …
  -/
  simp [applyComposition, this]
  /-
    🎉 no goals
  -/


/-- The right inverse to a formal multilinear series is indeed a right inverse, provided its linear
term is invertible and its constant term vanishes. -/
theorem comp_rightInv (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E)
    (h : p 1 = (continuousMultilinearCurryFin1 𝕜 E F).symm i) :
    p.comp (rightInv p i x) = id 𝕜 F (p 0 0) := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    h : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E F).symm ↑i)
    ⊢ Eq (p.comp (p.rightInv i x)) (FormalMultilinearSeries.id 𝕜 F ((p 0) 0))
  -/
  ext (n v)
  match n with
  | 0 =>
    simp only [comp_coeff_zero', Matrix.zero_empty, id_apply_zero]
    congr
    ext i
    exact i.elim0
  | 1 =>
    simp only [comp_coeff_one, h, rightInv_coeff_one, ContinuousLinearEquiv.apply_symm_apply,
      id_apply_one, ContinuousLinearEquiv.coe_apply, continuousMultilinearCurryFin1_symm_apply]
  | n + 2 =>
    have N : 0 < n + 2 := by norm_num
    simp [comp_rightInv_aux1 N, h, rightInv, lt_irrefl n, show n + 2 ≠ 1 by omega,
      ← sub_eq_add_neg, sub_eq_zero, comp_rightInv_aux2, -Set.toFinset_setOf]


theorem rightInv_coeff (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E)
    (n : ℕ) (hn : 2 ≤ n) :
    p.rightInv i x n =
      -(i.symm : F →L[𝕜] E).compContinuousMultilinearMap
          (∑ c ∈ ({c | 1 < Composition.length c}.toFinset : Finset (Composition n)),
            p.compAlongComposition (p.rightInv i x) c) := by
  match n with
  | 0 => exact False.elim (zero_lt_two.not_le hn)
  | 1 => exact False.elim (one_lt_two.not_le hn)
  | n + 2 =>
    simp only [rightInv, neg_inj]
    congr (config := { closePost := false }) 1
    ext v
    have N : 0 < n + 2 := by norm_num
    have : ((p 1) fun _ : Fin 1 => 0) = 0 := ContinuousMultilinearMap.map_zero _
    simp [comp_rightInv_aux1 N, lt_irrefl n, this, comp_rightInv_aux2, -Set.toFinset_setOf]


theorem leftInv_eq_rightInv (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) (x : E)
    (h : p 1 = (continuousMultilinearCurryFin1 𝕜 E F).symm i) :
    leftInv p i x = rightInv p i x :=
  calc
                                                                /-
                                                                  𝕜 : Type u_1
                                                                  inst✝⁴ : NontriviallyNormedField 𝕜
                                                                  E : Type u_2
                                                                  inst✝³ : NormedAddCommGroup E
                                                                  inst✝² : NormedSpace 𝕜 E
                                                                  F : Type u_3
                                                                  inst✝¹ : NormedAddCommGroup F
                                                                  inst✝ : NormedSpace 𝕜 F
                                                                  p : FormalMultilinearSeries 𝕜 E F
                                                                  i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
                                                                  x : E
                                                                  h : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E F).symm ↑i)
                                                                  ⊢ Eq (p.leftInv i x) ((p.leftInv i x).comp (FormalMultilinearSeries.id 𝕜 F ((p …
                                                                -/
    leftInv p i x = (leftInv p i x).comp (id 𝕜 F (p 0 0)) := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/
                                                             /-
                                                               𝕜 : Type u_1
                                                               inst✝⁴ : NontriviallyNormedField 𝕜
                                                               E : Type u_2
                                                               inst✝³ : NormedAddCommGroup E
                                                               inst✝² : NormedSpace 𝕜 E
                                                               F : Type u_3
                                                               inst✝¹ : NormedAddCommGroup F
                                                               inst✝ : NormedSpace 𝕜 F
                                                               p : FormalMultilinearSeries 𝕜 E F
                                                               i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
                                                               x : E
                                                               h : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E F).symm ↑i)
                                                               ⊢ Eq ((p.leftInv i x).comp (FormalMultilinearSeries.id 𝕜 F ((p 0) 0))) ((p.lef …
                                                             -/
    _ = (leftInv p i x).comp (p.comp (rightInv p i x)) := by rw [comp_rightInv p i _ h]
                                                             /-
                                                               🎉 no goals
                                                             -/
                                                             /-
                                                               𝕜 : Type u_1
                                                               inst✝⁴ : NontriviallyNormedField 𝕜
                                                               E : Type u_2
                                                               inst✝³ : NormedAddCommGroup E
                                                               inst✝² : NormedSpace 𝕜 E
                                                               F : Type u_3
                                                               inst✝¹ : NormedAddCommGroup F
                                                               inst✝ : NormedSpace 𝕜 F
                                                               p : FormalMultilinearSeries 𝕜 E F
                                                               i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
                                                               x : E
                                                               h : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E F).symm ↑i)
                                                               ⊢ Eq ((p.leftInv i x).comp (p.comp (p.rightInv i x))) (((p.leftInv i x).comp p …
                                                             -/
    _ = ((leftInv p i x).comp p).comp (rightInv p i x) := by rw [comp_assoc]
                                                             /-
                                                               🎉 no goals
                                                             -/
                                               /-
                                                 𝕜 : Type u_1
                                                 inst✝⁴ : NontriviallyNormedField 𝕜
                                                 E : Type u_2
                                                 inst✝³ : NormedAddCommGroup E
                                                 inst✝² : NormedSpace 𝕜 E
                                                 F : Type u_3
                                                 inst✝¹ : NormedAddCommGroup F
                                                 inst✝ : NormedSpace 𝕜 F
                                                 p : FormalMultilinearSeries 𝕜 E F
                                                 i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
                                                 x : E
                                                 h : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E F).symm ↑i)
                                                 ⊢ Eq (((p.leftInv i x).comp p).comp (p.rightInv i x)) ((FormalMultilinearSerie …
                                               -/
    _ = (id 𝕜 E x).comp (rightInv p i x) := by rw [leftInv_comp p i _ h]
                                               /-
                                                 🎉 no goals
                                               -/
                             /-
                               𝕜 : Type u_1
                               inst✝⁴ : NontriviallyNormedField 𝕜
                               E : Type u_2
                               inst✝³ : NormedAddCommGroup E
                               inst✝² : NormedSpace 𝕜 E
                               F : Type u_3
                               inst✝¹ : NormedAddCommGroup F
                               inst✝ : NormedSpace 𝕜 F
                               p : FormalMultilinearSeries 𝕜 E F
                               i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
                               x : E
                               h : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E F).symm ↑i)
                               ⊢ Eq ((FormalMultilinearSeries.id 𝕜 E x).comp (p.rightInv i x)) (p.rightInv i x)
                             -/
    _ = rightInv p i x := by simp [id_comp' _ _ 0]
                             /-
                               🎉 no goals
                             -/


/-- First technical lemma to control the growth of coefficients of the inverse. Bound the explicit
expression for `∑_{k<n+1} aᵏ Qₖ` in terms of a sum of powers of the same sum one step before,
in a general abstract setup. -/
theorem radius_right_inv_pos_of_radius_pos_aux1 (n : ℕ) (p : ℕ → ℝ) (hp : ∀ k, 0 ≤ p k) {r a : ℝ}
    (hr : 0 ≤ r) (ha : 0 ≤ a) :
    ∑ k ∈ Ico 2 (n + 1),
        a ^ k *
          ∑ c ∈ ({c | 1 < Composition.length c}.toFinset : Finset (Composition k)),
            r ^ c.length * ∏ j, p (c.blocksFun j) ≤
      ∑ j ∈ Ico 2 (n + 1), r ^ j * (∑ k ∈ Ico 1 n, a ^ k * p k) ^ j :=
  calc
    ∑ k ∈ Ico 2 (n + 1),
          a ^ k *
            ∑ c ∈ ({c | 1 < Composition.length c}.toFinset : Finset (Composition k)),
              r ^ c.length * ∏ j, p (c.blocksFun j) =
        ∑ k ∈ Ico 2 (n + 1),
          ∑ c ∈ ({c | 1 < Composition.length c}.toFinset : Finset (Composition k)),
            ∏ j, r * (a ^ c.blocksFun j * p (c.blocksFun j)) := by
      /-
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        ⊢ Eq ((Finset.Ico 2 (HAdd.hAdd n 1)).sum fun k => HMul.hMul (HPow.hPow a k) (( …
      -/
      simp_rw [mul_sum]
      /-
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        ⊢ Eq ((Finset.Ico 2 (HAdd.hAdd n 1)).sum fun x => (setOf fun c => LT.lt 1 c.le …
      -/
      congr! with k _ c
      rw [prod_mul_distrib, prod_mul_distrib, prod_pow_eq_pow_sum, Composition.sum_blocksFun,
        prod_const, card_fin]
      /-
        case a.a
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        k : Nat
        a✝¹ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) k
        c : Composition k
        a✝ : Membership.mem (setOf fun c => LT.lt 1 c.length).toFinset c
        ⊢ Eq (HMul.hMul (HPow.hPow a k) (HMul.hMul (HPow.hPow r c.length) (Finset.univ …
      -/
      ring
      /-
        🎉 no goals
      -/
    _ ≤
        ∑ d ∈ compPartialSumTarget 2 (n + 1) n,
          ∏ j : Fin d.2.length, r * (a ^ d.2.blocksFun j * p (d.2.blocksFun j)) := by
      /-
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        ⊢ LE.le ((Finset.Ico 2 (HAdd.hAdd n 1)).sum fun k => (setOf fun c => LT.lt 1 c …
      -/
      rw [sum_sigma']
      refine
        sum_le_sum_of_subset_of_nonneg ?_ fun x _ _ =>
          prod_nonneg fun j _ => mul_nonneg hr (mul_nonneg (pow_nonneg ha _) (hp _))
      /-
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        ⊢ HasSubset.Subset ((Finset.Ico 2 (HAdd.hAdd n 1)).sigma fun k => (setOf fun c …
      -/
      rintro ⟨k, c⟩ hd
      simp only [Set.mem_toFinset (s := {c | 1 < Composition.length c}), mem_Ico, mem_sigma,
        Set.mem_setOf_eq] at hd
      /-
        case mk
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        k : Nat
        c : Composition k
        hd : And (And (LE.le 2 k) (LT.lt k (HAdd.hAdd n 1))) (LT.lt 1 c.length)
        ⊢ Membership.mem (FormalMultilinearSeries.compPartialSumTarget 2 (HAdd.hAdd n  …
      -/
      simp only [mem_compPartialSumTarget_iff]
      /-
        case mk
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        k : Nat
        c : Composition k
        hd : And (And (LE.le 2 k) (LT.lt k (HAdd.hAdd n 1))) (LT.lt 1 c.length)
        ⊢ And (LE.le 2 c.length) (And (LT.lt c.length (HAdd.hAdd n 1)) (∀ (j : Fin c.l …
      -/
      refine ⟨hd.2, c.length_le.trans_lt hd.1.2, fun j => ?_⟩
      have : c ≠ Composition.single k (zero_lt_two.trans_le hd.1.1) := by
        simp [Composition.eq_single_iff_length, ne_of_gt hd.2]
      /-
        case mk
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        k : Nat
        c : Composition k
        hd : And (And (LE.le 2 k) (LT.lt k (HAdd.hAdd n 1))) (LT.lt 1 c.length)
        j : Fin c.length
        this : Ne c (Composition.single k ⋯)
        ⊢ LT.lt (c.blocksFun j) n
      -/
      rw [Composition.ne_single_iff] at this
      /-
        case mk
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        k : Nat
        c : Composition k
        hd : And (And (LE.le 2 k) (LT.lt k (HAdd.hAdd n 1))) (LT.lt 1 c.length)
        j : Fin c.length
        this : ∀ (i : Fin c.length), LT.lt (c.blocksFun i) k
        ⊢ LT.lt (c.blocksFun j) n
      -/
      exact (this j).trans_le (Nat.lt_succ_iff.mp hd.1.2)
      /-
        🎉 no goals
      -/
    _ = ∑ e ∈ compPartialSumSource 2 (n + 1) n, ∏ j : Fin e.1, r * (a ^ e.2 j * p (e.2 j)) := by
      /-
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        ⊢ Eq ((FormalMultilinearSeries.compPartialSumTarget 2 (HAdd.hAdd n 1) n).sum f …
      -/
      symm
      /-
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        ⊢ Eq ((FormalMultilinearSeries.compPartialSumSource 2 (HAdd.hAdd n 1) n).sum f …
      -/
      apply compChangeOfVariables_sum
      /-
        case h
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        ⊢ ∀ (e : Sigma fun n => Fin n → Nat) (he : Membership.mem (FormalMultilinearSe …
      -/
      rintro ⟨k, blocksFun⟩ H
      /-
        case h.mk
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        k : Nat
        blocksFun : Fin k → Nat
        H : Membership.mem (FormalMultilinearSeries.compPartialSumSource 2 (HAdd.hAdd  …
        ⊢ Eq (Finset.univ.prod fun j => HMul.hMul r (HMul.hMul (HPow.hPow a (⟨k, block …
      -/
      have K : (compChangeOfVariables 2 (n + 1) n ⟨k, blocksFun⟩ H).snd.length = k := by simp
      /-
        case h.mk
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        k : Nat
        blocksFun : Fin k → Nat
        H : Membership.mem (FormalMultilinearSeries.compPartialSumSource 2 (HAdd.hAdd  …
        K : Eq (FormalMultilinearSeries.compChangeOfVariables 2 (HAdd.hAdd n 1) n ⟨k,  …
        ⊢ Eq (Finset.univ.prod fun j => HMul.hMul r (HMul.hMul (HPow.hPow a (⟨k, block …
      -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
                  /-
                    🎉 no goals
                  -/
      congr 2 <;> try rw [K]
      /-
        case h.mk.h.e_5
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        k : Nat
        blocksFun : Fin k → Nat
        H : Membership.mem (FormalMultilinearSeries.compPartialSumSource 2 (HAdd.hAdd  …
        K : Eq (FormalMultilinearSeries.compChangeOfVariables 2 (HAdd.hAdd n 1) n ⟨k,  …
        ⊢ HEq (fun j => HMul.hMul r (HMul.hMul (HPow.hPow a (⟨k, blocksFun⟩.snd j)) (p …
      -/
      rw [Fin.heq_fun_iff K.symm]
      /-
        case h.mk.h.e_5
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        k : Nat
        blocksFun : Fin k → Nat
        H : Membership.mem (FormalMultilinearSeries.compPartialSumSource 2 (HAdd.hAdd  …
        K : Eq (FormalMultilinearSeries.compChangeOfVariables 2 (HAdd.hAdd n 1) n ⟨k,  …
        ⊢ ∀ (i : Fin k), Eq (HMul.hMul r (HMul.hMul (HPow.hPow a (⟨k, blocksFun⟩.snd i …
      -/
      intro j
      /-
        case h.mk.h.e_5
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        k : Nat
        blocksFun : Fin k → Nat
        H : Membership.mem (FormalMultilinearSeries.compPartialSumSource 2 (HAdd.hAdd  …
        K : Eq (FormalMultilinearSeries.compChangeOfVariables 2 (HAdd.hAdd n 1) n ⟨k,  …
        j : Fin k
        ⊢ Eq (HMul.hMul r (HMul.hMul (HPow.hPow a (⟨k, blocksFun⟩.snd j)) (p (⟨k, bloc …
      -/
      rw [compChangeOfVariables_blocksFun]
      /-
        🎉 no goals
      -/
    _ = ∑ j ∈ Ico 2 (n + 1), r ^ j * (∑ k ∈ Ico 1 n, a ^ k * p k) ^ j := by
      rw [compPartialSumSource,
        ← sum_sigma' (Ico 2 (n + 1))
          (fun k : ℕ => (Fintype.piFinset fun _ : Fin k => Ico 1 n : Finset (Fin k → ℕ)))
          (fun n e => ∏ j : Fin n, r * (a ^ e j * p (e j)))]
      /-
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        ⊢ Eq ((Finset.Ico 2 (HAdd.hAdd n 1)).sum fun a_1 => (Fintype.piFinset fun x => …
      -/
      congr! with j
      /-
        case a
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        j : Nat
        a✝ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        ⊢ Eq ((Fintype.piFinset fun x => Finset.Ico 1 n).sum fun s => Finset.univ.prod …
      -/
      simp only [← @MultilinearMap.mkPiAlgebra_apply ℝ (Fin j) _ ℝ]
      simp only [←
        MultilinearMap.map_sum_finset (MultilinearMap.mkPiAlgebra ℝ (Fin j) ℝ) fun _ (m : ℕ) =>
          r * (a ^ m * p m)]
      /-
        case a
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        j : Nat
        a✝ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        ⊢ Eq ((MultilinearMap.mkPiAlgebra Real (Fin j) Real) fun i => (Finset.Ico 1 n) …
      -/
      simp only [MultilinearMap.mkPiAlgebra_apply]
      /-
        case a
        n : Nat
        p : Nat → Real
        hp : ∀ (k : Nat), LE.le 0 (p k)
        r a : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        j : Nat
        a✝ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        ⊢ Eq (Finset.univ.prod fun i => (Finset.Ico 1 n).sum fun j => HMul.hMul r (HMu …
      -/
      simp [prod_const, ← mul_sum, mul_pow]
      /-
        🎉 no goals
      -/


/-- Second technical lemma to control the growth of coefficients of the inverse. Bound the explicit
expression for `∑_{k<n+1} aᵏ Qₖ` in terms of a sum of powers of the same sum one step before,
in the specific setup we are interesting in, by reducing to the general bound in
`radius_rightInv_pos_of_radius_pos_aux1`. -/
theorem radius_rightInv_pos_of_radius_pos_aux2 {x : E} {n : ℕ} (hn : 2 ≤ n + 1)
    (p : FormalMultilinearSeries 𝕜 E F) (i : E ≃L[𝕜] F) {r a C : ℝ} (hr : 0 ≤ r) (ha : 0 ≤ a)
    (hC : 0 ≤ C) (hp : ∀ n, ‖p n‖ ≤ C * r ^ n) :
    ∑ k ∈ Ico 1 (n + 1), a ^ k * ‖p.rightInv i x k‖ ≤
      ‖(i.symm : F →L[𝕜] E)‖ * a +
        ‖(i.symm : F →L[𝕜] E)‖ * C *
          ∑ k ∈ Ico 2 (n + 1), (r * ∑ j ∈ Ico 1 n, a ^ j * ‖p.rightInv i x j‖) ^ k :=
  let I := ‖(i.symm : F →L[𝕜] E)‖
  calc
    ∑ k ∈ Ico 1 (n + 1), a ^ k * ‖p.rightInv i x k‖ =
        a * I + ∑ k ∈ Ico 2 (n + 1), a ^ k * ‖p.rightInv i x k‖ := by
      simp only [I, LinearIsometryEquiv.norm_map, pow_one, rightInv_coeff_one,
        show Ico (1 : ℕ) 2 = {1} from Nat.Ico_succ_singleton 1,
        sum_singleton, ← sum_Ico_consecutive _ one_le_two hn]
    _ =
        a * I +
          ∑ k ∈ Ico 2 (n + 1),
            a ^ k *
              ‖(i.symm : F →L[𝕜] E).compContinuousMultilinearMap
                  (∑ c ∈ ({c | 1 < Composition.length c}.toFinset : Finset (Composition k)),
                    p.compAlongComposition (p.rightInv i x) c)‖ := by
      /-
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        ⊢ Eq (HAdd.hAdd (HMul.hMul a I) ((Finset.Ico 2 (HAdd.hAdd n 1)).sum fun k => H …
      -/
      congr! 2 with j hj
      /-
        case h.e'_6.a
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        j : Nat
        hj : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        ⊢ Eq (HMul.hMul (HPow.hPow a j) (Norm.norm (p.rightInv i x j))) (HMul.hMul (HP …
      -/
      rw [rightInv_coeff _ _ _ _ (mem_Ico.1 hj).1, norm_neg]
      /-
        🎉 no goals
      -/
    _ ≤
        a * ‖(i.symm : F →L[𝕜] E)‖ +
          ∑ k ∈ Ico 2 (n + 1),
            a ^ k *
              (I *
                ∑ c ∈ ({c | 1 < Composition.length c}.toFinset : Finset (Composition k)),
                  C * r ^ c.length * ∏ j, ‖p.rightInv i x (c.blocksFun j)‖) := by
      /-
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        ⊢ LE.le (HAdd.hAdd (HMul.hMul a I) ((Finset.Ico 2 (HAdd.hAdd n 1)).sum fun k = …
      -/
      gcongr with j
      /-
        case bc.h.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        j : Nat
        a✝ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        ⊢ LE.le (Norm.norm ((↑i.symm).compContinuousMultilinearMap ((setOf fun c => LT …
      -/
      apply (ContinuousLinearMap.norm_compContinuousMultilinearMap_le _ _).trans
      /-
        case bc.h.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        j : Nat
        a✝ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        ⊢ LE.le (HMul.hMul (Norm.norm ↑i.symm) (Norm.norm ((setOf fun c => LT.lt 1 c.l …
      -/
      gcongr
      /-
        case bc.h.h.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        j : Nat
        a✝ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        ⊢ LE.le (Norm.norm ((setOf fun c => LT.lt 1 c.length).toFinset.sum fun c => p. …
      -/
      apply (norm_sum_le _ _).trans
      /-
        case bc.h.h.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        j : Nat
        a✝ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        ⊢ LE.le ((setOf fun c => LT.lt 1 c.length).toFinset.sum fun i_1 => Norm.norm ( …
      -/
      gcongr
      /-
        case bc.h.h.h.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        j : Nat
        a✝¹ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        i✝ : Composition j
        a✝ : Membership.mem (setOf fun c => LT.lt 1 c.length).toFinset i✝
        ⊢ LE.le (Norm.norm (p.compAlongComposition (p.rightInv i x) i✝)) (HMul.hMul (H …
      -/
      apply (compAlongComposition_norm _ _ _).trans
      /-
        case bc.h.h.h.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        j : Nat
        a✝¹ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        i✝ : Composition j
        a✝ : Membership.mem (setOf fun c => LT.lt 1 c.length).toFinset i✝
        ⊢ LE.le (HMul.hMul (Norm.norm (p i✝.length)) (Finset.univ.prod fun i_1 => Norm …
      -/
      gcongr
      /-
        case bc.h.h.h.h.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        j : Nat
        a✝¹ : Membership.mem (Finset.Ico 2 (HAdd.hAdd n 1)) j
        i✝ : Composition j
        a✝ : Membership.mem (setOf fun c => LT.lt 1 c.length).toFinset i✝
        ⊢ LE.le (Norm.norm (p i✝.length)) (HMul.hMul C (HPow.hPow r i✝.length))
      -/
      apply hp
      /-
        🎉 no goals
      -/
    _ = I * a + I * C * ∑ k ∈ Ico 2 (n + 1), a ^ k *
          ∑ c ∈ ({c | 1 < Composition.length c}.toFinset : Finset (Composition k)),
            r ^ c.length * ∏ j, ‖p.rightInv i x (c.blocksFun j)‖ := by
      simp_rw [I, mul_assoc C, ← mul_sum, ← mul_assoc, mul_comm _ ‖(i.symm : F →L[𝕜] E)‖,
        mul_assoc, ← mul_sum, ← mul_assoc, mul_comm _ C, mul_assoc, ← mul_sum]
      /-
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        ⊢ Eq (HAdd.hAdd (HMul.hMul (Norm.norm ↑i.symm) a) (HMul.hMul (Norm.norm ↑i.sym …
      -/
      ring
      /-
        🎉 no goals
      -/
    _ ≤ I * a + I * C *
        ∑ k ∈ Ico 2 (n + 1), (r * ∑ j ∈ Ico 1 n, a ^ j * ‖p.rightInv i x j‖) ^ k := by
      /-
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        ⊢ LE.le (HAdd.hAdd (HMul.hMul I a) (HMul.hMul (HMul.hMul I C) ((Finset.Ico 2 ( …
      -/
      gcongr _ + _ * _ * ?_
      /-
        case bc.h
        𝕜 : Type u_1
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type u_2
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type u_3
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        x : E
        n : Nat
        hn : LE.le 2 (HAdd.hAdd n 1)
        p : FormalMultilinearSeries 𝕜 E F
        i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
        r a C : Real
        hr : LE.le 0 r
        ha : LE.le 0 a
        hC : LE.le 0 C
        hp : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
        I : Real := Norm.norm ↑i.symm
        ⊢ LE.le ((Finset.Ico 2 (HAdd.hAdd n 1)).sum fun k => HMul.hMul (HPow.hPow a k) …
      -/
      simp_rw [mul_pow]
      apply
        radius_right_inv_pos_of_radius_pos_aux1 n (fun k => ‖p.rightInv i x k‖)
          (fun k => norm_nonneg _) hr ha


/-- If a a formal multilinear series has a positive radius of convergence, then its right inverse
also has a positive radius of convergence. -/
theorem radius_rightInv_pos_of_radius_pos
    {p : FormalMultilinearSeries 𝕜 E F} {i : E ≃L[𝕜] F} {x : E}
    (hp : 0 < p.radius) : 0 < (p.rightInv i x).radius := by
  obtain ⟨C, r, Cpos, rpos, ple⟩ :
    ∃ (C r : _) (_ : 0 < C) (_ : 0 < r), ∀ n : ℕ, ‖p n‖ ≤ C * r ^ n :=
    le_mul_pow_of_radius_pos p hp
  /-
    case intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    hp : LT.lt 0 p.radius
    C r : Real
    Cpos : LT.lt 0 C
    rpos : LT.lt 0 r
    ple : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
    ⊢ LT.lt 0 (p.rightInv i x).radius
  -/
  let I := ‖(i.symm : F →L[𝕜] E)‖
  -- choose `a` small enough to make sure that `∑_{k ≤ n} aᵏ Qₖ` will be controllable by
  -- induction
  obtain ⟨a, apos, ha1, ha2⟩ :
    ∃ (a : _) (apos : 0 < a),
      2 * I * C * r ^ 2 * (I + 1) ^ 2 * a ≤ 1 ∧ r * (I + 1) * a ≤ 1 / 2 := by
    have :
      Tendsto (fun a => 2 * I * C * r ^ 2 * (I + 1) ^ 2 * a) (𝓝 0)
        (𝓝 (2 * I * C * r ^ 2 * (I + 1) ^ 2 * 0)) :=
      tendsto_const_nhds.mul tendsto_id
    have A : ∀ᶠ a in 𝓝 0, 2 * I * C * r ^ 2 * (I + 1) ^ 2 * a < 1 := by
      apply (tendsto_order.1 this).2; simp [zero_lt_one]
    have : Tendsto (fun a => r * (I + 1) * a) (𝓝 0) (𝓝 (r * (I + 1) * 0)) :=
      tendsto_const_nhds.mul tendsto_id
    have B : ∀ᶠ a in 𝓝 0, r * (I + 1) * a < 1 / 2 := by
      apply (tendsto_order.1 this).2; simp [zero_lt_one]
    have C : ∀ᶠ a in 𝓝[>] (0 : ℝ), (0 : ℝ) < a := by
      filter_upwards [self_mem_nhdsWithin] with _ ha using ha
    rcases (C.and ((A.and B).filter_mono inf_le_left)).exists with ⟨a, ha⟩
    exact ⟨a, ha.1, ha.2.1.le, ha.2.2.le⟩
  -- check by induction that the partial sums are suitably bounded, using the choice of `a` and the
  -- inductive control from Lemma `radius_rightInv_pos_of_radius_pos_aux2`.
  /-
    case intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    hp : LT.lt 0 p.radius
    C r : Real
    Cpos : LT.lt 0 C
    rpos : LT.lt 0 r
    ple : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
    I : Real := Norm.norm ↑i.symm
    a : Real
    apos : LT.lt 0 a
    ha1 : LE.le (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 I) C) (HP …
    ha2 : LE.le (HMul.hMul (HMul.hMul r (HAdd.hAdd I 1)) a) (1 / 2)
    ⊢ LT.lt 0 (p.rightInv i x).radius
  -/
  let S n := ∑ k ∈ Ico 1 n, a ^ k * ‖p.rightInv i x k‖
  have IRec : ∀ n, 1 ≤ n → S n ≤ (I + 1) * a := by
    apply Nat.le_induction
    · simp only [S]
      rw [Ico_eq_empty_of_le (le_refl 1), sum_empty]
      exact mul_nonneg (add_nonneg (norm_nonneg _) zero_le_one) apos.le
    · intro n one_le_n hn
      have In : 2 ≤ n + 1 := by omega
      have rSn : r * S n ≤ 1 / 2 :=
        calc
          r * S n ≤ r * ((I + 1) * a) := by gcongr
          _ ≤ 1 / 2 := by rwa [← mul_assoc]
      calc
        S (n + 1) ≤ I * a + I * C * ∑ k ∈ Ico 2 (n + 1), (r * S n) ^ k :=
          radius_rightInv_pos_of_radius_pos_aux2 In p i rpos.le apos.le Cpos.le ple
        _ = I * a + I * C * (((r * S n) ^ 2 - (r * S n) ^ (n + 1)) / (1 - r * S n)) := by
          rw [geom_sum_Ico' _ In]; exact ne_of_lt (rSn.trans_lt (by norm_num))
        _ ≤ I * a + I * C * ((r * S n) ^ 2 / (1 / 2)) := by
          gcongr
          · simp only [sub_le_self_iff]
            positivity
          · linarith only [rSn]
        _ = I * a + 2 * I * C * (r * S n) ^ 2 := by ring
        _ ≤ I * a + 2 * I * C * (r * ((I + 1) * a)) ^ 2 := by gcongr
        _ = (I + 2 * I * C * r ^ 2 * (I + 1) ^ 2 * a) * a := by ring
        _ ≤ (I + 1) * a := by gcongr
  -- conclude that all coefficients satisfy `aⁿ Qₙ ≤ (I + 1) a`.
  /-
    case intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    hp : LT.lt 0 p.radius
    C r : Real
    Cpos : LT.lt 0 C
    rpos : LT.lt 0 r
    ple : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
    I : Real := Norm.norm ↑i.symm
    a : Real
    apos : LT.lt 0 a
    ha1 : LE.le (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 I) C) (HP …
    ha2 : LE.le (HMul.hMul (HMul.hMul r (HAdd.hAdd I 1)) a) (1 / 2)
    S : Nat → Real := fun n => (Finset.Ico 1 n).sum fun k => HMul.hMul (HPow.hPow  …
    IRec : ∀ (n : Nat), LE.le 1 n → LE.le (S n) (HMul.hMul (HAdd.hAdd I 1) a)
    ⊢ LT.lt 0 (p.rightInv i x).radius
  -/
  let a' : NNReal := ⟨a, apos.le⟩
  suffices H : (a' : ENNReal) ≤ (p.rightInv i x).radius by
    apply lt_of_lt_of_le _ H
    -- Prior to https://github.com/leanprover/lean4/pull/2734, this was `exact_mod_cast apos`.
    simpa only [ENNReal.coe_pos]
  /-
    case intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    hp : LT.lt 0 p.radius
    C r : Real
    Cpos : LT.lt 0 C
    rpos : LT.lt 0 r
    ple : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
    I : Real := Norm.norm ↑i.symm
    a : Real
    apos : LT.lt 0 a
    ha1 : LE.le (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 I) C) (HP …
    ha2 : LE.le (HMul.hMul (HMul.hMul r (HAdd.hAdd I 1)) a) (1 / 2)
    S : Nat → Real := fun n => (Finset.Ico 1 n).sum fun k => HMul.hMul (HPow.hPow  …
    IRec : ∀ (n : Nat), LE.le 1 n → LE.le (S n) (HMul.hMul (HAdd.hAdd I 1) a)
    a' : NNReal := ⟨a, ⋯⟩
    ⊢ LE.le (↑a') (p.rightInv i x).radius
  -/
  apply le_radius_of_eventually_le _ ((I + 1) * a)
  /-
    case intro.intro.intro.intro.intro.intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    hp : LT.lt 0 p.radius
    C r : Real
    Cpos : LT.lt 0 C
    rpos : LT.lt 0 r
    ple : ∀ (n : Nat), LE.le (Norm.norm (p n)) (HMul.hMul C (HPow.hPow r n))
    I : Real := Norm.norm ↑i.symm
    a : Real
    apos : LT.lt 0 a
    ha1 : LE.le (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul 2 I) C) (HP …
    ha2 : LE.le (HMul.hMul (HMul.hMul r (HAdd.hAdd I 1)) a) (1 / 2)
    S : Nat → Real := fun n => (Finset.Ico 1 n).sum fun k => HMul.hMul (HPow.hPow  …
    IRec : ∀ (n : Nat), LE.le 1 n → LE.le (S n) (HMul.hMul (HAdd.hAdd I 1) a)
    a' : NNReal := ⟨a, ⋯⟩
    ⊢ Filter.Eventually (fun n => LE.le (HMul.hMul (Norm.norm (p.rightInv i x n))  …
  -/
  filter_upwards [Ici_mem_atTop 1] with n (hn : 1 ≤ n)
  calc
    ‖p.rightInv i x n‖ * (a' : ℝ) ^ n = a ^ n * ‖p.rightInv i x n‖ := mul_comm _ _
    _ ≤ ∑ k ∈ Ico 1 (n + 1), a ^ k * ‖p.rightInv i x k‖ :=
      (haveI : ∀ k ∈ Ico 1 (n + 1), 0 ≤ a ^ k * ‖p.rightInv i x k‖ := fun k _ => by positivity
      single_le_sum this (by simp [hn]))
    _ ≤ (I + 1) * a := IRec (n + 1) (by norm_num)


/-- If a a formal multilinear series has a positive radius of convergence, then its left inverse
also has a positive radius of convergence. -/
theorem radius_leftInv_pos_of_radius_pos
    {p : FormalMultilinearSeries 𝕜 E F} {i : E ≃L[𝕜] F} {x : E}
    (hp : 0 < p.radius) (h : p 1 = (continuousMultilinearCurryFin1 𝕜 E F).symm i) :
    0 < (p.leftInv i x).radius := by
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    hp : LT.lt 0 p.radius
    h : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E F).symm ↑i)
    ⊢ LT.lt 0 (p.leftInv i x).radius
  -/
  rw [leftInv_eq_rightInv _ _ _ h]
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E F
    x : E
    hp : LT.lt 0 p.radius
    h : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E F).symm ↑i)
    ⊢ LT.lt 0 (p.rightInv i x).radius
  -/
  exact radius_rightInv_pos_of_radius_pos hp
  /-
    🎉 no goals
  -/


lemma HasFPowerSeriesAt.tendsto_partialSum_prod_of_comp
    {f : E → G} {q : FormalMultilinearSeries 𝕜 F G}
    {p : FormalMultilinearSeries 𝕜 E F} {x : E}
    (hf : HasFPowerSeriesAt f (q.comp p) x) (hq : 0 < q.radius) (hp : 0 < p.radius) :
    ∀ᶠ y in 𝓝 0, Tendsto (fun (a : ℕ × ℕ) ↦ q.partialSum a.1 (p.partialSum a.2 y
      - p 0 (fun _ ↦ 0))) atTop (𝓝 (f (x + y))) := by
  /-
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hf : HasFPowerSeriesAt f (q.comp p) x
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    ⊢ Filter.Eventually (fun y => Filter.Tendsto (fun a => q.partialSum a.1 (HSub. …
  -/
  rcases hf with ⟨r0, h0⟩
  /-
    case intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    r0 : ENNReal
    h0 : HasFPowerSeriesOnBall f (q.comp p) x r0
    ⊢ Filter.Eventually (fun y => Filter.Tendsto (fun a => q.partialSum a.1 (HSub. …
  -/
  rcases q.comp_summable_nnreal p hq hp with ⟨r1, r1_pos : 0 < r1, hr1⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    r0 : ENNReal
    h0 : HasFPowerSeriesOnBall f (q.comp p) x r0
    r1 : NNReal
    r1_pos : LT.lt 0 r1
    hr1 : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.s …
    ⊢ Filter.Eventually (fun y => Filter.Tendsto (fun a => q.partialSum a.1 (HSub. …
  -/
  let r : ℝ≥0∞ := min r0 r1
  have : EMetric.ball (0 : E) r ∈ 𝓝 0 :=
    EMetric.ball_mem_nhds 0 (lt_min h0.r_pos (by exact_mod_cast r1_pos))
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    r0 : ENNReal
    h0 : HasFPowerSeriesOnBall f (q.comp p) x r0
    r1 : NNReal
    r1_pos : LT.lt 0 r1
    hr1 : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.s …
    r : ENNReal := Min.min r0 ↑r1
    this : Membership.mem (nhds 0) (EMetric.ball 0 r)
    ⊢ Filter.Eventually (fun y => Filter.Tendsto (fun a => q.partialSum a.1 (HSub. …
  -/
  filter_upwards [this] with y hy
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    r0 : ENNReal
    h0 : HasFPowerSeriesOnBall f (q.comp p) x r0
    r1 : NNReal
    r1_pos : LT.lt 0 r1
    hr1 : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.s …
    r : ENNReal := Min.min r0 ↑r1
    this : Membership.mem (nhds 0) (EMetric.ball 0 r)
    y : E
    hy : Membership.mem (EMetric.ball 0 r) y
    ⊢ Filter.Tendsto (fun a => q.partialSum a.1 (HSub.hSub (p.partialSum a.2 y) (( …
  -/
  have hy0 : y ∈ EMetric.ball 0 r0 := EMetric.ball_subset_ball (min_le_left _ _) hy
  have A : HasSum (fun i : Σ n, Composition n => q.compAlongComposition p i.2 fun _j => y)
      (f (x + y)) := by
    have cau : CauchySeq fun s : Finset (Σ n, Composition n) =>
        ∑ i ∈ s, q.compAlongComposition p i.2 fun _j => y := by
      apply cauchySeq_finset_of_norm_bounded _ (NNReal.summable_coe.2 hr1) _
      simp only [coe_nnnorm, NNReal.coe_mul, NNReal.coe_pow]
      rintro ⟨n, c⟩
      calc
        ‖(compAlongComposition q p c) fun _j : Fin n => y‖ ≤
            ‖compAlongComposition q p c‖ * ∏ _j : Fin n, ‖y‖ := by
          apply ContinuousMultilinearMap.le_opNorm
        _ ≤ ‖compAlongComposition q p c‖ * (r1 : ℝ) ^ n := by
          apply mul_le_mul_of_nonneg_left _ (norm_nonneg _)
          rw [Finset.prod_const, Finset.card_fin]
          gcongr
          rw [EMetric.mem_ball, edist_eq_coe_nnnorm] at hy
          have := le_trans (le_of_lt hy) (min_le_right _ _)
          rwa [ENNReal.coe_le_coe, ← NNReal.coe_le_coe, coe_nnnorm] at this
    apply HasSum.of_sigma (fun b ↦ hasSum_fintype _) ?_ cau
    simpa [FormalMultilinearSeries.comp] using h0.hasSum hy0
  have B : Tendsto (fun (n : ℕ × ℕ) => ∑ i ∈ compPartialSumTarget 0 n.1 n.2,
      q.compAlongComposition p i.2 fun _j => y) atTop (𝓝 (f (x + y))) := by
    apply Tendsto.comp A compPartialSumTarget_tendsto_prod_atTop
  have C : Tendsto (fun (n : ℕ × ℕ) => q.partialSum n.1 (∑ a ∈ Finset.Ico 1 n.2, p a fun _b ↦ y))
      atTop (𝓝 (f (x + y))) := by simpa [comp_partialSum] using B
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    r0 : ENNReal
    h0 : HasFPowerSeriesOnBall f (q.comp p) x r0
    r1 : NNReal
    r1_pos : LT.lt 0 r1
    hr1 : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.s …
    r : ENNReal := Min.min r0 ↑r1
    this : Membership.mem (nhds 0) (EMetric.ball 0 r)
    y : E
    hy : Membership.mem (EMetric.ball 0 r) y
    hy0 : Membership.mem (EMetric.ball 0 r0) y
    A : HasSum (fun i => (q.compAlongComposition p i.snd) fun _j => y) (f (HAdd.hA …
    B : Filter.Tendsto (fun n => (FormalMultilinearSeries.compPartialSumTarget 0 n …
    C : Filter.Tendsto (fun n => q.partialSum n.1 ((Finset.Ico 1 n.2).sum fun a => …
    ⊢ Filter.Tendsto (fun a => q.partialSum a.1 (HSub.hSub (p.partialSum a.2 y) (( …
  -/
  apply C.congr'
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    r0 : ENNReal
    h0 : HasFPowerSeriesOnBall f (q.comp p) x r0
    r1 : NNReal
    r1_pos : LT.lt 0 r1
    hr1 : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.s …
    r : ENNReal := Min.min r0 ↑r1
    this : Membership.mem (nhds 0) (EMetric.ball 0 r)
    y : E
    hy : Membership.mem (EMetric.ball 0 r) y
    hy0 : Membership.mem (EMetric.ball 0 r0) y
    A : HasSum (fun i => (q.compAlongComposition p i.snd) fun _j => y) (f (HAdd.hA …
    B : Filter.Tendsto (fun n => (FormalMultilinearSeries.compPartialSumTarget 0 n …
    C : Filter.Tendsto (fun n => q.partialSum n.1 ((Finset.Ico 1 n.2).sum fun a => …
    ⊢ Filter.atTop.EventuallyEq (fun n => q.partialSum n.1 ((Finset.Ico 1 n.2).sum …
  -/
  filter_upwards [Ici_mem_atTop (0, 1)]
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    r0 : ENNReal
    h0 : HasFPowerSeriesOnBall f (q.comp p) x r0
    r1 : NNReal
    r1_pos : LT.lt 0 r1
    hr1 : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.s …
    r : ENNReal := Min.min r0 ↑r1
    this : Membership.mem (nhds 0) (EMetric.ball 0 r)
    y : E
    hy : Membership.mem (EMetric.ball 0 r) y
    hy0 : Membership.mem (EMetric.ball 0 r0) y
    A : HasSum (fun i => (q.compAlongComposition p i.snd) fun _j => y) (f (HAdd.hA …
    B : Filter.Tendsto (fun n => (FormalMultilinearSeries.compPartialSumTarget 0 n …
    C : Filter.Tendsto (fun n => q.partialSum n.1 ((Finset.Ico 1 n.2).sum fun a => …
    ⊢ ∀ (a : Prod Nat Nat), Membership.mem (Set.Ici { fst := 0, snd := 1 }) a → Eq …
  -/
  rintro ⟨-, n⟩ ⟨-, (hn : 1 ≤ n)⟩
  /-
    case h.mk.intro
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    r0 : ENNReal
    h0 : HasFPowerSeriesOnBall f (q.comp p) x r0
    r1 : NNReal
    r1_pos : LT.lt 0 r1
    hr1 : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.s …
    r : ENNReal := Min.min r0 ↑r1
    this : Membership.mem (nhds 0) (EMetric.ball 0 r)
    y : E
    hy : Membership.mem (EMetric.ball 0 r) y
    hy0 : Membership.mem (EMetric.ball 0 r0) y
    A : HasSum (fun i => (q.compAlongComposition p i.snd) fun _j => y) (f (HAdd.hA …
    B : Filter.Tendsto (fun n => (FormalMultilinearSeries.compPartialSumTarget 0 n …
    C : Filter.Tendsto (fun n => q.partialSum n.1 ((Finset.Ico 1 n.2).sum fun a => …
    fst✝ n : Nat
    hn : LE.le 1 n
    ⊢ Eq (q.partialSum { fst := fst✝, snd := n }.1 ((Finset.Ico 1 { fst := fst✝, s …
  -/
  congr
  rw [partialSum, eq_sub_iff_add_eq', Finset.range_eq_Ico,
        Finset.sum_eq_sum_Ico_succ_bot hn]
  /-
    case h.mk.intro.e_x
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    r0 : ENNReal
    h0 : HasFPowerSeriesOnBall f (q.comp p) x r0
    r1 : NNReal
    r1_pos : LT.lt 0 r1
    hr1 : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.s …
    r : ENNReal := Min.min r0 ↑r1
    this : Membership.mem (nhds 0) (EMetric.ball 0 r)
    y : E
    hy : Membership.mem (EMetric.ball 0 r) y
    hy0 : Membership.mem (EMetric.ball 0 r0) y
    A : HasSum (fun i => (q.compAlongComposition p i.snd) fun _j => y) (f (HAdd.hA …
    B : Filter.Tendsto (fun n => (FormalMultilinearSeries.compPartialSumTarget 0 n …
    C : Filter.Tendsto (fun n => q.partialSum n.1 ((Finset.Ico 1 n.2).sum fun a => …
    fst✝ n : Nat
    hn : LE.le 1 n
    ⊢ Eq (HAdd.hAdd ((p 0) fun x => 0) ((Finset.Ico 1 { fst := fst✝, snd := n }.2) …
  -/
  congr with i
  /-
    case h.mk.intro.e_x.e_a.h.e_6.h.h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hq : LT.lt 0 q.radius
    hp : LT.lt 0 p.radius
    r0 : ENNReal
    h0 : HasFPowerSeriesOnBall f (q.comp p) x r0
    r1 : NNReal
    r1_pos : LT.lt 0 r1
    hr1 : Summable fun i => HMul.hMul (NNNorm.nnnorm (q.compAlongComposition p i.s …
    r : ENNReal := Min.min r0 ↑r1
    this : Membership.mem (nhds 0) (EMetric.ball 0 r)
    y : E
    hy : Membership.mem (EMetric.ball 0 r) y
    hy0 : Membership.mem (EMetric.ball 0 r0) y
    A : HasSum (fun i => (q.compAlongComposition p i.snd) fun _j => y) (f (HAdd.hA …
    B : Filter.Tendsto (fun n => (FormalMultilinearSeries.compPartialSumTarget 0 n …
    C : Filter.Tendsto (fun n => q.partialSum n.1 ((Finset.Ico 1 n.2).sum fun a => …
    fst✝ n : Nat
    hn : LE.le 1 n
    i : Fin 0
    ⊢ Eq 0 y
  -/
  exact i.elim0
  /-
    🎉 no goals
  -/


lemma HasFPowerSeriesAt.eventually_hasSum_of_comp  {f : E → F} {g : F → G}
    {q : FormalMultilinearSeries 𝕜 F G} {p : FormalMultilinearSeries 𝕜 E F} {x : E}
    (hgf : HasFPowerSeriesAt (g ∘ f) (q.comp p) x) (hf : HasFPowerSeriesAt f p x)
    (hq : 0 < q.radius) :
    ∀ᶠ y in 𝓝 0, HasSum (fun n : ℕ => q n fun _ : Fin n => (f (x + y) - f x)) (g (f (x + y))) := by
  have : ∀ᶠ y in 𝓝 (0 : E), f (x + y) - f x ∈ EMetric.ball 0 q.radius := by
    have A : ContinuousAt (fun y ↦ f (x + y) - f x) 0 := by
      apply ContinuousAt.sub _ continuousAt_const
      exact hf.continuousAt.comp_of_eq (continuous_add_left x).continuousAt (by simp)
    have B : EMetric.ball 0 q.radius ∈ 𝓝 (f (x + 0) - f x) := by
      simpa using EMetric.ball_mem_nhds _ hq
    exact A.preimage_mem_nhds B
  filter_upwards [hgf.tendsto_partialSum_prod_of_comp hq (hf.radius_pos),
    hf.tendsto_partialSum, this] with y hy h'y h''y
  have L : Tendsto (fun n ↦ q.partialSum n (f (x + y) - f x)) atTop (𝓝 (g (f (x + y)))) := by
    apply (closed_nhds_basis (g (f (x + y)))).tendsto_right_iff.2
    rintro u ⟨hu, u_closed⟩
    simp only [id_eq, eventually_atTop, ge_iff_le]
    rcases mem_nhds_iff.1 hu with ⟨v, vu, v_open, hv⟩
    obtain ⟨a₀, b₀, hab⟩ : ∃ a₀ b₀, ∀ (a b : ℕ), a₀ ≤ a → b₀ ≤ b →
        q.partialSum a (p.partialSum b y - (p 0) fun _ ↦ 0) ∈ v := by
      simpa using hy (v_open.mem_nhds hv)
    refine ⟨a₀, fun a ha ↦ ?_⟩
    have : Tendsto (fun b ↦ q.partialSum a (p.partialSum b y - (p 0) fun _ ↦ 0)) atTop
        (𝓝 (q.partialSum a (f (x + y) - f x))) := by
      have : ContinuousAt (q.partialSum a) (f (x + y) - f x) :=
        (partialSum_continuous q a).continuousAt
      apply this.tendsto.comp
      apply Tendsto.sub h'y
      convert tendsto_const_nhds
      exact (HasFPowerSeriesAt.coeff_zero hf fun _ ↦ 0).symm
    apply u_closed.mem_of_tendsto this
    filter_upwards [Ici_mem_atTop b₀] with b hb using vu (hab _ _ ha hb)
  have C : CauchySeq (fun (s : Finset ℕ) ↦ ∑ n ∈ s, q n fun _ : Fin n => (f (x + y) - f x)) := by
    have Z := q.summable_norm_apply (x := f (x + y) - f x) h''y
    exact cauchySeq_finset_of_norm_bounded _ Z (fun i ↦ le_rfl)
  /-
    case h
    𝕜 : Type u_1
    inst✝⁶ : NontriviallyNormedField 𝕜
    E : Type u_2
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    F : Type u_3
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    G : Type u_4
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → F
    g : F → G
    q : FormalMultilinearSeries 𝕜 F G
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    hgf : HasFPowerSeriesAt (Function.comp g f) (q.comp p) x
    hf : HasFPowerSeriesAt f p x
    hq : LT.lt 0 q.radius
    this : Filter.Eventually (fun y => Membership.mem (EMetric.ball 0 q.radius) (H …
    y : E
    hy : Filter.Tendsto (fun a => q.partialSum a.1 (HSub.hSub (p.partialSum a.2 y) …
    h'y : Filter.Tendsto (fun n => p.partialSum n y) Filter.atTop (nhds (f (HAdd.h …
    h''y : Membership.mem (EMetric.ball 0 q.radius) (HSub.hSub (f (HAdd.hAdd x y)) …
    L : Filter.Tendsto (fun n => q.partialSum n (HSub.hSub (f (HAdd.hAdd x y)) (f  …
    C : CauchySeq fun s => s.sum fun n => (q n) fun x_1 => HSub.hSub (f (HAdd.hAdd …
    ⊢ HasSum (fun n => (q n) fun x_1 => HSub.hSub (f (HAdd.hAdd x y)) (f x)) (g (f …
  -/
  exact tendsto_nhds_of_cauchySeq_of_subseq C tendsto_finset_range L
  /-
    🎉 no goals
  -/


/-- If a partial homeomorphism `f` is defined at `a` and has a power series expansion there with
invertible linear term, then `f.symm` has a power series expansion at `f a`, given by the inverse
of the initial power series. -/
theorem PartialHomeomorph.hasFPowerSeriesAt_symm (f : PartialHomeomorph E F) {a : E}
    {i : E ≃L[𝕜] F} (h0 : a ∈ f.source) {p : FormalMultilinearSeries 𝕜 E F}
    (h : HasFPowerSeriesAt f p a) (hp : p 1 = (continuousMultilinearCurryFin1 𝕜 E F).symm i) :
    HasFPowerSeriesAt f.symm (p.leftInv i a) (f a) := by
  have A : HasFPowerSeriesAt (f.symm ∘ f) ((p.leftInv i a).comp p) a := by
    have : HasFPowerSeriesAt (ContinuousLinearMap.id 𝕜 E) ((p.leftInv i a).comp p) a := by
      rw [leftInv_comp _ _ _ hp]
      exact (ContinuousLinearMap.id 𝕜 E).hasFPowerSeriesAt a
    apply this.congr
    filter_upwards [f.open_source.mem_nhds h0] with x hx using by simp [hx]
  have B : ∀ᶠ (y : E) in 𝓝 0, HasSum (fun n ↦ (p.leftInv i a n) fun _ ↦ f (a + y) - f a)
      (f.symm (f (a + y))) := by
    simpa using A.eventually_hasSum_of_comp h (radius_leftInv_pos_of_radius_pos h.radius_pos hp)
  have C : ∀ᶠ (y : E) in 𝓝 a, HasSum (fun n ↦ (p.leftInv i a n) fun _ ↦ f y - f a)
      (f.symm (f y)) := by
    rw [← sub_eq_zero_of_eq (a := a) rfl] at B
    have : ContinuousAt (fun x ↦ x - a) a := by fun_prop
    simpa using this.preimage_mem_nhds B
  have D : ∀ᶠ (y : E) in 𝓝 (f.symm (f a)),
      HasSum (fun n ↦ (p.leftInv i a n) fun _ ↦ f y - f a) y := by
    simp only [h0, PartialHomeomorph.left_inv]
    filter_upwards [C, f.open_source.mem_nhds h0] with x hx h'x
    simpa [h'x] using hx
  have E : ∀ᶠ z in 𝓝 (f a), HasSum (fun n ↦ (p.leftInv i a n) fun _ ↦ f (f.symm z) - f a)
      (f.symm z) := by
    have : ContinuousAt f.symm (f a) := f.continuousAt_symm (f.map_source h0)
    exact this D
  have F : ∀ᶠ z in 𝓝 (f a), HasSum (fun n ↦ (p.leftInv i a n) fun _ ↦ z - f a) (f.symm z) := by
    filter_upwards [f.open_target.mem_nhds (f.map_source h0), E] with z hz h'z
    simpa [hz] using h'z
  /-
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E✝ : Type u_2
    inst✝³ : NormedAddCommGroup E✝
    inst✝² : NormedSpace 𝕜 E✝
    F✝ : Type u_3
    inst✝¹ : NormedAddCommGroup F✝
    inst✝ : NormedSpace 𝕜 F✝
    f : PartialHomeomorph E✝ F✝
    a : E✝
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E✝ F✝
    h0 : Membership.mem f.source a
    p : FormalMultilinearSeries 𝕜 E✝ F✝
    h : HasFPowerSeriesAt (↑f) p a
    hp : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E✝ F✝).symm ↑i)
    A : HasFPowerSeriesAt (Function.comp ↑f.symm ↑f) ((p.leftInv i a).comp p) a
    B : Filter.Eventually (fun y => HasSum (fun n => (p.leftInv i a n) fun x => HS …
    C : Filter.Eventually (fun y => HasSum (fun n => (p.leftInv i a n) fun x => HS …
    D : Filter.Eventually (fun y => HasSum (fun n => (p.leftInv i a n) fun x => HS …
    E : Filter.Eventually (fun z => HasSum (fun n => (p.leftInv i a n) fun x => HS …
    F : Filter.Eventually (fun z => HasSum (fun n => (p.leftInv i a n) fun x => HS …
    ⊢ HasFPowerSeriesAt (↑f.symm) (p.leftInv i a) (↑f a)
  -/
  rcases EMetric.mem_nhds_iff.1 F with ⟨r, r_pos, hr⟩
  refine ⟨min r (p.leftInv i a).radius, min_le_right _ _,
    lt_min r_pos (radius_leftInv_pos_of_radius_pos h.radius_pos hp), fun {y} hy ↦ ?_⟩
  have : y + f a ∈ EMetric.ball (f a) r := by
    simp only [EMetric.mem_ball, edist_eq_coe_nnnorm_sub, sub_zero, lt_min_iff,
      add_sub_cancel_right] at hy ⊢
    exact hy.1
  /-
    case intro.intro
    𝕜 : Type u_1
    inst✝⁴ : NontriviallyNormedField 𝕜
    E✝ : Type u_2
    inst✝³ : NormedAddCommGroup E✝
    inst✝² : NormedSpace 𝕜 E✝
    F✝ : Type u_3
    inst✝¹ : NormedAddCommGroup F✝
    inst✝ : NormedSpace 𝕜 F✝
    f : PartialHomeomorph E✝ F✝
    a : E✝
    i : ContinuousLinearEquiv (RingHom.id 𝕜) E✝ F✝
    h0 : Membership.mem f.source a
    p : FormalMultilinearSeries 𝕜 E✝ F✝
    h : HasFPowerSeriesAt (↑f) p a
    hp : Eq (p 1) ((continuousMultilinearCurryFin1 𝕜 E✝ F✝).symm ↑i)
    A : HasFPowerSeriesAt (Function.comp ↑f.symm ↑f) ((p.leftInv i a).comp p) a
    B : Filter.Eventually (fun y => HasSum (fun n => (p.leftInv i a n) fun x => HS …
    C : Filter.Eventually (fun y => HasSum (fun n => (p.leftInv i a n) fun x => HS …
    D : Filter.Eventually (fun y => HasSum (fun n => (p.leftInv i a n) fun x => HS …
    E : Filter.Eventually (fun z => HasSum (fun n => (p.leftInv i a n) fun x => HS …
    F : Filter.Eventually (fun z => HasSum (fun n => (p.leftInv i a n) fun x => HS …
    r : ENNReal
    r_pos : GT.gt r 0
    hr : HasSubset.Subset (EMetric.ball (↑f a) r) (setOf fun x => (fun z => HasSum …
    y : F✝
    hy : Membership.mem (EMetric.ball 0 (Min.min r (p.leftInv i a).radius)) y
    this : Membership.mem (EMetric.ball (↑f a) r) (HAdd.hAdd y (↑f a))
    ⊢ HasSum (fun n => (p.leftInv i a n) fun x => y) (↑f.symm (HAdd.hAdd (↑f a) y))
  -/
  simpa [add_comm] using hr this
  /-
    🎉 no goals
  -/

