/-- Given an ordered basis, produce a bilinear form associated with the quadratic form.

Unlike `QuadraticMap.associated`, this is not symmetric; however, as a result it can be used even
in characteristic two. When considered as a matrix, the form is triangular. -/
noncomputable def toBilin (Q : QuadraticMap R M N) (bm : Basis ι R M) : LinearMap.BilinMap R M N :=
  bm.constr (S := R) fun i =>
    bm.constr (S := R) fun j =>
      if i = j then Q (bm i) else if i < j then polar Q (bm i) (bm j) else 0


theorem toBilin_apply (Q : QuadraticMap R M N) (bm : Basis ι R M) (i j : ι) :
    Q.toBilin bm (bm i) (bm j) =
      if i = j then Q (bm i) else if i < j then polar Q (bm i) (bm j) else 0 := by
  /-
    ι : Type u_4
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticMap R M N
    bm : Basis ι R M
    i j : ι
    ⊢ Eq (((Q.toBilin bm) (bm i)) (bm j)) (ite (Eq i j) (Q (bm i)) (ite (LT.lt i j …
  -/
  simp [toBilin]
  /-
    🎉 no goals
  -/


theorem toQuadraticMap_toBilin (Q : QuadraticMap R M N) (bm : Basis ι R M) :
    (Q.toBilin bm).toQuadraticMap = Q := by
  /-
    ι : Type u_4
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticMap R M N
    bm : Basis ι R M
    ⊢ Eq (Q.toBilin bm).toQuadraticMap Q
  -/
  ext x
  rw [← bm.linearCombination_repr x, LinearMap.BilinMap.toQuadraticMap_apply,
      Finsupp.linearCombination_apply, Finsupp.sum]
  simp_rw [LinearMap.map_sum₂, map_sum, LinearMap.map_smul₂, _root_.map_smul, toBilin_apply,
    smul_ite, smul_zero, ← Finset.sum_product', ← Finset.diag_union_offDiag,
    Finset.sum_union (Finset.disjoint_diag_offDiag _), Finset.sum_diag, if_true]
  /-
    case H
    ι : Type u_4
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    Q : QuadraticMap R M N
    bm : Basis ι R M
    x : M
    ⊢ Eq (HAdd.hAdd ((bm.repr x).support.sum fun x_1 => HSMul.hSMul ((bm.repr x) x …
  -/
  rw [Finset.sum_ite_of_false, QuadraticMap.map_sum, ← Finset.sum_filter]
  · simp_rw [← polar_smul_right _ (bm.repr x <| Prod.snd _),
      ← polar_smul_left _ (bm.repr x <| Prod.fst _)]
    /-
      case H
      ι : Type u_4
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      Q : QuadraticMap R M N
      bm : Basis ι R M
      x : M
      ⊢ Eq (HAdd.hAdd ((bm.repr x).support.sum fun x_1 => HSMul.hSMul ((bm.repr x) x …
    -/
    simp_rw [QuadraticMap.map_smul, mul_smul, Finset.sum_sym2_filter_not_isDiag]
    /-
      case H
      ι : Type u_4
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      Q : QuadraticMap R M N
      bm : Basis ι R M
      x : M
      ⊢ Eq (HAdd.hAdd ((bm.repr x).support.sum fun x_1 => HSMul.hSMul ((bm.repr x) x …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case H.h
      ι : Type u_4
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      Q : QuadraticMap R M N
      bm : Basis ι R M
      x : M
      ⊢ ∀ (x_1 : Prod ι ι), Membership.mem (bm.repr x).support.offDiag x_1 → Not (Eq …
    -/
  · intro x hx
    /-
      case H.h
      ι : Type u_4
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      Q : QuadraticMap R M N
      bm : Basis ι R M
      x✝ : M
      x : Prod ι ι
      hx : Membership.mem (bm.repr x✝).support.offDiag x
      ⊢ Not (Eq x.1 x.2)
    -/
    rw [Finset.mem_offDiag] at hx
    /-
      case H.h
      ι : Type u_4
      R : Type u_1
      M : Type u_2
      N : Type u_3
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      Q : QuadraticMap R M N
      bm : Basis ι R M
      x✝ : M
      x : Prod ι ι
      hx : And (Membership.mem (bm.repr x✝).support x.1) (And (Membership.mem (bm.re …
      ⊢ Not (Eq x.1 x.2)
    -/
    simpa using hx.2.2
    /-
      🎉 no goals
    -/


/-- From a free module, every quadratic map can be built from a bilinear form.

See `BilinMap.not_forall_toQuadraticMap_surjective` for a counterexample when the module is
not free. -/
theorem _root_.LinearMap.BilinMap.toQuadraticMap_surjective [Module.Free R M] :
    Function.Surjective (LinearMap.BilinMap.toQuadraticMap : LinearMap.BilinMap R M N → _) := by
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module.Free R M
    ⊢ Function.Surjective LinearMap.BilinMap.toQuadraticMap
  -/
  intro Q
  /-
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module.Free R M
    Q : QuadraticMap R M N
    ⊢ Exists fun a => Eq a.toQuadraticMap Q
  -/
  obtain ⟨ι, b⟩ := Module.Free.exists_basis (R := R) (M := M)
  /-
    case intro.mk
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module.Free R M
    Q : QuadraticMap R M N
    ι : Type u_2
    b : Basis ι R M
    ⊢ Exists fun a => Eq a.toQuadraticMap Q
  -/
  letI : LinearOrder ι := IsWellOrder.linearOrder WellOrderingRel
  /-
    case intro.mk
    R : Type u_1
    M : Type u_2
    N : Type u_3
    inst✝⁵ : CommRing R
    inst✝⁴ : AddCommGroup M
    inst✝³ : AddCommGroup N
    inst✝² : Module R M
    inst✝¹ : Module R N
    inst✝ : Module.Free R M
    Q : QuadraticMap R M N
    ι : Type u_2
    b : Basis ι R M
    this : LinearOrder ι := IsWellOrder.linearOrder WellOrderingRel
    ⊢ Exists fun a => Eq a.toQuadraticMap Q
  -/
  exact ⟨_, toQuadraticMap_toBilin _ b⟩
  /-
    🎉 no goals
  -/


@[simp]
lemma add_toBilin (bm : Basis ι R M) (Q₁ Q₂ : QuadraticMap R M N) :
    (Q₁ + Q₂).toBilin bm = Q₁.toBilin bm + Q₂.toBilin bm := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    bm : Basis ι R M
    Q₁ Q₂ : QuadraticMap R M N
    ⊢ Eq ((HAdd.hAdd Q₁ Q₂).toBilin bm) (HAdd.hAdd (Q₁.toBilin bm) (Q₂.toBilin bm))
  -/
  refine bm.ext fun i => bm.ext fun j => ?_
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁵ : LinearOrder ι
    inst✝⁴ : CommRing R
    inst✝³ : AddCommGroup M
    inst✝² : AddCommGroup N
    inst✝¹ : Module R M
    inst✝ : Module R N
    bm : Basis ι R M
    Q₁ Q₂ : QuadraticMap R M N
    i j : ι
    ⊢ Eq ((((HAdd.hAdd Q₁ Q₂).toBilin bm) (bm i)) (bm j)) (((HAdd.hAdd (Q₁.toBilin …
  -/
  obtain h | rfl | h := lt_trichotomy i j
    /-
      case inl
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      bm : Basis ι R M
      Q₁ Q₂ : QuadraticMap R M N
      i j : ι
      h : LT.lt i j
      ⊢ Eq ((((HAdd.hAdd Q₁ Q₂).toBilin bm) (bm i)) (bm j)) (((HAdd.hAdd (Q₁.toBilin …
    -/
  · simp [h.ne, h, toBilin_apply, polar_add]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      bm : Basis ι R M
      Q₁ Q₂ : QuadraticMap R M N
      i : ι
      ⊢ Eq ((((HAdd.hAdd Q₁ Q₂).toBilin bm) (bm i)) (bm i)) (((HAdd.hAdd (Q₁.toBilin …
    -/
  · simp [toBilin_apply]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁵ : LinearOrder ι
      inst✝⁴ : CommRing R
      inst✝³ : AddCommGroup M
      inst✝² : AddCommGroup N
      inst✝¹ : Module R M
      inst✝ : Module R N
      bm : Basis ι R M
      Q₁ Q₂ : QuadraticMap R M N
      i j : ι
      h : LT.lt j i
      ⊢ Eq ((((HAdd.hAdd Q₁ Q₂).toBilin bm) (bm i)) (bm j)) (((HAdd.hAdd (Q₁.toBilin …
    -/
  · simp [h.ne', h.not_lt, toBilin_apply, polar_add]
    /-
      🎉 no goals
    -/


@[simp]
lemma smul_toBilin (bm : Basis ι R M) (s : S) (Q : QuadraticMap R M N) :
    (s • Q).toBilin bm = s • Q.toBilin bm := by
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁹ : LinearOrder ι
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : Module R M
    inst✝⁴ : Module R N
    S : Type u_5
    inst✝³ : CommSemiring S
    inst✝² : Algebra S R
    inst✝¹ : Module S N
    inst✝ : IsScalarTower S R N
    bm : Basis ι R M
    s : S
    Q : QuadraticMap R M N
    ⊢ Eq ((HSMul.hSMul s Q).toBilin bm) (HSMul.hSMul s (Q.toBilin bm))
  -/
  refine bm.ext fun i => bm.ext fun j => ?_
  /-
    ι : Type u_1
    R : Type u_2
    M : Type u_3
    N : Type u_4
    inst✝⁹ : LinearOrder ι
    inst✝⁸ : CommRing R
    inst✝⁷ : AddCommGroup M
    inst✝⁶ : AddCommGroup N
    inst✝⁵ : Module R M
    inst✝⁴ : Module R N
    S : Type u_5
    inst✝³ : CommSemiring S
    inst✝² : Algebra S R
    inst✝¹ : Module S N
    inst✝ : IsScalarTower S R N
    bm : Basis ι R M
    s : S
    Q : QuadraticMap R M N
    i j : ι
    ⊢ Eq ((((HSMul.hSMul s Q).toBilin bm) (bm i)) (bm j)) (((HSMul.hSMul s (Q.toBi …
  -/
  obtain h | rfl | h := lt_trichotomy i j
    /-
      case inl
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁹ : LinearOrder ι
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      S : Type u_5
      inst✝³ : CommSemiring S
      inst✝² : Algebra S R
      inst✝¹ : Module S N
      inst✝ : IsScalarTower S R N
      bm : Basis ι R M
      s : S
      Q : QuadraticMap R M N
      i j : ι
      h : LT.lt i j
      ⊢ Eq ((((HSMul.hSMul s Q).toBilin bm) (bm i)) (bm j)) (((HSMul.hSMul s (Q.toBi …
    -/
  · simp [h.ne, h, toBilin_apply, polar_smul]
    /-
      🎉 no goals
    -/
    /-
      case inr.inl
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁹ : LinearOrder ι
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      S : Type u_5
      inst✝³ : CommSemiring S
      inst✝² : Algebra S R
      inst✝¹ : Module S N
      inst✝ : IsScalarTower S R N
      bm : Basis ι R M
      s : S
      Q : QuadraticMap R M N
      i : ι
      ⊢ Eq ((((HSMul.hSMul s Q).toBilin bm) (bm i)) (bm i)) (((HSMul.hSMul s (Q.toBi …
    -/
  · simp [toBilin_apply]
    /-
      🎉 no goals
    -/
    /-
      case inr.inr
      ι : Type u_1
      R : Type u_2
      M : Type u_3
      N : Type u_4
      inst✝⁹ : LinearOrder ι
      inst✝⁸ : CommRing R
      inst✝⁷ : AddCommGroup M
      inst✝⁶ : AddCommGroup N
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      S : Type u_5
      inst✝³ : CommSemiring S
      inst✝² : Algebra S R
      inst✝¹ : Module S N
      inst✝ : IsScalarTower S R N
      bm : Basis ι R M
      s : S
      Q : QuadraticMap R M N
      i j : ι
      h : LT.lt j i
      ⊢ Eq ((((HSMul.hSMul s Q).toBilin bm) (bm i)) (bm j)) (((HSMul.hSMul s (Q.toBi …
    -/
  · simp [h.ne', h.not_lt, toBilin_apply]
    /-
      🎉 no goals
    -/


/-- `QuadraticMap.toBilin` as an S-linear map -/
@[simps]
noncomputable def toBilinHom (bm : Basis ι R M) : QuadraticMap R M N →ₗ[S] BilinMap R M N where
  toFun Q := Q.toBilin bm
  map_add' := add_toBilin bm
  map_smul' := smul_toBilin S bm


