/-- The rank of a matrix is the rank of its image. -/
noncomputable def rank (A : Matrix m n R) : ℕ :=
  finrank R <| LinearMap.range A.mulVecLin


@[simp]
theorem rank_one [StrongRankCondition R] [DecidableEq n] :
    rank (1 : Matrix n n R) = Fintype.card n := by
  /-
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : StrongRankCondition R
    inst✝ : DecidableEq n
    ⊢ Eq (Matrix.rank 1) (Fintype.card n)
  -/
  rw [rank, mulVecLin_one, LinearMap.range_id, finrank_top, finrank_pi]
  /-
    🎉 no goals
  -/


@[simp]
theorem rank_zero [Nontrivial R] : rank (0 : Matrix m n R) = 0 := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : CommRing R
    inst✝ : Nontrivial R
    ⊢ Eq (Matrix.rank 0) 0
  -/
  rw [rank, mulVecLin_zero, LinearMap.range_zero, finrank_bot]
  /-
    🎉 no goals
  -/


theorem rank_le_card_width [StrongRankCondition R] (A : Matrix m n R) :
    A.rank ≤ Fintype.card n := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : CommRing R
    inst✝ : StrongRankCondition R
    A : Matrix m n R
    ⊢ LE.le A.rank (Fintype.card n)
  -/
  haveI : Module.Finite R (n → R) := Module.Finite.pi
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : CommRing R
    inst✝ : StrongRankCondition R
    A : Matrix m n R
    this : Module.Finite R (n → R)
    ⊢ LE.le A.rank (Fintype.card n)
  -/
  haveI : Module.Free R (n → R) := Module.Free.pi _ _
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : CommRing R
    inst✝ : StrongRankCondition R
    A : Matrix m n R
    this✝ : Module.Finite R (n → R)
    this : Module.Free R (n → R)
    ⊢ LE.le A.rank (Fintype.card n)
  -/
  exact A.mulVecLin.finrank_range_le.trans_eq (finrank_pi _)
  /-
    🎉 no goals
  -/


theorem rank_le_width [StrongRankCondition R] {m n : ℕ} (A : Matrix (Fin m) (Fin n) R) :
    A.rank ≤ n :=
  A.rank_le_card_width.trans <| (Fintype.card_fin n).le


theorem rank_mul_le_left [StrongRankCondition R] (A : Matrix m n R) (B : Matrix n o R) :
    (A * B).rank ≤ A.rank := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : Fintype o
    inst✝¹ : CommRing R
    inst✝ : StrongRankCondition R
    A : Matrix m n R
    B : Matrix n o R
    ⊢ LE.le (HMul.hMul A B).rank A.rank
  -/
  rw [rank, rank, mulVecLin_mul]
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : Fintype o
    inst✝¹ : CommRing R
    inst✝ : StrongRankCondition R
    A : Matrix m n R
    B : Matrix n o R
    ⊢ LE.le (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.range (A …
  -/
  exact Cardinal.toNat_le_toNat (LinearMap.rank_comp_le_left _ _) (rank_lt_aleph0 _ _)
  /-
    🎉 no goals
  -/


theorem rank_mul_le_right [StrongRankCondition R] (A : Matrix m n R) (B : Matrix n o R) :
    (A * B).rank ≤ B.rank := by
  /-
    m : Type u_2
    n : Type u_3
    o : Type u_4
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : Fintype o
    inst✝¹ : CommRing R
    inst✝ : StrongRankCondition R
    A : Matrix m n R
    B : Matrix n o R
    ⊢ LE.le (HMul.hMul A B).rank B.rank
  -/
  rw [rank, rank, mulVecLin_mul]
  exact finrank_le_finrank_of_rank_le_rank (LinearMap.lift_rank_comp_le_right _ _)
    (rank_lt_aleph0 _ _)


theorem rank_mul_le [StrongRankCondition R] (A : Matrix m n R) (B : Matrix n o R) :
    (A * B).rank ≤ min A.rank B.rank :=
  le_min (rank_mul_le_left _ _) (rank_mul_le_right _ _)


theorem rank_unit [StrongRankCondition R] [DecidableEq n] (A : (Matrix n n R)ˣ) :
    (A : Matrix n n R).rank = Fintype.card n := by
  /-
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : StrongRankCondition R
    inst✝ : DecidableEq n
    A : Units (Matrix n n R)
    ⊢ Eq (↑A).rank (Fintype.card n)
  -/
  apply le_antisymm (rank_le_card_width (A : Matrix n n R)) _
  /-
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : StrongRankCondition R
    inst✝ : DecidableEq n
    A : Units (Matrix n n R)
    ⊢ LE.le (Fintype.card n) (↑A).rank
  -/
  have := rank_mul_le_left (A : Matrix n n R) (↑A⁻¹ : Matrix n n R)
  /-
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : StrongRankCondition R
    inst✝ : DecidableEq n
    A : Units (Matrix n n R)
    this : LE.le (HMul.hMul ↑A ↑(Inv.inv A)).rank (↑A).rank
    ⊢ LE.le (Fintype.card n) (↑A).rank
  -/
  rwa [← Units.val_mul, mul_inv_cancel, Units.val_one, rank_one] at this
  /-
    🎉 no goals
  -/


theorem rank_of_isUnit [StrongRankCondition R] [DecidableEq n] (A : Matrix n n R) (h : IsUnit A) :
    A.rank = Fintype.card n := by
  /-
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : StrongRankCondition R
    inst✝ : DecidableEq n
    A : Matrix n n R
    h : IsUnit A
    ⊢ Eq A.rank (Fintype.card n)
  -/
  obtain ⟨A, rfl⟩ := h
  /-
    case intro
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : StrongRankCondition R
    inst✝ : DecidableEq n
    A : Units (Matrix n n R)
    ⊢ Eq (↑A).rank (Fintype.card n)
  -/
  exact rank_unit A
  /-
    🎉 no goals
  -/


/-- Right multiplying by an invertible matrix does not change the rank -/
@[simp]
lemma rank_mul_eq_left_of_isUnit_det [DecidableEq n]
    (A : Matrix n n R) (B : Matrix m n R) (hA : IsUnit A.det) :
    (B * A).rank = B.rank := by
  suffices Function.Surjective A.mulVecLin by
    rw [rank, mulVecLin_mul, LinearMap.range_comp_of_range_eq_top _
      (LinearMap.range_eq_top.mpr this), ← rank]
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : CommRing R
    inst✝ : DecidableEq n
    A : Matrix n n R
    B : Matrix m n R
    hA : IsUnit A.det
    ⊢ Function.Surjective ⇑A.mulVecLin
  -/
  intro v
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : CommRing R
    inst✝ : DecidableEq n
    A : Matrix n n R
    B : Matrix m n R
    hA : IsUnit A.det
    v : n → R
    ⊢ Exists fun a => Eq (A.mulVecLin a) v
  -/
  exact ⟨(A⁻¹).mulVecLin v, by simp [mul_nonsing_inv _ hA]⟩
  /-
    🎉 no goals
  -/


/-- Left multiplying by an invertible matrix does not change the rank -/
@[simp]
lemma rank_mul_eq_right_of_isUnit_det [Fintype m] [DecidableEq m]
    (A : Matrix m m R) (B : Matrix m n R) (hA : IsUnit A.det) :
    (A * B).rank = B.rank := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix m m R
    B : Matrix m n R
    hA : IsUnit A.det
    ⊢ Eq (HMul.hMul A B).rank B.rank
  -/
  let b : Basis m R (m → R) := Pi.basisFun R m
  replace hA : IsUnit (LinearMap.toMatrix b b A.mulVecLin).det := by
    convert hA; rw [← LinearEquiv.eq_symm_apply]; rfl
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix m m R
    B : Matrix m n R
    b : Basis m R (m → R) := Pi.basisFun R m
    hA : IsUnit ((LinearMap.toMatrix b b) A.mulVecLin).det
    ⊢ Eq (HMul.hMul A B).rank B.rank
  -/
  have hAB : mulVecLin (A * B) = (LinearEquiv.ofIsUnitDet hA).comp (mulVecLin B) := by ext; simp
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : Fintype m
    inst✝ : DecidableEq m
    A : Matrix m m R
    B : Matrix m n R
    b : Basis m R (m → R) := Pi.basisFun R m
    hA : IsUnit ((LinearMap.toMatrix b b) A.mulVecLin).det
    hAB : Eq (HMul.hMul A B).mulVecLin ((↑(LinearEquiv.ofIsUnitDet hA)).comp B.mul …
    ⊢ Eq (HMul.hMul A B).rank B.rank
  -/
  rw [rank, rank, hAB, LinearMap.range_comp, LinearEquiv.finrank_map_eq]
  /-
    🎉 no goals
  -/


/-- Taking a subset of the rows and permuting the columns reduces the rank. -/
theorem rank_submatrix_le [StrongRankCondition R] [Fintype m] (f : n → m) (e : n ≃ m)
    (A : Matrix m m R) : rank (A.submatrix f e) ≤ rank A := by
  rw [rank, rank, mulVecLin_submatrix, LinearMap.range_comp, LinearMap.range_comp,
    show LinearMap.funLeft R R e.symm = LinearEquiv.funCongrLeft R R e.symm from rfl,
    LinearEquiv.range, Submodule.map_top]
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : StrongRankCondition R
    inst✝ : Fintype m
    f : n → m
    e : Equiv n m
    A : Matrix m m R
    ⊢ LE.le (Module.finrank R (Subtype fun x => Membership.mem (Submodule.map (Lin …
  -/
  exact Submodule.finrank_map_le _ _
  /-
    🎉 no goals
  -/


theorem rank_reindex [Fintype m] (e₁ e₂ : m ≃ n) (A : Matrix m m R) :
    rank (reindex e₁ e₂ A) = rank A := by
  rw [rank, rank, mulVecLin_reindex, LinearMap.range_comp, LinearMap.range_comp,
    LinearEquiv.range, Submodule.map_top, LinearEquiv.finrank_map_eq]


@[simp]
theorem rank_submatrix [Fintype m] (A : Matrix m m R) (e₁ e₂ : n ≃ m) :
    rank (A.submatrix e₁ e₂) = rank A := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : CommRing R
    inst✝ : Fintype m
    A : Matrix m m R
    e₁ e₂ : Equiv n m
    ⊢ Eq (A.submatrix ⇑e₁ ⇑e₂).rank A.rank
  -/
  simpa only [reindex_apply] using rank_reindex e₁.symm e₂.symm A
  /-
    🎉 no goals
  -/


theorem rank_eq_finrank_range_toLin [Finite m] [DecidableEq n] {M₁ M₂ : Type*} [AddCommGroup M₁]
    [AddCommGroup M₂] [Module R M₁] [Module R M₂] (A : Matrix m n R) (v₁ : Basis m R M₁)
    (v₂ : Basis n R M₂) : A.rank = finrank R (LinearMap.range (toLin v₂ v₁ A)) := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    ⊢ Eq A.rank (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.rang …
  -/
  cases nonempty_fintype m
  /-
    case intro
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    ⊢ Eq A.rank (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.rang …
  -/
  let e₁ := (Pi.basisFun R m).equiv v₁ (Equiv.refl _)
  /-
    case intro
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    ⊢ Eq A.rank (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.rang …
  -/
  let e₂ := (Pi.basisFun R n).equiv v₂ (Equiv.refl _)
  have range_e₂ : LinearMap.range e₂ = ⊤ := by
    rw [LinearMap.range_eq_top]
    exact e₂.surjective
  /-
    case intro
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    e₂ : LinearEquiv (RingHom.id R) (n → R) M₂ := (Pi.basisFun R n).equiv v₂ (Equi …
    range_e₂ : Eq (LinearMap.range e₂) Top.top
    ⊢ Eq A.rank (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.rang …
  -/
  refine LinearEquiv.finrank_eq (e₁.ofSubmodules _ _ ?_)
  /-
    case intro
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    e₂ : LinearEquiv (RingHom.id R) (n → R) M₂ := (Pi.basisFun R n).equiv v₂ (Equi …
    range_e₂ : Eq (LinearMap.range e₂) Top.top
    ⊢ Eq (Submodule.map (↑e₁) (LinearMap.range A.mulVecLin)) (LinearMap.range ((Ma …
  -/
  rw [← LinearMap.range_comp, ← LinearMap.range_comp_of_range_eq_top (toLin v₂ v₁ A) range_e₂]
  /-
    case intro
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    e₂ : LinearEquiv (RingHom.id R) (n → R) M₂ := (Pi.basisFun R n).equiv v₂ (Equi …
    range_e₂ : Eq (LinearMap.range e₂) Top.top
    ⊢ Eq (LinearMap.range ((↑e₁).comp A.mulVecLin)) (LinearMap.range (((Matrix.toL …
  -/
  congr 1
  /-
    case intro.e_f
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    e₂ : LinearEquiv (RingHom.id R) (n → R) M₂ := (Pi.basisFun R n).equiv v₂ (Equi …
    range_e₂ : Eq (LinearMap.range e₂) Top.top
    ⊢ Eq ((↑e₁).comp A.mulVecLin) (((Matrix.toLin v₂ v₁) A).comp ↑e₂)
  -/
  apply LinearMap.pi_ext'
  /-
    case intro.e_f.h
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    e₂ : LinearEquiv (RingHom.id R) (n → R) M₂ := (Pi.basisFun R n).equiv v₂ (Equi …
    range_e₂ : Eq (LinearMap.range e₂) Top.top
    ⊢ ∀ (i : n), Eq (((↑e₁).comp A.mulVecLin).comp (LinearMap.single R (fun i => R …
  -/
  rintro i
  /-
    case intro.e_f.h
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    e₂ : LinearEquiv (RingHom.id R) (n → R) M₂ := (Pi.basisFun R n).equiv v₂ (Equi …
    range_e₂ : Eq (LinearMap.range e₂) Top.top
    i : n
    ⊢ Eq (((↑e₁).comp A.mulVecLin).comp (LinearMap.single R (fun i => R) i)) ((((M …
  -/
  apply LinearMap.ext_ring
  /-
    case intro.e_f.h.h
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    e₂ : LinearEquiv (RingHom.id R) (n → R) M₂ := (Pi.basisFun R n).equiv v₂ (Equi …
    range_e₂ : Eq (LinearMap.range e₂) Top.top
    i : n
    ⊢ Eq ((((↑e₁).comp A.mulVecLin).comp (LinearMap.single R (fun i => R) i)) 1) ( …
  -/
  have aux₁ := toLin_self (Pi.basisFun R n) (Pi.basisFun R m) A i
  /-
    case intro.e_f.h.h
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    e₂ : LinearEquiv (RingHom.id R) (n → R) M₂ := (Pi.basisFun R n).equiv v₂ (Equi …
    range_e₂ : Eq (LinearMap.range e₂) Top.top
    i : n
    aux₁ : Eq (((Matrix.toLin (Pi.basisFun R n) (Pi.basisFun R m)) A) ((Pi.basisFu …
    ⊢ Eq ((((↑e₁).comp A.mulVecLin).comp (LinearMap.single R (fun i => R) i)) 1) ( …
  -/
  have aux₂ := Basis.equiv_apply (Pi.basisFun R n) i v₂
  /-
    case intro.e_f.h.h
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    e₂ : LinearEquiv (RingHom.id R) (n → R) M₂ := (Pi.basisFun R n).equiv v₂ (Equi …
    range_e₂ : Eq (LinearMap.range e₂) Top.top
    i : n
    aux₁ : Eq (((Matrix.toLin (Pi.basisFun R n) (Pi.basisFun R m)) A) ((Pi.basisFu …
    aux₂ : ∀ (e : Equiv n n), Eq (((Pi.basisFun R n).equiv v₂ e) ((Pi.basisFun R n …
    ⊢ Eq ((((↑e₁).comp A.mulVecLin).comp (LinearMap.single R (fun i => R) i)) 1) ( …
  -/
  rw [toLin_eq_toLin', toLin'_apply'] at aux₁
  /-
    case intro.e_f.h.h
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁷ : Fintype n
    inst✝⁶ : CommRing R
    inst✝⁵ : Finite m
    inst✝⁴ : DecidableEq n
    M₁ : Type u_6
    M₂ : Type u_7
    inst✝³ : AddCommGroup M₁
    inst✝² : AddCommGroup M₂
    inst✝¹ : Module R M₁
    inst✝ : Module R M₂
    A : Matrix m n R
    v₁ : Basis m R M₁
    v₂ : Basis n R M₂
    val✝ : Fintype m
    e₁ : LinearEquiv (RingHom.id R) (m → R) M₁ := (Pi.basisFun R m).equiv v₁ (Equi …
    e₂ : LinearEquiv (RingHom.id R) (n → R) M₂ := (Pi.basisFun R n).equiv v₂ (Equi …
    range_e₂ : Eq (LinearMap.range e₂) Top.top
    i : n
    aux₁ : Eq (A.mulVecLin ((Pi.basisFun R n) i)) (Finset.univ.sum fun j => HSMul. …
    aux₂ : ∀ (e : Equiv n n), Eq (((Pi.basisFun R n).equiv v₂ e) ((Pi.basisFun R n …
    ⊢ Eq ((((↑e₁).comp A.mulVecLin).comp (LinearMap.single R (fun i => R) i)) 1) ( …
  -/
  rw [Pi.basisFun_apply] at aux₁ aux₂
  simp only [e₁, e₂, LinearMap.comp_apply, LinearEquiv.coe_coe, Equiv.refl_apply,
    aux₁, aux₂, LinearMap.coe_single, toLin_self, map_sum, LinearEquiv.map_smul, Basis.equiv_apply]


theorem rank_le_card_height [Fintype m] [StrongRankCondition R] (A : Matrix m n R) :
    A.rank ≤ Fintype.card m := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : Fintype m
    inst✝ : StrongRankCondition R
    A : Matrix m n R
    ⊢ LE.le A.rank (Fintype.card m)
  -/
  haveI : Module.Finite R (m → R) := Module.Finite.pi
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : Fintype m
    inst✝ : StrongRankCondition R
    A : Matrix m n R
    this : Module.Finite R (m → R)
    ⊢ LE.le A.rank (Fintype.card m)
  -/
  haveI : Module.Free R (m → R) := Module.Free.pi _ _
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝³ : Fintype n
    inst✝² : CommRing R
    inst✝¹ : Fintype m
    inst✝ : StrongRankCondition R
    A : Matrix m n R
    this✝ : Module.Finite R (m → R)
    this : Module.Free R (m → R)
    ⊢ LE.le A.rank (Fintype.card m)
  -/
  exact (Submodule.finrank_le _).trans (finrank_pi R).le
  /-
    🎉 no goals
  -/


theorem rank_le_height [StrongRankCondition R] {m n : ℕ} (A : Matrix (Fin m) (Fin n) R) :
    A.rank ≤ m :=
  A.rank_le_card_height.trans <| (Fintype.card_fin m).le


/-- The rank of a matrix is the rank of the space spanned by its columns. -/
theorem rank_eq_finrank_span_cols (A : Matrix m n R) :
                                                               /-
                                                                 m : Type u_2
                                                                 n : Type u_3
                                                                 R : Type u_5
                                                                 inst✝¹ : Fintype n
                                                                 inst✝ : CommRing R
                                                                 A : Matrix m n R
                                                                 ⊢ Eq A.rank (Module.finrank R (Subtype fun x => Membership.mem (Submodule.span …
                                                               -/
    A.rank = finrank R (Submodule.span R (Set.range Aᵀ)) := by rw [rank, Matrix.range_mulVecLin]
                                                               /-
                                                                 🎉 no goals
                                                               -/


/-- The rank of a diagonal matrix is the count of non-zero elements on its main diagonal -/
theorem rank_diagonal [Fintype m] [DecidableEq m] [DecidableEq R] (w : m → R) :
    (diagonal w).rank = Fintype.card {i // (w i) ≠ 0} := by
  rw [Matrix.rank, ← Matrix.toLin'_apply', Module.finrank, ← LinearMap.rank,
    LinearMap.rank_diagonal, Cardinal.toNat_natCast]


theorem ker_mulVecLin_conjTranspose_mul_self (A : Matrix m n R) :
    LinearMap.ker (Aᴴ * A).mulVecLin = LinearMap.ker (mulVecLin A) := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : Field R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    A : Matrix m n R
    ⊢ Eq (LinearMap.ker (HMul.hMul A.conjTranspose A).mulVecLin) (LinearMap.ker A. …
  -/
  ext x
  /-
    case h
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : Field R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    A : Matrix m n R
    x : n → R
    ⊢ Iff (Membership.mem (LinearMap.ker (HMul.hMul A.conjTranspose A).mulVecLin)  …
  -/
  simp only [LinearMap.mem_ker, mulVecLin_apply, conjTranspose_mul_self_mulVec_eq_zero]
  /-
    🎉 no goals
  -/


theorem rank_conjTranspose_mul_self (A : Matrix m n R) : (Aᴴ * A).rank = A.rank := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : Field R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    A : Matrix m n R
    ⊢ Eq (HMul.hMul A.conjTranspose A).rank A.rank
  -/
  dsimp only [rank]
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : Field R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    A : Matrix m n R
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.range (HMul …
  -/
  refine add_left_injective (finrank R (LinearMap.ker (mulVecLin A))) ?_
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝⁵ : Fintype n
    inst✝⁴ : Fintype m
    inst✝³ : Field R
    inst✝² : PartialOrder R
    inst✝¹ : StarRing R
    inst✝ : StarOrderedRing R
    A : Matrix m n R
    ⊢ Eq ((fun x => HAdd.hAdd x (Module.finrank R (Subtype fun x => Membership.mem …
  -/
  dsimp only
  trans finrank R { x // x ∈ LinearMap.range (mulVecLin (Aᴴ * A)) } +
    finrank R { x // x ∈ LinearMap.ker (mulVecLin (Aᴴ * A)) }
    /-
      m : Type u_2
      n : Type u_3
      R : Type u_5
      inst✝⁵ : Fintype n
      inst✝⁴ : Fintype m
      inst✝³ : Field R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : StarOrderedRing R
      A : Matrix m n R
      ⊢ Eq (HAdd.hAdd (Module.finrank R (Subtype fun x => Membership.mem (LinearMap. …
    -/
  · rw [ker_mulVecLin_conjTranspose_mul_self]
    /-
      🎉 no goals
    -/
    /-
      m : Type u_2
      n : Type u_3
      R : Type u_5
      inst✝⁵ : Fintype n
      inst✝⁴ : Fintype m
      inst✝³ : Field R
      inst✝² : PartialOrder R
      inst✝¹ : StarRing R
      inst✝ : StarOrderedRing R
      A : Matrix m n R
      ⊢ Eq (HAdd.hAdd (Module.finrank R (Subtype fun x => Membership.mem (LinearMap. …
    -/
  · simp only [LinearMap.finrank_range_add_finrank_ker]
    /-
      🎉 no goals
    -/

-- this follows the proof here https://math.stackexchange.com/a/81903/1896

/-- TODO: prove this in greater generality. -/
@[simp]
theorem rank_conjTranspose (A : Matrix m n R) : Aᴴ.rank = A.rank :=
  le_antisymm
    (((rank_conjTranspose_mul_self _).symm.trans_le <| rank_mul_le_left _ _).trans_eq <|
      congr_arg _ <| conjTranspose_conjTranspose _)
    ((rank_conjTranspose_mul_self _).symm.trans_le <| rank_mul_le_left _ _)


@[simp]
theorem rank_self_mul_conjTranspose (A : Matrix m n R) : (A * Aᴴ).rank = A.rank := by
  simpa only [rank_conjTranspose, conjTranspose_conjTranspose] using
    rank_conjTranspose_mul_self Aᴴ


theorem ker_mulVecLin_transpose_mul_self (A : Matrix m n R) :
    LinearMap.ker (Aᵀ * A).mulVecLin = LinearMap.ker (mulVecLin A) := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : Fintype m
    inst✝ : LinearOrderedField R
    A : Matrix m n R
    ⊢ Eq (LinearMap.ker (HMul.hMul A.transpose A).mulVecLin) (LinearMap.ker A.mulV …
  -/
  ext x
  /-
    case h
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : Fintype m
    inst✝ : LinearOrderedField R
    A : Matrix m n R
    x : n → R
    ⊢ Iff (Membership.mem (LinearMap.ker (HMul.hMul A.transpose A).mulVecLin) x) ( …
  -/
  simp only [LinearMap.mem_ker, mulVecLin_apply, ← mulVec_mulVec]
  /-
    case h
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : Fintype m
    inst✝ : LinearOrderedField R
    A : Matrix m n R
    x : n → R
    ⊢ Iff (Eq (A.transpose.mulVec (A.mulVec x)) 0) (Eq (A.mulVec x) 0)
  -/
  constructor
    /-
      case h.mp
      m : Type u_2
      n : Type u_3
      R : Type u_5
      inst✝² : Fintype n
      inst✝¹ : Fintype m
      inst✝ : LinearOrderedField R
      A : Matrix m n R
      x : n → R
      ⊢ Eq (A.transpose.mulVec (A.mulVec x)) 0 → Eq (A.mulVec x) 0
    -/
  · intro h
    /-
      case h.mp
      m : Type u_2
      n : Type u_3
      R : Type u_5
      inst✝² : Fintype n
      inst✝¹ : Fintype m
      inst✝ : LinearOrderedField R
      A : Matrix m n R
      x : n → R
      h : Eq (A.transpose.mulVec (A.mulVec x)) 0
      ⊢ Eq (A.mulVec x) 0
    -/
    replace h := congr_arg (dotProduct x) h
    /-
      case h.mp
      m : Type u_2
      n : Type u_3
      R : Type u_5
      inst✝² : Fintype n
      inst✝¹ : Fintype m
      inst✝ : LinearOrderedField R
      A : Matrix m n R
      x : n → R
      h : Eq (dotProduct x (A.transpose.mulVec (A.mulVec x))) (dotProduct x 0)
      ⊢ Eq (A.mulVec x) 0
    -/
    rwa [dotProduct_mulVec, dotProduct_zero, vecMul_transpose, dotProduct_self_eq_zero] at h
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      m : Type u_2
      n : Type u_3
      R : Type u_5
      inst✝² : Fintype n
      inst✝¹ : Fintype m
      inst✝ : LinearOrderedField R
      A : Matrix m n R
      x : n → R
      ⊢ Eq (A.mulVec x) 0 → Eq (A.transpose.mulVec (A.mulVec x)) 0
    -/
  · intro h
    /-
      case h.mpr
      m : Type u_2
      n : Type u_3
      R : Type u_5
      inst✝² : Fintype n
      inst✝¹ : Fintype m
      inst✝ : LinearOrderedField R
      A : Matrix m n R
      x : n → R
      h : Eq (A.mulVec x) 0
      ⊢ Eq (A.transpose.mulVec (A.mulVec x)) 0
    -/
    rw [h, mulVec_zero]
    /-
      🎉 no goals
    -/


theorem rank_transpose_mul_self (A : Matrix m n R) : (Aᵀ * A).rank = A.rank := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : Fintype m
    inst✝ : LinearOrderedField R
    A : Matrix m n R
    ⊢ Eq (HMul.hMul A.transpose A).rank A.rank
  -/
  dsimp only [rank]
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : Fintype m
    inst✝ : LinearOrderedField R
    A : Matrix m n R
    ⊢ Eq (Module.finrank R (Subtype fun x => Membership.mem (LinearMap.range (HMul …
  -/
  refine add_left_injective (finrank R <| LinearMap.ker A.mulVecLin) ?_
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : Fintype m
    inst✝ : LinearOrderedField R
    A : Matrix m n R
    ⊢ Eq ((fun x => HAdd.hAdd x (Module.finrank R (Subtype fun x => Membership.mem …
  -/
  dsimp only
  trans finrank R { x // x ∈ LinearMap.range (mulVecLin (Aᵀ * A)) } +
    finrank R { x // x ∈ LinearMap.ker (mulVecLin (Aᵀ * A)) }
    /-
      m : Type u_2
      n : Type u_3
      R : Type u_5
      inst✝² : Fintype n
      inst✝¹ : Fintype m
      inst✝ : LinearOrderedField R
      A : Matrix m n R
      ⊢ Eq (HAdd.hAdd (Module.finrank R (Subtype fun x => Membership.mem (LinearMap. …
    -/
  · rw [ker_mulVecLin_transpose_mul_self]
    /-
      🎉 no goals
    -/
    /-
      m : Type u_2
      n : Type u_3
      R : Type u_5
      inst✝² : Fintype n
      inst✝¹ : Fintype m
      inst✝ : LinearOrderedField R
      A : Matrix m n R
      ⊢ Eq (HAdd.hAdd (Module.finrank R (Subtype fun x => Membership.mem (LinearMap. …
    -/
  · simp only [LinearMap.finrank_range_add_finrank_ker]
    /-
      🎉 no goals
    -/


@[simp]
theorem rank_transpose [Field R] [Fintype m] (A : Matrix m n R) : Aᵀ.rank = A.rank := by
  classical
  rw [Aᵀ.rank_eq_finrank_range_toLin (Pi.basisFun R n).dualBasis (Pi.basisFun R m).dualBasis,
      toLin_transpose, ← LinearMap.dualMap_def, LinearMap.finrank_range_dualMap_eq_finrank_range,
      toLin_eq_toLin', toLin'_apply', rank]


@[simp]
theorem rank_self_mul_transpose [LinearOrderedField R] [Fintype m] (A : Matrix m n R) :
    (A * Aᵀ).rank = A.rank := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : LinearOrderedField R
    inst✝ : Fintype m
    A : Matrix m n R
    ⊢ Eq (HMul.hMul A A.transpose).rank A.rank
  -/
  simpa only [rank_transpose, transpose_transpose] using rank_transpose_mul_self Aᵀ
  /-
    🎉 no goals
  -/


/-- The rank of a matrix is the rank of the space spanned by its rows. -/
theorem rank_eq_finrank_span_row [Field R] [Finite m] (A : Matrix m n R) :
    A.rank = finrank R (Submodule.span R (Set.range A)) := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : Field R
    inst✝ : Finite m
    A : Matrix m n R
    ⊢ Eq A.rank (Module.finrank R (Subtype fun x => Membership.mem (Submodule.span …
  -/
  cases nonempty_fintype m
  /-
    case intro
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : Field R
    inst✝ : Finite m
    A : Matrix m n R
    val✝ : Fintype m
    ⊢ Eq A.rank (Module.finrank R (Subtype fun x => Membership.mem (Submodule.span …
  -/
  rw [← rank_transpose, rank_eq_finrank_span_cols, transpose_transpose]
  /-
    🎉 no goals
  -/


theorem _root_.LinearIndependent.rank_matrix [Field R] [Fintype m]
    {M : Matrix m n R} (h : LinearIndependent R M) : M.rank = Fintype.card m := by
  /-
    m : Type u_2
    n : Type u_3
    R : Type u_5
    inst✝² : Fintype n
    inst✝¹ : Field R
    inst✝ : Fintype m
    M : Matrix m n R
    h : LinearIndependent R M
    ⊢ Eq M.rank (Fintype.card m)
  -/
  rw [M.rank_eq_finrank_span_row, linearIndependent_iff_card_eq_finrank_span.mp h, Set.finrank]
  /-
    🎉 no goals
  -/


lemma rank_add_rank_le_card_of_mul_eq_zero [Field R] [Finite l] [Fintype m]
    {A : Matrix l m R} {B : Matrix m n R} (hAB : A * B = 0) :
    A.rank + B.rank ≤ Fintype.card m := by
  classical
  let el : Basis l R (l → R) := Pi.basisFun R l
  let em : Basis m R (m → R) := Pi.basisFun R m
  let en : Basis n R (n → R) := Pi.basisFun R n
  rw [Matrix.rank_eq_finrank_range_toLin A el em,
      Matrix.rank_eq_finrank_range_toLin B em en,
      ← Module.finrank_fintype_fun_eq_card R,
      ← LinearMap.finrank_range_add_finrank_ker (Matrix.toLin em el A),
      add_le_add_iff_left]
  apply Submodule.finrank_mono
  rw [LinearMap.range_le_ker_iff, ← Matrix.toLin_mul, hAB, map_zero]


