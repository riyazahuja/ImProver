/-- Two submodules `M` and `N` in an algebra `S` over `R` are linearly disjoint if the natural map
`M ⊗[R] N →ₗ[R] S` induced by multiplication in `S` is injective. -/
@[mk_iff]
protected structure LinearDisjoint : Prop where
  injective : Function.Injective (mulMap M N)


/-- If `M` and `N` are linearly disjoint submodules, then there is the natural isomorphism
`M ⊗[R] N ≃ₗ[R] M * N` induced by multiplication in `S`. -/
protected def LinearDisjoint.mulMap (H : M.LinearDisjoint N) : M ⊗[R] N ≃ₗ[R] M * N :=
  LinearEquiv.ofInjective (M.mulMap N) H.injective ≪≫ₗ LinearEquiv.ofEq _ _ (mulMap_range M N)


@[simp]
theorem LinearDisjoint.val_mulMap_tmul (H : M.LinearDisjoint N) (m : M) (n : N) :
    (H.mulMap (m ⊗ₜ[R] n) : S) = m.1 * n.1 := rfl


@[nontriviality]
theorem LinearDisjoint.of_subsingleton [Subsingleton R] : M.LinearDisjoint N :=
  haveI : Subsingleton S := Module.subsingleton R S
  ⟨Function.injective_of_subsingleton _⟩


@[nontriviality]
theorem LinearDisjoint.of_subsingleton_top [Subsingleton S] : M.LinearDisjoint N :=
  ⟨Function.injective_of_subsingleton _⟩


/-- Linear disjointness is preserved by taking multiplicative opposite. -/
theorem linearDisjoint_op :
    M.LinearDisjoint N ↔ (equivOpposite.symm (MulOpposite.op N)).LinearDisjoint
      (equivOpposite.symm (MulOpposite.op M)) := by
  simp only [linearDisjoint_iff, mulMap_op, LinearMap.coe_comp,
    LinearEquiv.coe_coe, EquivLike.comp_injective, EquivLike.injective_comp]


alias ⟨LinearDisjoint.op, LinearDisjoint.of_op⟩ := linearDisjoint_op


/-- Linear disjointness is symmetric if elements in the module commute. -/
theorem LinearDisjoint.symm_of_commute (H : M.LinearDisjoint N)
    (hc : ∀ (m : M) (n : N), Commute m.1 n.1) : N.LinearDisjoint M := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hc : ∀ (m : Subtype fun x => Membership.mem M x) (n : Subtype fun x => Members …
    ⊢ N.LinearDisjoint M
  -/
  rw [linearDisjoint_iff, mulMap_comm_of_commute M N hc]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hc : ∀ (m : Subtype fun x => Membership.mem M x) (n : Subtype fun x => Members …
    ⊢ Function.Injective ⇑((M.mulMap N).comp ↑(TensorProduct.comm R (Subtype fun x …
  -/
  exact ((TensorProduct.comm R N M).toEquiv.injective_comp _).2 H.injective
  /-
    🎉 no goals
  -/


/-- Linear disjointness is symmetric if elements in the module commute. -/
theorem linearDisjoint_comm_of_commute
    (hc : ∀ (m : M) (n : N), Commute m.1 n.1) : M.LinearDisjoint N ↔ N.LinearDisjoint M :=
  ⟨fun H ↦ H.symm_of_commute hc, fun H ↦ H.symm_of_commute fun _ _ ↦ (hc _ _).symm⟩


/-- Linear disjointness is preserved by injective algebra homomorphisms. -/
theorem map (H : M.LinearDisjoint N) {T : Type w} [Semiring T] [Algebra R T]
    {F : Type*} [FunLike F S T] [AlgHomClass F R S T] (f : F) (hf : Function.Injective f) :
    (M.map f).LinearDisjoint (N.map f) := by
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    T : Type w
    inst✝³ : Semiring T
    inst✝² : Algebra R T
    F : Type u_1
    inst✝¹ : FunLike F S T
    inst✝ : AlgHomClass F R S T
    f : F
    hf : Function.Injective ⇑f
    ⊢ (Submodule.map f M).LinearDisjoint (Submodule.map f N)
  -/
  rw [linearDisjoint_iff] at H ⊢
  have : _ ∘ₗ
    (TensorProduct.congr (M.equivMapOfInjective f hf) (N.equivMapOfInjective f hf)).toLinearMap
      = _ := M.mulMap_map_comp_eq N f
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : Algebra R S
    M N : Submodule R S
    H : Function.Injective ⇑(M.mulMap N)
    T : Type w
    inst✝³ : Semiring T
    inst✝² : Algebra R T
    F : Type u_1
    inst✝¹ : FunLike F S T
    inst✝ : AlgHomClass F R S T
    f : F
    hf : Function.Injective ⇑f
    this : Eq (((Submodule.map f M).mulMap (Submodule.map f N)).comp ↑(TensorProdu …
    ⊢ Function.Injective ⇑((Submodule.map f M).mulMap (Submodule.map f N))
  -/
  replace H : Function.Injective ((f : S →ₗ[R] T) ∘ₗ mulMap M N) := hf.comp H
  /-
    R : Type u
    S : Type v
    inst✝⁶ : CommSemiring R
    inst✝⁵ : Semiring S
    inst✝⁴ : Algebra R S
    M N : Submodule R S
    T : Type w
    inst✝³ : Semiring T
    inst✝² : Algebra R T
    F : Type u_1
    inst✝¹ : FunLike F S T
    inst✝ : AlgHomClass F R S T
    f : F
    hf : Function.Injective ⇑f
    this : Eq (((Submodule.map f M).mulMap (Submodule.map f N)).comp ↑(TensorProdu …
    H : Function.Injective ⇑((↑f).comp (M.mulMap N))
    ⊢ Function.Injective ⇑((Submodule.map f M).mulMap (Submodule.map f N))
  -/
  simpa only [← this, LinearMap.coe_comp, LinearEquiv.coe_coe, EquivLike.injective_comp] using H
  /-
    🎉 no goals
  -/


/-- If `{ m_i }` is an `R`-basis of `M`, which is also `N`-linearly independent
(in this result it is stated as `Submodule.mulLeftMap` is injective),
then `M` and `N` are linearly disjoint. -/
theorem of_basis_left' {ι : Type*} (m : Basis ι R M)
    (H : Function.Injective (mulLeftMap N m)) : M.LinearDisjoint N := by
  classical simp_rw [mulLeftMap_eq_mulMap_comp, ← Basis.coe_repr_symm,
    ← LinearEquiv.coe_rTensor, LinearEquiv.comp_coe, LinearMap.coe_comp,
    LinearEquiv.coe_coe, EquivLike.injective_comp] at H
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    ι : Type u_1
    m : Basis ι R (Subtype fun x => Membership.mem M x)
    H : Function.Injective ⇑(M.mulMap N)
    ⊢ M.LinearDisjoint N
  -/
  exact ⟨H⟩
  /-
    🎉 no goals
  -/


/-- If `{ n_i }` is an `R`-basis of `N`, which is also `M`-linearly independent
(in this result it is stated as `Submodule.mulRightMap` is injective),
then `M` and `N` are linearly disjoint. -/
theorem of_basis_right' {ι : Type*} (n : Basis ι R N)
    (H : Function.Injective (mulRightMap M n)) : M.LinearDisjoint N := by
  classical simp_rw [mulRightMap_eq_mulMap_comp, ← Basis.coe_repr_symm,
    ← LinearEquiv.coe_lTensor, LinearEquiv.comp_coe, LinearMap.coe_comp,
    LinearEquiv.coe_coe, EquivLike.injective_comp] at H
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    ι : Type u_1
    n : Basis ι R (Subtype fun x => Membership.mem N x)
    H : Function.Injective ⇑(M.mulMap N)
    ⊢ M.LinearDisjoint N
  -/
  exact ⟨H⟩
  /-
    🎉 no goals
  -/


/-- If `{ m_i }` is an `R`-basis of `M`, if `{ n_i }` is an `R`-basis of `N`,
such that the family `{ m_i * n_j }` in `S` is `R`-linearly independent
(in this result it is stated as the relevant `Finsupp.linearCombination` is injective),
then `M` and `N` are linearly disjoint. -/
theorem of_basis_mul' {κ ι : Type*} (m : Basis κ R M) (n : Basis ι R N)
    (H : Function.Injective (Finsupp.linearCombination R fun i : κ × ι ↦ (m i.1 * n i.2 : S))) :
    M.LinearDisjoint N := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    κ : Type u_1
    ι : Type u_2
    m : Basis κ R (Subtype fun x => Membership.mem M x)
    n : Basis ι R (Subtype fun x => Membership.mem N x)
    H : Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i. …
    ⊢ M.LinearDisjoint N
  -/
  let i0 := (finsuppTensorFinsupp' R κ ι).symm
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    κ : Type u_1
    ι : Type u_2
    m : Basis κ R (Subtype fun x => Membership.mem M x)
    n : Basis ι R (Subtype fun x => Membership.mem N x)
    H : Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i. …
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    ⊢ M.LinearDisjoint N
  -/
  let i1 := TensorProduct.congr m.repr n.repr
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    κ : Type u_1
    ι : Type u_2
    m : Basis κ R (Subtype fun x => Membership.mem M x)
    n : Basis ι R (Subtype fun x => Membership.mem N x)
    H : Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i. …
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearEquiv (RingHom.id R) (TensorProduct R (Subtype fun x => Membership. …
    ⊢ M.LinearDisjoint N
  -/
  let i := mulMap M N ∘ₗ (i0.trans i1.symm).toLinearMap
  have : i = Finsupp.linearCombination R fun i : κ × ι ↦ (m i.1 * n i.2 : S) := by
    ext x
    simp [i, i0, i1, finsuppTensorFinsupp'_symm_single_eq_single_one_tmul]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    κ : Type u_1
    ι : Type u_2
    m : Basis κ R (Subtype fun x => Membership.mem M x)
    n : Basis ι R (Subtype fun x => Membership.mem N x)
    H : Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i. …
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearEquiv (RingHom.id R) (TensorProduct R (Subtype fun x => Membership. …
    i : LinearMap (RingHom.id R) (Finsupp (Prod κ ι) R) S := (M.mulMap N).comp ↑(i …
    this : Eq i (Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) ↑(n i.2))
    ⊢ M.LinearDisjoint N
  -/
  simp_rw [← this, i, LinearMap.coe_comp, LinearEquiv.coe_coe, EquivLike.injective_comp] at H
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    κ : Type u_1
    ι : Type u_2
    m : Basis κ R (Subtype fun x => Membership.mem M x)
    n : Basis ι R (Subtype fun x => Membership.mem N x)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearEquiv (RingHom.id R) (TensorProduct R (Subtype fun x => Membership. …
    i : LinearMap (RingHom.id R) (Finsupp (Prod κ ι) R) S := (M.mulMap N).comp ↑(i …
    this : Eq i (Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) ↑(n i.2))
    H : Function.Injective ⇑(M.mulMap N)
    ⊢ M.LinearDisjoint N
  -/
  exact ⟨H⟩
  /-
    🎉 no goals
  -/


/-- The zero module is linearly disjoint with any other submodules. -/
theorem bot_left : (⊥ : Submodule R S).LinearDisjoint N :=
  ⟨Function.injective_of_subsingleton _⟩


/-- The zero module is linearly disjoint with any other submodules. -/
theorem bot_right : M.LinearDisjoint (⊥ : Submodule R S) :=
  ⟨Function.injective_of_subsingleton _⟩


/-- The image of `R` in `S` is linearly disjoint with any other submodules. -/
theorem one_left : (1 : Submodule R S).LinearDisjoint N := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    N : Submodule R S
    ⊢ Submodule.LinearDisjoint 1 N
  -/
  rw [linearDisjoint_iff, ← Algebra.toSubmodule_bot, mulMap_one_left_eq]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    N : Submodule R S
    ⊢ Function.Injective ⇑(N.subtype.comp ↑N.lTensorOne)
  -/
  exact N.injective_subtype.comp N.lTensorOne.injective
  /-
    🎉 no goals
  -/


/-- The image of `R` in `S` is linearly disjoint with any other submodules. -/
theorem one_right : M.LinearDisjoint (1 : Submodule R S) := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    ⊢ M.LinearDisjoint 1
  -/
  rw [linearDisjoint_iff, ← Algebra.toSubmodule_bot, mulMap_one_right_eq]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    ⊢ Function.Injective ⇑(M.subtype.comp ↑M.rTensorOne)
  -/
  exact M.injective_subtype.comp M.rTensorOne.injective
  /-
    🎉 no goals
  -/


/-- If for any finitely generated submodules `M'` of `M`, `M'` and `N` are linearly disjoint,
then `M` and `N` themselves are linearly disjoint. -/
theorem of_linearDisjoint_fg_left
    (H : ∀ M' : Submodule R S, M' ≤ M → M'.FG → M'.LinearDisjoint N) :
    M.LinearDisjoint N := (linearDisjoint_iff _ _).2 fun x y hxy ↦ by
  obtain ⟨M', hM, hFG, h⟩ :=
    TensorProduct.exists_finite_submodule_left_of_finite' {x, y} (Set.toFinite _)
  /-
    case intro.intro.intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : ∀ (M' : Submodule R S), LE.le M' M → M'.FG → M'.LinearDisjoint N
    x y : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x =>  …
    hxy : Eq ((M.mulMap N) x) ((M.mulMap N) y)
    M' : Submodule R S
    hM : LE.le M' M
    hFG : Module.Finite R (Subtype fun x => Membership.mem M' x)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    ⊢ Eq x y
  -/
  rw [Module.Finite.iff_fg] at hFG
  /-
    case intro.intro.intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : ∀ (M' : Submodule R S), LE.le M' M → M'.FG → M'.LinearDisjoint N
    x y : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x =>  …
    hxy : Eq ((M.mulMap N) x) ((M.mulMap N) y)
    M' : Submodule R S
    hM : LE.le M' M
    hFG : M'.FG
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    ⊢ Eq x y
  -/
  obtain ⟨x', hx'⟩ := h (show x ∈ {x, y} by simp)
  /-
    case intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : ∀ (M' : Submodule R S), LE.le M' M → M'.FG → M'.LinearDisjoint N
    x y : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x =>  …
    hxy : Eq ((M.mulMap N) x) ((M.mulMap N) y)
    M' : Submodule R S
    hM : LE.le M' M
    hFG : M'.FG
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    x' : TensorProduct R (Subtype fun x => Membership.mem M' x) (Subtype fun x =>  …
    hx' : Eq ((LinearMap.rTensor (Subtype fun x => Membership.mem N x) (Submodule. …
    ⊢ Eq x y
  -/
  obtain ⟨y', hy'⟩ := h (show y ∈ {x, y} by simp)
  /-
    case intro.intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : ∀ (M' : Submodule R S), LE.le M' M → M'.FG → M'.LinearDisjoint N
    x y : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x =>  …
    hxy : Eq ((M.mulMap N) x) ((M.mulMap N) y)
    M' : Submodule R S
    hM : LE.le M' M
    hFG : M'.FG
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    x' : TensorProduct R (Subtype fun x => Membership.mem M' x) (Subtype fun x =>  …
    hx' : Eq ((LinearMap.rTensor (Subtype fun x => Membership.mem N x) (Submodule. …
    y' : TensorProduct R (Subtype fun x => Membership.mem M' x) (Subtype fun x =>  …
    hy' : Eq ((LinearMap.rTensor (Subtype fun x => Membership.mem N x) (Submodule. …
    ⊢ Eq x y
  -/
  rw [← hx', ← hy']; congr
  /-
    case intro.intro.intro.intro.intro.h.e_6.h
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : ∀ (M' : Submodule R S), LE.le M' M → M'.FG → M'.LinearDisjoint N
    x y : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x =>  …
    hxy : Eq ((M.mulMap N) x) ((M.mulMap N) y)
    M' : Submodule R S
    hM : LE.le M' M
    hFG : M'.FG
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    x' : TensorProduct R (Subtype fun x => Membership.mem M' x) (Subtype fun x =>  …
    hx' : Eq ((LinearMap.rTensor (Subtype fun x => Membership.mem N x) (Submodule. …
    y' : TensorProduct R (Subtype fun x => Membership.mem M' x) (Subtype fun x =>  …
    hy' : Eq ((LinearMap.rTensor (Subtype fun x => Membership.mem N x) (Submodule. …
    ⊢ Eq x' y'
  -/
  exact (H M' hM hFG).injective (by simp [← mulMap_comp_rTensor _ hM, hx', hy', hxy])
  /-
    🎉 no goals
  -/


/-- If for any finitely generated submodules `N'` of `N`, `M` and `N'` are linearly disjoint,
then `M` and `N` themselves are linearly disjoint. -/
theorem of_linearDisjoint_fg_right
    (H : ∀ N' : Submodule R S, N' ≤ N → N'.FG → M.LinearDisjoint N') :
    M.LinearDisjoint N := (linearDisjoint_iff _ _).2 fun x y hxy ↦ by
  obtain ⟨N', hN, hFG, h⟩ :=
    TensorProduct.exists_finite_submodule_right_of_finite' {x, y} (Set.toFinite _)
  /-
    case intro.intro.intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : ∀ (N' : Submodule R S), LE.le N' N → N'.FG → M.LinearDisjoint N'
    x y : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x =>  …
    hxy : Eq ((M.mulMap N) x) ((M.mulMap N) y)
    N' : Submodule R S
    hN : LE.le N' N
    hFG : Module.Finite R (Subtype fun x => Membership.mem N' x)
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    ⊢ Eq x y
  -/
  rw [Module.Finite.iff_fg] at hFG
  /-
    case intro.intro.intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : ∀ (N' : Submodule R S), LE.le N' N → N'.FG → M.LinearDisjoint N'
    x y : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x =>  …
    hxy : Eq ((M.mulMap N) x) ((M.mulMap N) y)
    N' : Submodule R S
    hN : LE.le N' N
    hFG : N'.FG
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    ⊢ Eq x y
  -/
  obtain ⟨x', hx'⟩ := h (show x ∈ {x, y} by simp)
  /-
    case intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : ∀ (N' : Submodule R S), LE.le N' N → N'.FG → M.LinearDisjoint N'
    x y : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x =>  …
    hxy : Eq ((M.mulMap N) x) ((M.mulMap N) y)
    N' : Submodule R S
    hN : LE.le N' N
    hFG : N'.FG
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    x' : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x => M …
    hx' : Eq ((LinearMap.lTensor (Subtype fun x => Membership.mem M x) (Submodule. …
    ⊢ Eq x y
  -/
  obtain ⟨y', hy'⟩ := h (show y ∈ {x, y} by simp)
  /-
    case intro.intro.intro.intro.intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : ∀ (N' : Submodule R S), LE.le N' N → N'.FG → M.LinearDisjoint N'
    x y : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x =>  …
    hxy : Eq ((M.mulMap N) x) ((M.mulMap N) y)
    N' : Submodule R S
    hN : LE.le N' N
    hFG : N'.FG
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    x' : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x => M …
    hx' : Eq ((LinearMap.lTensor (Subtype fun x => Membership.mem M x) (Submodule. …
    y' : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x => M …
    hy' : Eq ((LinearMap.lTensor (Subtype fun x => Membership.mem M x) (Submodule. …
    ⊢ Eq x y
  -/
  rw [← hx', ← hy']; congr
  /-
    case intro.intro.intro.intro.intro.h.e_6.h
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : ∀ (N' : Submodule R S), LE.le N' N → N'.FG → M.LinearDisjoint N'
    x y : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x =>  …
    hxy : Eq ((M.mulMap N) x) ((M.mulMap N) y)
    N' : Submodule R S
    hN : LE.le N' N
    hFG : N'.FG
    h : HasSubset.Subset (Insert.insert x (Singleton.singleton y)) ↑(LinearMap.ran …
    x' : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x => M …
    hx' : Eq ((LinearMap.lTensor (Subtype fun x => Membership.mem M x) (Submodule. …
    y' : TensorProduct R (Subtype fun x => Membership.mem M x) (Subtype fun x => M …
    hy' : Eq ((LinearMap.lTensor (Subtype fun x => Membership.mem M x) (Submodule. …
    ⊢ Eq x' y'
  -/
  exact (H N' hN hFG).injective (by simp [← mulMap_comp_lTensor _ hN, hx', hy', hxy])
  /-
    🎉 no goals
  -/


/-- If for any finitely generated submodules `M'` and `N'` of `M` and `N`, respectively,
`M'` and `N'` are linearly disjoint, then `M` and `N` themselves are linearly disjoint. -/
theorem of_linearDisjoint_fg
    (H : ∀ (M' N' : Submodule R S), M' ≤ M → N' ≤ N → M'.FG → N'.FG → M'.LinearDisjoint N') :
    M.LinearDisjoint N :=
  of_linearDisjoint_fg_left _ _ fun _ hM hM' ↦
    of_linearDisjoint_fg_right _ _ fun _ hN hN' ↦ H _ _ hM hN hM' hN'


/-- Linear disjointness is symmetric in a commutative ring. -/
theorem LinearDisjoint.symm (H : M.LinearDisjoint N) : N.LinearDisjoint M :=
  H.symm_of_commute fun _ _ ↦ mul_comm _ _


/-- Linear disjointness is symmetric in a commutative ring. -/
theorem linearDisjoint_comm : M.LinearDisjoint N ↔ N.LinearDisjoint M :=
  ⟨LinearDisjoint.symm, LinearDisjoint.symm⟩


variable {M N} in
/-- If `M` and `N` are linearly disjoint, if `N` is a flat `R`-module, then for any family of
`R`-linearly independent elements `{ m_i }` of `M`, they are also `N`-linearly independent,
in the sense that the `R`-linear map from `ι →₀ N` to `S` which maps `{ n_i }`
to the sum of `m_i * n_i` (`Submodule.mulLeftMap N m`) has trivial kernel. -/
theorem linearIndependent_left_of_flat (H : M.LinearDisjoint N) [Module.Flat R N]
    {ι : Type*} {m : ι → M} (hm : LinearIndependent R m) : LinearMap.ker (mulLeftMap N m) = ⊥ := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    ι : Type u_1
    m : ι → Subtype fun x => Membership.mem M x
    hm : LinearIndependent R m
    ⊢ Eq (LinearMap.ker (Submodule.mulLeftMap N m)) Bot.bot
  -/
  refine LinearMap.ker_eq_bot_of_injective ?_
  classical simp_rw [mulLeftMap_eq_mulMap_comp, LinearMap.coe_comp, LinearEquiv.coe_coe,
    ← Function.comp_assoc, EquivLike.injective_comp]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    ι : Type u_1
    m : ι → Subtype fun x => Membership.mem M x
    hm : LinearIndependent R m
    ⊢ Function.Injective (Function.comp ⇑(M.mulMap N) ⇑(LinearMap.rTensor (Subtype …
  -/
  rw [LinearIndependent] at hm
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    ι : Type u_1
    m : ι → Subtype fun x => Membership.mem M x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    ⊢ Function.Injective (Function.comp ⇑(M.mulMap N) ⇑(LinearMap.rTensor (Subtype …
  -/
  exact H.injective.comp (Module.Flat.rTensor_preserves_injective_linearMap (M := N) _ hm)
  /-
    🎉 no goals
  -/


/-- If `{ m_i }` is an `R`-basis of `M`, which is also `N`-linearly independent,
then `M` and `N` are linearly disjoint. -/
theorem of_basis_left {ι : Type*} (m : Basis ι R M)
    (H : LinearMap.ker (mulLeftMap N m) = ⊥) : M.LinearDisjoint N := by
  -- need this instance otherwise `LinearMap.ker_eq_bot` does not work
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    ι : Type u_1
    m : Basis ι R (Subtype fun x => Membership.mem M x)
    H : Eq (LinearMap.ker (Submodule.mulLeftMap N ⇑m)) Bot.bot
    ⊢ M.LinearDisjoint N
  -/
  letI : AddCommGroup (ι →₀ N) := Finsupp.instAddCommGroup
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    ι : Type u_1
    m : Basis ι R (Subtype fun x => Membership.mem M x)
    H : Eq (LinearMap.ker (Submodule.mulLeftMap N ⇑m)) Bot.bot
    this : AddCommGroup (Finsupp ι (Subtype fun x => Membership.mem N x)) := Finsu …
    ⊢ M.LinearDisjoint N
  -/
  exact of_basis_left' M N m (LinearMap.ker_eq_bot.1 H)
  /-
    🎉 no goals
  -/


variable {M N} in
/-- If `M` and `N` are linearly disjoint, if `M` is a flat `R`-module, then for any family of
`R`-linearly independent elements `{ n_i }` of `N`, they are also `M`-linearly independent,
in the sense that the `R`-linear map from `ι →₀ M` to `S` which maps `{ m_i }`
to the sum of `m_i * n_i` (`Submodule.mulRightMap M n`) has trivial kernel. -/
theorem linearIndependent_right_of_flat (H : M.LinearDisjoint N) [Module.Flat R M]
    {ι : Type*} {n : ι → N} (hn : LinearIndependent R n) : LinearMap.ker (mulRightMap M n) = ⊥ := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    ι : Type u_1
    n : ι → Subtype fun x => Membership.mem N x
    hn : LinearIndependent R n
    ⊢ Eq (LinearMap.ker (M.mulRightMap n)) Bot.bot
  -/
  refine LinearMap.ker_eq_bot_of_injective ?_
  classical simp_rw [mulRightMap_eq_mulMap_comp, LinearMap.coe_comp, LinearEquiv.coe_coe,
    ← Function.comp_assoc, EquivLike.injective_comp]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    ι : Type u_1
    n : ι → Subtype fun x => Membership.mem N x
    hn : LinearIndependent R n
    ⊢ Function.Injective (Function.comp ⇑(M.mulMap N) ⇑(LinearMap.lTensor (Subtype …
  -/
  rw [LinearIndependent] at hn
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    ι : Type u_1
    n : ι → Subtype fun x => Membership.mem N x
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    ⊢ Function.Injective (Function.comp ⇑(M.mulMap N) ⇑(LinearMap.lTensor (Subtype …
  -/
  exact H.injective.comp (Module.Flat.lTensor_preserves_injective_linearMap (M := M) _ hn)
  /-
    🎉 no goals
  -/


/-- If `{ n_i }` is an `R`-basis of `N`, which is also `M`-linearly independent,
then `M` and `N` are linearly disjoint. -/
theorem of_basis_right {ι : Type*} (n : Basis ι R N)
    (H : LinearMap.ker (mulRightMap M n) = ⊥) : M.LinearDisjoint N := by
  -- need this instance otherwise `LinearMap.ker_eq_bot` does not work
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    ι : Type u_1
    n : Basis ι R (Subtype fun x => Membership.mem N x)
    H : Eq (LinearMap.ker (M.mulRightMap ⇑n)) Bot.bot
    ⊢ M.LinearDisjoint N
  -/
  letI : AddCommGroup (ι →₀ M) := Finsupp.instAddCommGroup
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    ι : Type u_1
    n : Basis ι R (Subtype fun x => Membership.mem N x)
    H : Eq (LinearMap.ker (M.mulRightMap ⇑n)) Bot.bot
    this : AddCommGroup (Finsupp ι (Subtype fun x => Membership.mem M x)) := Finsu …
    ⊢ M.LinearDisjoint N
  -/
  exact of_basis_right' M N n (LinearMap.ker_eq_bot.1 H)
  /-
    🎉 no goals
  -/


variable {M N} in
/-- If `M` and `N` are linearly disjoint, if `M` is flat, then for any family of
`R`-linearly independent elements `{ m_i }` of `M`, and any family of
`R`-linearly independent elements `{ n_j }` of `N`, the family `{ m_i * n_j }` in `S` is
also `R`-linearly independent. -/
theorem linearIndependent_mul_of_flat_left (H : M.LinearDisjoint N) [Module.Flat R M]
    {κ ι : Type*} {m : κ → M} {n : ι → N} (hm : LinearIndependent R m)
    (hn : LinearIndependent R n) : LinearIndependent R fun (i : κ × ι) ↦ (m i.1).1 * (n i.2).1 := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : LinearIndependent R m
    hn : LinearIndependent R n
    ⊢ LinearIndependent R fun i => HMul.hMul ↑(m i.1) ↑(n i.2)
  -/
  rw [LinearIndependent] at hm hn ⊢
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  let i0 := (finsuppTensorFinsupp' R κ ι).symm
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  let i1 := LinearMap.rTensor (ι →₀ R) (Finsupp.linearCombination R m)
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  let i2 := LinearMap.lTensor M (Finsupp.linearCombination R n)
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    i2 : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.me …
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  let i := mulMap M N ∘ₗ i2 ∘ₗ i1 ∘ₗ i0.toLinearMap
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    i2 : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.me …
    i : LinearMap (RingHom.id R) (Finsupp (Prod κ ι) R) S := (M.mulMap N).comp (i2 …
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  have h1 : Function.Injective i1 := Module.Flat.rTensor_preserves_injective_linearMap _ hm
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    i2 : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.me …
    i : LinearMap (RingHom.id R) (Finsupp (Prod κ ι) R) S := (M.mulMap N).comp (i2 …
    h1 : Function.Injective ⇑i1
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  have h2 : Function.Injective i2 := Module.Flat.lTensor_preserves_injective_linearMap _ hn
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    i2 : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.me …
    i : LinearMap (RingHom.id R) (Finsupp (Prod κ ι) R) S := (M.mulMap N).comp (i2 …
    h1 : Function.Injective ⇑i1
    h2 : Function.Injective ⇑i2
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  have h : Function.Injective i := H.injective.comp h2 |>.comp h1 |>.comp i0.injective
  have : i = Finsupp.linearCombination R fun i ↦ (m i.1).1 * (n i.2).1 := by
    ext x
    simp [i, i0, i1, i2, finsuppTensorFinsupp'_symm_single_eq_single_one_tmul]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    i2 : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.me …
    i : LinearMap (RingHom.id R) (Finsupp (Prod κ ι) R) S := (M.mulMap N).comp (i2 …
    h1 : Function.Injective ⇑i1
    h2 : Function.Injective ⇑i2
    h : Function.Injective ⇑i
    this : Eq i (Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) ↑(n i.2))
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  rwa [this] at h
  /-
    🎉 no goals
  -/


variable {M N} in
/-- If `M` and `N` are linearly disjoint, if `N` is flat, then for any family of
`R`-linearly independent elements `{ m_i }` of `M`, and any family of
`R`-linearly independent elements `{ n_j }` of `N`, the family `{ m_i * n_j }` in `S` is
also `R`-linearly independent. -/
theorem linearIndependent_mul_of_flat_right (H : M.LinearDisjoint N) [Module.Flat R N]
    {κ ι : Type*} {m : κ → M} {n : ι → N} (hm : LinearIndependent R m)
    (hn : LinearIndependent R n) : LinearIndependent R fun (i : κ × ι) ↦ (m i.1).1 * (n i.2).1 := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : LinearIndependent R m
    hn : LinearIndependent R n
    ⊢ LinearIndependent R fun i => HMul.hMul ↑(m i.1) ↑(n i.2)
  -/
  rw [LinearIndependent] at hm hn ⊢
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  let i0 := (finsuppTensorFinsupp' R κ ι).symm
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  let i1 := LinearMap.lTensor (κ →₀ R) (Finsupp.linearCombination R n)
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  let i2 := LinearMap.rTensor N (Finsupp.linearCombination R m)
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    i2 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Subtype fun x => …
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  let i := mulMap M N ∘ₗ i2 ∘ₗ i1 ∘ₗ i0.toLinearMap
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    i2 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Subtype fun x => …
    i : LinearMap (RingHom.id R) (Finsupp (Prod κ ι) R) S := (M.mulMap N).comp (i2 …
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  have h1 : Function.Injective i1 := Module.Flat.lTensor_preserves_injective_linearMap _ hn
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    i2 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Subtype fun x => …
    i : LinearMap (RingHom.id R) (Finsupp (Prod κ ι) R) S := (M.mulMap N).comp (i2 …
    h1 : Function.Injective ⇑i1
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  have h2 : Function.Injective i2 := Module.Flat.rTensor_preserves_injective_linearMap _ hm
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    i2 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Subtype fun x => …
    i : LinearMap (RingHom.id R) (Finsupp (Prod κ ι) R) S := (M.mulMap N).comp (i2 …
    h1 : Function.Injective ⇑i1
    h2 : Function.Injective ⇑i2
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  have h : Function.Injective i := H.injective.comp h2 |>.comp h1 |>.comp i0.injective
  have : i = Finsupp.linearCombination R fun i ↦ (m i.1).1 * (n i.2).1 := by
    ext x
    simp [i, i0, i1, i2, finsuppTensorFinsupp'_symm_single_eq_single_one_tmul]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : Function.Injective ⇑(Finsupp.linearCombination R m)
    hn : Function.Injective ⇑(Finsupp.linearCombination R n)
    i0 : LinearEquiv (RingHom.id R) (Finsupp (Prod κ ι) R) (TensorProduct R (Finsu …
    i1 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Finsupp ι R)) (T …
    i2 : LinearMap (RingHom.id R) (TensorProduct R (Finsupp κ R) (Subtype fun x => …
    i : LinearMap (RingHom.id R) (Finsupp (Prod κ ι) R) S := (M.mulMap N).comp (i2 …
    h1 : Function.Injective ⇑i1
    h2 : Function.Injective ⇑i2
    h : Function.Injective ⇑i
    this : Eq i (Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) ↑(n i.2))
    ⊢ Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i.1) …
  -/
  rwa [this] at h
  /-
    🎉 no goals
  -/


variable {M N} in
/-- If `M` and `N` are linearly disjoint, if one of `M` and `N` is flat, then for any family of
`R`-linearly independent elements `{ m_i }` of `M`, and any family of
`R`-linearly independent elements `{ n_j }` of `N`, the family `{ m_i * n_j }` in `S` is
also `R`-linearly independent. -/
theorem linearIndependent_mul_of_flat (H : M.LinearDisjoint N)
    (hf : Module.Flat R M ∨ Module.Flat R N)
    {κ ι : Type*} {m : κ → M} {n : ι → N} (hm : LinearIndependent R m)
    (hn : LinearIndependent R n) : LinearIndependent R fun (i : κ × ι) ↦ (m i.1).1 * (n i.2).1 := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    κ : Type u_1
    ι : Type u_2
    m : κ → Subtype fun x => Membership.mem M x
    n : ι → Subtype fun x => Membership.mem N x
    hm : LinearIndependent R m
    hn : LinearIndependent R n
    ⊢ LinearIndependent R fun i => HMul.hMul ↑(m i.1) ↑(n i.2)
  -/
  rcases hf with _ | _
    /-
      case inl
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      M N : Submodule R S
      H : M.LinearDisjoint N
      κ : Type u_1
      ι : Type u_2
      m : κ → Subtype fun x => Membership.mem M x
      n : ι → Subtype fun x => Membership.mem N x
      hm : LinearIndependent R m
      hn : LinearIndependent R n
      h✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
      ⊢ LinearIndependent R fun i => HMul.hMul ↑(m i.1) ↑(n i.2)
    -/
  · exact H.linearIndependent_mul_of_flat_left hm hn
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      S : Type v
      inst✝² : CommRing R
      inst✝¹ : Ring S
      inst✝ : Algebra R S
      M N : Submodule R S
      H : M.LinearDisjoint N
      κ : Type u_1
      ι : Type u_2
      m : κ → Subtype fun x => Membership.mem M x
      n : ι → Subtype fun x => Membership.mem N x
      hm : LinearIndependent R m
      hn : LinearIndependent R n
      h✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
      ⊢ LinearIndependent R fun i => HMul.hMul ↑(m i.1) ↑(n i.2)
    -/
  · exact H.linearIndependent_mul_of_flat_right hm hn
    /-
      🎉 no goals
    -/


/-- If `{ m_i }` is an `R`-basis of `M`, if `{ n_j }` is an `R`-basis of `N`,
such that the family `{ m_i * n_j }` in `S` is `R`-linearly independent,
then `M` and `N` are linearly disjoint. -/
theorem of_basis_mul {κ ι : Type*} (m : Basis κ R M) (n : Basis ι R N)
    (H : LinearIndependent R fun (i : κ × ι) ↦ (m i.1).1 * (n i.2).1) : M.LinearDisjoint N := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    κ : Type u_1
    ι : Type u_2
    m : Basis κ R (Subtype fun x => Membership.mem M x)
    n : Basis ι R (Subtype fun x => Membership.mem N x)
    H : LinearIndependent R fun i => HMul.hMul ↑(m i.1) ↑(n i.2)
    ⊢ M.LinearDisjoint N
  -/
  rw [LinearIndependent] at H
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    κ : Type u_1
    ι : Type u_2
    m : Basis κ R (Subtype fun x => Membership.mem M x)
    n : Basis ι R (Subtype fun x => Membership.mem N x)
    H : Function.Injective ⇑(Finsupp.linearCombination R fun i => HMul.hMul ↑(m i. …
    ⊢ M.LinearDisjoint N
  -/
  exact of_basis_mul' M N m n H
  /-
    🎉 no goals
  -/


variable {M N} in
/-- If `M` and `N` are linearly disjoint, if `N` is flat, then for any submodule `M'` of `M`,
`M'` and `N` are also linearly disjoint. -/
theorem of_le_left_of_flat (H : M.LinearDisjoint N) {M' : Submodule R S}
    (h : M' ≤ M) [Module.Flat R N] : M'.LinearDisjoint N := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    M' : Submodule R S
    h : LE.le M' M
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    ⊢ M'.LinearDisjoint N
  -/
  let i := mulMap M N ∘ₗ (inclusion h).rTensor N
  have hi : Function.Injective i := H.injective.comp <|
    Module.Flat.rTensor_preserves_injective_linearMap _ <| inclusion_injective h
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    M' : Submodule R S
    h : LE.le M' M
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    i : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    hi : Function.Injective ⇑i
    ⊢ M'.LinearDisjoint N
  -/
  have : i = mulMap M' N := by ext; simp [i]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    M' : Submodule R S
    h : LE.le M' M
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    i : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    hi : Function.Injective ⇑i
    this : Eq i (M'.mulMap N)
    ⊢ M'.LinearDisjoint N
  -/
  exact ⟨this ▸ hi⟩
  /-
    🎉 no goals
  -/


variable {M N} in
/-- If `M` and `N` are linearly disjoint, if `M` is flat, then for any submodule `N'` of `N`,
`M` and `N'` are also linearly disjoint. -/
theorem of_le_right_of_flat (H : M.LinearDisjoint N) {N' : Submodule R S}
    (h : N' ≤ N) [Module.Flat R M] : M.LinearDisjoint N' := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    N' : Submodule R S
    h : LE.le N' N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    ⊢ M.LinearDisjoint N'
  -/
  let i := mulMap M N ∘ₗ (inclusion h).lTensor M
  have hi : Function.Injective i := H.injective.comp <|
    Module.Flat.lTensor_preserves_injective_linearMap _ <| inclusion_injective h
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    N' : Submodule R S
    h : LE.le N' N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    i : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    hi : Function.Injective ⇑i
    ⊢ M.LinearDisjoint N'
  -/
  have : i = mulMap M N' := by ext; simp [i]
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    N' : Submodule R S
    h : LE.le N' N
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    i : LinearMap (RingHom.id R) (TensorProduct R (Subtype fun x => Membership.mem …
    hi : Function.Injective ⇑i
    this : Eq i (M.mulMap N')
    ⊢ M.LinearDisjoint N'
  -/
  exact ⟨this ▸ hi⟩
  /-
    🎉 no goals
  -/


variable {M N} in
/-- If `M` and `N` are linearly disjoint, `M'` and `N'` are submodules of `M` and `N`,
respectively, such that `N` and `M'` are flat, then `M'` and `N'` are also linearly disjoint. -/
theorem of_le_of_flat_right (H : M.LinearDisjoint N) {M' N' : Submodule R S}
    (hm : M' ≤ M) (hn : N' ≤ N) [Module.Flat R N] [Module.Flat R M'] :
    M'.LinearDisjoint N' := (H.of_le_left_of_flat hm).of_le_right_of_flat hn


variable {M N} in
/-- If `M` and `N` are linearly disjoint, `M'` and `N'` are submodules of `M` and `N`,
respectively, such that `M` and `N'` are flat, then `M'` and `N'` are also linearly disjoint. -/
theorem of_le_of_flat_left (H : M.LinearDisjoint N) {M' N' : Submodule R S}
    (hm : M' ≤ M) (hn : N' ≤ N) [Module.Flat R M] [Module.Flat R N'] :
    M'.LinearDisjoint N' := (H.of_le_right_of_flat hn).of_le_left_of_flat hm


/-- If `N` is flat, `M` is contained in `i(R)`, where `i : R → S` is the structure map,
then `M` and `N` are linearly disjoint. -/
theorem of_left_le_one_of_flat (h : M ≤ 1) [Module.Flat R N] :
    M.LinearDisjoint N := (one_left N).of_le_left_of_flat h


/-- If `M` is flat, `N` is contained in `i(R)`, where `i : R → S` is the structure map,
then `M` and `N` are linearly disjoint. -/
theorem of_right_le_one_of_flat (h : N ≤ 1) [Module.Flat R M] :
    M.LinearDisjoint N := (one_right M).of_le_right_of_flat h


/-- If `M` and `N` are linearly disjoint, if `M` is flat, then any two commutative
elements of `↥(M ⊓ N)` are not `R`-linearly independent (namely, their span is not `R ^ 2`). -/
theorem not_linearIndependent_pair_of_commute_of_flat_left [Module.Flat R M]
    (a b : ↥(M ⊓ N)) (hc : Commute a.1 b.1) : ¬LinearIndependent R ![a, b] := fun h ↦ by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    ⊢ False
  -/
  let n : Fin 2 → N := (inclusion inf_le_right) ∘ ![a, b]
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    n : Fin 2 → Subtype fun x => Membership.mem N x := Function.comp (⇑(Submodule. …
    ⊢ False
  -/
  have hn : LinearIndependent R n := h.map' _ (ker_inclusion _ _ _)
  -- need this instance otherwise it only has semigroup structure
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    n : Fin 2 → Subtype fun x => Membership.mem N x := Function.comp (⇑(Submodule. …
    hn : LinearIndependent R n
    ⊢ False
  -/
  letI : AddCommGroup (Fin 2 →₀ M) := Finsupp.instAddCommGroup
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    n : Fin 2 → Subtype fun x => Membership.mem N x := Function.comp (⇑(Submodule. …
    hn : LinearIndependent R n
    this : AddCommGroup (Finsupp (Fin 2) (Subtype fun x => Membership.mem M x)) := …
    ⊢ False
  -/
  let m : Fin 2 →₀ M := .single 0 ⟨b.1, b.2.1⟩ - .single 1 ⟨a.1, a.2.1⟩
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    n : Fin 2 → Subtype fun x => Membership.mem N x := Function.comp (⇑(Submodule. …
    hn : LinearIndependent R n
    this : AddCommGroup (Finsupp (Fin 2) (Subtype fun x => Membership.mem M x)) := …
    m : Finsupp (Fin 2) (Subtype fun x => Membership.mem M x) := HSub.hSub (Finsup …
    ⊢ False
  -/
  have hm : mulRightMap M n m = 0 := by simp [m, n, show _ * _ = _ * _ from hc]
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    n : Fin 2 → Subtype fun x => Membership.mem N x := Function.comp (⇑(Submodule. …
    hn : LinearIndependent R n
    this : AddCommGroup (Finsupp (Fin 2) (Subtype fun x => Membership.mem M x)) := …
    m : Finsupp (Fin 2) (Subtype fun x => Membership.mem M x) := HSub.hSub (Finsup …
    hm : Eq ((M.mulRightMap n) m) 0
    ⊢ False
  -/
  rw [← LinearMap.mem_ker, H.linearIndependent_right_of_flat hn, mem_bot] at hm
  simp only [Fin.isValue, sub_eq_zero, Finsupp.single_eq_single_iff, zero_ne_one, Subtype.mk.injEq,
    SetLike.coe_eq_coe, false_and, false_or, m] at hm
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    n : Fin 2 → Subtype fun x => Membership.mem N x := Function.comp (⇑(Submodule. …
    hn : LinearIndependent R n
    this : AddCommGroup (Finsupp (Fin 2) (Subtype fun x => Membership.mem M x)) := …
    m : Finsupp (Fin 2) (Subtype fun x => Membership.mem M x) := HSub.hSub (Finsup …
    hm : And (Eq ⟨↑b, ⋯⟩ 0) (Eq ⟨↑a, ⋯⟩ 0)
    ⊢ False
  -/
  repeat rw [AddSubmonoid.mk_eq_zero, ZeroMemClass.coe_eq_zero] at hm
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    n : Fin 2 → Subtype fun x => Membership.mem N x := Function.comp (⇑(Submodule. …
    hn : LinearIndependent R n
    this : AddCommGroup (Finsupp (Fin 2) (Subtype fun x => Membership.mem M x)) := …
    m : Finsupp (Fin 2) (Subtype fun x => Membership.mem M x) := HSub.hSub (Finsup …
    hm : And (Eq b 0) (Eq a 0)
    ⊢ False
  -/
  exact h.ne_zero 0 hm.2
  /-
    🎉 no goals
  -/


/-- If `M` and `N` are linearly disjoint, if `N` is flat, then any two commutative
elements of `↥(M ⊓ N)` are not `R`-linearly independent (namely, their span is not `R ^ 2`). -/
theorem not_linearIndependent_pair_of_commute_of_flat_right [Module.Flat R N]
    (a b : ↥(M ⊓ N)) (hc : Commute a.1 b.1) : ¬LinearIndependent R ![a, b] := fun h ↦ by
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    ⊢ False
  -/
  let m : Fin 2 → M := (inclusion inf_le_left) ∘ ![a, b]
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    m : Fin 2 → Subtype fun x => Membership.mem M x := Function.comp (⇑(Submodule. …
    ⊢ False
  -/
  have hm : LinearIndependent R m := h.map' _ (ker_inclusion _ _ _)
  -- need this instance otherwise it only has semigroup structure
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    m : Fin 2 → Subtype fun x => Membership.mem M x := Function.comp (⇑(Submodule. …
    hm : LinearIndependent R m
    ⊢ False
  -/
  letI : AddCommGroup (Fin 2 →₀ N) := Finsupp.instAddCommGroup
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    m : Fin 2 → Subtype fun x => Membership.mem M x := Function.comp (⇑(Submodule. …
    hm : LinearIndependent R m
    this : AddCommGroup (Finsupp (Fin 2) (Subtype fun x => Membership.mem N x)) := …
    ⊢ False
  -/
  let n : Fin 2 →₀ N := .single 0 ⟨b.1, b.2.2⟩ - .single 1 ⟨a.1, a.2.2⟩
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    m : Fin 2 → Subtype fun x => Membership.mem M x := Function.comp (⇑(Submodule. …
    hm : LinearIndependent R m
    this : AddCommGroup (Finsupp (Fin 2) (Subtype fun x => Membership.mem N x)) := …
    n : Finsupp (Fin 2) (Subtype fun x => Membership.mem N x) := HSub.hSub (Finsup …
    ⊢ False
  -/
  have hn : mulLeftMap N m n = 0 := by simp [m, n, show _ * _ = _ * _ from hc]
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    m : Fin 2 → Subtype fun x => Membership.mem M x := Function.comp (⇑(Submodule. …
    hm : LinearIndependent R m
    this : AddCommGroup (Finsupp (Fin 2) (Subtype fun x => Membership.mem N x)) := …
    n : Finsupp (Fin 2) (Subtype fun x => Membership.mem N x) := HSub.hSub (Finsup …
    hn : Eq ((Submodule.mulLeftMap N m) n) 0
    ⊢ False
  -/
  rw [← LinearMap.mem_ker, H.linearIndependent_left_of_flat hm, mem_bot] at hn
  simp only [Fin.isValue, sub_eq_zero, Finsupp.single_eq_single_iff, zero_ne_one, Subtype.mk.injEq,
    SetLike.coe_eq_coe, false_and, false_or, n] at hn
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    m : Fin 2 → Subtype fun x => Membership.mem M x := Function.comp (⇑(Submodule. …
    hm : LinearIndependent R m
    this : AddCommGroup (Finsupp (Fin 2) (Subtype fun x => Membership.mem N x)) := …
    n : Finsupp (Fin 2) (Subtype fun x => Membership.mem N x) := HSub.hSub (Finsup …
    hn : And (Eq ⟨↑b, ⋯⟩ 0) (Eq ⟨↑a, ⋯⟩ 0)
    ⊢ False
  -/
  repeat rw [AddSubmonoid.mk_eq_zero, ZeroMemClass.coe_eq_zero] at hn
  /-
    R : Type u
    S : Type v
    inst✝⁴ : CommRing R
    inst✝³ : Ring S
    inst✝² : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝¹ : Nontrivial R
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    h : LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty))
    m : Fin 2 → Subtype fun x => Membership.mem M x := Function.comp (⇑(Submodule. …
    hm : LinearIndependent R m
    this : AddCommGroup (Finsupp (Fin 2) (Subtype fun x => Membership.mem N x)) := …
    n : Finsupp (Fin 2) (Subtype fun x => Membership.mem N x) := HSub.hSub (Finsup …
    hn : And (Eq b 0) (Eq a 0)
    ⊢ False
  -/
  exact h.ne_zero 0 hn.2
  /-
    🎉 no goals
  -/


/-- If `M` and `N` are linearly disjoint, if one of `M` and `N` is flat, then any two commutative
elements of `↥(M ⊓ N)` are not `R`-linearly independent (namely, their span is not `R ^ 2`). -/
theorem not_linearIndependent_pair_of_commute_of_flat (hf : Module.Flat R M ∨ Module.Flat R N)
    (a b : ↥(M ⊓ N)) (hc : Commute a.1 b.1) : ¬LinearIndependent R ![a, b] := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    inst✝ : Nontrivial R
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    a b : Subtype fun x => Membership.mem (Min.min M N) x
    hc : Commute ↑a ↑b
    ⊢ Not (LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty …
  -/
  rcases hf with _ | _
    /-
      case inl
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      inst✝¹ : Algebra R S
      M N : Submodule R S
      H : M.LinearDisjoint N
      inst✝ : Nontrivial R
      a b : Subtype fun x => Membership.mem (Min.min M N) x
      hc : Commute ↑a ↑b
      h✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
      ⊢ Not (LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty …
    -/
  · exact H.not_linearIndependent_pair_of_commute_of_flat_left a b hc
    /-
      🎉 no goals
    -/
    /-
      case inr
      R : Type u
      S : Type v
      inst✝³ : CommRing R
      inst✝² : Ring S
      inst✝¹ : Algebra R S
      M N : Submodule R S
      H : M.LinearDisjoint N
      inst✝ : Nontrivial R
      a b : Subtype fun x => Membership.mem (Min.min M N) x
      hc : Commute ↑a ↑b
      h✝ : Module.Flat R (Subtype fun x => Membership.mem N x)
      ⊢ Not (LinearIndependent R (Matrix.vecCons a (Matrix.vecCons b Matrix.vecEmpty …
    -/
  · exact H.not_linearIndependent_pair_of_commute_of_flat_right a b hc
    /-
      🎉 no goals
    -/


/-- If `M` and `N` are linearly disjoint, if one of `M` and `N` is flat,
if any two elements of `↥(M ⊓ N)` are commutative, then the rank of `↥(M ⊓ N)` is at most one. -/
theorem rank_inf_le_one_of_commute_of_flat (hf : Module.Flat R M ∨ Module.Flat R N)
    (hc : ∀ (m n : ↥(M ⊓ N)), Commute m.1 n.1) : Module.rank R ↥(M ⊓ N) ≤ 1 := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    hc : ∀ (m n : Subtype fun x => Membership.mem (Min.min M N) x), Commute ↑m ↑n
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min M N) x)) 1
  -/
  nontriviality R
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    hc : ∀ (m n : Subtype fun x => Membership.mem (Min.min M N) x), Commute ↑m ↑n
    a✝ : Nontrivial R
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min M N) x)) 1
  -/
  refine _root_.rank_le fun s h ↦ ?_
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    hc : ∀ (m n : Subtype fun x => Membership.mem (Min.min M N) x), Commute ↑m ↑n
    a✝ : Nontrivial R
    s : Finset (Subtype fun x => Membership.mem (Min.min M N) x)
    h : LinearIndependent R fun i => ↑i
    ⊢ LE.le s.card 1
  -/
  by_contra hs
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    hc : ∀ (m n : Subtype fun x => Membership.mem (Min.min M N) x), Commute ↑m ↑n
    a✝ : Nontrivial R
    s : Finset (Subtype fun x => Membership.mem (Min.min M N) x)
    h : LinearIndependent R fun i => ↑i
    hs : Not (LE.le s.card 1)
    ⊢ False
  -/
  rw [not_le, ← Fintype.card_coe, Fintype.one_lt_card_iff_nontrivial] at hs
  /-
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    hc : ∀ (m n : Subtype fun x => Membership.mem (Min.min M N) x), Commute ↑m ↑n
    a✝ : Nontrivial R
    s : Finset (Subtype fun x => Membership.mem (Min.min M N) x)
    h : LinearIndependent R fun i => ↑i
    hs : Nontrivial (Subtype fun x => Membership.mem s x)
    ⊢ False
  -/
  obtain ⟨a, b, hab⟩ := hs.exists_pair_ne
  /-
    case intro.intro
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    hc : ∀ (m n : Subtype fun x => Membership.mem (Min.min M N) x), Commute ↑m ↑n
    a✝ : Nontrivial R
    s : Finset (Subtype fun x => Membership.mem (Min.min M N) x)
    h : LinearIndependent R fun i => ↑i
    hs : Nontrivial (Subtype fun x => Membership.mem s x)
    a b : Subtype fun x => Membership.mem s x
    hab : Ne a b
    ⊢ False
  -/
  refine H.not_linearIndependent_pair_of_commute_of_flat hf a.1 b.1 (hc a.1 b.1) ?_
  have := h.comp ![a, b] fun i j hij ↦ by
    fin_cases i <;> fin_cases j
    · rfl
    · simp [hab] at hij
    · simp [hab.symm] at hij
    · rfl
  /-
    case intro.intro
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    hc : ∀ (m n : Subtype fun x => Membership.mem (Min.min M N) x), Commute ↑m ↑n
    a✝ : Nontrivial R
    s : Finset (Subtype fun x => Membership.mem (Min.min M N) x)
    h : LinearIndependent R fun i => ↑i
    hs : Nontrivial (Subtype fun x => Membership.mem s x)
    a b : Subtype fun x => Membership.mem s x
    hab : Ne a b
    this : LinearIndependent R (Function.comp (fun i => ↑i) (Matrix.vecCons a (Mat …
    ⊢ LinearIndependent R (Matrix.vecCons (↑a) (Matrix.vecCons (↑b) Matrix.vecEmpt …
  -/
  convert this
  /-
    case h.e'_4
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    hc : ∀ (m n : Subtype fun x => Membership.mem (Min.min M N) x), Commute ↑m ↑n
    a✝ : Nontrivial R
    s : Finset (Subtype fun x => Membership.mem (Min.min M N) x)
    h : LinearIndependent R fun i => ↑i
    hs : Nontrivial (Subtype fun x => Membership.mem s x)
    a b : Subtype fun x => Membership.mem s x
    hab : Ne a b
    this : LinearIndependent R (Function.comp (fun i => ↑i) (Matrix.vecCons a (Mat …
    ⊢ Eq (Matrix.vecCons (↑a) (Matrix.vecCons (↑b) Matrix.vecEmpty)) (Function.com …
  -/
  ext i
  /-
    case h.e'_4.h.a
    R : Type u
    S : Type v
    inst✝² : CommRing R
    inst✝¹ : Ring S
    inst✝ : Algebra R S
    M N : Submodule R S
    H : M.LinearDisjoint N
    hf : Or (Module.Flat R (Subtype fun x => Membership.mem M x)) (Module.Flat R ( …
    hc : ∀ (m n : Subtype fun x => Membership.mem (Min.min M N) x), Commute ↑m ↑n
    a✝ : Nontrivial R
    s : Finset (Subtype fun x => Membership.mem (Min.min M N) x)
    h : LinearIndependent R fun i => ↑i
    hs : Nontrivial (Subtype fun x => Membership.mem s x)
    a b : Subtype fun x => Membership.mem s x
    hab : Ne a b
    this : LinearIndependent R (Function.comp (fun i => ↑i) (Matrix.vecCons a (Mat …
    i : Fin (Nat.succ 0).succ
    ⊢ Eq ↑(Matrix.vecCons (↑a) (Matrix.vecCons (↑b) Matrix.vecEmpty) i) ↑(Function …
  -/
                  /-
                    🎉 no goals
                  -/
  fin_cases i <;> simp
                  /-
                    🎉 no goals
                  -/


/-- If `M` and `N` are linearly disjoint, if `M` is flat,
if any two elements of `↥(M ⊓ N)` are commutative, then the rank of `↥(M ⊓ N)` is at most one. -/
theorem rank_inf_le_one_of_commute_of_flat_left [Module.Flat R M]
    (hc : ∀ (m n : ↥(M ⊓ N)), Commute m.1 n.1) : Module.rank R ↥(M ⊓ N) ≤ 1 :=
  H.rank_inf_le_one_of_commute_of_flat (Or.inl ‹_›) hc


/-- If `M` and `N` are linearly disjoint, if `N` is flat,
if any two elements of `↥(M ⊓ N)` are commutative, then the rank of `↥(M ⊓ N)` is at most one. -/
theorem rank_inf_le_one_of_commute_of_flat_right [Module.Flat R N]
    (hc : ∀ (m n : ↥(M ⊓ N)), Commute m.1 n.1) : Module.rank R ↥(M ⊓ N) ≤ 1 :=
  H.rank_inf_le_one_of_commute_of_flat (Or.inr ‹_›) hc


/-- If `M` and itself are linearly disjoint, if `M` is flat,
if any two elements of `M` are commutative, then the rank of `M` is at most one. -/
theorem rank_le_one_of_commute_of_flat_of_self (H : M.LinearDisjoint M) [Module.Flat R M]
    (hc : ∀ (m n : M), Commute m.1 n.1) : Module.rank R M ≤ 1 := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M : Submodule R S
    H : M.LinearDisjoint M
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    hc : ∀ (m n : Subtype fun x => Membership.mem M x), Commute ↑m ↑n
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem M x)) 1
  -/
  rw [← inf_of_le_left (le_refl M)] at hc ⊢
  /-
    R : Type u
    S : Type v
    inst✝³ : CommRing R
    inst✝² : Ring S
    inst✝¹ : Algebra R S
    M : Submodule R S
    H : M.LinearDisjoint M
    inst✝ : Module.Flat R (Subtype fun x => Membership.mem M x)
    hc : ∀ (m n : Subtype fun x => Membership.mem (Min.min M M) x), Commute ↑m ↑n
    ⊢ LE.le (Module.rank R (Subtype fun x => Membership.mem (Min.min M M) x)) 1
  -/
  exact H.rank_inf_le_one_of_commute_of_flat_left hc
  /-
    🎉 no goals
  -/


/-- The `Submodule.LinearDisjoint.not_linearIndependent_pair_of_commute_of_flat_left`
for commutative rings. -/
theorem not_linearIndependent_pair_of_flat_left [Module.Flat R M]
    (a b : ↥(M ⊓ N)) : ¬LinearIndependent R ![a, b] :=
  H.not_linearIndependent_pair_of_commute_of_flat_left a b (mul_comm _ _)


/-- The `Submodule.LinearDisjoint.not_linearIndependent_pair_of_commute_of_flat_right`
for commutative rings. -/
theorem not_linearIndependent_pair_of_flat_right [Module.Flat R N]
    (a b : ↥(M ⊓ N)) : ¬LinearIndependent R ![a, b] :=
  H.not_linearIndependent_pair_of_commute_of_flat_right a b (mul_comm _ _)


/-- The `Submodule.LinearDisjoint.not_linearIndependent_pair_of_commute_of_flat`
for commutative rings. -/
theorem not_linearIndependent_pair_of_flat (hf : Module.Flat R M ∨ Module.Flat R N)
    (a b : ↥(M ⊓ N)) : ¬LinearIndependent R ![a, b] :=
  H.not_linearIndependent_pair_of_commute_of_flat hf a b (mul_comm _ _)


/-- The `Submodule.LinearDisjoint.rank_inf_le_one_of_commute_of_flat`
for commutative rings. -/
theorem rank_inf_le_one_of_flat (hf : Module.Flat R M ∨ Module.Flat R N) :
    Module.rank R ↥(M ⊓ N) ≤ 1 :=
  H.rank_inf_le_one_of_commute_of_flat hf fun _ _ ↦ mul_comm _ _


/-- The `Submodule.LinearDisjoint.rank_inf_le_one_of_commute_of_flat_left`
for commutative rings. -/
theorem rank_inf_le_one_of_flat_left [Module.Flat R M] : Module.rank R ↥(M ⊓ N) ≤ 1 :=
  H.rank_inf_le_one_of_commute_of_flat_left fun _ _ ↦ mul_comm _ _


/-- The `Submodule.LinearDisjoint.rank_inf_le_one_of_commute_of_flat_right`
for commutative rings. -/
theorem rank_inf_le_one_of_flat_right [Module.Flat R N] : Module.rank R ↥(M ⊓ N) ≤ 1 :=
  H.rank_inf_le_one_of_commute_of_flat_right fun _ _ ↦ mul_comm _ _


/-- The `Submodule.LinearDisjoint.rank_le_one_of_commute_of_flat_of_self`
for commutative rings. -/
theorem rank_le_one_of_flat_of_self (H : M.LinearDisjoint M) [Module.Flat R M] :
    Module.rank R M ≤ 1 :=
  H.rank_le_one_of_commute_of_flat_of_self fun _ _ ↦ mul_comm _ _


