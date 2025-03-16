/-- Given a bilinear map `f : M₁ →ₗ[R] M₂ →ₗ[R] M`, `IsTensorProduct f` means that
`M` is the tensor product of `M₁` and `M₂` via `f`.
This is defined by requiring the lift `M₁ ⊗[R] M₂ → M` to be bijective.
-/
def IsTensorProduct : Prop :=
  Function.Bijective (TensorProduct.lift f)


theorem TensorProduct.isTensorProduct : IsTensorProduct (TensorProduct.mk R M N) := by
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_4
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_8
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ⊢ IsTensorProduct (TensorProduct.mk R M N)
  -/
  delta IsTensorProduct
  /-
    R : Type u_1
    inst✝⁴ : CommSemiring R
    M : Type u_4
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M
    N : Type u_8
    inst✝¹ : AddCommMonoid N
    inst✝ : Module R N
    ⊢ Function.Bijective ⇑(TensorProduct.lift (TensorProduct.mk R M N))
  -/
  convert_to Function.Bijective (LinearMap.id : M ⊗[R] N →ₗ[R] M ⊗[R] N) using 2
    /-
      case h.e'_3.h.e'_5
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_8
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      ⊢ Eq (TensorProduct.lift (TensorProduct.mk R M N)) LinearMap.id
    -/
  · apply TensorProduct.ext'
    /-
      case h.e'_3.h.e'_5.H
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_8
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      ⊢ ∀ (x : M) (y : N), Eq ((TensorProduct.lift (TensorProduct.mk R M N)) (Tensor …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      R : Type u_1
      inst✝⁴ : CommSemiring R
      M : Type u_4
      inst✝³ : AddCommMonoid M
      inst✝² : Module R M
      N : Type u_8
      inst✝¹ : AddCommMonoid N
      inst✝ : Module R N
      ⊢ Function.Bijective ⇑LinearMap.id
    -/
  · exact Function.bijective_id
    /-
      🎉 no goals
    -/


/-- If `M` is the tensor product of `M₁` and `M₂`, it is linearly equivalent to `M₁ ⊗[R] M₂`. -/
@[simps! apply]
noncomputable def IsTensorProduct.equiv (h : IsTensorProduct f) : M₁ ⊗[R] M₂ ≃ₗ[R] M :=
  LinearEquiv.ofBijective _ h


@[simp]
theorem IsTensorProduct.equiv_toLinearMap (h : IsTensorProduct f) :
    h.equiv.toLinearMap = TensorProduct.lift f :=
  rfl


@[simp]
theorem IsTensorProduct.equiv_symm_apply (h : IsTensorProduct f) (x₁ : M₁) (x₂ : M₂) :
    h.equiv.symm (f x₁ x₂) = x₁ ⊗ₜ x₂ := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    h : IsTensorProduct f
    x₁ : M₁
    x₂ : M₂
    ⊢ Eq (h.equiv.symm ((f x₁) x₂)) (TensorProduct.tmul R x₁ x₂)
  -/
  apply h.equiv.injective
  /-
    case a
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    h : IsTensorProduct f
    x₁ : M₁
    x₂ : M₂
    ⊢ Eq (h.equiv (h.equiv.symm ((f x₁) x₂))) (h.equiv (TensorProduct.tmul R x₁ x₂))
  -/
  refine (h.equiv.apply_symm_apply _).trans ?_
  /-
    case a
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    h : IsTensorProduct f
    x₁ : M₁
    x₂ : M₂
    ⊢ Eq ((f x₁) x₂) (h.equiv (TensorProduct.tmul R x₁ x₂))
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `M` is the tensor product of `M₁` and `M₂`, we may lift a bilinear map `M₁ →ₗ[R] M₂ →ₗ[R] M'`
to a `M →ₗ[R] M'`. -/
noncomputable def IsTensorProduct.lift (h : IsTensorProduct f) (f' : M₁ →ₗ[R] M₂ →ₗ[R] M') :
    M →ₗ[R] M' :=
  (TensorProduct.lift f').comp h.equiv.symm.toLinearMap


theorem IsTensorProduct.lift_eq (h : IsTensorProduct f) (f' : M₁ →ₗ[R] M₂ →ₗ[R] M') (x₁ : M₁)
    (x₂ : M₂) : h.lift f' (f x₁ x₂) = f' x₁ x₂ := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    M' : Type u_5
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M'
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    inst✝¹ : Module R M
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    h : IsTensorProduct f
    f' : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M')
    x₁ : M₁
    x₂ : M₂
    ⊢ Eq ((h.lift f') ((f x₁) x₂)) ((f' x₁) x₂)
  -/
  delta IsTensorProduct.lift
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    M' : Type u_5
    inst✝⁷ : AddCommMonoid M₁
    inst✝⁶ : AddCommMonoid M₂
    inst✝⁵ : AddCommMonoid M
    inst✝⁴ : AddCommMonoid M'
    inst✝³ : Module R M₁
    inst✝² : Module R M₂
    inst✝¹ : Module R M
    inst✝ : Module R M'
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    h : IsTensorProduct f
    f' : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M')
    x₁ : M₁
    x₂ : M₂
    ⊢ Eq (((TensorProduct.lift f').comp ↑h.equiv.symm) ((f x₁) x₂)) ((f' x₁) x₂)
  -/
  simp
  /-
    🎉 no goals
  -/


/-- The tensor product of a pair of linear maps between modules. -/
noncomputable def IsTensorProduct.map (hf : IsTensorProduct f) (hg : IsTensorProduct g)
    (i₁ : M₁ →ₗ[R] N₁) (i₂ : M₂ →ₗ[R] N₂) : M →ₗ[R] N :=
  hg.equiv.toLinearMap.comp ((TensorProduct.map i₁ i₂).comp hf.equiv.symm.toLinearMap)


theorem IsTensorProduct.map_eq (hf : IsTensorProduct f) (hg : IsTensorProduct g) (i₁ : M₁ →ₗ[R] N₁)
    (i₂ : M₂ →ₗ[R] N₂) (x₁ : M₁) (x₂ : M₂) : hf.map hg i₁ i₂ (f x₁ x₂) = g (i₁ x₁) (i₂ x₂) := by
  /-
    R : Type u_1
    inst✝¹² : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    inst✝¹¹ : AddCommMonoid M₁
    inst✝¹⁰ : AddCommMonoid M₂
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M₁
    inst✝⁷ : Module R M₂
    inst✝⁶ : Module R M
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    N₁ : Type u_6
    N₂ : Type u_7
    N : Type u_8
    inst✝⁵ : AddCommMonoid N₁
    inst✝⁴ : AddCommMonoid N₂
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N₁
    inst✝¹ : Module R N₂
    inst✝ : Module R N
    g : LinearMap (RingHom.id R) N₁ (LinearMap (RingHom.id R) N₂ N)
    hf : IsTensorProduct f
    hg : IsTensorProduct g
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    x₁ : M₁
    x₂ : M₂
    ⊢ Eq ((hf.map hg i₁ i₂) ((f x₁) x₂)) ((g (i₁ x₁)) (i₂ x₂))
  -/
  delta IsTensorProduct.map
  /-
    R : Type u_1
    inst✝¹² : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    inst✝¹¹ : AddCommMonoid M₁
    inst✝¹⁰ : AddCommMonoid M₂
    inst✝⁹ : AddCommMonoid M
    inst✝⁸ : Module R M₁
    inst✝⁷ : Module R M₂
    inst✝⁶ : Module R M
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    N₁ : Type u_6
    N₂ : Type u_7
    N : Type u_8
    inst✝⁵ : AddCommMonoid N₁
    inst✝⁴ : AddCommMonoid N₂
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N₁
    inst✝¹ : Module R N₂
    inst✝ : Module R N
    g : LinearMap (RingHom.id R) N₁ (LinearMap (RingHom.id R) N₂ N)
    hf : IsTensorProduct f
    hg : IsTensorProduct g
    i₁ : LinearMap (RingHom.id R) M₁ N₁
    i₂ : LinearMap (RingHom.id R) M₂ N₂
    x₁ : M₁
    x₂ : M₂
    ⊢ Eq (((↑hg.equiv).comp ((TensorProduct.map i₁ i₂).comp ↑hf.equiv.symm)) ((f x …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem IsTensorProduct.inductionOn (h : IsTensorProduct f) {C : M → Prop} (m : M) (h0 : C 0)
    (htmul : ∀ x y, C (f x y)) (hadd : ∀ x y, C x → C y → C (x + y)) : C m := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    h : IsTensorProduct f
    C : M → Prop
    m : M
    h0 : C 0
    htmul : ∀ (x : M₁) (y : M₂), C ((f x) y)
    hadd : ∀ (x y : M), C x → C y → C (HAdd.hAdd x y)
    ⊢ C m
  -/
  rw [← h.equiv.right_inv m]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    h : IsTensorProduct f
    C : M → Prop
    m : M
    h0 : C 0
    htmul : ∀ (x : M₁) (y : M₂), C ((f x) y)
    hadd : ∀ (x y : M), C x → C y → C (HAdd.hAdd x y)
    ⊢ C ((↑h.equiv).toFun (h.equiv.invFun m))
  -/
  generalize h.equiv.invFun m = y
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    h : IsTensorProduct f
    C : M → Prop
    m : M
    h0 : C 0
    htmul : ∀ (x : M₁) (y : M₂), C ((f x) y)
    hadd : ∀ (x y : M), C x → C y → C (HAdd.hAdd x y)
    y : TensorProduct R M₁ M₂
    ⊢ C ((↑h.equiv).toFun y)
  -/
  change C (TensorProduct.lift f y)
  induction y with
  | zero => rwa [map_zero]
  | tmul _ _ =>
    rw [TensorProduct.lift.tmul]
    apply htmul
  | add _ _ _ _ =>
    rw [map_add]
    apply hadd <;> assumption


lemma IsTensorProduct.of_equiv (e : M₁ ⊗[R] M₂ ≃ₗ[R] M) (he : ∀ x y, e (x ⊗ₜ y) = f x y) :
    IsTensorProduct f := by
  have : TensorProduct.lift f = e := by
    ext x y
    simp [he]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    M₁ : Type u_2
    M₂ : Type u_3
    M : Type u_4
    inst✝⁵ : AddCommMonoid M₁
    inst✝⁴ : AddCommMonoid M₂
    inst✝³ : AddCommMonoid M
    inst✝² : Module R M₁
    inst✝¹ : Module R M₂
    inst✝ : Module R M
    f : LinearMap (RingHom.id R) M₁ (LinearMap (RingHom.id R) M₂ M)
    e : LinearEquiv (RingHom.id R) (TensorProduct R M₁ M₂) M
    he : ∀ (x : M₁) (y : M₂), Eq (e (TensorProduct.tmul R x y)) ((f x) y)
    this : Eq (TensorProduct.lift f) ↑e
    ⊢ IsTensorProduct f
  -/
  simpa [IsTensorProduct, this] using e.bijective
  /-
    🎉 no goals
  -/


/-- Given an `R`-algebra `S` and an `R`-module `M`, an `S`-module `N` together with a map
`f : M →ₗ[R] N` is the base change of `M` to `S` if the map `S × M → N, (s, m) ↦ s • f m` is the
tensor product. -/
def IsBaseChange : Prop :=
  IsTensorProduct
    (((Algebra.linearMap S <| Module.End S (M →ₗ[R] N)).flip f).restrictScalars R)

-- Porting note: split `variable`

/-- Suppose `f : M →ₗ[R] N` is the base change of `M` along `R → S`. Then any `R`-linear map from
`M` to an `S`-module factors through `f`. -/
noncomputable nonrec def IsBaseChange.lift (g : M →ₗ[R] Q) : N →ₗ[S] Q :=
  { h.lift
      (((Algebra.linearMap S <| Module.End S (M →ₗ[R] Q)).flip g).restrictScalars R) with
    map_smul' := fun r x => by
      /-
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝¹⁴ : AddCommMonoid M
        inst✝¹³ : AddCommMonoid N
        inst✝¹² : CommSemiring R
        inst✝¹¹ : CommSemiring S
        inst✝¹⁰ : Algebra R S
        inst✝⁹ : Module R M
        inst✝⁸ : Module R N
        inst✝⁷ : Module S N
        inst✝⁶ : IsScalarTower R S N
        f : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f
        P : Type u_2
        Q : Type u_3
        inst✝⁵ : AddCommMonoid P
        inst✝⁴ : Module R P
        inst✝³ : AddCommMonoid Q
        inst✝² : Module S Q
        inst✝¹ : Module R Q
        inst✝ : IsScalarTower R S Q
        g : LinearMap (RingHom.id R) M Q
        r : S
        x : N
        ⊢ Eq (__src✝.toFun (HSMul.hSMul r x)) (HSMul.hSMul ((RingHom.id S) r) (__src✝. …
      -/
      let F := ((Algebra.linearMap S <| Module.End S (M →ₗ[R] Q)).flip g).restrictScalars R
      /-
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝¹⁴ : AddCommMonoid M
        inst✝¹³ : AddCommMonoid N
        inst✝¹² : CommSemiring R
        inst✝¹¹ : CommSemiring S
        inst✝¹⁰ : Algebra R S
        inst✝⁹ : Module R M
        inst✝⁸ : Module R N
        inst✝⁷ : Module S N
        inst✝⁶ : IsScalarTower R S N
        f : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f
        P : Type u_2
        Q : Type u_3
        inst✝⁵ : AddCommMonoid P
        inst✝⁴ : Module R P
        inst✝³ : AddCommMonoid Q
        inst✝² : Module S Q
        inst✝¹ : Module R Q
        inst✝ : IsScalarTower R S Q
        g : LinearMap (RingHom.id R) M Q
        r : S
        x : N
        F : LinearMap (RingHom.id R) S (LinearMap (RingHom.id R) M Q) := ↑R ((Algebra. …
        ⊢ Eq (__src✝.toFun (HSMul.hSMul r x)) (HSMul.hSMul ((RingHom.id S) r) (__src✝. …
      -/
      have hF : ∀ (s : S) (m : M), h.lift F (s • f m) = s • g m := h.lift_eq F
      /-
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝¹⁴ : AddCommMonoid M
        inst✝¹³ : AddCommMonoid N
        inst✝¹² : CommSemiring R
        inst✝¹¹ : CommSemiring S
        inst✝¹⁰ : Algebra R S
        inst✝⁹ : Module R M
        inst✝⁸ : Module R N
        inst✝⁷ : Module S N
        inst✝⁶ : IsScalarTower R S N
        f : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f
        P : Type u_2
        Q : Type u_3
        inst✝⁵ : AddCommMonoid P
        inst✝⁴ : Module R P
        inst✝³ : AddCommMonoid Q
        inst✝² : Module S Q
        inst✝¹ : Module R Q
        inst✝ : IsScalarTower R S Q
        g : LinearMap (RingHom.id R) M Q
        r : S
        x : N
        F : LinearMap (RingHom.id R) S (LinearMap (RingHom.id R) M Q) := ↑R ((Algebra. …
        hF : ∀ (s : S) (m : M), Eq ((IsTensorProduct.lift h F) (HSMul.hSMul s (f m)))  …
        ⊢ Eq (__src✝.toFun (HSMul.hSMul r x)) (HSMul.hSMul ((RingHom.id S) r) (__src✝. …
      -/
      change h.lift F (r • x) = r • h.lift F x
      /-
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝¹⁴ : AddCommMonoid M
        inst✝¹³ : AddCommMonoid N
        inst✝¹² : CommSemiring R
        inst✝¹¹ : CommSemiring S
        inst✝¹⁰ : Algebra R S
        inst✝⁹ : Module R M
        inst✝⁸ : Module R N
        inst✝⁷ : Module S N
        inst✝⁶ : IsScalarTower R S N
        f : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f
        P : Type u_2
        Q : Type u_3
        inst✝⁵ : AddCommMonoid P
        inst✝⁴ : Module R P
        inst✝³ : AddCommMonoid Q
        inst✝² : Module S Q
        inst✝¹ : Module R Q
        inst✝ : IsScalarTower R S Q
        g : LinearMap (RingHom.id R) M Q
        r : S
        x : N
        F : LinearMap (RingHom.id R) S (LinearMap (RingHom.id R) M Q) := ↑R ((Algebra. …
        hF : ∀ (s : S) (m : M), Eq ((IsTensorProduct.lift h F) (HSMul.hSMul s (f m)))  …
        ⊢ Eq ((IsTensorProduct.lift h F) (HSMul.hSMul r x)) (HSMul.hSMul r ((IsTensorP …
      -/
      apply h.inductionOn x
        /-
          case h0
          R : Type u_1
          M : Type v₁
          N : Type v₂
          S : Type v₃
          inst✝¹⁴ : AddCommMonoid M
          inst✝¹³ : AddCommMonoid N
          inst✝¹² : CommSemiring R
          inst✝¹¹ : CommSemiring S
          inst✝¹⁰ : Algebra R S
          inst✝⁹ : Module R M
          inst✝⁸ : Module R N
          inst✝⁷ : Module S N
          inst✝⁶ : IsScalarTower R S N
          f : LinearMap (RingHom.id R) M N
          h : IsBaseChange S f
          P : Type u_2
          Q : Type u_3
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : Module R P
          inst✝³ : AddCommMonoid Q
          inst✝² : Module S Q
          inst✝¹ : Module R Q
          inst✝ : IsScalarTower R S Q
          g : LinearMap (RingHom.id R) M Q
          r : S
          x : N
          F : LinearMap (RingHom.id R) S (LinearMap (RingHom.id R) M Q) := ↑R ((Algebra. …
          hF : ∀ (s : S) (m : M), Eq ((IsTensorProduct.lift h F) (HSMul.hSMul s (f m)))  …
          ⊢ Eq ((IsTensorProduct.lift h F) (HSMul.hSMul r 0)) (HSMul.hSMul r ((IsTensorP …
        -/
      · rw [smul_zero, map_zero, smul_zero]
        /-
          🎉 no goals
        -/
        /-
          case htmul
          R : Type u_1
          M : Type v₁
          N : Type v₂
          S : Type v₃
          inst✝¹⁴ : AddCommMonoid M
          inst✝¹³ : AddCommMonoid N
          inst✝¹² : CommSemiring R
          inst✝¹¹ : CommSemiring S
          inst✝¹⁰ : Algebra R S
          inst✝⁹ : Module R M
          inst✝⁸ : Module R N
          inst✝⁷ : Module S N
          inst✝⁶ : IsScalarTower R S N
          f : LinearMap (RingHom.id R) M N
          h : IsBaseChange S f
          P : Type u_2
          Q : Type u_3
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : Module R P
          inst✝³ : AddCommMonoid Q
          inst✝² : Module S Q
          inst✝¹ : Module R Q
          inst✝ : IsScalarTower R S Q
          g : LinearMap (RingHom.id R) M Q
          r : S
          x : N
          F : LinearMap (RingHom.id R) S (LinearMap (RingHom.id R) M Q) := ↑R ((Algebra. …
          hF : ∀ (s : S) (m : M), Eq ((IsTensorProduct.lift h F) (HSMul.hSMul s (f m)))  …
          ⊢ ∀ (x : S) (y : M), Eq ((IsTensorProduct.lift h F) (HSMul.hSMul r (((↑R ((Alg …
        -/
      · intro s m
        /-
          case htmul
          R : Type u_1
          M : Type v₁
          N : Type v₂
          S : Type v₃
          inst✝¹⁴ : AddCommMonoid M
          inst✝¹³ : AddCommMonoid N
          inst✝¹² : CommSemiring R
          inst✝¹¹ : CommSemiring S
          inst✝¹⁰ : Algebra R S
          inst✝⁹ : Module R M
          inst✝⁸ : Module R N
          inst✝⁷ : Module S N
          inst✝⁶ : IsScalarTower R S N
          f : LinearMap (RingHom.id R) M N
          h : IsBaseChange S f
          P : Type u_2
          Q : Type u_3
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : Module R P
          inst✝³ : AddCommMonoid Q
          inst✝² : Module S Q
          inst✝¹ : Module R Q
          inst✝ : IsScalarTower R S Q
          g : LinearMap (RingHom.id R) M Q
          r : S
          x : N
          F : LinearMap (RingHom.id R) S (LinearMap (RingHom.id R) M Q) := ↑R ((Algebra. …
          hF : ∀ (s : S) (m : M), Eq ((IsTensorProduct.lift h F) (HSMul.hSMul s (f m)))  …
          s : S
          m : M
          ⊢ Eq ((IsTensorProduct.lift h F) (HSMul.hSMul r (((↑R ((Algebra.linearMap S (M …
        -/
        change h.lift F (r • s • f m) = r • h.lift F (s • f m)
        /-
          case htmul
          R : Type u_1
          M : Type v₁
          N : Type v₂
          S : Type v₃
          inst✝¹⁴ : AddCommMonoid M
          inst✝¹³ : AddCommMonoid N
          inst✝¹² : CommSemiring R
          inst✝¹¹ : CommSemiring S
          inst✝¹⁰ : Algebra R S
          inst✝⁹ : Module R M
          inst✝⁸ : Module R N
          inst✝⁷ : Module S N
          inst✝⁶ : IsScalarTower R S N
          f : LinearMap (RingHom.id R) M N
          h : IsBaseChange S f
          P : Type u_2
          Q : Type u_3
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : Module R P
          inst✝³ : AddCommMonoid Q
          inst✝² : Module S Q
          inst✝¹ : Module R Q
          inst✝ : IsScalarTower R S Q
          g : LinearMap (RingHom.id R) M Q
          r : S
          x : N
          F : LinearMap (RingHom.id R) S (LinearMap (RingHom.id R) M Q) := ↑R ((Algebra. …
          hF : ∀ (s : S) (m : M), Eq ((IsTensorProduct.lift h F) (HSMul.hSMul s (f m)))  …
          s : S
          m : M
          ⊢ Eq ((IsTensorProduct.lift h F) (HSMul.hSMul r (HSMul.hSMul s (f m)))) (HSMul …
        -/
        rw [← mul_smul, hF, hF, mul_smul]
        /-
          🎉 no goals
        -/
        /-
          case hadd
          R : Type u_1
          M : Type v₁
          N : Type v₂
          S : Type v₃
          inst✝¹⁴ : AddCommMonoid M
          inst✝¹³ : AddCommMonoid N
          inst✝¹² : CommSemiring R
          inst✝¹¹ : CommSemiring S
          inst✝¹⁰ : Algebra R S
          inst✝⁹ : Module R M
          inst✝⁸ : Module R N
          inst✝⁷ : Module S N
          inst✝⁶ : IsScalarTower R S N
          f : LinearMap (RingHom.id R) M N
          h : IsBaseChange S f
          P : Type u_2
          Q : Type u_3
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : Module R P
          inst✝³ : AddCommMonoid Q
          inst✝² : Module S Q
          inst✝¹ : Module R Q
          inst✝ : IsScalarTower R S Q
          g : LinearMap (RingHom.id R) M Q
          r : S
          x : N
          F : LinearMap (RingHom.id R) S (LinearMap (RingHom.id R) M Q) := ↑R ((Algebra. …
          hF : ∀ (s : S) (m : M), Eq ((IsTensorProduct.lift h F) (HSMul.hSMul s (f m)))  …
          ⊢ ∀ (x y : N), Eq ((IsTensorProduct.lift h F) (HSMul.hSMul r x)) (HSMul.hSMul  …
        -/
      · intro x₁ x₂ e₁ e₂
        /-
          case hadd
          R : Type u_1
          M : Type v₁
          N : Type v₂
          S : Type v₃
          inst✝¹⁴ : AddCommMonoid M
          inst✝¹³ : AddCommMonoid N
          inst✝¹² : CommSemiring R
          inst✝¹¹ : CommSemiring S
          inst✝¹⁰ : Algebra R S
          inst✝⁹ : Module R M
          inst✝⁸ : Module R N
          inst✝⁷ : Module S N
          inst✝⁶ : IsScalarTower R S N
          f : LinearMap (RingHom.id R) M N
          h : IsBaseChange S f
          P : Type u_2
          Q : Type u_3
          inst✝⁵ : AddCommMonoid P
          inst✝⁴ : Module R P
          inst✝³ : AddCommMonoid Q
          inst✝² : Module S Q
          inst✝¹ : Module R Q
          inst✝ : IsScalarTower R S Q
          g : LinearMap (RingHom.id R) M Q
          r : S
          x : N
          F : LinearMap (RingHom.id R) S (LinearMap (RingHom.id R) M Q) := ↑R ((Algebra. …
          hF : ∀ (s : S) (m : M), Eq ((IsTensorProduct.lift h F) (HSMul.hSMul s (f m)))  …
          x₁ x₂ : N
          e₁ : Eq ((IsTensorProduct.lift h F) (HSMul.hSMul r x₁)) (HSMul.hSMul r ((IsTen …
          e₂ : Eq ((IsTensorProduct.lift h F) (HSMul.hSMul r x₂)) (HSMul.hSMul r ((IsTen …
          ⊢ Eq ((IsTensorProduct.lift h F) (HSMul.hSMul r (HAdd.hAdd x₁ x₂))) (HSMul.hSM …
        -/
        rw [map_add, smul_add, map_add, smul_add, e₁, e₂] }
        /-
          🎉 no goals
        -/


nonrec theorem IsBaseChange.lift_eq (g : M →ₗ[R] Q) (x : M) : h.lift g (f x) = g x := by
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝¹² : AddCommMonoid M
    inst✝¹¹ : AddCommMonoid N
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    f : LinearMap (RingHom.id R) M N
    h : IsBaseChange S f
    Q : Type u_3
    inst✝³ : AddCommMonoid Q
    inst✝² : Module S Q
    inst✝¹ : Module R Q
    inst✝ : IsScalarTower R S Q
    g : LinearMap (RingHom.id R) M Q
    x : M
    ⊢ Eq ((h.lift g) (f x)) (g x)
  -/
  have hF : ∀ (s : S) (m : M), h.lift g (s • f m) = s • g m := h.lift_eq _
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝¹² : AddCommMonoid M
    inst✝¹¹ : AddCommMonoid N
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    inst✝⁷ : Module R M
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    f : LinearMap (RingHom.id R) M N
    h : IsBaseChange S f
    Q : Type u_3
    inst✝³ : AddCommMonoid Q
    inst✝² : Module S Q
    inst✝¹ : Module R Q
    inst✝ : IsScalarTower R S Q
    g : LinearMap (RingHom.id R) M Q
    x : M
    hF : ∀ (s : S) (m : M), Eq ((h.lift g) (HSMul.hSMul s (f m))) (HSMul.hSMul s ( …
    ⊢ Eq ((h.lift g) (f x)) (g x)
  -/
                     /-
                       🎉 no goals
                     -/
  convert hF 1 x <;> rw [one_smul]
                     /-
                       🎉 no goals
                     -/


theorem IsBaseChange.lift_comp (g : M →ₗ[R] Q) : ((h.lift g).restrictScalars R).comp f = g :=
  LinearMap.ext (h.lift_eq g)


@[elab_as_elim]
nonrec theorem IsBaseChange.inductionOn (x : N) (P : N → Prop) (h₁ : P 0) (h₂ : ∀ m : M, P (f m))
    (h₃ : ∀ (s : S) (n), P n → P (s • n)) (h₄ : ∀ n₁ n₂, P n₁ → P n₂ → P (n₁ + n₂)) : P x :=
  h.inductionOn x h₁ (fun _ _ => h₃ _ _ (h₂ _)) h₄


theorem IsBaseChange.algHom_ext (g₁ g₂ : N →ₗ[S] Q) (e : ∀ x, g₁ (f x) = g₂ (f x)) : g₁ = g₂ := by
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝¹⁰ : AddCommMonoid M
    inst✝⁹ : AddCommMonoid N
    inst✝⁸ : CommSemiring R
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module R M
    inst✝⁴ : Module R N
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    f : LinearMap (RingHom.id R) M N
    h : IsBaseChange S f
    Q : Type u_3
    inst✝¹ : AddCommMonoid Q
    inst✝ : Module S Q
    g₁ g₂ : LinearMap (RingHom.id S) N Q
    e : ∀ (x : M), Eq (g₁ (f x)) (g₂ (f x))
    ⊢ Eq g₁ g₂
  -/
  ext x
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝¹⁰ : AddCommMonoid M
    inst✝⁹ : AddCommMonoid N
    inst✝⁸ : CommSemiring R
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    inst✝⁵ : Module R M
    inst✝⁴ : Module R N
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    f : LinearMap (RingHom.id R) M N
    h : IsBaseChange S f
    Q : Type u_3
    inst✝¹ : AddCommMonoid Q
    inst✝ : Module S Q
    g₁ g₂ : LinearMap (RingHom.id S) N Q
    e : ∀ (x : M), Eq (g₁ (f x)) (g₂ (f x))
    x : N
    ⊢ Eq (g₁ x) (g₂ x)
  -/
  refine h.inductionOn x _ ?_ ?_ ?_ ?_
    /-
      case h.refine_1
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid N
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f
      Q : Type u_3
      inst✝¹ : AddCommMonoid Q
      inst✝ : Module S Q
      g₁ g₂ : LinearMap (RingHom.id S) N Q
      e : ∀ (x : M), Eq (g₁ (f x)) (g₂ (f x))
      x : N
      ⊢ Eq (g₁ 0) (g₂ 0)
    -/
  · rw [map_zero, map_zero]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid N
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f
      Q : Type u_3
      inst✝¹ : AddCommMonoid Q
      inst✝ : Module S Q
      g₁ g₂ : LinearMap (RingHom.id S) N Q
      e : ∀ (x : M), Eq (g₁ (f x)) (g₂ (f x))
      x : N
      ⊢ ∀ (m : M), Eq (g₁ (f m)) (g₂ (f m))
    -/
  · assumption
    /-
      🎉 no goals
    -/
    /-
      case h.refine_3
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid N
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f
      Q : Type u_3
      inst✝¹ : AddCommMonoid Q
      inst✝ : Module S Q
      g₁ g₂ : LinearMap (RingHom.id S) N Q
      e : ∀ (x : M), Eq (g₁ (f x)) (g₂ (f x))
      x : N
      ⊢ ∀ (s : S) (n : N), Eq (g₁ n) (g₂ n) → Eq (g₁ (HSMul.hSMul s n)) (g₂ (HSMul.h …
    -/
  · intro s n e'
    /-
      case h.refine_3
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid N
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f
      Q : Type u_3
      inst✝¹ : AddCommMonoid Q
      inst✝ : Module S Q
      g₁ g₂ : LinearMap (RingHom.id S) N Q
      e : ∀ (x : M), Eq (g₁ (f x)) (g₂ (f x))
      x : N
      s : S
      n : N
      e' : Eq (g₁ n) (g₂ n)
      ⊢ Eq (g₁ (HSMul.hSMul s n)) (g₂ (HSMul.hSMul s n))
    -/
    rw [g₁.map_smul, g₂.map_smul, e']
    /-
      🎉 no goals
    -/
    /-
      case h.refine_4
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid N
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f
      Q : Type u_3
      inst✝¹ : AddCommMonoid Q
      inst✝ : Module S Q
      g₁ g₂ : LinearMap (RingHom.id S) N Q
      e : ∀ (x : M), Eq (g₁ (f x)) (g₂ (f x))
      x : N
      ⊢ ∀ (n₁ n₂ : N), Eq (g₁ n₁) (g₂ n₁) → Eq (g₁ n₂) (g₂ n₂) → Eq (g₁ (HAdd.hAdd n …
    -/
  · intro x y e₁ e₂
    /-
      case h.refine_4
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝¹⁰ : AddCommMonoid M
      inst✝⁹ : AddCommMonoid N
      inst✝⁸ : CommSemiring R
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      inst✝⁵ : Module R M
      inst✝⁴ : Module R N
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f
      Q : Type u_3
      inst✝¹ : AddCommMonoid Q
      inst✝ : Module S Q
      g₁ g₂ : LinearMap (RingHom.id S) N Q
      e : ∀ (x : M), Eq (g₁ (f x)) (g₂ (f x))
      x✝ x y : N
      e₁ : Eq (g₁ x) (g₂ x)
      e₂ : Eq (g₁ y) (g₂ y)
      ⊢ Eq (g₁ (HAdd.hAdd x y)) (g₂ (HAdd.hAdd x y))
    -/
    rw [map_add, map_add, e₁, e₂]
    /-
      🎉 no goals
    -/


theorem IsBaseChange.algHom_ext' [Module R Q] [IsScalarTower R S Q] (g₁ g₂ : N →ₗ[S] Q)
    (e : (g₁.restrictScalars R).comp f = (g₂.restrictScalars R).comp f) : g₁ = g₂ :=
  h.algHom_ext g₁ g₂ (LinearMap.congr_fun e)


theorem TensorProduct.isBaseChange : IsBaseChange S (TensorProduct.mk R S M 1) := by
  /-
    R : Type u_1
    M : Type v₁
    S : Type v₃
    inst✝⁴ : AddCommMonoid M
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : Module R M
    ⊢ IsBaseChange S ((TensorProduct.mk R S M) 1)
  -/
  delta IsBaseChange
  /-
    R : Type u_1
    M : Type v₁
    S : Type v₃
    inst✝⁴ : AddCommMonoid M
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : Module R M
    ⊢ IsTensorProduct (↑R ((Algebra.linearMap S (Module.End S (LinearMap (RingHom. …
  -/
  convert TensorProduct.isTensorProduct R S M using 1
  /-
    case h.e'_12
    R : Type u_1
    M : Type v₁
    S : Type v₃
    inst✝⁴ : AddCommMonoid M
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : Module R M
    ⊢ Eq (↑R ((Algebra.linearMap S (Module.End S (LinearMap (RingHom.id R) M (Tens …
  -/
  ext s x
  /-
    case h.e'_12.h.h
    R : Type u_1
    M : Type v₁
    S : Type v₃
    inst✝⁴ : AddCommMonoid M
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : Module R M
    s : S
    x : M
    ⊢ Eq (((↑R ((Algebra.linearMap S (Module.End S (LinearMap (RingHom.id R) M (Te …
  -/
  change s • (1 : S) ⊗ₜ[R] x = s ⊗ₜ[R] x
  /-
    case h.e'_12.h.h
    R : Type u_1
    M : Type v₁
    S : Type v₃
    inst✝⁴ : AddCommMonoid M
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : Module R M
    s : S
    x : M
    ⊢ Eq (HSMul.hSMul s (TensorProduct.tmul R 1 x)) (TensorProduct.tmul R s x)
  -/
  rw [TensorProduct.smul_tmul']
  /-
    case h.e'_12.h.h
    R : Type u_1
    M : Type v₁
    S : Type v₃
    inst✝⁴ : AddCommMonoid M
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : Module R M
    s : S
    x : M
    ⊢ Eq (TensorProduct.tmul R (HSMul.hSMul s 1) x) (TensorProduct.tmul R s x)
  -/
  congr 1
  /-
    case h.e'_12.h.h.e_m
    R : Type u_1
    M : Type v₁
    S : Type v₃
    inst✝⁴ : AddCommMonoid M
    inst✝³ : CommSemiring R
    inst✝² : CommSemiring S
    inst✝¹ : Algebra R S
    inst✝ : Module R M
    s : S
    x : M
    ⊢ Eq (HSMul.hSMul s 1) s
  -/
  exact mul_one _
  /-
    🎉 no goals
  -/


/-- The base change of `M` along `R → S` is linearly equivalent to `S ⊗[R] M`. -/
noncomputable nonrec def IsBaseChange.equiv : S ⊗[R] M ≃ₗ[S] N :=
  { h.equiv with
    map_smul' := fun r x => by
      /-
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝¹² : AddCommMonoid M
        inst✝¹¹ : AddCommMonoid N
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Algebra R S
        inst✝⁷ : Module R M
        inst✝⁶ : Module R N
        inst✝⁵ : Module S N
        inst✝⁴ : IsScalarTower R S N
        f : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f
        P : Type u_2
        Q : Type u_3
        inst✝³ : AddCommMonoid P
        inst✝² : Module R P
        inst✝¹ : AddCommMonoid Q
        inst✝ : Module S Q
        r : S
        x : TensorProduct R S M
        ⊢ Eq ((↑__src✝).toFun (HSMul.hSMul r x)) (HSMul.hSMul ((RingHom.id S) r) ((↑__ …
      -/
      change h.equiv (r • x) = r • h.equiv x
      /-
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝¹² : AddCommMonoid M
        inst✝¹¹ : AddCommMonoid N
        inst✝¹⁰ : CommSemiring R
        inst✝⁹ : CommSemiring S
        inst✝⁸ : Algebra R S
        inst✝⁷ : Module R M
        inst✝⁶ : Module R N
        inst✝⁵ : Module S N
        inst✝⁴ : IsScalarTower R S N
        f : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f
        P : Type u_2
        Q : Type u_3
        inst✝³ : AddCommMonoid P
        inst✝² : Module R P
        inst✝¹ : AddCommMonoid Q
        inst✝ : Module S Q
        r : S
        x : TensorProduct R S M
        ⊢ Eq ((IsTensorProduct.equiv h) (HSMul.hSMul r x)) (HSMul.hSMul r ((IsTensorPr …
      -/
      refine TensorProduct.induction_on x ?_ ?_ ?_
        /-
          case refine_1
          R : Type u_1
          M : Type v₁
          N : Type v₂
          S : Type v₃
          inst✝¹² : AddCommMonoid M
          inst✝¹¹ : AddCommMonoid N
          inst✝¹⁰ : CommSemiring R
          inst✝⁹ : CommSemiring S
          inst✝⁸ : Algebra R S
          inst✝⁷ : Module R M
          inst✝⁶ : Module R N
          inst✝⁵ : Module S N
          inst✝⁴ : IsScalarTower R S N
          f : LinearMap (RingHom.id R) M N
          h : IsBaseChange S f
          P : Type u_2
          Q : Type u_3
          inst✝³ : AddCommMonoid P
          inst✝² : Module R P
          inst✝¹ : AddCommMonoid Q
          inst✝ : Module S Q
          r : S
          x : TensorProduct R S M
          ⊢ Eq ((IsTensorProduct.equiv h) (HSMul.hSMul r 0)) (HSMul.hSMul r ((IsTensorPr …
        -/
      · rw [smul_zero, map_zero, smul_zero]
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          R : Type u_1
          M : Type v₁
          N : Type v₂
          S : Type v₃
          inst✝¹² : AddCommMonoid M
          inst✝¹¹ : AddCommMonoid N
          inst✝¹⁰ : CommSemiring R
          inst✝⁹ : CommSemiring S
          inst✝⁸ : Algebra R S
          inst✝⁷ : Module R M
          inst✝⁶ : Module R N
          inst✝⁵ : Module S N
          inst✝⁴ : IsScalarTower R S N
          f : LinearMap (RingHom.id R) M N
          h : IsBaseChange S f
          P : Type u_2
          Q : Type u_3
          inst✝³ : AddCommMonoid P
          inst✝² : Module R P
          inst✝¹ : AddCommMonoid Q
          inst✝ : Module S Q
          r : S
          x : TensorProduct R S M
          ⊢ ∀ (x : S) (y : M), Eq ((IsTensorProduct.equiv h) (HSMul.hSMul r (TensorProdu …
        -/
      · intro x y
        -- Porting note (https://github.com/leanprover-community/mathlib4/issues/10745): was simp [smul_tmul', Algebra.ofId_apply]
        simp only [Algebra.linearMap_apply, lift.tmul, smul_eq_mul, LinearMap.mul_apply,
          LinearMap.smul_apply, IsTensorProduct.equiv_apply, Module.algebraMap_end_apply, map_mul,
          smul_tmul', eq_self_iff_true, LinearMap.coe_restrictScalars, LinearMap.flip_apply]
        /-
          case refine_3
          R : Type u_1
          M : Type v₁
          N : Type v₂
          S : Type v₃
          inst✝¹² : AddCommMonoid M
          inst✝¹¹ : AddCommMonoid N
          inst✝¹⁰ : CommSemiring R
          inst✝⁹ : CommSemiring S
          inst✝⁸ : Algebra R S
          inst✝⁷ : Module R M
          inst✝⁶ : Module R N
          inst✝⁵ : Module S N
          inst✝⁴ : IsScalarTower R S N
          f : LinearMap (RingHom.id R) M N
          h : IsBaseChange S f
          P : Type u_2
          Q : Type u_3
          inst✝³ : AddCommMonoid P
          inst✝² : Module R P
          inst✝¹ : AddCommMonoid Q
          inst✝ : Module S Q
          r : S
          x : TensorProduct R S M
          ⊢ ∀ (x y : TensorProduct R S M), Eq ((IsTensorProduct.equiv h) (HSMul.hSMul r  …
        -/
      · intro x y hx hy
        /-
          case refine_3
          R : Type u_1
          M : Type v₁
          N : Type v₂
          S : Type v₃
          inst✝¹² : AddCommMonoid M
          inst✝¹¹ : AddCommMonoid N
          inst✝¹⁰ : CommSemiring R
          inst✝⁹ : CommSemiring S
          inst✝⁸ : Algebra R S
          inst✝⁷ : Module R M
          inst✝⁶ : Module R N
          inst✝⁵ : Module S N
          inst✝⁴ : IsScalarTower R S N
          f : LinearMap (RingHom.id R) M N
          h : IsBaseChange S f
          P : Type u_2
          Q : Type u_3
          inst✝³ : AddCommMonoid P
          inst✝² : Module R P
          inst✝¹ : AddCommMonoid Q
          inst✝ : Module S Q
          r : S
          x✝ x y : TensorProduct R S M
          hx : Eq ((IsTensorProduct.equiv h) (HSMul.hSMul r x)) (HSMul.hSMul r ((IsTenso …
          hy : Eq ((IsTensorProduct.equiv h) (HSMul.hSMul r y)) (HSMul.hSMul r ((IsTenso …
          ⊢ Eq ((IsTensorProduct.equiv h) (HSMul.hSMul r (HAdd.hAdd x y))) (HSMul.hSMul  …
        -/
        rw [map_add, smul_add, map_add, smul_add, hx, hy] }
        /-
          🎉 no goals
        -/


theorem IsBaseChange.equiv_tmul (s : S) (m : M) : h.equiv (s ⊗ₜ m) = s • f m :=
  TensorProduct.lift.tmul s m


theorem IsBaseChange.equiv_symm_apply (m : M) : h.equiv.symm (f m) = 1 ⊗ₜ m := by
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module S N
    inst✝ : IsScalarTower R S N
    f : LinearMap (RingHom.id R) M N
    h : IsBaseChange S f
    m : M
    ⊢ Eq (h.equiv.symm (f m)) (TensorProduct.tmul R 1 m)
  -/
  rw [h.equiv.symm_apply_eq, h.equiv_tmul, one_smul]
  /-
    🎉 no goals
  -/


lemma IsBaseChange.of_equiv (e : S ⊗[R] M ≃ₗ[S] N) (he : ∀ x, e (1 ⊗ₜ x) = f x) :
    IsBaseChange S f := by
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module S N
    inst✝ : IsScalarTower R S N
    f : LinearMap (RingHom.id R) M N
    e : LinearEquiv (RingHom.id S) (TensorProduct R S M) N
    he : ∀ (x : M), Eq (e (TensorProduct.tmul R 1 x)) (f x)
    ⊢ IsBaseChange S f
  -/
  apply IsTensorProduct.of_equiv (e.restrictScalars R)
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module S N
    inst✝ : IsScalarTower R S N
    f : LinearMap (RingHom.id R) M N
    e : LinearEquiv (RingHom.id S) (TensorProduct R S M) N
    he : ∀ (x : M), Eq (e (TensorProduct.tmul R 1 x)) (f x)
    ⊢ ∀ (x : S) (y : M), Eq ((LinearEquiv.restrictScalars R e) (TensorProduct.tmul …
  -/
  intro x y
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module S N
    inst✝ : IsScalarTower R S N
    f : LinearMap (RingHom.id R) M N
    e : LinearEquiv (RingHom.id S) (TensorProduct R S M) N
    he : ∀ (x : M), Eq (e (TensorProduct.tmul R 1 x)) (f x)
    x : S
    y : M
    ⊢ Eq ((LinearEquiv.restrictScalars R e) (TensorProduct.tmul R x y)) (((↑R ((Al …
  -/
  simp [show x ⊗ₜ[R] y = x • (1 ⊗ₜ[R] y) by simp [smul_tmul'], he]
  /-
    🎉 no goals
  -/


/-- If `N` is the base change of `M` to `A`, then `N ⊗[R] P` is the base change
of `M ⊗[R] P` to `A`. This is simply the isomorphism
`A ⊗[S] (M ⊗[R] P) ≃ₗ[A] (A ⊗[S] M) ⊗[R] P`. -/
lemma isBaseChange_tensorProduct_map {f : M →ₗ[S] N} (hf : IsBaseChange A f) :
    IsBaseChange A (AlgebraTensorModule.map f (LinearMap.id (R := R) (M := P))) := by
  let e : A ⊗[S] M ⊗[R] P ≃ₗ[A] N ⊗[R] P := (AlgebraTensorModule.assoc R S A A M P).symm.trans
    (AlgebraTensorModule.congr hf.equiv (LinearEquiv.refl R P))
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝¹⁹ : AddCommMonoid M
    inst✝¹⁸ : AddCommMonoid N
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : CommSemiring S
    inst✝¹⁵ : Algebra R S
    inst✝¹⁴ : Module R M
    inst✝¹³ : Module R N
    inst✝¹² : Module S N
    inst✝¹¹ : IsScalarTower R S N
    P : Type u_2
    inst✝¹⁰ : AddCommMonoid P
    inst✝⁹ : Module R P
    A : Type u_4
    inst✝⁸ : CommSemiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : Algebra S A
    inst✝⁵ : IsScalarTower R S A
    inst✝⁴ : Module S M
    inst✝³ : IsScalarTower R S M
    inst✝² : Module A N
    inst✝¹ : IsScalarTower S A N
    inst✝ : IsScalarTower R A N
    f : LinearMap (RingHom.id S) M N
    hf : IsBaseChange A f
    e : LinearEquiv (RingHom.id A) (TensorProduct S A (TensorProduct R M P)) (Tens …
    ⊢ IsBaseChange A (TensorProduct.AlgebraTensorModule.map f LinearMap.id)
  -/
  refine IsBaseChange.of_equiv e (fun x ↦ ?_)
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝¹⁹ : AddCommMonoid M
    inst✝¹⁸ : AddCommMonoid N
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : CommSemiring S
    inst✝¹⁵ : Algebra R S
    inst✝¹⁴ : Module R M
    inst✝¹³ : Module R N
    inst✝¹² : Module S N
    inst✝¹¹ : IsScalarTower R S N
    P : Type u_2
    inst✝¹⁰ : AddCommMonoid P
    inst✝⁹ : Module R P
    A : Type u_4
    inst✝⁸ : CommSemiring A
    inst✝⁷ : Algebra R A
    inst✝⁶ : Algebra S A
    inst✝⁵ : IsScalarTower R S A
    inst✝⁴ : Module S M
    inst✝³ : IsScalarTower R S M
    inst✝² : Module A N
    inst✝¹ : IsScalarTower S A N
    inst✝ : IsScalarTower R A N
    f : LinearMap (RingHom.id S) M N
    hf : IsBaseChange A f
    e : LinearEquiv (RingHom.id A) (TensorProduct S A (TensorProduct R M P)) (Tens …
    x : TensorProduct R M P
    ⊢ Eq (e (TensorProduct.tmul S 1 x)) ((TensorProduct.AlgebraTensorModule.map f  …
  -/
  induction' x with m p _ _ h1 h2
    /-
      case zero
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝¹⁹ : AddCommMonoid M
      inst✝¹⁸ : AddCommMonoid N
      inst✝¹⁷ : CommSemiring R
      inst✝¹⁶ : CommSemiring S
      inst✝¹⁵ : Algebra R S
      inst✝¹⁴ : Module R M
      inst✝¹³ : Module R N
      inst✝¹² : Module S N
      inst✝¹¹ : IsScalarTower R S N
      P : Type u_2
      inst✝¹⁰ : AddCommMonoid P
      inst✝⁹ : Module R P
      A : Type u_4
      inst✝⁸ : CommSemiring A
      inst✝⁷ : Algebra R A
      inst✝⁶ : Algebra S A
      inst✝⁵ : IsScalarTower R S A
      inst✝⁴ : Module S M
      inst✝³ : IsScalarTower R S M
      inst✝² : Module A N
      inst✝¹ : IsScalarTower S A N
      inst✝ : IsScalarTower R A N
      f : LinearMap (RingHom.id S) M N
      hf : IsBaseChange A f
      e : LinearEquiv (RingHom.id A) (TensorProduct S A (TensorProduct R M P)) (Tens …
      ⊢ Eq (e (TensorProduct.tmul S 1 0)) ((TensorProduct.AlgebraTensorModule.map f  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case tmul
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝¹⁹ : AddCommMonoid M
      inst✝¹⁸ : AddCommMonoid N
      inst✝¹⁷ : CommSemiring R
      inst✝¹⁶ : CommSemiring S
      inst✝¹⁵ : Algebra R S
      inst✝¹⁴ : Module R M
      inst✝¹³ : Module R N
      inst✝¹² : Module S N
      inst✝¹¹ : IsScalarTower R S N
      P : Type u_2
      inst✝¹⁰ : AddCommMonoid P
      inst✝⁹ : Module R P
      A : Type u_4
      inst✝⁸ : CommSemiring A
      inst✝⁷ : Algebra R A
      inst✝⁶ : Algebra S A
      inst✝⁵ : IsScalarTower R S A
      inst✝⁴ : Module S M
      inst✝³ : IsScalarTower R S M
      inst✝² : Module A N
      inst✝¹ : IsScalarTower S A N
      inst✝ : IsScalarTower R A N
      f : LinearMap (RingHom.id S) M N
      hf : IsBaseChange A f
      e : LinearEquiv (RingHom.id A) (TensorProduct S A (TensorProduct R M P)) (Tens …
      m : M
      p : P
      ⊢ Eq (e (TensorProduct.tmul S 1 (TensorProduct.tmul R m p))) ((TensorProduct.A …
    -/
  · simp [e, IsBaseChange.equiv_tmul]
    /-
      🎉 no goals
    -/
    /-
      case add
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝¹⁹ : AddCommMonoid M
      inst✝¹⁸ : AddCommMonoid N
      inst✝¹⁷ : CommSemiring R
      inst✝¹⁶ : CommSemiring S
      inst✝¹⁵ : Algebra R S
      inst✝¹⁴ : Module R M
      inst✝¹³ : Module R N
      inst✝¹² : Module S N
      inst✝¹¹ : IsScalarTower R S N
      P : Type u_2
      inst✝¹⁰ : AddCommMonoid P
      inst✝⁹ : Module R P
      A : Type u_4
      inst✝⁸ : CommSemiring A
      inst✝⁷ : Algebra R A
      inst✝⁶ : Algebra S A
      inst✝⁵ : IsScalarTower R S A
      inst✝⁴ : Module S M
      inst✝³ : IsScalarTower R S M
      inst✝² : Module A N
      inst✝¹ : IsScalarTower S A N
      inst✝ : IsScalarTower R A N
      f : LinearMap (RingHom.id S) M N
      hf : IsBaseChange A f
      e : LinearEquiv (RingHom.id A) (TensorProduct S A (TensorProduct R M P)) (Tens …
      x✝ y✝ : TensorProduct R M P
      h1 : Eq (e (TensorProduct.tmul S 1 x✝)) ((TensorProduct.AlgebraTensorModule.ma …
      h2 : Eq (e (TensorProduct.tmul S 1 y✝)) ((TensorProduct.AlgebraTensorModule.ma …
      ⊢ Eq (e (TensorProduct.tmul S 1 (HAdd.hAdd x✝ y✝))) ((TensorProduct.AlgebraTen …
    -/
  · simp [tmul_add, h1, h2]
    /-
      🎉 no goals
    -/


theorem IsBaseChange.of_lift_unique
    (h : ∀ (Q : Type max v₁ v₂ v₃) [AddCommMonoid Q],
      ∀ [Module R Q] [Module S Q], ∀ [IsScalarTower R S Q],
        ∀ g : M →ₗ[R] Q, ∃! g' : N →ₗ[S] Q, (g'.restrictScalars R).comp f = g) :
    IsBaseChange S f := by
  obtain ⟨g, hg, -⟩ :=
    h (ULift.{v₂} <| S ⊗[R] M)
      (ULift.moduleEquiv.symm.toLinearMap.comp <| TensorProduct.mk R S M 1)
  let f' : S ⊗[R] M →ₗ[R] N :=
    TensorProduct.lift (((LinearMap.flip (AlgHom.toLinearMap (Algebra.ofId S
      (Module.End S (M →ₗ[R] N))))) f).restrictScalars R)
  /-
    case intro.intro
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module S N
    inst✝ : IsScalarTower R S N
    f : LinearMap (RingHom.id R) M N
    h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
    g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
    hg : Eq ((↑R g).comp f) ((↑ULift.moduleEquiv.symm).comp ((TensorProduct.mk R S …
    f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
    ⊢ IsBaseChange S f
  -/
  change Function.Bijective f'
  let f'' : S ⊗[R] M →ₗ[S] N := by
    refine
      { f' with
        map_smul' := fun s x =>
          TensorProduct.induction_on x ?_ (fun s' y => smul_assoc s s' _) fun x y hx hy => ?_ }
    · dsimp; rw [map_zero, smul_zero, map_zero, smul_zero]
    · dsimp at *; rw [smul_add, map_add, map_add, smul_add, hx, hy]
  /-
    case intro.intro
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝⁸ : AddCommMonoid M
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : CommSemiring R
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    inst✝³ : Module R M
    inst✝² : Module R N
    inst✝¹ : Module S N
    inst✝ : IsScalarTower R S N
    f : LinearMap (RingHom.id R) M N
    h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
    g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
    hg : Eq ((↑R g).comp f) ((↑ULift.moduleEquiv.symm).comp ((TensorProduct.mk R S …
    f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
    f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
    ⊢ Function.Bijective ⇑f'
  -/
  simp_rw [DFunLike.ext_iff, LinearMap.comp_apply, LinearMap.restrictScalars_apply] at hg
  let fe : S ⊗[R] M ≃ₗ[S] N :=
    LinearEquiv.ofLinear f'' (ULift.moduleEquiv.toLinearMap.comp g) ?_ ?_
    /-
      case intro.intro.refine_3
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
      g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
      f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
      f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
      hg : ∀ (x : M), Eq (g (f x)) (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S  …
      fe : LinearEquiv (RingHom.id S) (TensorProduct R S M) N := LinearEquiv.ofLinea …
      ⊢ Function.Bijective ⇑f'
    -/
  · exact fe.bijective
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_1
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
      g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
      f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
      f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
      hg : ∀ (x : M), Eq (g (f x)) (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S  …
      ⊢ Eq (f''.comp ((↑ULift.moduleEquiv).comp g)) LinearMap.id
    -/
  · rw [← LinearMap.cancel_left (ULift.moduleEquiv : ULift.{max v₁ v₃} N ≃ₗ[S] N).symm.injective]
    /-
      case intro.intro.refine_1
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
      g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
      f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
      f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
      hg : ∀ (x : M), Eq (g (f x)) (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S  …
      ⊢ Eq ((↑ULift.moduleEquiv.symm).comp (f''.comp ((↑ULift.moduleEquiv).comp g))) …
    -/
    refine (h (ULift.{max v₁ v₃} N) <| ULift.moduleEquiv.symm.toLinearMap.comp f).unique ?_ rfl
    /-
      case intro.intro.refine_1
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
      g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
      f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
      f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
      hg : ∀ (x : M), Eq (g (f x)) (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S  …
      ⊢ Eq ((↑R ((↑ULift.moduleEquiv.symm).comp (f''.comp ((↑ULift.moduleEquiv).comp …
    -/
    ext x
    /-
      case intro.intro.refine_1.h.h
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
      g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
      f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
      f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
      hg : ∀ (x : M), Eq (g (f x)) (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S  …
      x : M
      ⊢ Eq (((↑R ((↑ULift.moduleEquiv.symm).comp (f''.comp ((↑ULift.moduleEquiv).com …
    -/
    simp only [LinearMap.comp_apply, LinearMap.restrictScalars_apply, hg]
    /-
      case intro.intro.refine_1.h.h
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
      g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
      f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
      f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
      hg : ∀ (x : M), Eq (g (f x)) (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S  …
      x : M
      ⊢ Eq (↑ULift.moduleEquiv.symm (f'' (↑ULift.moduleEquiv (↑ULift.moduleEquiv.sym …
    -/
    apply one_smul
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
      g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
      f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
      f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
      hg : ∀ (x : M), Eq (g (f x)) (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S  …
      ⊢ Eq (((↑ULift.moduleEquiv).comp g).comp f'') LinearMap.id
    -/
  · ext x
    /-
      case intro.intro.refine_2.a.h.h
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
      g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
      f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
      f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
      hg : ∀ (x : M), Eq (g (f x)) (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S  …
      x : M
      ⊢ Eq (((TensorProduct.AlgebraTensorModule.curry (((↑ULift.moduleEquiv).comp g) …
    -/
    change (g <| (1 : S) • f x).down = _
    /-
      case intro.intro.refine_2.a.h.h
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
      g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
      f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
      f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
      hg : ∀ (x : M), Eq (g (f x)) (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S  …
      x : M
      ⊢ Eq (g (HSMul.hSMul 1 (f x))).down (((TensorProduct.AlgebraTensorModule.curry …
    -/
    rw [one_smul, hg]
    /-
      case intro.intro.refine_2.a.h.h
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
      g : LinearMap (RingHom.id S) N (ULift.{v₂, max v₁ v₃} (TensorProduct R S M))
      f' : LinearMap (RingHom.id R) (TensorProduct R S M) N := TensorProduct.lift (↑ …
      f'' : LinearMap (RingHom.id S) (TensorProduct R S M) N := { toAddHom := f'.toA …
      hg : ∀ (x : M), Eq (g (f x)) (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S  …
      x : M
      ⊢ Eq (↑ULift.moduleEquiv.symm (((TensorProduct.mk R S M) 1) x)).down (((Tensor …
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem IsBaseChange.iff_lift_unique :
    IsBaseChange S f ↔
      ∀ (Q : Type max v₁ v₂ v₃) [AddCommMonoid Q],
        ∀ [Module R Q] [Module S Q],
          ∀ [IsScalarTower R S Q],
            ∀ g : M →ₗ[R] Q, ∃! g' : N →ₗ[S] Q, (g'.restrictScalars R).comp f = g :=
  ⟨fun h => by
    /-
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝⁸ : AddCommMonoid M
      inst✝⁷ : AddCommMonoid N
      inst✝⁶ : CommSemiring R
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      inst✝³ : Module R M
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f
      ⊢ ∀ (Q : Type (max v₁ v₂ v₃)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] [ …
    -/
    intros Q _ _ _ _ g
    /-
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝¹² : AddCommMonoid M
      inst✝¹¹ : AddCommMonoid N
      inst✝¹⁰ : CommSemiring R
      inst✝⁹ : CommSemiring S
      inst✝⁸ : Algebra R S
      inst✝⁷ : Module R M
      inst✝⁶ : Module R N
      inst✝⁵ : Module S N
      inst✝⁴ : IsScalarTower R S N
      f : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f
      Q : Type (max v₁ v₂ v₃)
      inst✝³ : AddCommMonoid Q
      inst✝² : Module R Q
      inst✝¹ : Module S Q
      inst✝ : IsScalarTower R S Q
      g : LinearMap (RingHom.id R) M Q
      ⊢ ExistsUnique fun g' => Eq ((↑R g').comp f) g
    -/
    exact ⟨h.lift g, h.lift_comp g, fun g' e => h.algHom_ext' _ _ (e.trans (h.lift_comp g).symm)⟩,
    /-
      🎉 no goals
    -/
    IsBaseChange.of_lift_unique f⟩


theorem IsBaseChange.ofEquiv (e : M ≃ₗ[R] N) : IsBaseChange R e.toLinearMap := by
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : CommSemiring R
    inst✝¹ : Module R M
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) M N
    ⊢ IsBaseChange R ↑e
  -/
  apply IsBaseChange.of_lift_unique
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : CommSemiring R
    inst✝¹ : Module R M
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) M N
    ⊢ ∀ (Q : Type (max v₁ v₂ u_1)) [inst : AddCommMonoid Q] [inst_1 : Module R Q]  …
  -/
  intro Q I₁ I₂ I₃ I₄ g
  have : I₂ = I₃ := by
    ext r q
    show (by let _ := I₂; exact r • q) = (by let _ := I₃; exact r • q)
    dsimp
    rw [← one_smul R q, smul_smul, ← @smul_assoc _ _ _ (id _) (id _) (id _) I₄, smul_eq_mul]
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : CommSemiring R
    inst✝¹ : Module R M
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) M N
    Q : Type (max v₁ v₂ u_1)
    I₁ : AddCommMonoid Q
    I₂ I₃ : Module R Q
    I₄ : IsScalarTower R R Q
    g : LinearMap (RingHom.id R) M Q
    this : Eq I₂ I₃
    ⊢ ExistsUnique fun g' => Eq ((↑R g').comp ↑e) g
  -/
  cases this
  refine
    ⟨g.comp e.symm.toLinearMap, by
      ext
      simp, ?_⟩
  /-
    case h.refl
    R : Type u_1
    M : Type v₁
    N : Type v₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : CommSemiring R
    inst✝¹ : Module R M
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) M N
    Q : Type (max v₁ v₂ u_1)
    I₁ : AddCommMonoid Q
    I₂ : Module R Q
    g : LinearMap (RingHom.id R) M Q
    I₄ : IsScalarTower R R Q
    ⊢ ∀ (y : LinearMap (RingHom.id R) N Q), (fun g' => Eq ((↑R g').comp ↑e) g) y → …
  -/
  rintro y (rfl : _ = _)
  /-
    case h.refl
    R : Type u_1
    M : Type v₁
    N : Type v₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : CommSemiring R
    inst✝¹ : Module R M
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) M N
    Q : Type (max v₁ v₂ u_1)
    I₁ : AddCommMonoid Q
    I₂ : Module R Q
    I₄ : IsScalarTower R R Q
    y : LinearMap (RingHom.id R) N Q
    ⊢ Eq y (((↑R y).comp ↑e).comp ↑e.symm)
  -/
  ext
  /-
    case h.refl.h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    inst✝⁴ : AddCommMonoid M
    inst✝³ : AddCommMonoid N
    inst✝² : CommSemiring R
    inst✝¹ : Module R M
    inst✝ : Module R N
    e : LinearEquiv (RingHom.id R) M N
    Q : Type (max v₁ v₂ u_1)
    I₁ : AddCommMonoid Q
    I₂ : Module R Q
    I₄ : IsScalarTower R R Q
    y : LinearMap (RingHom.id R) N Q
    x✝ : N
    ⊢ Eq (y x✝) ((((↑R y).comp ↑e).comp ↑e.symm) x✝)
  -/
  simp
  /-
    🎉 no goals
  -/


theorem IsBaseChange.comp {f : M →ₗ[R] N} (hf : IsBaseChange S f) {g : N →ₗ[S] O}
    (hg : IsBaseChange T g) : IsBaseChange T ((g.restrictScalars R).comp f) := by
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝¹⁹ : AddCommMonoid M
    inst✝¹⁸ : AddCommMonoid N
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : CommSemiring S
    inst✝¹⁵ : Algebra R S
    inst✝¹⁴ : Module R M
    inst✝¹³ : Module R N
    inst✝¹² : Module S N
    inst✝¹¹ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁰ : CommSemiring T
    inst✝⁹ : Algebra R T
    inst✝⁸ : Algebra S T
    inst✝⁷ : IsScalarTower R S T
    inst✝⁶ : AddCommMonoid O
    inst✝⁵ : Module R O
    inst✝⁴ : Module S O
    inst✝³ : Module T O
    inst✝² : IsScalarTower S T O
    inst✝¹ : IsScalarTower R S O
    inst✝ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    g : LinearMap (RingHom.id S) N O
    hg : IsBaseChange T g
    ⊢ IsBaseChange T ((↑R g).comp f)
  -/
  apply IsBaseChange.of_lift_unique
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝¹⁹ : AddCommMonoid M
    inst✝¹⁸ : AddCommMonoid N
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : CommSemiring S
    inst✝¹⁵ : Algebra R S
    inst✝¹⁴ : Module R M
    inst✝¹³ : Module R N
    inst✝¹² : Module S N
    inst✝¹¹ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁰ : CommSemiring T
    inst✝⁹ : Algebra R T
    inst✝⁸ : Algebra S T
    inst✝⁷ : IsScalarTower R S T
    inst✝⁶ : AddCommMonoid O
    inst✝⁵ : Module R O
    inst✝⁴ : Module S O
    inst✝³ : Module T O
    inst✝² : IsScalarTower S T O
    inst✝¹ : IsScalarTower R S O
    inst✝ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    g : LinearMap (RingHom.id S) N O
    hg : IsBaseChange T g
    ⊢ ∀ (Q : Type (max v₁ u_5 u_4)) [inst : AddCommMonoid Q] [inst_1 : Module R Q] …
  -/
  intro Q _ _ _ _ i
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    g : LinearMap (RingHom.id S) N O
    hg : IsBaseChange T g
    Q : Type (max v₁ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower R T Q
    i : LinearMap (RingHom.id R) M Q
    ⊢ ExistsUnique fun g' => Eq ((↑R g').comp ((↑R g).comp f)) i
  -/
  letI := Module.compHom Q (algebraMap S T)
  haveI : IsScalarTower S T Q :=
    ⟨fun x y z => by
      rw [Algebra.smul_def, mul_smul]
      rfl⟩
  have : IsScalarTower R S Q := by
    refine ⟨fun x y z => ?_⟩
    change (IsScalarTower.toAlgHom R S T) (x • y) • z = x • algebraMap S T y • z
    rw [map_smul, smul_assoc]
    rfl
  refine
    ⟨hg.lift (hf.lift i), by
      ext
      simp [IsBaseChange.lift_eq], ?_⟩
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    g : LinearMap (RingHom.id S) N O
    hg : IsBaseChange T g
    Q : Type (max v₁ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower R T Q
    i : LinearMap (RingHom.id R) M Q
    this✝¹ : Module S Q := Module.compHom Q (algebraMap S T)
    this✝ : IsScalarTower S T Q
    this : IsScalarTower R S Q
    ⊢ ∀ (y : LinearMap (RingHom.id T) O Q), (fun g' => Eq ((↑R g').comp ((↑R g).co …
  -/
  rintro g' (e : _ = _)
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    g : LinearMap (RingHom.id S) N O
    hg : IsBaseChange T g
    Q : Type (max v₁ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower R T Q
    i : LinearMap (RingHom.id R) M Q
    this✝¹ : Module S Q := Module.compHom Q (algebraMap S T)
    this✝ : IsScalarTower S T Q
    this : IsScalarTower R S Q
    g' : LinearMap (RingHom.id T) O Q
    e : Eq ((↑R g').comp ((↑R g).comp f)) i
    ⊢ Eq g' (hg.lift (hf.lift i))
  -/
  refine hg.algHom_ext' _ _ (hf.algHom_ext' _ _ ?_)
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    g : LinearMap (RingHom.id S) N O
    hg : IsBaseChange T g
    Q : Type (max v₁ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower R T Q
    i : LinearMap (RingHom.id R) M Q
    this✝¹ : Module S Q := Module.compHom Q (algebraMap S T)
    this✝ : IsScalarTower S T Q
    this : IsScalarTower R S Q
    g' : LinearMap (RingHom.id T) O Q
    e : Eq ((↑R g').comp ((↑R g).comp f)) i
    ⊢ Eq ((↑R ((↑S g').comp g)).comp f) ((↑R ((↑S (hg.lift (hf.lift i))).comp g)). …
  -/
  rw [IsBaseChange.lift_comp, IsBaseChange.lift_comp, ← e]
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    g : LinearMap (RingHom.id S) N O
    hg : IsBaseChange T g
    Q : Type (max v₁ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower R T Q
    i : LinearMap (RingHom.id R) M Q
    this✝¹ : Module S Q := Module.compHom Q (algebraMap S T)
    this✝ : IsScalarTower S T Q
    this : IsScalarTower R S Q
    g' : LinearMap (RingHom.id T) O Q
    e : Eq ((↑R g').comp ((↑R g).comp f)) i
    ⊢ Eq ((↑R ((↑S g').comp g)).comp f) ((↑R g').comp ((↑R g).comp f))
  -/
  ext
  /-
    case h.h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    g : LinearMap (RingHom.id S) N O
    hg : IsBaseChange T g
    Q : Type (max v₁ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module R Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower R T Q
    i : LinearMap (RingHom.id R) M Q
    this✝¹ : Module S Q := Module.compHom Q (algebraMap S T)
    this✝ : IsScalarTower S T Q
    this : IsScalarTower R S Q
    g' : LinearMap (RingHom.id T) O Q
    e : Eq ((↑R g').comp ((↑R g).comp f)) i
    x✝ : M
    ⊢ Eq (((↑R ((↑S g').comp g)).comp f) x✝) (((↑R g').comp ((↑R g).comp f)) x✝)
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `N` is the base change of `M` to `S` and `O` the base change of `M` to `T`, then
`O` is the base change of `N` to `T`. -/
lemma IsBaseChange.of_comp {f : M →ₗ[R] N} (hf : IsBaseChange S f) {h : N →ₗ[S] O}
    (hc : IsBaseChange T ((h : N →ₗ[R] O) ∘ₗ f)) :
    IsBaseChange T h := by
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝¹⁹ : AddCommMonoid M
    inst✝¹⁸ : AddCommMonoid N
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : CommSemiring S
    inst✝¹⁵ : Algebra R S
    inst✝¹⁴ : Module R M
    inst✝¹³ : Module R N
    inst✝¹² : Module S N
    inst✝¹¹ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁰ : CommSemiring T
    inst✝⁹ : Algebra R T
    inst✝⁸ : Algebra S T
    inst✝⁷ : IsScalarTower R S T
    inst✝⁶ : AddCommMonoid O
    inst✝⁵ : Module R O
    inst✝⁴ : Module S O
    inst✝³ : Module T O
    inst✝² : IsScalarTower S T O
    inst✝¹ : IsScalarTower R S O
    inst✝ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    h : LinearMap (RingHom.id S) N O
    hc : IsBaseChange T ((↑R h).comp f)
    ⊢ IsBaseChange T h
  -/
  apply IsBaseChange.of_lift_unique
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝¹⁹ : AddCommMonoid M
    inst✝¹⁸ : AddCommMonoid N
    inst✝¹⁷ : CommSemiring R
    inst✝¹⁶ : CommSemiring S
    inst✝¹⁵ : Algebra R S
    inst✝¹⁴ : Module R M
    inst✝¹³ : Module R N
    inst✝¹² : Module S N
    inst✝¹¹ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁰ : CommSemiring T
    inst✝⁹ : Algebra R T
    inst✝⁸ : Algebra S T
    inst✝⁷ : IsScalarTower R S T
    inst✝⁶ : AddCommMonoid O
    inst✝⁵ : Module R O
    inst✝⁴ : Module S O
    inst✝³ : Module T O
    inst✝² : IsScalarTower S T O
    inst✝¹ : IsScalarTower R S O
    inst✝ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    h : LinearMap (RingHom.id S) N O
    hc : IsBaseChange T ((↑R h).comp f)
    ⊢ ∀ (Q : Type (max v₂ u_5 u_4)) [inst : AddCommMonoid Q] [inst_1 : Module S Q] …
  -/
  intro Q _ _ _ _ r
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    h : LinearMap (RingHom.id S) N O
    hc : IsBaseChange T ((↑R h).comp f)
    Q : Type (max v₂ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module S Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower S T Q
    r : LinearMap (RingHom.id S) N Q
    ⊢ ExistsUnique fun g' => Eq ((↑S g').comp h) r
  -/
  letI : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    h : LinearMap (RingHom.id S) N O
    hc : IsBaseChange T ((↑R h).comp f)
    Q : Type (max v₂ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module S Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower S T Q
    r : LinearMap (RingHom.id S) N Q
    this : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
    ⊢ ExistsUnique fun g' => Eq ((↑S g').comp h) r
  -/
  haveI : IsScalarTower R S Q := IsScalarTower.of_algebraMap_smul fun r ↦ congrFun rfl
  haveI : IsScalarTower R T Q := IsScalarTower.of_algebraMap_smul fun r x ↦ by
    simp [IsScalarTower.algebraMap_apply R S T]
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    h : LinearMap (RingHom.id S) N O
    hc : IsBaseChange T ((↑R h).comp f)
    Q : Type (max v₂ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module S Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower S T Q
    r : LinearMap (RingHom.id S) N Q
    this✝¹ : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
    this✝ : IsScalarTower R S Q
    this : IsScalarTower R T Q
    ⊢ ExistsUnique fun g' => Eq ((↑S g').comp h) r
  -/
  let r' : M →ₗ[R] Q := r ∘ₗ f
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    h : LinearMap (RingHom.id S) N O
    hc : IsBaseChange T ((↑R h).comp f)
    Q : Type (max v₂ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module S Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower S T Q
    r : LinearMap (RingHom.id S) N Q
    this✝¹ : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
    this✝ : IsScalarTower R S Q
    this : IsScalarTower R T Q
    r' : LinearMap (RingHom.id R) M Q := (↑R r).comp f
    ⊢ ExistsUnique fun g' => Eq ((↑S g').comp h) r
  -/
  let q : O →ₗ[T] Q := hc.lift r'
  /-
    case h
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝²³ : AddCommMonoid M
    inst✝²² : AddCommMonoid N
    inst✝²¹ : CommSemiring R
    inst✝²⁰ : CommSemiring S
    inst✝¹⁹ : Algebra R S
    inst✝¹⁸ : Module R M
    inst✝¹⁷ : Module R N
    inst✝¹⁶ : Module S N
    inst✝¹⁵ : IsScalarTower R S N
    T : Type u_4
    O : Type u_5
    inst✝¹⁴ : CommSemiring T
    inst✝¹³ : Algebra R T
    inst✝¹² : Algebra S T
    inst✝¹¹ : IsScalarTower R S T
    inst✝¹⁰ : AddCommMonoid O
    inst✝⁹ : Module R O
    inst✝⁸ : Module S O
    inst✝⁷ : Module T O
    inst✝⁶ : IsScalarTower S T O
    inst✝⁵ : IsScalarTower R S O
    inst✝⁴ : IsScalarTower R T O
    f : LinearMap (RingHom.id R) M N
    hf : IsBaseChange S f
    h : LinearMap (RingHom.id S) N O
    hc : IsBaseChange T ((↑R h).comp f)
    Q : Type (max v₂ u_5 u_4)
    inst✝³ : AddCommMonoid Q
    inst✝² : Module S Q
    inst✝¹ : Module T Q
    inst✝ : IsScalarTower S T Q
    r : LinearMap (RingHom.id S) N Q
    this✝¹ : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
    this✝ : IsScalarTower R S Q
    this : IsScalarTower R T Q
    r' : LinearMap (RingHom.id R) M Q := (↑R r).comp f
    q : LinearMap (RingHom.id T) O Q := hc.lift r'
    ⊢ ExistsUnique fun g' => Eq ((↑S g').comp h) r
  -/
  refine ⟨q, ?_, ?_⟩
    /-
      case h.refine_1
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝²³ : AddCommMonoid M
      inst✝²² : AddCommMonoid N
      inst✝²¹ : CommSemiring R
      inst✝²⁰ : CommSemiring S
      inst✝¹⁹ : Algebra R S
      inst✝¹⁸ : Module R M
      inst✝¹⁷ : Module R N
      inst✝¹⁶ : Module S N
      inst✝¹⁵ : IsScalarTower R S N
      T : Type u_4
      O : Type u_5
      inst✝¹⁴ : CommSemiring T
      inst✝¹³ : Algebra R T
      inst✝¹² : Algebra S T
      inst✝¹¹ : IsScalarTower R S T
      inst✝¹⁰ : AddCommMonoid O
      inst✝⁹ : Module R O
      inst✝⁸ : Module S O
      inst✝⁷ : Module T O
      inst✝⁶ : IsScalarTower S T O
      inst✝⁵ : IsScalarTower R S O
      inst✝⁴ : IsScalarTower R T O
      f : LinearMap (RingHom.id R) M N
      hf : IsBaseChange S f
      h : LinearMap (RingHom.id S) N O
      hc : IsBaseChange T ((↑R h).comp f)
      Q : Type (max v₂ u_5 u_4)
      inst✝³ : AddCommMonoid Q
      inst✝² : Module S Q
      inst✝¹ : Module T Q
      inst✝ : IsScalarTower S T Q
      r : LinearMap (RingHom.id S) N Q
      this✝¹ : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
      this✝ : IsScalarTower R S Q
      this : IsScalarTower R T Q
      r' : LinearMap (RingHom.id R) M Q := (↑R r).comp f
      q : LinearMap (RingHom.id T) O Q := hc.lift r'
      ⊢ (fun g' => Eq ((↑S g').comp h) r) q
    -/
  · apply hf.algHom_ext'
    /-
      case h.refine_1.e
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝²³ : AddCommMonoid M
      inst✝²² : AddCommMonoid N
      inst✝²¹ : CommSemiring R
      inst✝²⁰ : CommSemiring S
      inst✝¹⁹ : Algebra R S
      inst✝¹⁸ : Module R M
      inst✝¹⁷ : Module R N
      inst✝¹⁶ : Module S N
      inst✝¹⁵ : IsScalarTower R S N
      T : Type u_4
      O : Type u_5
      inst✝¹⁴ : CommSemiring T
      inst✝¹³ : Algebra R T
      inst✝¹² : Algebra S T
      inst✝¹¹ : IsScalarTower R S T
      inst✝¹⁰ : AddCommMonoid O
      inst✝⁹ : Module R O
      inst✝⁸ : Module S O
      inst✝⁷ : Module T O
      inst✝⁶ : IsScalarTower S T O
      inst✝⁵ : IsScalarTower R S O
      inst✝⁴ : IsScalarTower R T O
      f : LinearMap (RingHom.id R) M N
      hf : IsBaseChange S f
      h : LinearMap (RingHom.id S) N O
      hc : IsBaseChange T ((↑R h).comp f)
      Q : Type (max v₂ u_5 u_4)
      inst✝³ : AddCommMonoid Q
      inst✝² : Module S Q
      inst✝¹ : Module T Q
      inst✝ : IsScalarTower S T Q
      r : LinearMap (RingHom.id S) N Q
      this✝¹ : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
      this✝ : IsScalarTower R S Q
      this : IsScalarTower R T Q
      r' : LinearMap (RingHom.id R) M Q := (↑R r).comp f
      q : LinearMap (RingHom.id T) O Q := hc.lift r'
      ⊢ Eq ((↑R ((↑S q).comp h)).comp f) ((↑R r).comp f)
    -/
    simp [r', q, LinearMap.comp_assoc, hc.lift_comp]
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝²³ : AddCommMonoid M
      inst✝²² : AddCommMonoid N
      inst✝²¹ : CommSemiring R
      inst✝²⁰ : CommSemiring S
      inst✝¹⁹ : Algebra R S
      inst✝¹⁸ : Module R M
      inst✝¹⁷ : Module R N
      inst✝¹⁶ : Module S N
      inst✝¹⁵ : IsScalarTower R S N
      T : Type u_4
      O : Type u_5
      inst✝¹⁴ : CommSemiring T
      inst✝¹³ : Algebra R T
      inst✝¹² : Algebra S T
      inst✝¹¹ : IsScalarTower R S T
      inst✝¹⁰ : AddCommMonoid O
      inst✝⁹ : Module R O
      inst✝⁸ : Module S O
      inst✝⁷ : Module T O
      inst✝⁶ : IsScalarTower S T O
      inst✝⁵ : IsScalarTower R S O
      inst✝⁴ : IsScalarTower R T O
      f : LinearMap (RingHom.id R) M N
      hf : IsBaseChange S f
      h : LinearMap (RingHom.id S) N O
      hc : IsBaseChange T ((↑R h).comp f)
      Q : Type (max v₂ u_5 u_4)
      inst✝³ : AddCommMonoid Q
      inst✝² : Module S Q
      inst✝¹ : Module T Q
      inst✝ : IsScalarTower S T Q
      r : LinearMap (RingHom.id S) N Q
      this✝¹ : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
      this✝ : IsScalarTower R S Q
      this : IsScalarTower R T Q
      r' : LinearMap (RingHom.id R) M Q := (↑R r).comp f
      q : LinearMap (RingHom.id T) O Q := hc.lift r'
      ⊢ ∀ (y : LinearMap (RingHom.id T) O Q), (fun g' => Eq ((↑S g').comp h) r) y →  …
    -/
  · intro q' hq'
    /-
      case h.refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝²³ : AddCommMonoid M
      inst✝²² : AddCommMonoid N
      inst✝²¹ : CommSemiring R
      inst✝²⁰ : CommSemiring S
      inst✝¹⁹ : Algebra R S
      inst✝¹⁸ : Module R M
      inst✝¹⁷ : Module R N
      inst✝¹⁶ : Module S N
      inst✝¹⁵ : IsScalarTower R S N
      T : Type u_4
      O : Type u_5
      inst✝¹⁴ : CommSemiring T
      inst✝¹³ : Algebra R T
      inst✝¹² : Algebra S T
      inst✝¹¹ : IsScalarTower R S T
      inst✝¹⁰ : AddCommMonoid O
      inst✝⁹ : Module R O
      inst✝⁸ : Module S O
      inst✝⁷ : Module T O
      inst✝⁶ : IsScalarTower S T O
      inst✝⁵ : IsScalarTower R S O
      inst✝⁴ : IsScalarTower R T O
      f : LinearMap (RingHom.id R) M N
      hf : IsBaseChange S f
      h : LinearMap (RingHom.id S) N O
      hc : IsBaseChange T ((↑R h).comp f)
      Q : Type (max v₂ u_5 u_4)
      inst✝³ : AddCommMonoid Q
      inst✝² : Module S Q
      inst✝¹ : Module T Q
      inst✝ : IsScalarTower S T Q
      r : LinearMap (RingHom.id S) N Q
      this✝¹ : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
      this✝ : IsScalarTower R S Q
      this : IsScalarTower R T Q
      r' : LinearMap (RingHom.id R) M Q := (↑R r).comp f
      q : LinearMap (RingHom.id T) O Q := hc.lift r'
      q' : LinearMap (RingHom.id T) O Q
      hq' : Eq ((↑S q').comp h) r
      ⊢ Eq q' q
    -/
    apply hc.algHom_ext'
    /-
      case h.refine_2.e
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝²³ : AddCommMonoid M
      inst✝²² : AddCommMonoid N
      inst✝²¹ : CommSemiring R
      inst✝²⁰ : CommSemiring S
      inst✝¹⁹ : Algebra R S
      inst✝¹⁸ : Module R M
      inst✝¹⁷ : Module R N
      inst✝¹⁶ : Module S N
      inst✝¹⁵ : IsScalarTower R S N
      T : Type u_4
      O : Type u_5
      inst✝¹⁴ : CommSemiring T
      inst✝¹³ : Algebra R T
      inst✝¹² : Algebra S T
      inst✝¹¹ : IsScalarTower R S T
      inst✝¹⁰ : AddCommMonoid O
      inst✝⁹ : Module R O
      inst✝⁸ : Module S O
      inst✝⁷ : Module T O
      inst✝⁶ : IsScalarTower S T O
      inst✝⁵ : IsScalarTower R S O
      inst✝⁴ : IsScalarTower R T O
      f : LinearMap (RingHom.id R) M N
      hf : IsBaseChange S f
      h : LinearMap (RingHom.id S) N O
      hc : IsBaseChange T ((↑R h).comp f)
      Q : Type (max v₂ u_5 u_4)
      inst✝³ : AddCommMonoid Q
      inst✝² : Module S Q
      inst✝¹ : Module T Q
      inst✝ : IsScalarTower S T Q
      r : LinearMap (RingHom.id S) N Q
      this✝¹ : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
      this✝ : IsScalarTower R S Q
      this : IsScalarTower R T Q
      r' : LinearMap (RingHom.id R) M Q := (↑R r).comp f
      q : LinearMap (RingHom.id T) O Q := hc.lift r'
      q' : LinearMap (RingHom.id T) O Q
      hq' : Eq ((↑S q').comp h) r
      ⊢ Eq ((↑R q').comp ((↑R h).comp f)) ((↑R q).comp ((↑R h).comp f))
    -/
    apply_fun LinearMap.restrictScalars R at hq'
    /-
      case h.refine_2.e
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝²³ : AddCommMonoid M
      inst✝²² : AddCommMonoid N
      inst✝²¹ : CommSemiring R
      inst✝²⁰ : CommSemiring S
      inst✝¹⁹ : Algebra R S
      inst✝¹⁸ : Module R M
      inst✝¹⁷ : Module R N
      inst✝¹⁶ : Module S N
      inst✝¹⁵ : IsScalarTower R S N
      T : Type u_4
      O : Type u_5
      inst✝¹⁴ : CommSemiring T
      inst✝¹³ : Algebra R T
      inst✝¹² : Algebra S T
      inst✝¹¹ : IsScalarTower R S T
      inst✝¹⁰ : AddCommMonoid O
      inst✝⁹ : Module R O
      inst✝⁸ : Module S O
      inst✝⁷ : Module T O
      inst✝⁶ : IsScalarTower S T O
      inst✝⁵ : IsScalarTower R S O
      inst✝⁴ : IsScalarTower R T O
      f : LinearMap (RingHom.id R) M N
      hf : IsBaseChange S f
      h : LinearMap (RingHom.id S) N O
      hc : IsBaseChange T ((↑R h).comp f)
      Q : Type (max v₂ u_5 u_4)
      inst✝³ : AddCommMonoid Q
      inst✝² : Module S Q
      inst✝¹ : Module T Q
      inst✝ : IsScalarTower S T Q
      r : LinearMap (RingHom.id S) N Q
      this✝¹ : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
      this✝ : IsScalarTower R S Q
      this : IsScalarTower R T Q
      r' : LinearMap (RingHom.id R) M Q := (↑R r).comp f
      q : LinearMap (RingHom.id T) O Q := hc.lift r'
      q' : LinearMap (RingHom.id T) O Q
      hq' : Eq (↑R ((↑S q').comp h)) (↑R r)
      ⊢ Eq ((↑R q').comp ((↑R h).comp f)) ((↑R q).comp ((↑R h).comp f))
    -/
    rw [← LinearMap.comp_assoc]
    /-
      case h.refine_2.e
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝²³ : AddCommMonoid M
      inst✝²² : AddCommMonoid N
      inst✝²¹ : CommSemiring R
      inst✝²⁰ : CommSemiring S
      inst✝¹⁹ : Algebra R S
      inst✝¹⁸ : Module R M
      inst✝¹⁷ : Module R N
      inst✝¹⁶ : Module S N
      inst✝¹⁵ : IsScalarTower R S N
      T : Type u_4
      O : Type u_5
      inst✝¹⁴ : CommSemiring T
      inst✝¹³ : Algebra R T
      inst✝¹² : Algebra S T
      inst✝¹¹ : IsScalarTower R S T
      inst✝¹⁰ : AddCommMonoid O
      inst✝⁹ : Module R O
      inst✝⁸ : Module S O
      inst✝⁷ : Module T O
      inst✝⁶ : IsScalarTower S T O
      inst✝⁵ : IsScalarTower R S O
      inst✝⁴ : IsScalarTower R T O
      f : LinearMap (RingHom.id R) M N
      hf : IsBaseChange S f
      h : LinearMap (RingHom.id S) N O
      hc : IsBaseChange T ((↑R h).comp f)
      Q : Type (max v₂ u_5 u_4)
      inst✝³ : AddCommMonoid Q
      inst✝² : Module S Q
      inst✝¹ : Module T Q
      inst✝ : IsScalarTower S T Q
      r : LinearMap (RingHom.id S) N Q
      this✝¹ : Module R Q := inferInstanceAs (Module R (RestrictScalars R S Q))
      this✝ : IsScalarTower R S Q
      this : IsScalarTower R T Q
      r' : LinearMap (RingHom.id R) M Q := (↑R r).comp f
      q : LinearMap (RingHom.id T) O Q := hc.lift r'
      q' : LinearMap (RingHom.id T) O Q
      hq' : Eq (↑R ((↑S q').comp h)) (↑R r)
      ⊢ Eq (((↑R q').comp (↑R h)).comp f) ((↑R q).comp ((↑R h).comp f))
    -/
    rw [show q'.restrictScalars R ∘ₗ h.restrictScalars R = _ from hq', hc.lift_comp]
    /-
      🎉 no goals
    -/


/-- If `N` is the base change `M` to `S`, then `O` is the base change of `M` to `T` if and
only if `O` is the base change of `N` to `T`. -/
lemma IsBaseChange.comp_iff {f : M →ₗ[R] N} (hf : IsBaseChange S f) {h : N →ₗ[S] O} :
    IsBaseChange T ((h : N →ₗ[R] O) ∘ₗ f) ↔ IsBaseChange T h :=
  ⟨fun hc ↦ IsBaseChange.of_comp hf hc, fun hh ↦ IsBaseChange.comp hf hh⟩


/-- A type-class stating that the following diagram of scalar towers
R  →  S
↓     ↓
R' →  S'
is a pushout diagram (i.e. `S' = S ⊗[R] R'`)
-/
@[mk_iff]
class Algebra.IsPushout : Prop where
  out : IsBaseChange S (toAlgHom R R' S').toLinearMap


@[symm]
theorem Algebra.IsPushout.symm (h : Algebra.IsPushout R S R' S') : Algebra.IsPushout R R' S S' := by
  /-
    R : Type u_1
    S : Type v₃
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    R' : Type u_6
    S' : Type u_7
    inst✝⁷ : CommSemiring R'
    inst✝⁶ : CommSemiring S'
    inst✝⁵ : Algebra R R'
    inst✝⁴ : Algebra S S'
    inst✝³ : Algebra R' S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    h : Algebra.IsPushout R S R' S'
    ⊢ Algebra.IsPushout R R' S S'
  -/
  let _ := (Algebra.TensorProduct.includeRight : R' →ₐ[R] S ⊗ R').toRingHom.toAlgebra
  let e : R' ⊗[R] S ≃ₗ[R'] S' := by
    refine { (_root_.TensorProduct.comm R R' S).trans <|
      h.1.equiv.restrictScalars R with map_smul' := ?_ }
    intro r x
    change
      h.1.equiv (TensorProduct.comm R R' S (r • x)) = r • h.1.equiv (TensorProduct.comm R R' S x)
    refine TensorProduct.induction_on x ?_ ?_ ?_
    · simp only [smul_zero, map_zero]
    · intro x y
      simp only [smul_tmul', smul_eq_mul, TensorProduct.comm_tmul, smul_def,
        TensorProduct.algebraMap_apply, id.map_eq_id, RingHom.id_apply, TensorProduct.tmul_mul_tmul,
        one_mul, h.1.equiv_tmul, AlgHom.toLinearMap_apply, map_mul, IsScalarTower.coe_toAlgHom']
      ring
    · intro x y hx hy
      rw [map_add, map_add, smul_add, map_add, map_add, hx, hy, smul_add]
  have :
    (toAlgHom R S S').toLinearMap =
      (e.toLinearMap.restrictScalars R).comp (TensorProduct.mk R R' S 1) := by
    ext
    simp [e, h.1.equiv_tmul, Algebra.smul_def]
  /-
    R : Type u_1
    S : Type v₃
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    R' : Type u_6
    S' : Type u_7
    inst✝⁷ : CommSemiring R'
    inst✝⁶ : CommSemiring S'
    inst✝⁵ : Algebra R R'
    inst✝⁴ : Algebra S S'
    inst✝³ : Algebra R' S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    h : Algebra.IsPushout R S R' S'
    x✝ : Algebra R' (TensorProduct R S R') := Algebra.TensorProduct.includeRight.t …
    e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' :=
      let __src := (_root_.TensorProduct.comm R R' S).trans (LinearEquiv.restrictS …
      { toAddHom := (↑__src).toAddHom, map_smul' := ⋯, invFun := __src.invFun, lef …
    this : Eq (IsScalarTower.toAlgHom R S S').toLinearMap ((↑R ↑e).comp ((TensorPr …
    ⊢ Algebra.IsPushout R R' S S'
  -/
  constructor
  /-
    case out
    R : Type u_1
    S : Type v₃
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    R' : Type u_6
    S' : Type u_7
    inst✝⁷ : CommSemiring R'
    inst✝⁶ : CommSemiring S'
    inst✝⁵ : Algebra R R'
    inst✝⁴ : Algebra S S'
    inst✝³ : Algebra R' S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    h : Algebra.IsPushout R S R' S'
    x✝ : Algebra R' (TensorProduct R S R') := Algebra.TensorProduct.includeRight.t …
    e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' :=
      let __src := (_root_.TensorProduct.comm R R' S).trans (LinearEquiv.restrictS …
      { toAddHom := (↑__src).toAddHom, map_smul' := ⋯, invFun := __src.invFun, lef …
    this : Eq (IsScalarTower.toAlgHom R S S').toLinearMap ((↑R ↑e).comp ((TensorPr …
    ⊢ IsBaseChange R' (IsScalarTower.toAlgHom R S S').toLinearMap
  -/
  rw [this]
  /-
    case out
    R : Type u_1
    S : Type v₃
    inst✝¹⁰ : CommSemiring R
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    R' : Type u_6
    S' : Type u_7
    inst✝⁷ : CommSemiring R'
    inst✝⁶ : CommSemiring S'
    inst✝⁵ : Algebra R R'
    inst✝⁴ : Algebra S S'
    inst✝³ : Algebra R' S'
    inst✝² : Algebra R S'
    inst✝¹ : IsScalarTower R R' S'
    inst✝ : IsScalarTower R S S'
    h : Algebra.IsPushout R S R' S'
    x✝ : Algebra R' (TensorProduct R S R') := Algebra.TensorProduct.includeRight.t …
    e : LinearEquiv (RingHom.id R') (TensorProduct R R' S) S' :=
      let __src := (_root_.TensorProduct.comm R R' S).trans (LinearEquiv.restrictS …
      { toAddHom := (↑__src).toAddHom, map_smul' := ⋯, invFun := __src.invFun, lef …
    this : Eq (IsScalarTower.toAlgHom R S S').toLinearMap ((↑R ↑e).comp ((TensorPr …
    ⊢ IsBaseChange R' ((↑R ↑e).comp ((TensorProduct.mk R R' S) 1))
  -/
  exact (TensorProduct.isBaseChange R S R').comp (IsBaseChange.ofEquiv e)
  /-
    🎉 no goals
  -/


theorem Algebra.IsPushout.comm : Algebra.IsPushout R S R' S' ↔ Algebra.IsPushout R R' S S' :=
  ⟨Algebra.IsPushout.symm, Algebra.IsPushout.symm⟩


instance TensorProduct.isPushout {R S T : Type*} [CommRing R] [CommRing S] [CommRing T]
    [Algebra R S] [Algebra R T] : Algebra.IsPushout R S T (TensorProduct R S T) :=
  ⟨TensorProduct.isBaseChange R T S⟩


instance TensorProduct.isPushout' {R S T : Type*} [CommRing R] [CommRing S] [CommRing T]
    [Algebra R S] [Algebra R T] : Algebra.IsPushout R T S (TensorProduct R S T) :=
  Algebra.IsPushout.symm inferInstance


/-- If `S' = S ⊗[R] R'`, then any pair of `R`-algebra homomorphisms `f : S → A` and `g : R' → A`
such that `f x` and `g y` commutes for all `x, y` descends to a (unique) homomorphism `S' → A`.
-/
@[simps! (config := .lemmasOnly) apply]
noncomputable def Algebra.pushoutDesc [H : Algebra.IsPushout R S R' S'] {A : Type*} [Semiring A]
    [Algebra R A] (f : S →ₐ[R] A) (g : R' →ₐ[R] A) (hf : ∀ x y, f x * g y = g y * f x) :
    S' →ₐ[R] A := by
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝³³ : AddCommMonoid M
    inst✝³² : AddCommMonoid N
    inst✝³¹ : CommSemiring R
    inst✝³⁰ : CommSemiring S
    inst✝²⁹ : Algebra R S
    inst✝²⁸ : Module R M
    inst✝²⁷ : Module R N
    inst✝²⁶ : Module S N
    inst✝²⁵ : IsScalarTower R S N
    f✝ : LinearMap (RingHom.id R) M N
    h : IsBaseChange S f✝
    P : Type u_2
    Q : Type u_3
    inst✝²⁴ : AddCommMonoid P
    inst✝²³ : Module R P
    inst✝²² : AddCommMonoid Q
    inst✝²¹ : Module S Q
    T : Type u_4
    O : Type u_5
    inst✝²⁰ : CommSemiring T
    inst✝¹⁹ : Algebra R T
    inst✝¹⁸ : Algebra S T
    inst✝¹⁷ : IsScalarTower R S T
    inst✝¹⁶ : AddCommMonoid O
    inst✝¹⁵ : Module R O
    inst✝¹⁴ : Module S O
    inst✝¹³ : Module T O
    inst✝¹² : IsScalarTower S T O
    inst✝¹¹ : IsScalarTower R S O
    inst✝¹⁰ : IsScalarTower R T O
    R' : Type u_6
    S' : Type u_7
    inst✝⁹ : CommSemiring R'
    inst✝⁸ : CommSemiring S'
    inst✝⁷ : Algebra R R'
    inst✝⁶ : Algebra S S'
    inst✝⁵ : Algebra R' S'
    inst✝⁴ : Algebra R S'
    inst✝³ : IsScalarTower R R' S'
    inst✝² : IsScalarTower R S S'
    H : Algebra.IsPushout R S R' S'
    A : Type u_8
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : AlgHom R S A
    g : AlgHom R R' A
    hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
    ⊢ AlgHom R S' A
  -/
  letI := Module.compHom A f.toRingHom
  haveI : IsScalarTower R S A :=
    { smul_assoc := fun r s a =>
        show f (r • s) * a = r • (f s * a) by rw [map_smul, smul_mul_assoc] }
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝³³ : AddCommMonoid M
    inst✝³² : AddCommMonoid N
    inst✝³¹ : CommSemiring R
    inst✝³⁰ : CommSemiring S
    inst✝²⁹ : Algebra R S
    inst✝²⁸ : Module R M
    inst✝²⁷ : Module R N
    inst✝²⁶ : Module S N
    inst✝²⁵ : IsScalarTower R S N
    f✝ : LinearMap (RingHom.id R) M N
    h : IsBaseChange S f✝
    P : Type u_2
    Q : Type u_3
    inst✝²⁴ : AddCommMonoid P
    inst✝²³ : Module R P
    inst✝²² : AddCommMonoid Q
    inst✝²¹ : Module S Q
    T : Type u_4
    O : Type u_5
    inst✝²⁰ : CommSemiring T
    inst✝¹⁹ : Algebra R T
    inst✝¹⁸ : Algebra S T
    inst✝¹⁷ : IsScalarTower R S T
    inst✝¹⁶ : AddCommMonoid O
    inst✝¹⁵ : Module R O
    inst✝¹⁴ : Module S O
    inst✝¹³ : Module T O
    inst✝¹² : IsScalarTower S T O
    inst✝¹¹ : IsScalarTower R S O
    inst✝¹⁰ : IsScalarTower R T O
    R' : Type u_6
    S' : Type u_7
    inst✝⁹ : CommSemiring R'
    inst✝⁸ : CommSemiring S'
    inst✝⁷ : Algebra R R'
    inst✝⁶ : Algebra S S'
    inst✝⁵ : Algebra R' S'
    inst✝⁴ : Algebra R S'
    inst✝³ : IsScalarTower R R' S'
    inst✝² : IsScalarTower R S S'
    H : Algebra.IsPushout R S R' S'
    A : Type u_8
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : AlgHom R S A
    g : AlgHom R R' A
    hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
    this✝ : Module S A := Module.compHom A f.toRingHom
    this : IsScalarTower R S A
    ⊢ AlgHom R S' A
  -/
  haveI : IsScalarTower S A A := { smul_assoc := fun r a b => mul_assoc _ _ _ }
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝³³ : AddCommMonoid M
    inst✝³² : AddCommMonoid N
    inst✝³¹ : CommSemiring R
    inst✝³⁰ : CommSemiring S
    inst✝²⁹ : Algebra R S
    inst✝²⁸ : Module R M
    inst✝²⁷ : Module R N
    inst✝²⁶ : Module S N
    inst✝²⁵ : IsScalarTower R S N
    f✝ : LinearMap (RingHom.id R) M N
    h : IsBaseChange S f✝
    P : Type u_2
    Q : Type u_3
    inst✝²⁴ : AddCommMonoid P
    inst✝²³ : Module R P
    inst✝²² : AddCommMonoid Q
    inst✝²¹ : Module S Q
    T : Type u_4
    O : Type u_5
    inst✝²⁰ : CommSemiring T
    inst✝¹⁹ : Algebra R T
    inst✝¹⁸ : Algebra S T
    inst✝¹⁷ : IsScalarTower R S T
    inst✝¹⁶ : AddCommMonoid O
    inst✝¹⁵ : Module R O
    inst✝¹⁴ : Module S O
    inst✝¹³ : Module T O
    inst✝¹² : IsScalarTower S T O
    inst✝¹¹ : IsScalarTower R S O
    inst✝¹⁰ : IsScalarTower R T O
    R' : Type u_6
    S' : Type u_7
    inst✝⁹ : CommSemiring R'
    inst✝⁸ : CommSemiring S'
    inst✝⁷ : Algebra R R'
    inst✝⁶ : Algebra S S'
    inst✝⁵ : Algebra R' S'
    inst✝⁴ : Algebra R S'
    inst✝³ : IsScalarTower R R' S'
    inst✝² : IsScalarTower R S S'
    H : Algebra.IsPushout R S R' S'
    A : Type u_8
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : AlgHom R S A
    g : AlgHom R R' A
    hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
    this✝¹ : Module S A := Module.compHom A f.toRingHom
    this✝ : IsScalarTower R S A
    this : IsScalarTower S A A
    ⊢ AlgHom R S' A
  -/
  have : ∀ x, H.out.lift g.toLinearMap (algebraMap R' S' x) = g x := H.out.lift_eq _
  /-
    R : Type u_1
    M : Type v₁
    N : Type v₂
    S : Type v₃
    inst✝³³ : AddCommMonoid M
    inst✝³² : AddCommMonoid N
    inst✝³¹ : CommSemiring R
    inst✝³⁰ : CommSemiring S
    inst✝²⁹ : Algebra R S
    inst✝²⁸ : Module R M
    inst✝²⁷ : Module R N
    inst✝²⁶ : Module S N
    inst✝²⁵ : IsScalarTower R S N
    f✝ : LinearMap (RingHom.id R) M N
    h : IsBaseChange S f✝
    P : Type u_2
    Q : Type u_3
    inst✝²⁴ : AddCommMonoid P
    inst✝²³ : Module R P
    inst✝²² : AddCommMonoid Q
    inst✝²¹ : Module S Q
    T : Type u_4
    O : Type u_5
    inst✝²⁰ : CommSemiring T
    inst✝¹⁹ : Algebra R T
    inst✝¹⁸ : Algebra S T
    inst✝¹⁷ : IsScalarTower R S T
    inst✝¹⁶ : AddCommMonoid O
    inst✝¹⁵ : Module R O
    inst✝¹⁴ : Module S O
    inst✝¹³ : Module T O
    inst✝¹² : IsScalarTower S T O
    inst✝¹¹ : IsScalarTower R S O
    inst✝¹⁰ : IsScalarTower R T O
    R' : Type u_6
    S' : Type u_7
    inst✝⁹ : CommSemiring R'
    inst✝⁸ : CommSemiring S'
    inst✝⁷ : Algebra R R'
    inst✝⁶ : Algebra S S'
    inst✝⁵ : Algebra R' S'
    inst✝⁴ : Algebra R S'
    inst✝³ : IsScalarTower R R' S'
    inst✝² : IsScalarTower R S S'
    H : Algebra.IsPushout R S R' S'
    A : Type u_8
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : AlgHom R S A
    g : AlgHom R R' A
    hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
    this✝² : Module S A := Module.compHom A f.toRingHom
    this✝¹ : IsScalarTower R S A
    this✝ : IsScalarTower S A A
    this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
    ⊢ AlgHom R S' A
  -/
  refine AlgHom.ofLinearMap ((H.out.lift g.toLinearMap).restrictScalars R) ?_ ?_
    /-
      case refine_1
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝³³ : AddCommMonoid M
      inst✝³² : AddCommMonoid N
      inst✝³¹ : CommSemiring R
      inst✝³⁰ : CommSemiring S
      inst✝²⁹ : Algebra R S
      inst✝²⁸ : Module R M
      inst✝²⁷ : Module R N
      inst✝²⁶ : Module S N
      inst✝²⁵ : IsScalarTower R S N
      f✝ : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f✝
      P : Type u_2
      Q : Type u_3
      inst✝²⁴ : AddCommMonoid P
      inst✝²³ : Module R P
      inst✝²² : AddCommMonoid Q
      inst✝²¹ : Module S Q
      T : Type u_4
      O : Type u_5
      inst✝²⁰ : CommSemiring T
      inst✝¹⁹ : Algebra R T
      inst✝¹⁸ : Algebra S T
      inst✝¹⁷ : IsScalarTower R S T
      inst✝¹⁶ : AddCommMonoid O
      inst✝¹⁵ : Module R O
      inst✝¹⁴ : Module S O
      inst✝¹³ : Module T O
      inst✝¹² : IsScalarTower S T O
      inst✝¹¹ : IsScalarTower R S O
      inst✝¹⁰ : IsScalarTower R T O
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : AlgHom R S A
      g : AlgHom R R' A
      hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
      this✝² : Module S A := Module.compHom A f.toRingHom
      this✝¹ : IsScalarTower R S A
      this✝ : IsScalarTower S A A
      this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
      ⊢ Eq ((↑R (⋯.lift g.toLinearMap)) 1) 1
    -/
  · dsimp only [LinearMap.restrictScalars_apply]
    /-
      case refine_1
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝³³ : AddCommMonoid M
      inst✝³² : AddCommMonoid N
      inst✝³¹ : CommSemiring R
      inst✝³⁰ : CommSemiring S
      inst✝²⁹ : Algebra R S
      inst✝²⁸ : Module R M
      inst✝²⁷ : Module R N
      inst✝²⁶ : Module S N
      inst✝²⁵ : IsScalarTower R S N
      f✝ : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f✝
      P : Type u_2
      Q : Type u_3
      inst✝²⁴ : AddCommMonoid P
      inst✝²³ : Module R P
      inst✝²² : AddCommMonoid Q
      inst✝²¹ : Module S Q
      T : Type u_4
      O : Type u_5
      inst✝²⁰ : CommSemiring T
      inst✝¹⁹ : Algebra R T
      inst✝¹⁸ : Algebra S T
      inst✝¹⁷ : IsScalarTower R S T
      inst✝¹⁶ : AddCommMonoid O
      inst✝¹⁵ : Module R O
      inst✝¹⁴ : Module S O
      inst✝¹³ : Module T O
      inst✝¹² : IsScalarTower S T O
      inst✝¹¹ : IsScalarTower R S O
      inst✝¹⁰ : IsScalarTower R T O
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : AlgHom R S A
      g : AlgHom R R' A
      hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
      this✝² : Module S A := Module.compHom A f.toRingHom
      this✝¹ : IsScalarTower R S A
      this✝ : IsScalarTower S A A
      this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
      ⊢ Eq ((⋯.lift g.toLinearMap) 1) 1
    -/
    rw [← (algebraMap R' S').map_one, this, map_one]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝³³ : AddCommMonoid M
      inst✝³² : AddCommMonoid N
      inst✝³¹ : CommSemiring R
      inst✝³⁰ : CommSemiring S
      inst✝²⁹ : Algebra R S
      inst✝²⁸ : Module R M
      inst✝²⁷ : Module R N
      inst✝²⁶ : Module S N
      inst✝²⁵ : IsScalarTower R S N
      f✝ : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f✝
      P : Type u_2
      Q : Type u_3
      inst✝²⁴ : AddCommMonoid P
      inst✝²³ : Module R P
      inst✝²² : AddCommMonoid Q
      inst✝²¹ : Module S Q
      T : Type u_4
      O : Type u_5
      inst✝²⁰ : CommSemiring T
      inst✝¹⁹ : Algebra R T
      inst✝¹⁸ : Algebra S T
      inst✝¹⁷ : IsScalarTower R S T
      inst✝¹⁶ : AddCommMonoid O
      inst✝¹⁵ : Module R O
      inst✝¹⁴ : Module S O
      inst✝¹³ : Module T O
      inst✝¹² : IsScalarTower S T O
      inst✝¹¹ : IsScalarTower R S O
      inst✝¹⁰ : IsScalarTower R T O
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : AlgHom R S A
      g : AlgHom R R' A
      hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
      this✝² : Module S A := Module.compHom A f.toRingHom
      this✝¹ : IsScalarTower R S A
      this✝ : IsScalarTower S A A
      this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
      ⊢ ∀ (x y : S'), Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul x y)) (HMul.hMul (( …
    -/
  · intro x y
    /-
      case refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝³³ : AddCommMonoid M
      inst✝³² : AddCommMonoid N
      inst✝³¹ : CommSemiring R
      inst✝³⁰ : CommSemiring S
      inst✝²⁹ : Algebra R S
      inst✝²⁸ : Module R M
      inst✝²⁷ : Module R N
      inst✝²⁶ : Module S N
      inst✝²⁵ : IsScalarTower R S N
      f✝ : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f✝
      P : Type u_2
      Q : Type u_3
      inst✝²⁴ : AddCommMonoid P
      inst✝²³ : Module R P
      inst✝²² : AddCommMonoid Q
      inst✝²¹ : Module S Q
      T : Type u_4
      O : Type u_5
      inst✝²⁰ : CommSemiring T
      inst✝¹⁹ : Algebra R T
      inst✝¹⁸ : Algebra S T
      inst✝¹⁷ : IsScalarTower R S T
      inst✝¹⁶ : AddCommMonoid O
      inst✝¹⁵ : Module R O
      inst✝¹⁴ : Module S O
      inst✝¹³ : Module T O
      inst✝¹² : IsScalarTower S T O
      inst✝¹¹ : IsScalarTower R S O
      inst✝¹⁰ : IsScalarTower R T O
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : AlgHom R S A
      g : AlgHom R R' A
      hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
      this✝² : Module S A := Module.compHom A f.toRingHom
      this✝¹ : IsScalarTower R S A
      this✝ : IsScalarTower S A A
      this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
      x y : S'
      ⊢ Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul x y)) (HMul.hMul ((↑R (⋯.lift g.t …
    -/
    refine H.out.inductionOn x _ ?_ ?_ ?_ ?_
      /-
        case refine_2.refine_1
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x y : S'
        ⊢ Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul 0 y)) (HMul.hMul ((↑R (⋯.lift g.t …
      -/
    · rw [zero_mul, map_zero, zero_mul]
      /-
        🎉 no goals
      -/
    /-
      case refine_2.refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝³³ : AddCommMonoid M
      inst✝³² : AddCommMonoid N
      inst✝³¹ : CommSemiring R
      inst✝³⁰ : CommSemiring S
      inst✝²⁹ : Algebra R S
      inst✝²⁸ : Module R M
      inst✝²⁷ : Module R N
      inst✝²⁶ : Module S N
      inst✝²⁵ : IsScalarTower R S N
      f✝ : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f✝
      P : Type u_2
      Q : Type u_3
      inst✝²⁴ : AddCommMonoid P
      inst✝²³ : Module R P
      inst✝²² : AddCommMonoid Q
      inst✝²¹ : Module S Q
      T : Type u_4
      O : Type u_5
      inst✝²⁰ : CommSemiring T
      inst✝¹⁹ : Algebra R T
      inst✝¹⁸ : Algebra S T
      inst✝¹⁷ : IsScalarTower R S T
      inst✝¹⁶ : AddCommMonoid O
      inst✝¹⁵ : Module R O
      inst✝¹⁴ : Module S O
      inst✝¹³ : Module T O
      inst✝¹² : IsScalarTower S T O
      inst✝¹¹ : IsScalarTower R S O
      inst✝¹⁰ : IsScalarTower R T O
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : AlgHom R S A
      g : AlgHom R R' A
      hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
      this✝² : Module S A := Module.compHom A f.toRingHom
      this✝¹ : IsScalarTower R S A
      this✝ : IsScalarTower S A A
      this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
      x y : S'
      ⊢ ∀ (m : R'), Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul ((IsScalarTower.toAlg …
    -/
    rotate_left
      /-
        case refine_2.refine_3
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x y : S'
        ⊢ ∀ (s : S) (n : S'), Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul n y)) (HMul.h …
      -/
    · intro s s' e
      /-
        case refine_2.refine_3
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x y : S'
        s : S
        s' : S'
        e : Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul s' y)) (HMul.hMul ((↑R (⋯.lift  …
        ⊢ Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul (HSMul.hSMul s s') y)) (HMul.hMul …
      -/
      dsimp only [LinearMap.restrictScalars_apply] at e ⊢
      /-
        case refine_2.refine_3
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x y : S'
        s : S
        s' : S'
        e : Eq ((⋯.lift g.toLinearMap) (HMul.hMul s' y)) (HMul.hMul ((⋯.lift g.toLinea …
        ⊢ Eq ((⋯.lift g.toLinearMap) (HMul.hMul (HSMul.hSMul s s') y)) (HMul.hMul ((⋯. …
      -/
      rw [LinearMap.map_smul, smul_mul_assoc, LinearMap.map_smul, e, smul_mul_assoc]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_4
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x y : S'
        ⊢ ∀ (n₁ n₂ : S'), Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul n₁ y)) (HMul.hMul …
      -/
    · intro s s' e₁ e₂
      /-
        case refine_2.refine_4
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x y s s' : S'
        e₁ : Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul s y)) (HMul.hMul ((↑R (⋯.lift  …
        e₂ : Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul s' y)) (HMul.hMul ((↑R (⋯.lift …
        ⊢ Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul (HAdd.hAdd s s') y)) (HMul.hMul ( …
      -/
      dsimp only [LinearMap.restrictScalars_apply] at e₁ e₂ ⊢
      /-
        case refine_2.refine_4
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x y s s' : S'
        e₁ : Eq ((⋯.lift g.toLinearMap) (HMul.hMul s y)) (HMul.hMul ((⋯.lift g.toLinea …
        e₂ : Eq ((⋯.lift g.toLinearMap) (HMul.hMul s' y)) (HMul.hMul ((⋯.lift g.toLine …
        ⊢ Eq ((⋯.lift g.toLinearMap) (HMul.hMul (HAdd.hAdd s s') y)) (HMul.hMul ((⋯.li …
      -/
      rw [add_mul, map_add, map_add, add_mul, e₁, e₂]
      /-
        🎉 no goals
      -/
    /-
      case refine_2.refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝³³ : AddCommMonoid M
      inst✝³² : AddCommMonoid N
      inst✝³¹ : CommSemiring R
      inst✝³⁰ : CommSemiring S
      inst✝²⁹ : Algebra R S
      inst✝²⁸ : Module R M
      inst✝²⁷ : Module R N
      inst✝²⁶ : Module S N
      inst✝²⁵ : IsScalarTower R S N
      f✝ : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f✝
      P : Type u_2
      Q : Type u_3
      inst✝²⁴ : AddCommMonoid P
      inst✝²³ : Module R P
      inst✝²² : AddCommMonoid Q
      inst✝²¹ : Module S Q
      T : Type u_4
      O : Type u_5
      inst✝²⁰ : CommSemiring T
      inst✝¹⁹ : Algebra R T
      inst✝¹⁸ : Algebra S T
      inst✝¹⁷ : IsScalarTower R S T
      inst✝¹⁶ : AddCommMonoid O
      inst✝¹⁵ : Module R O
      inst✝¹⁴ : Module S O
      inst✝¹³ : Module T O
      inst✝¹² : IsScalarTower S T O
      inst✝¹¹ : IsScalarTower R S O
      inst✝¹⁰ : IsScalarTower R T O
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : AlgHom R S A
      g : AlgHom R R' A
      hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
      this✝² : Module S A := Module.compHom A f.toRingHom
      this✝¹ : IsScalarTower R S A
      this✝ : IsScalarTower S A A
      this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
      x y : S'
      ⊢ ∀ (m : R'), Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul ((IsScalarTower.toAlg …
    -/
    intro x
    /-
      case refine_2.refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝³³ : AddCommMonoid M
      inst✝³² : AddCommMonoid N
      inst✝³¹ : CommSemiring R
      inst✝³⁰ : CommSemiring S
      inst✝²⁹ : Algebra R S
      inst✝²⁸ : Module R M
      inst✝²⁷ : Module R N
      inst✝²⁶ : Module S N
      inst✝²⁵ : IsScalarTower R S N
      f✝ : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f✝
      P : Type u_2
      Q : Type u_3
      inst✝²⁴ : AddCommMonoid P
      inst✝²³ : Module R P
      inst✝²² : AddCommMonoid Q
      inst✝²¹ : Module S Q
      T : Type u_4
      O : Type u_5
      inst✝²⁰ : CommSemiring T
      inst✝¹⁹ : Algebra R T
      inst✝¹⁸ : Algebra S T
      inst✝¹⁷ : IsScalarTower R S T
      inst✝¹⁶ : AddCommMonoid O
      inst✝¹⁵ : Module R O
      inst✝¹⁴ : Module S O
      inst✝¹³ : Module T O
      inst✝¹² : IsScalarTower S T O
      inst✝¹¹ : IsScalarTower R S O
      inst✝¹⁰ : IsScalarTower R T O
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : AlgHom R S A
      g : AlgHom R R' A
      hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
      this✝² : Module S A := Module.compHom A f.toRingHom
      this✝¹ : IsScalarTower R S A
      this✝ : IsScalarTower S A A
      this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
      x✝ y : S'
      x : R'
      ⊢ Eq ((↑R (⋯.lift g.toLinearMap)) (HMul.hMul ((IsScalarTower.toAlgHom R R' S') …
    -/
    dsimp
    /-
      case refine_2.refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝³³ : AddCommMonoid M
      inst✝³² : AddCommMonoid N
      inst✝³¹ : CommSemiring R
      inst✝³⁰ : CommSemiring S
      inst✝²⁹ : Algebra R S
      inst✝²⁸ : Module R M
      inst✝²⁷ : Module R N
      inst✝²⁶ : Module S N
      inst✝²⁵ : IsScalarTower R S N
      f✝ : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f✝
      P : Type u_2
      Q : Type u_3
      inst✝²⁴ : AddCommMonoid P
      inst✝²³ : Module R P
      inst✝²² : AddCommMonoid Q
      inst✝²¹ : Module S Q
      T : Type u_4
      O : Type u_5
      inst✝²⁰ : CommSemiring T
      inst✝¹⁹ : Algebra R T
      inst✝¹⁸ : Algebra S T
      inst✝¹⁷ : IsScalarTower R S T
      inst✝¹⁶ : AddCommMonoid O
      inst✝¹⁵ : Module R O
      inst✝¹⁴ : Module S O
      inst✝¹³ : Module T O
      inst✝¹² : IsScalarTower S T O
      inst✝¹¹ : IsScalarTower R S O
      inst✝¹⁰ : IsScalarTower R T O
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : AlgHom R S A
      g : AlgHom R R' A
      hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
      this✝² : Module S A := Module.compHom A f.toRingHom
      this✝¹ : IsScalarTower R S A
      this✝ : IsScalarTower S A A
      this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
      x✝ y : S'
      x : R'
      ⊢ Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) y)) (HMul.hMul  …
    -/
    rw [this]
    /-
      case refine_2.refine_2
      R : Type u_1
      M : Type v₁
      N : Type v₂
      S : Type v₃
      inst✝³³ : AddCommMonoid M
      inst✝³² : AddCommMonoid N
      inst✝³¹ : CommSemiring R
      inst✝³⁰ : CommSemiring S
      inst✝²⁹ : Algebra R S
      inst✝²⁸ : Module R M
      inst✝²⁷ : Module R N
      inst✝²⁶ : Module S N
      inst✝²⁵ : IsScalarTower R S N
      f✝ : LinearMap (RingHom.id R) M N
      h : IsBaseChange S f✝
      P : Type u_2
      Q : Type u_3
      inst✝²⁴ : AddCommMonoid P
      inst✝²³ : Module R P
      inst✝²² : AddCommMonoid Q
      inst✝²¹ : Module S Q
      T : Type u_4
      O : Type u_5
      inst✝²⁰ : CommSemiring T
      inst✝¹⁹ : Algebra R T
      inst✝¹⁸ : Algebra S T
      inst✝¹⁷ : IsScalarTower R S T
      inst✝¹⁶ : AddCommMonoid O
      inst✝¹⁵ : Module R O
      inst✝¹⁴ : Module S O
      inst✝¹³ : Module T O
      inst✝¹² : IsScalarTower S T O
      inst✝¹¹ : IsScalarTower R S O
      inst✝¹⁰ : IsScalarTower R T O
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f : AlgHom R S A
      g : AlgHom R R' A
      hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
      this✝² : Module S A := Module.compHom A f.toRingHom
      this✝¹ : IsScalarTower R S A
      this✝ : IsScalarTower S A A
      this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
      x✝ y : S'
      x : R'
      ⊢ Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) y)) (HMul.hMul  …
    -/
    refine H.out.inductionOn y _ ?_ ?_ ?_ ?_
      /-
        case refine_2.refine_2.refine_1
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x✝ y : S'
        x : R'
        ⊢ Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) 0)) (HMul.hMul  …
      -/
    · rw [mul_zero, map_zero, mul_zero]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2.refine_2
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x✝ y : S'
        x : R'
        ⊢ ∀ (m : R'), Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) ((I …
      -/
    · intro y
      /-
        case refine_2.refine_2.refine_2
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x✝ y✝ : S'
        x y : R'
        ⊢ Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) ((IsScalarTower …
      -/
      dsimp
      /-
        case refine_2.refine_2.refine_2
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x✝ y✝ : S'
        x y : R'
        ⊢ Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) ((algebraMap R' …
      -/
      rw [← map_mul, this, this, map_mul]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2.refine_3
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x✝ y : S'
        x : R'
        ⊢ ∀ (s : S) (n : S'), Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S' …
      -/
    · intro s s' e
      /-
        case refine_2.refine_2.refine_3
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x✝ y : S'
        x : R'
        s : S
        s' : S'
        e : Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) s')) (HMul.hM …
        ⊢ Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) (HSMul.hSMul s  …
      -/
      rw [mul_comm, smul_mul_assoc, LinearMap.map_smul, LinearMap.map_smul, mul_comm, e]
      /-
        case refine_2.refine_2.refine_3
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x✝ y : S'
        x : R'
        s : S
        s' : S'
        e : Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) s')) (HMul.hM …
        ⊢ Eq (HSMul.hSMul s (HMul.hMul (g x) ((⋯.lift g.toLinearMap) s'))) (HMul.hMul  …
      -/
      change f s * (g x * _) = g x * (f s * _)
      /-
        case refine_2.refine_2.refine_3
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x✝ y : S'
        x : R'
        s : S
        s' : S'
        e : Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) s')) (HMul.hM …
        ⊢ Eq (HMul.hMul (f s) (HMul.hMul (g x) ((⋯.lift g.toLinearMap) s'))) (HMul.hMu …
      -/
      rw [← mul_assoc, ← mul_assoc, hf]
      /-
        🎉 no goals
      -/
      /-
        case refine_2.refine_2.refine_4
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x✝ y : S'
        x : R'
        ⊢ ∀ (n₁ n₂ : S'), Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) …
      -/
    · intro s s' e₁ e₂
      /-
        case refine_2.refine_2.refine_4
        R : Type u_1
        M : Type v₁
        N : Type v₂
        S : Type v₃
        inst✝³³ : AddCommMonoid M
        inst✝³² : AddCommMonoid N
        inst✝³¹ : CommSemiring R
        inst✝³⁰ : CommSemiring S
        inst✝²⁹ : Algebra R S
        inst✝²⁸ : Module R M
        inst✝²⁷ : Module R N
        inst✝²⁶ : Module S N
        inst✝²⁵ : IsScalarTower R S N
        f✝ : LinearMap (RingHom.id R) M N
        h : IsBaseChange S f✝
        P : Type u_2
        Q : Type u_3
        inst✝²⁴ : AddCommMonoid P
        inst✝²³ : Module R P
        inst✝²² : AddCommMonoid Q
        inst✝²¹ : Module S Q
        T : Type u_4
        O : Type u_5
        inst✝²⁰ : CommSemiring T
        inst✝¹⁹ : Algebra R T
        inst✝¹⁸ : Algebra S T
        inst✝¹⁷ : IsScalarTower R S T
        inst✝¹⁶ : AddCommMonoid O
        inst✝¹⁵ : Module R O
        inst✝¹⁴ : Module S O
        inst✝¹³ : Module T O
        inst✝¹² : IsScalarTower S T O
        inst✝¹¹ : IsScalarTower R S O
        inst✝¹⁰ : IsScalarTower R T O
        R' : Type u_6
        S' : Type u_7
        inst✝⁹ : CommSemiring R'
        inst✝⁸ : CommSemiring S'
        inst✝⁷ : Algebra R R'
        inst✝⁶ : Algebra S S'
        inst✝⁵ : Algebra R' S'
        inst✝⁴ : Algebra R S'
        inst✝³ : IsScalarTower R R' S'
        inst✝² : IsScalarTower R S S'
        H : Algebra.IsPushout R S R' S'
        A : Type u_8
        inst✝¹ : Semiring A
        inst✝ : Algebra R A
        f : AlgHom R S A
        g : AlgHom R R' A
        hf : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
        this✝² : Module S A := Module.compHom A f.toRingHom
        this✝¹ : IsScalarTower R S A
        this✝ : IsScalarTower S A A
        this : ∀ (x : R'), Eq ((⋯.lift g.toLinearMap) ((algebraMap R' S') x)) (g x)
        x✝ y : S'
        x : R'
        s s' : S'
        e₁ : Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) s)) (HMul.hM …
        e₂ : Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) s')) (HMul.h …
        ⊢ Eq ((⋯.lift g.toLinearMap) (HMul.hMul ((algebraMap R' S') x) (HAdd.hAdd s s' …
      -/
      rw [mul_add, map_add, map_add, mul_add, e₁, e₂]
      /-
        🎉 no goals
      -/


@[simp]
theorem Algebra.pushoutDesc_left [Algebra.IsPushout R S R' S'] {A : Type*} [Semiring A]
    [Algebra R A] (f : S →ₐ[R] A) (g : R' →ₐ[R] A) (H) (x : S) :
    Algebra.pushoutDesc S' f g H (algebraMap S S' x) = f x := by
  /-
    R : Type u_1
    S : Type v₃
    inst✝¹³ : CommSemiring R
    inst✝¹² : CommSemiring S
    inst✝¹¹ : Algebra R S
    R' : Type u_6
    S' : Type u_7
    inst✝¹⁰ : CommSemiring R'
    inst✝⁹ : CommSemiring S'
    inst✝⁸ : Algebra R R'
    inst✝⁷ : Algebra S S'
    inst✝⁶ : Algebra R' S'
    inst✝⁵ : Algebra R S'
    inst✝⁴ : IsScalarTower R R' S'
    inst✝³ : IsScalarTower R S S'
    inst✝² : Algebra.IsPushout R S R' S'
    A : Type u_8
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : AlgHom R S A
    g : AlgHom R R' A
    H : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
    x : S
    ⊢ Eq ((Algebra.pushoutDesc S' f g H) ((algebraMap S S') x)) (f x)
  -/
  letI := Module.compHom A f.toRingHom
  haveI : IsScalarTower R S A :=
    { smul_assoc := fun r s a =>
        show f (r • s) * a = r • (f s * a) by rw [map_smul, smul_mul_assoc] }
  /-
    R : Type u_1
    S : Type v₃
    inst✝¹³ : CommSemiring R
    inst✝¹² : CommSemiring S
    inst✝¹¹ : Algebra R S
    R' : Type u_6
    S' : Type u_7
    inst✝¹⁰ : CommSemiring R'
    inst✝⁹ : CommSemiring S'
    inst✝⁸ : Algebra R R'
    inst✝⁷ : Algebra S S'
    inst✝⁶ : Algebra R' S'
    inst✝⁵ : Algebra R S'
    inst✝⁴ : IsScalarTower R R' S'
    inst✝³ : IsScalarTower R S S'
    inst✝² : Algebra.IsPushout R S R' S'
    A : Type u_8
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : AlgHom R S A
    g : AlgHom R R' A
    H : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
    x : S
    this✝ : Module S A := Module.compHom A f.toRingHom
    this : IsScalarTower R S A
    ⊢ Eq ((Algebra.pushoutDesc S' f g H) ((algebraMap S S') x)) (f x)
  -/
  haveI : IsScalarTower S A A := { smul_assoc := fun r a b => mul_assoc _ _ _ }
  rw [Algebra.algebraMap_eq_smul_one, pushoutDesc_apply, map_smul, ←
    Algebra.pushoutDesc_apply S' f g H, map_one]
  /-
    R : Type u_1
    S : Type v₃
    inst✝¹³ : CommSemiring R
    inst✝¹² : CommSemiring S
    inst✝¹¹ : Algebra R S
    R' : Type u_6
    S' : Type u_7
    inst✝¹⁰ : CommSemiring R'
    inst✝⁹ : CommSemiring S'
    inst✝⁸ : Algebra R R'
    inst✝⁷ : Algebra S S'
    inst✝⁶ : Algebra R' S'
    inst✝⁵ : Algebra R S'
    inst✝⁴ : IsScalarTower R R' S'
    inst✝³ : IsScalarTower R S S'
    inst✝² : Algebra.IsPushout R S R' S'
    A : Type u_8
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f : AlgHom R S A
    g : AlgHom R R' A
    H : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
    x : S
    this✝¹ : Module S A := Module.compHom A f.toRingHom
    this✝ : IsScalarTower R S A
    this : IsScalarTower S A A
    ⊢ Eq (HSMul.hSMul x 1) (f x)
  -/
  exact mul_one (f x)
  /-
    🎉 no goals
  -/


theorem Algebra.lift_algHom_comp_left [Algebra.IsPushout R S R' S'] {A : Type*} [Semiring A]
    [Algebra R A] (f : S →ₐ[R] A) (g : R' →ₐ[R] A) (H) :
    (Algebra.pushoutDesc S' f g H).comp (toAlgHom R S S') = f :=
  AlgHom.ext fun x => (Algebra.pushoutDesc_left S' f g H x : _)


@[simp]
theorem Algebra.pushoutDesc_right [Algebra.IsPushout R S R' S'] {A : Type*} [Semiring A]
    [Algebra R A] (f : S →ₐ[R] A) (g : R' →ₐ[R] A) (H) (x : R') :
    Algebra.pushoutDesc S' f g H (algebraMap R' S' x) = g x :=
  letI := Module.compHom A f.toRingHom
  haveI : IsScalarTower R S A :=
    { smul_assoc := fun r s a =>
                                              /-
                                                R : Type u_1
                                                S : Type v₃
                                                inst✝¹³ : CommSemiring R
                                                inst✝¹² : CommSemiring S
                                                inst✝¹¹ : Algebra R S
                                                R' : Type u_6
                                                S' : Type u_7
                                                inst✝¹⁰ : CommSemiring R'
                                                inst✝⁹ : CommSemiring S'
                                                inst✝⁸ : Algebra R R'
                                                inst✝⁷ : Algebra S S'
                                                inst✝⁶ : Algebra R' S'
                                                inst✝⁵ : Algebra R S'
                                                inst✝⁴ : IsScalarTower R R' S'
                                                inst✝³ : IsScalarTower R S S'
                                                inst✝² : Algebra.IsPushout R S R' S'
                                                A : Type u_8
                                                inst✝¹ : Semiring A
                                                inst✝ : Algebra R A
                                                f : AlgHom R S A
                                                g : AlgHom R R' A
                                                H : ∀ (x : S) (y : R'), Eq (HMul.hMul (f x) (g y)) (HMul.hMul (g y) (f x))
                                                x : R'
                                                this : Module S A := Module.compHom A f.toRingHom
                                                r : R
                                                s : S
                                                a : A
                                                ⊢ Eq (HMul.hMul (f (HSMul.hSMul r s)) a) (HSMul.hSMul r (HMul.hMul (f s) a))
                                              -/
        show f (r • s) * a = r • (f s * a) by rw [map_smul, smul_mul_assoc] }
                                              /-
                                                🎉 no goals
                                              -/
  IsBaseChange.lift_eq _ _ _


theorem Algebra.lift_algHom_comp_right [Algebra.IsPushout R S R' S'] {A : Type*} [Semiring A]
    [Algebra R A] (f : S →ₐ[R] A) (g : R' →ₐ[R] A) (H) :
    (Algebra.pushoutDesc S' f g H).comp (toAlgHom R R' S') = g :=
  AlgHom.ext fun x => (Algebra.pushoutDesc_right S' f g H x : _)


@[ext (iff := false)]
theorem Algebra.IsPushout.algHom_ext [H : Algebra.IsPushout R S R' S'] {A : Type*} [Semiring A]
    [Algebra R A] {f g : S' →ₐ[R] A} (h₁ : f.comp (toAlgHom R R' S') = g.comp (toAlgHom R R' S'))
    (h₂ : f.comp (toAlgHom R S S') = g.comp (toAlgHom R S S')) : f = g := by
  /-
    R : Type u_1
    S : Type v₃
    inst✝¹² : CommSemiring R
    inst✝¹¹ : CommSemiring S
    inst✝¹⁰ : Algebra R S
    R' : Type u_6
    S' : Type u_7
    inst✝⁹ : CommSemiring R'
    inst✝⁸ : CommSemiring S'
    inst✝⁷ : Algebra R R'
    inst✝⁶ : Algebra S S'
    inst✝⁵ : Algebra R' S'
    inst✝⁴ : Algebra R S'
    inst✝³ : IsScalarTower R R' S'
    inst✝² : IsScalarTower R S S'
    H : Algebra.IsPushout R S R' S'
    A : Type u_8
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R S' A
    h₁ : Eq (f.comp (IsScalarTower.toAlgHom R R' S')) (g.comp (IsScalarTower.toAlg …
    h₂ : Eq (f.comp (IsScalarTower.toAlgHom R S S')) (g.comp (IsScalarTower.toAlgH …
    ⊢ Eq f g
  -/
  ext x
  /-
    case H
    R : Type u_1
    S : Type v₃
    inst✝¹² : CommSemiring R
    inst✝¹¹ : CommSemiring S
    inst✝¹⁰ : Algebra R S
    R' : Type u_6
    S' : Type u_7
    inst✝⁹ : CommSemiring R'
    inst✝⁸ : CommSemiring S'
    inst✝⁷ : Algebra R R'
    inst✝⁶ : Algebra S S'
    inst✝⁵ : Algebra R' S'
    inst✝⁴ : Algebra R S'
    inst✝³ : IsScalarTower R R' S'
    inst✝² : IsScalarTower R S S'
    H : Algebra.IsPushout R S R' S'
    A : Type u_8
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    f g : AlgHom R S' A
    h₁ : Eq (f.comp (IsScalarTower.toAlgHom R R' S')) (g.comp (IsScalarTower.toAlg …
    h₂ : Eq (f.comp (IsScalarTower.toAlgHom R S S')) (g.comp (IsScalarTower.toAlgH …
    x : S'
    ⊢ Eq (f x) (g x)
  -/
  refine H.1.inductionOn x _ ?_ ?_ ?_ ?_
    /-
      case H.refine_1
      R : Type u_1
      S : Type v₃
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f g : AlgHom R S' A
      h₁ : Eq (f.comp (IsScalarTower.toAlgHom R R' S')) (g.comp (IsScalarTower.toAlg …
      h₂ : Eq (f.comp (IsScalarTower.toAlgHom R S S')) (g.comp (IsScalarTower.toAlgH …
      x : S'
      ⊢ Eq (f 0) (g 0)
    -/
  · simp only [map_zero]
    /-
      🎉 no goals
    -/
    /-
      case H.refine_2
      R : Type u_1
      S : Type v₃
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f g : AlgHom R S' A
      h₁ : Eq (f.comp (IsScalarTower.toAlgHom R R' S')) (g.comp (IsScalarTower.toAlg …
      h₂ : Eq (f.comp (IsScalarTower.toAlgHom R S S')) (g.comp (IsScalarTower.toAlgH …
      x : S'
      ⊢ ∀ (m : R'), Eq (f ((IsScalarTower.toAlgHom R R' S').toLinearMap m)) (g ((IsS …
    -/
  · exact AlgHom.congr_fun h₁
    /-
      🎉 no goals
    -/
    /-
      case H.refine_3
      R : Type u_1
      S : Type v₃
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f g : AlgHom R S' A
      h₁ : Eq (f.comp (IsScalarTower.toAlgHom R R' S')) (g.comp (IsScalarTower.toAlg …
      h₂ : Eq (f.comp (IsScalarTower.toAlgHom R S S')) (g.comp (IsScalarTower.toAlgH …
      x : S'
      ⊢ ∀ (s : S) (n : S'), Eq (f n) (g n) → Eq (f (HSMul.hSMul s n)) (g (HSMul.hSMu …
    -/
  · intro s s' e
    /-
      case H.refine_3
      R : Type u_1
      S : Type v₃
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f g : AlgHom R S' A
      h₁ : Eq (f.comp (IsScalarTower.toAlgHom R R' S')) (g.comp (IsScalarTower.toAlg …
      h₂ : Eq (f.comp (IsScalarTower.toAlgHom R S S')) (g.comp (IsScalarTower.toAlgH …
      x : S'
      s : S
      s' : S'
      e : Eq (f s') (g s')
      ⊢ Eq (f (HSMul.hSMul s s')) (g (HSMul.hSMul s s'))
    -/
    rw [Algebra.smul_def, map_mul, map_mul, e]
    /-
      case H.refine_3
      R : Type u_1
      S : Type v₃
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f g : AlgHom R S' A
      h₁ : Eq (f.comp (IsScalarTower.toAlgHom R R' S')) (g.comp (IsScalarTower.toAlg …
      h₂ : Eq (f.comp (IsScalarTower.toAlgHom R S S')) (g.comp (IsScalarTower.toAlgH …
      x : S'
      s : S
      s' : S'
      e : Eq (f s') (g s')
      ⊢ Eq (HMul.hMul (f ((algebraMap S S') s)) (g s')) (HMul.hMul (g ((algebraMap S …
    -/
    congr 1
    /-
      case H.refine_3.e_a
      R : Type u_1
      S : Type v₃
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f g : AlgHom R S' A
      h₁ : Eq (f.comp (IsScalarTower.toAlgHom R R' S')) (g.comp (IsScalarTower.toAlg …
      h₂ : Eq (f.comp (IsScalarTower.toAlgHom R S S')) (g.comp (IsScalarTower.toAlgH …
      x : S'
      s : S
      s' : S'
      e : Eq (f s') (g s')
      ⊢ Eq (f ((algebraMap S S') s)) (g ((algebraMap S S') s))
    -/
    exact (AlgHom.congr_fun h₂ s : _)
    /-
      🎉 no goals
    -/
    /-
      case H.refine_4
      R : Type u_1
      S : Type v₃
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f g : AlgHom R S' A
      h₁ : Eq (f.comp (IsScalarTower.toAlgHom R R' S')) (g.comp (IsScalarTower.toAlg …
      h₂ : Eq (f.comp (IsScalarTower.toAlgHom R S S')) (g.comp (IsScalarTower.toAlgH …
      x : S'
      ⊢ ∀ (n₁ n₂ : S'), Eq (f n₁) (g n₁) → Eq (f n₂) (g n₂) → Eq (f (HAdd.hAdd n₁ n₂ …
    -/
  · intro s₁ s₂ e₁ e₂
    /-
      case H.refine_4
      R : Type u_1
      S : Type v₃
      inst✝¹² : CommSemiring R
      inst✝¹¹ : CommSemiring S
      inst✝¹⁰ : Algebra R S
      R' : Type u_6
      S' : Type u_7
      inst✝⁹ : CommSemiring R'
      inst✝⁸ : CommSemiring S'
      inst✝⁷ : Algebra R R'
      inst✝⁶ : Algebra S S'
      inst✝⁵ : Algebra R' S'
      inst✝⁴ : Algebra R S'
      inst✝³ : IsScalarTower R R' S'
      inst✝² : IsScalarTower R S S'
      H : Algebra.IsPushout R S R' S'
      A : Type u_8
      inst✝¹ : Semiring A
      inst✝ : Algebra R A
      f g : AlgHom R S' A
      h₁ : Eq (f.comp (IsScalarTower.toAlgHom R R' S')) (g.comp (IsScalarTower.toAlg …
      h₂ : Eq (f.comp (IsScalarTower.toAlgHom R S S')) (g.comp (IsScalarTower.toAlgH …
      x s₁ s₂ : S'
      e₁ : Eq (f s₁) (g s₁)
      e₂ : Eq (f s₂) (g s₂)
      ⊢ Eq (f (HAdd.hAdd s₁ s₂)) (g (HAdd.hAdd s₁ s₂))
    -/
    rw [map_add, map_add, e₁, e₂]
    /-
      🎉 no goals
    -/


/--
Let the following be a commutative diagram of rings
```
  R  →  S  →  T
  ↓     ↓     ↓
  R' →  S' →  T'
```
where the left-hand square is a pushout. Then the following are equivalent:
- the big rectangle is a pushout.
- the right-hand square is a pushout.

Note that this is essentially the isomorphism `T ⊗[S] (S ⊗[R] R') ≃ₐ[T] T ⊗[R] R'`.
-/
lemma Algebra.IsPushout.comp_iff {T' : Type*} [CommRing T'] [Algebra R T']
    [Algebra S' T'] [Algebra S T'] [Algebra T T'] [Algebra R' T']
    [IsScalarTower R T T'] [IsScalarTower S T T'] [IsScalarTower S S' T']
    [IsScalarTower R R' T'] [IsScalarTower R S' T'] [IsScalarTower R' S' T']
    [Algebra.IsPushout R S R' S'] :
    Algebra.IsPushout R T R' T' ↔ Algebra.IsPushout S T S' T' := by
  /-
    R : Type u_1
    S : Type v₃
    inst✝²⁷ : CommSemiring R
    inst✝²⁶ : CommSemiring S
    inst✝²⁵ : Algebra R S
    T : Type u_4
    inst✝²⁴ : CommSemiring T
    inst✝²³ : Algebra R T
    inst✝²² : Algebra S T
    inst✝²¹ : IsScalarTower R S T
    R' : Type u_6
    S' : Type u_7
    inst✝²⁰ : CommSemiring R'
    inst✝¹⁹ : CommSemiring S'
    inst✝¹⁸ : Algebra R R'
    inst✝¹⁷ : Algebra S S'
    inst✝¹⁶ : Algebra R' S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    inst✝¹³ : IsScalarTower R S S'
    T' : Type u_8
    inst✝¹² : CommRing T'
    inst✝¹¹ : Algebra R T'
    inst✝¹⁰ : Algebra S' T'
    inst✝⁹ : Algebra S T'
    inst✝⁸ : Algebra T T'
    inst✝⁷ : Algebra R' T'
    inst✝⁶ : IsScalarTower R T T'
    inst✝⁵ : IsScalarTower S T T'
    inst✝⁴ : IsScalarTower S S' T'
    inst✝³ : IsScalarTower R R' T'
    inst✝² : IsScalarTower R S' T'
    inst✝¹ : IsScalarTower R' S' T'
    inst✝ : Algebra.IsPushout R S R' S'
    ⊢ Iff (Algebra.IsPushout R T R' T') (Algebra.IsPushout S T S' T')
  -/
  let f : R' →ₗ[R] S' := (IsScalarTower.toAlgHom R R' S').toLinearMap
  haveI : IsScalarTower R S T' := IsScalarTower.of_algebraMap_eq <| fun x ↦ by
    rw [algebraMap_apply R S' T', algebraMap_apply R S S', ← algebraMap_apply S S' T']
  have heq : (toAlgHom S S' T').toLinearMap.restrictScalars R ∘ₗ f =
      (toAlgHom R R' T').toLinearMap := by
    ext x
    simp [f, ← IsScalarTower.algebraMap_apply]
  /-
    R : Type u_1
    S : Type v₃
    inst✝²⁷ : CommSemiring R
    inst✝²⁶ : CommSemiring S
    inst✝²⁵ : Algebra R S
    T : Type u_4
    inst✝²⁴ : CommSemiring T
    inst✝²³ : Algebra R T
    inst✝²² : Algebra S T
    inst✝²¹ : IsScalarTower R S T
    R' : Type u_6
    S' : Type u_7
    inst✝²⁰ : CommSemiring R'
    inst✝¹⁹ : CommSemiring S'
    inst✝¹⁸ : Algebra R R'
    inst✝¹⁷ : Algebra S S'
    inst✝¹⁶ : Algebra R' S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    inst✝¹³ : IsScalarTower R S S'
    T' : Type u_8
    inst✝¹² : CommRing T'
    inst✝¹¹ : Algebra R T'
    inst✝¹⁰ : Algebra S' T'
    inst✝⁹ : Algebra S T'
    inst✝⁸ : Algebra T T'
    inst✝⁷ : Algebra R' T'
    inst✝⁶ : IsScalarTower R T T'
    inst✝⁵ : IsScalarTower S T T'
    inst✝⁴ : IsScalarTower S S' T'
    inst✝³ : IsScalarTower R R' T'
    inst✝² : IsScalarTower R S' T'
    inst✝¹ : IsScalarTower R' S' T'
    inst✝ : Algebra.IsPushout R S R' S'
    f : LinearMap (RingHom.id R) R' S' := (IsScalarTower.toAlgHom R R' S').toLinea …
    this : IsScalarTower R S T'
    heq : Eq ((↑R (IsScalarTower.toAlgHom S S' T').toLinearMap).comp f) (IsScalarT …
    ⊢ Iff (Algebra.IsPushout R T R' T') (Algebra.IsPushout S T S' T')
  -/
  rw [isPushout_iff, isPushout_iff, ← heq, IsBaseChange.comp_iff]
  /-
    case hf
    R : Type u_1
    S : Type v₃
    inst✝²⁷ : CommSemiring R
    inst✝²⁶ : CommSemiring S
    inst✝²⁵ : Algebra R S
    T : Type u_4
    inst✝²⁴ : CommSemiring T
    inst✝²³ : Algebra R T
    inst✝²² : Algebra S T
    inst✝²¹ : IsScalarTower R S T
    R' : Type u_6
    S' : Type u_7
    inst✝²⁰ : CommSemiring R'
    inst✝¹⁹ : CommSemiring S'
    inst✝¹⁸ : Algebra R R'
    inst✝¹⁷ : Algebra S S'
    inst✝¹⁶ : Algebra R' S'
    inst✝¹⁵ : Algebra R S'
    inst✝¹⁴ : IsScalarTower R R' S'
    inst✝¹³ : IsScalarTower R S S'
    T' : Type u_8
    inst✝¹² : CommRing T'
    inst✝¹¹ : Algebra R T'
    inst✝¹⁰ : Algebra S' T'
    inst✝⁹ : Algebra S T'
    inst✝⁸ : Algebra T T'
    inst✝⁷ : Algebra R' T'
    inst✝⁶ : IsScalarTower R T T'
    inst✝⁵ : IsScalarTower S T T'
    inst✝⁴ : IsScalarTower S S' T'
    inst✝³ : IsScalarTower R R' T'
    inst✝² : IsScalarTower R S' T'
    inst✝¹ : IsScalarTower R' S' T'
    inst✝ : Algebra.IsPushout R S R' S'
    f : LinearMap (RingHom.id R) R' S' := (IsScalarTower.toAlgHom R R' S').toLinea …
    this : IsScalarTower R S T'
    heq : Eq ((↑R (IsScalarTower.toAlgHom S S' T').toLinearMap).comp f) (IsScalarT …
    ⊢ IsBaseChange S f
  -/
  exact Algebra.IsPushout.out
  /-
    🎉 no goals
  -/


