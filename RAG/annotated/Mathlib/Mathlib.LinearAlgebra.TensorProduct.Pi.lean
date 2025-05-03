private noncomputable def piRightHomBil : N →ₗ[S] (∀ i, M i) →ₗ[R] ∀ i, N ⊗[R] M i where
  toFun n := LinearMap.pi (fun i ↦ mk R N (M i) n ∘ₗ LinearMap.proj i)
  map_add' _ _ := by
    /-
      R : Type u_1
      inst✝⁸ : CommSemiring R
      S : Type u_2
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      N : Type u_3
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      ι : Type u_4
      M : ι → Type u_5
      inst✝¹ : (i : ι) → AddCommMonoid (M i)
      inst✝ : (i : ι) → Module R (M i)
      x✝¹ x✝ : N
      ⊢ Eq ((fun n => LinearMap.pi fun i => ((TensorProduct.mk R N (M i)) n).comp (L …
    -/
    ext
    /-
      case h.h
      R : Type u_1
      inst✝⁸ : CommSemiring R
      S : Type u_2
      inst✝⁷ : CommSemiring S
      inst✝⁶ : Algebra R S
      N : Type u_3
      inst✝⁵ : AddCommMonoid N
      inst✝⁴ : Module R N
      inst✝³ : Module S N
      inst✝² : IsScalarTower R S N
      ι : Type u_4
      M : ι → Type u_5
      inst✝¹ : (i : ι) → AddCommMonoid (M i)
      inst✝ : (i : ι) → Module R (M i)
      x✝³ x✝² : N
      x✝¹ : (i : ι) → M i
      x✝ : ι
      ⊢ Eq (((fun n => LinearMap.pi fun i => ((TensorProduct.mk R N (M i)) n).comp ( …
    -/
    simp [add_tmul]
    /-
      🎉 no goals
    -/
  map_smul' _ _ := rfl


/-- For any `R`-module `N`, index type `ι` and family of `R`-modules `Mᵢ`, there is a natural
linear map `N ⊗[R] (∀ i, M i) →ₗ ∀ i, N ⊗[R] M i`. This map is an isomorphism if `ι` is finite. -/
noncomputable def piRightHom : N ⊗[R] (∀ i, M i) →ₗ[S] ∀ i, N ⊗[R] M i :=
  AlgebraTensorModule.lift <| piRightHomBil R S N M


@[simp]
lemma piRightHom_tmul (x : N) (f : ∀ i, M i) :
    piRightHom R S N M (x ⊗ₜ f) = (fun j ↦ x ⊗ₜ f j) :=
  rfl


private noncomputable
def piRightInv : (∀ i, N ⊗[R] M i) →ₗ[S] N ⊗[R] ∀ i, M i :=
  LinearMap.lsum S (fun i ↦ N ⊗[R] M i) S <| fun i ↦
    AlgebraTensorModule.map LinearMap.id (single R M i)


@[simp]
private lemma piRightInv_apply (x : N) (m : ∀ i, M i) :
    piRightInv R S N M (fun i ↦ x ⊗ₜ m i) = x ⊗ₜ m := by
  simp only [piRightInv, lsum_apply, coeFn_sum, coe_comp, coe_proj, Finset.sum_apply,
    Function.comp_apply, Function.eval, AlgebraTensorModule.map_tmul, id_coe, id_eq, coe_single]
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Type u_2
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    N : Type u_3
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    ι : Type u_4
    M : ι → Type u_5
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : N
    m : (i : ι) → M i
    ⊢ Eq (Finset.univ.sum fun x_1 => TensorProduct.tmul R x (Pi.single x_1 (m x_1) …
  -/
  rw [← tmul_sum]
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Type u_2
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    N : Type u_3
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    ι : Type u_4
    M : ι → Type u_5
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : N
    m : (i : ι) → M i
    ⊢ Eq (TensorProduct.tmul R x (Finset.univ.sum fun a => Pi.single a (m a))) (Te …
  -/
  congr
  /-
    case e_n
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Type u_2
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    N : Type u_3
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    ι : Type u_4
    M : ι → Type u_5
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : N
    m : (i : ι) → M i
    ⊢ Eq (Finset.univ.sum fun a => Pi.single a (m a)) m
  -/
  ext j
  /-
    case e_n.h
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Type u_2
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    N : Type u_3
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    ι : Type u_4
    M : ι → Type u_5
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : N
    m : (i : ι) → M i
    j : ι
    ⊢ Eq (Finset.univ.sum (fun a => Pi.single a (m a)) j) (m j)
  -/
  simp
  /-
    🎉 no goals
  -/


@[simp]
private lemma piRightInv_single (x : N) (i : ι) (m : M i) :
    piRightInv R S N M (Pi.single i (x ⊗ₜ m)) = x ⊗ₜ Pi.single i m := by
  have : Pi.single i (x ⊗ₜ m) = fun j ↦ x ⊗ₜ[R] (Pi.single i m j) := by
    ext j
    rw [← tmul_single]
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Type u_2
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    N : Type u_3
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    ι : Type u_4
    M : ι → Type u_5
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : N
    i : ι
    m : M i
    this : Eq (Pi.single i (TensorProduct.tmul R x m)) fun j => TensorProduct.tmul …
    ⊢ Eq ((TensorProduct.piRightInv R S N M) (Pi.single i (TensorProduct.tmul R x  …
  -/
  rw [this]
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Type u_2
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    N : Type u_3
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    ι : Type u_4
    M : ι → Type u_5
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : N
    i : ι
    m : M i
    this : Eq (Pi.single i (TensorProduct.tmul R x m)) fun j => TensorProduct.tmul …
    ⊢ Eq ((TensorProduct.piRightInv R S N M) fun j => TensorProduct.tmul R x (Pi.s …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Tensor product commutes with finite products on the right. -/
noncomputable def piRight : N ⊗[R] (∀ i, M i) ≃ₗ[S] ∀ i, N ⊗[R] M i :=
  LinearEquiv.ofLinear
    (piRightHom R S N M)
    (piRightInv R S N M)
        /-
          R : Type u_1
          inst✝¹⁰ : CommSemiring R
          S : Type u_2
          inst✝⁹ : CommSemiring S
          inst✝⁸ : Algebra R S
          N : Type u_3
          inst✝⁷ : AddCommMonoid N
          inst✝⁶ : Module R N
          inst✝⁵ : Module S N
          inst✝⁴ : IsScalarTower R S N
          ι : Type u_4
          M : ι → Type u_5
          inst✝³ : (i : ι) → AddCommMonoid (M i)
          inst✝² : (i : ι) → Module R (M i)
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          ⊢ Eq ((TensorProduct.piRightHom R S N M).comp (TensorProduct.piRightInv R S N  …
        -/
    (by ext i x m j; simp [tmul_single])
                     /-
                       🎉 no goals
                     -/
        /-
          R : Type u_1
          inst✝¹⁰ : CommSemiring R
          S : Type u_2
          inst✝⁹ : CommSemiring S
          inst✝⁸ : Algebra R S
          N : Type u_3
          inst✝⁷ : AddCommMonoid N
          inst✝⁶ : Module R N
          inst✝⁵ : Module S N
          inst✝⁴ : IsScalarTower R S N
          ι : Type u_4
          M : ι → Type u_5
          inst✝³ : (i : ι) → AddCommMonoid (M i)
          inst✝² : (i : ι) → Module R (M i)
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          ⊢ Eq ((TensorProduct.piRightInv R S N M).comp (TensorProduct.piRightHom R S N  …
        -/
    (by ext x j m; simp)
                   /-
                     🎉 no goals
                   -/


@[simp]
lemma piRight_apply (x : N ⊗[R] (∀ i, M i)) :
    piRight R S N M x = piRightHom R S N M x := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Type u_2
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    N : Type u_3
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    ι : Type u_4
    M : ι → Type u_5
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : TensorProduct R N ((i : ι) → M i)
    ⊢ Eq ((TensorProduct.piRight R S N M) x) ((TensorProduct.piRightHom R S N M) x)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma piRight_symm_apply (x : N) (m : ∀ i, M i) :
    (piRight R S N M).symm (fun i ↦ x ⊗ₜ m i) = x ⊗ₜ m := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Type u_2
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    N : Type u_3
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    ι : Type u_4
    M : ι → Type u_5
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : N
    m : (i : ι) → M i
    ⊢ Eq ((TensorProduct.piRight R S N M).symm fun i => TensorProduct.tmul R x (m  …
  -/
  simp [piRight]
  /-
    🎉 no goals
  -/


@[simp]
lemma piRight_symm_single (x : N) (i : ι) (m : M i) :
    (piRight R S N M).symm (Pi.single i (x ⊗ₜ m)) = x ⊗ₜ Pi.single i m := by
  /-
    R : Type u_1
    inst✝¹⁰ : CommSemiring R
    S : Type u_2
    inst✝⁹ : CommSemiring S
    inst✝⁸ : Algebra R S
    N : Type u_3
    inst✝⁷ : AddCommMonoid N
    inst✝⁶ : Module R N
    inst✝⁵ : Module S N
    inst✝⁴ : IsScalarTower R S N
    ι : Type u_4
    M : ι → Type u_5
    inst✝³ : (i : ι) → AddCommMonoid (M i)
    inst✝² : (i : ι) → Module R (M i)
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : N
    i : ι
    m : M i
    ⊢ Eq ((TensorProduct.piRight R S N M).symm (Pi.single i (TensorProduct.tmul R  …
  -/
  simp [piRight]
  /-
    🎉 no goals
  -/


private def piScalarRightHomBil : N →ₗ[S] (ι → R) →ₗ[R] (ι → N) where
  toFun n := LinearMap.compLeft (toSpanSingleton R N n) ι
  map_add' x y := by
    /-
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Type u_2
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      N : Type u_3
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      ι : Type u_4
      x y : N
      ⊢ Eq ((fun n => (LinearMap.toSpanSingleton R N n).compLeft ι) (HAdd.hAdd x y)) …
    -/
    ext i j
    /-
      case h.h
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Type u_2
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      N : Type u_3
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      ι : Type u_4
      x y : N
      i : ι → R
      j : ι
      ⊢ Eq (((fun n => (LinearMap.toSpanSingleton R N n).compLeft ι) (HAdd.hAdd x y) …
    -/
    simp
    /-
      🎉 no goals
    -/
  map_smul' s x := by
    /-
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Type u_2
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      N : Type u_3
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      ι : Type u_4
      s : S
      x : N
      ⊢ Eq ({ toFun := fun n => (LinearMap.toSpanSingleton R N n).compLeft ι, map_ad …
    -/
    ext i j
    dsimp only [coe_comp, coe_single, Function.comp_apply, compLeft_apply, toSpanSingleton_apply,
      RingHom.id_apply, smul_apply, Pi.smul_apply]
    /-
      case h.h
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Type u_2
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      N : Type u_3
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      ι : Type u_4
      s : S
      x : N
      i : ι → R
      j : ι
      ⊢ Eq (HSMul.hSMul (i j) (HSMul.hSMul s x)) (HSMul.hSMul s (HSMul.hSMul (i j) x))
    -/
    rw [← IsScalarTower.smul_assoc, _root_.Algebra.smul_def, mul_comm, mul_smul]
    /-
      case h.h
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Type u_2
      inst✝⁵ : CommSemiring S
      inst✝⁴ : Algebra R S
      N : Type u_3
      inst✝³ : AddCommMonoid N
      inst✝² : Module R N
      inst✝¹ : Module S N
      inst✝ : IsScalarTower R S N
      ι : Type u_4
      s : S
      x : N
      i : ι → R
      j : ι
      ⊢ Eq (HSMul.hSMul s (HSMul.hSMul ((algebraMap R S) (i j)) x)) (HSMul.hSMul s ( …
    -/
    simp
    /-
      🎉 no goals
    -/


/-- For any `R`-module `N` and index type `ι`, there is a natural
linear map `N ⊗[R] (ι → R) →ₗ (ι → N)`. This map is an isomorphism if `ι` is finite. -/
noncomputable def piScalarRightHom : N ⊗[R] (ι → R) →ₗ[S] (ι → N) :=
  AlgebraTensorModule.lift <| piScalarRightHomBil R S N ι


@[simp]
lemma piScalarRightHom_tmul (x : N) (f : ι → R) :
    piScalarRightHom R S N ι (x ⊗ₜ f) = (fun j ↦ f j • x) := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Type u_2
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    N : Type u_3
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : Module S N
    inst✝ : IsScalarTower R S N
    ι : Type u_4
    x : N
    f : ι → R
    ⊢ Eq ((TensorProduct.piScalarRightHom R S N ι) (TensorProduct.tmul R x f)) fun …
  -/
  ext j
  /-
    case h
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Type u_2
    inst✝⁵ : CommSemiring S
    inst✝⁴ : Algebra R S
    N : Type u_3
    inst✝³ : AddCommMonoid N
    inst✝² : Module R N
    inst✝¹ : Module S N
    inst✝ : IsScalarTower R S N
    ι : Type u_4
    x : N
    f : ι → R
    j : ι
    ⊢ Eq ((TensorProduct.piScalarRightHom R S N ι) (TensorProduct.tmul R x f) j) ( …
  -/
  simp [piScalarRightHom, piScalarRightHomBil]
  /-
    🎉 no goals
  -/


private noncomputable
def piScalarRightInv : (ι → N) →ₗ[S] N ⊗[R] (ι → R) :=
  LinearMap.lsum S (fun _ ↦ N) S <| fun i ↦ {
    toFun := fun n ↦ n ⊗ₜ Pi.single i 1
                             /-
                               R : Type u_1
                               inst✝⁸ : CommSemiring R
                               S : Type u_2
                               inst✝⁷ : CommSemiring S
                               inst✝⁶ : Algebra R S
                               N : Type u_3
                               inst✝⁵ : AddCommMonoid N
                               inst✝⁴ : Module R N
                               inst✝³ : Module S N
                               inst✝² : IsScalarTower R S N
                               ι : Type u_4
                               inst✝¹ : Fintype ι
                               inst✝ : DecidableEq ι
                               i : ι
                               x y : N
                               ⊢ Eq ((fun n => TensorProduct.tmul R n (Pi.single i 1)) (HAdd.hAdd x y)) (HAdd …
                             -/
    map_add' := fun x y ↦ by simp [add_tmul]
                             /-
                               🎉 no goals
                             -/
    map_smul' := fun _ _ ↦ rfl
  }


@[simp]
private lemma piScalarRightInv_single (x : N) (i : ι) :
    piScalarRightInv R S N ι (Pi.single i x) = x ⊗ₜ Pi.single i 1 := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Type u_3
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : N
    i : ι
    ⊢ Eq ((TensorProduct.piScalarRightInv R S N ι) (Pi.single i x)) (TensorProduct …
  -/
  simp [piScalarRightInv, Pi.single_apply, TensorProduct.ite_tmul]
  /-
    🎉 no goals
  -/


/-- For any `R`-module `N` and finite index type `ι`, `N ⊗[R] (ι → R)` is canonically
isomorphic to `ι → N`. -/
noncomputable def piScalarRight : N ⊗[R] (ι → R) ≃ₗ[S] (ι → N) :=
  LinearEquiv.ofLinear
    (piScalarRightHom R S N ι)
    (piScalarRightInv R S N ι)
        /-
          R : Type u_1
          inst✝⁸ : CommSemiring R
          S : Type u_2
          inst✝⁷ : CommSemiring S
          inst✝⁶ : Algebra R S
          N : Type u_3
          inst✝⁵ : AddCommMonoid N
          inst✝⁴ : Module R N
          inst✝³ : Module S N
          inst✝² : IsScalarTower R S N
          ι : Type u_4
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          ⊢ Eq ((TensorProduct.piScalarRightHom R S N ι).comp (TensorProduct.piScalarRig …
        -/
    (by ext i x j; simp [Pi.single_apply])
                   /-
                     🎉 no goals
                   -/
        /-
          R : Type u_1
          inst✝⁸ : CommSemiring R
          S : Type u_2
          inst✝⁷ : CommSemiring S
          inst✝⁶ : Algebra R S
          N : Type u_3
          inst✝⁵ : AddCommMonoid N
          inst✝⁴ : Module R N
          inst✝³ : Module S N
          inst✝² : IsScalarTower R S N
          ι : Type u_4
          inst✝¹ : Fintype ι
          inst✝ : DecidableEq ι
          ⊢ Eq ((TensorProduct.piScalarRightInv R S N ι).comp (TensorProduct.piScalarRig …
        -/
    (by ext x i; simp [Pi.single_apply_smul])
                 /-
                   🎉 no goals
                 -/


@[simp]
lemma piScalarRight_apply (x : N ⊗[R] (ι → R)) :
    piScalarRight R S N ι x = piScalarRightHom R S N ι x := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Type u_3
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : TensorProduct R N (ι → R)
    ⊢ Eq ((TensorProduct.piScalarRight R S N ι) x) ((TensorProduct.piScalarRightHo …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
lemma piScalarRight_symm_single (x : N) (i : ι) :
    (piScalarRight R S N ι).symm (Pi.single i x) = x ⊗ₜ Pi.single i 1 := by
  /-
    R : Type u_1
    inst✝⁸ : CommSemiring R
    S : Type u_2
    inst✝⁷ : CommSemiring S
    inst✝⁶ : Algebra R S
    N : Type u_3
    inst✝⁵ : AddCommMonoid N
    inst✝⁴ : Module R N
    inst✝³ : Module S N
    inst✝² : IsScalarTower R S N
    ι : Type u_4
    inst✝¹ : Fintype ι
    inst✝ : DecidableEq ι
    x : N
    i : ι
    ⊢ Eq ((TensorProduct.piScalarRight R S N ι).symm (Pi.single i x)) (TensorProdu …
  -/
  simp [piScalarRight]
  /-
    🎉 no goals
  -/


