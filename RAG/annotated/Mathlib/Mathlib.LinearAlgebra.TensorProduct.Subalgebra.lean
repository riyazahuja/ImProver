/-- If `A` is a subalgebra of `S/R`, there is the natural `R`-algebra isomorphism between
`i(R) ⊗[R] A` and `A` induced by multiplication in `S`, here `i : R → S` is the structure map.
This generalizes `Algebra.TensorProduct.lid` as `i(R)` is not necessarily isomorphic to `R`.

This is the `Subalgebra` version of `Submodule.lTensorOne` -/
def lTensorBot : (⊥ : Subalgebra R S) ⊗[R] A ≃ₐ[R] A := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring S
    inst✝² : Algebra R S
    inst✝¹ : Semiring T
    inst✝ : Algebra R T
    A : Subalgebra R S
    ⊢ AlgEquiv R (TensorProduct R (Subtype fun x => Membership.mem Bot.bot x) (Sub …
  -/
  refine Algebra.TensorProduct.algEquivOfLinearEquivTensorProduct (toSubmodule A).lTensorOne ?_ ?_
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      ⊢ ∀ (a₁ a₂ : Subtype fun x => Membership.mem Bot.bot x) (b₁ b₂ : Subtype fun x …
    -/
  · rintro x y a b
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      x y : Subtype fun x => Membership.mem Bot.bot x
      a b : Subtype fun x => Membership.mem A x
      ⊢ Eq ((Subalgebra.toSubmodule A).lTensorOne (TensorProduct.tmul R (HMul.hMul x …
    -/
    obtain ⟨x', hx⟩ := Algebra.mem_bot.1 x.2
    /-
      case refine_1.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      x y : Subtype fun x => Membership.mem Bot.bot x
      a b : Subtype fun x => Membership.mem A x
      x' : R
      hx : Eq ((algebraMap R S) x') ↑x
      ⊢ Eq ((Subalgebra.toSubmodule A).lTensorOne (TensorProduct.tmul R (HMul.hMul x …
    -/
    replace hx : algebraMap R _ x' = x := Subtype.val_injective hx
    /-
      case refine_1.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      x y : Subtype fun x => Membership.mem Bot.bot x
      a b : Subtype fun x => Membership.mem A x
      x' : R
      hx : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x') x
      ⊢ Eq ((Subalgebra.toSubmodule A).lTensorOne (TensorProduct.tmul R (HMul.hMul x …
    -/
    obtain ⟨y', hy⟩ := Algebra.mem_bot.1 y.2
    /-
      case refine_1.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      x y : Subtype fun x => Membership.mem Bot.bot x
      a b : Subtype fun x => Membership.mem A x
      x' : R
      hx : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x') x
      y' : R
      hy : Eq ((algebraMap R S) y') ↑y
      ⊢ Eq ((Subalgebra.toSubmodule A).lTensorOne (TensorProduct.tmul R (HMul.hMul x …
    -/
    replace hy : algebraMap R _ y' = y := Subtype.val_injective hy
    /-
      case refine_1.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      x y : Subtype fun x => Membership.mem Bot.bot x
      a b : Subtype fun x => Membership.mem A x
      x' : R
      hx : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x') x
      y' : R
      hy : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) y') y
      ⊢ Eq ((Subalgebra.toSubmodule A).lTensorOne (TensorProduct.tmul R (HMul.hMul x …
    -/
    rw [← hx, ← hy, ← map_mul]
    erw [(toSubmodule A).lTensorOne_tmul x' a,
      (toSubmodule A).lTensorOne_tmul y' b,
      (toSubmodule A).lTensorOne_tmul (x' * y') (a * b)]
    /-
      case refine_1.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      x y : Subtype fun x => Membership.mem Bot.bot x
      a b : Subtype fun x => Membership.mem A x
      x' : R
      hx : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x') x
      y' : R
      hy : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) y') y
      ⊢ Eq (HSMul.hSMul (HMul.hMul x' y') (HMul.hMul a b)) (HMul.hMul (HSMul.hSMul x …
    -/
    rw [Algebra.mul_smul_comm, Algebra.smul_mul_assoc, smul_smul, mul_comm x' y']
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      ⊢ Eq ((Subalgebra.toSubmodule A).lTensorOne (TensorProduct.tmul R 1 1)) 1
    -/
  · exact Submodule.lTensorOne_one_tmul _
    /-
      🎉 no goals
    -/


@[simp]
theorem lTensorBot_tmul (x : R) (a : A) : A.lTensorBot (algebraMap R _ x ⊗ₜ[R] a) = x • a :=
  (toSubmodule A).lTensorOne_tmul x a


@[simp]
theorem lTensorBot_one_tmul (a : A) : A.lTensorBot (1 ⊗ₜ[R] a) = a :=
  (toSubmodule A).lTensorOne_one_tmul a


@[simp]
theorem lTensorBot_symm_apply (a : A) : A.lTensorBot.symm a = 1 ⊗ₜ[R] a := rfl


/-- If `A` is a subalgebra of `S/R`, there is the natural `R`-algebra isomorphism between
`A ⊗[R] i(R)` and `A` induced by multiplication in `S`, here `i : R → S` is the structure map.
This generalizes `Algebra.TensorProduct.rid` as `i(R)` is not necessarily isomorphic to `R`.

This is the `Subalgebra` version of `Submodule.rTensorOne` -/
def rTensorBot : A ⊗[R] (⊥ : Subalgebra R S) ≃ₐ[R] A := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : Semiring S
    inst✝² : Algebra R S
    inst✝¹ : Semiring T
    inst✝ : Algebra R T
    A : Subalgebra R S
    ⊢ AlgEquiv R (TensorProduct R (Subtype fun x => Membership.mem A x) (Subtype f …
  -/
  refine Algebra.TensorProduct.algEquivOfLinearEquivTensorProduct (toSubmodule A).rTensorOne ?_ ?_
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      ⊢ ∀ (a₁ a₂ : Subtype fun x => Membership.mem A x) (b₁ b₂ : Subtype fun x => Me …
    -/
  · rintro a b x y
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      a b : Subtype fun x => Membership.mem A x
      x y : Subtype fun x => Membership.mem Bot.bot x
      ⊢ Eq ((Subalgebra.toSubmodule A).rTensorOne (TensorProduct.tmul R (HMul.hMul a …
    -/
    obtain ⟨x', hx⟩ := Algebra.mem_bot.1 x.2
    /-
      case refine_1.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      a b : Subtype fun x => Membership.mem A x
      x y : Subtype fun x => Membership.mem Bot.bot x
      x' : R
      hx : Eq ((algebraMap R S) x') ↑x
      ⊢ Eq ((Subalgebra.toSubmodule A).rTensorOne (TensorProduct.tmul R (HMul.hMul a …
    -/
    replace hx : algebraMap R _ x' = x := Subtype.val_injective hx
    /-
      case refine_1.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      a b : Subtype fun x => Membership.mem A x
      x y : Subtype fun x => Membership.mem Bot.bot x
      x' : R
      hx : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x') x
      ⊢ Eq ((Subalgebra.toSubmodule A).rTensorOne (TensorProduct.tmul R (HMul.hMul a …
    -/
    obtain ⟨y', hy⟩ := Algebra.mem_bot.1 y.2
    /-
      case refine_1.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      a b : Subtype fun x => Membership.mem A x
      x y : Subtype fun x => Membership.mem Bot.bot x
      x' : R
      hx : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x') x
      y' : R
      hy : Eq ((algebraMap R S) y') ↑y
      ⊢ Eq ((Subalgebra.toSubmodule A).rTensorOne (TensorProduct.tmul R (HMul.hMul a …
    -/
    replace hy : algebraMap R _ y' = y := Subtype.val_injective hy
    /-
      case refine_1.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      a b : Subtype fun x => Membership.mem A x
      x y : Subtype fun x => Membership.mem Bot.bot x
      x' : R
      hx : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x') x
      y' : R
      hy : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) y') y
      ⊢ Eq ((Subalgebra.toSubmodule A).rTensorOne (TensorProduct.tmul R (HMul.hMul a …
    -/
    rw [← hx, ← hy, ← map_mul]
    erw [(toSubmodule A).rTensorOne_tmul x' a,
      (toSubmodule A).rTensorOne_tmul y' b,
      (toSubmodule A).rTensorOne_tmul (x' * y') (a * b)]
    /-
      case refine_1.intro.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      a b : Subtype fun x => Membership.mem A x
      x y : Subtype fun x => Membership.mem Bot.bot x
      x' : R
      hx : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x') x
      y' : R
      hy : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) y') y
      ⊢ Eq (HSMul.hSMul (HMul.hMul x' y') (HMul.hMul a b)) (HMul.hMul (HSMul.hSMul x …
    -/
    rw [Algebra.mul_smul_comm, Algebra.smul_mul_assoc, smul_smul, mul_comm x' y']
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      A : Subalgebra R S
      ⊢ Eq ((Subalgebra.toSubmodule A).rTensorOne (TensorProduct.tmul R 1 1)) 1
    -/
  · exact Submodule.rTensorOne_tmul_one _
    /-
      🎉 no goals
    -/


@[simp]
theorem rTensorBot_tmul (x : R) (a : A) : A.rTensorBot (a ⊗ₜ[R] algebraMap R _ x) = x • a :=
  (toSubmodule A).rTensorOne_tmul x a


@[simp]
theorem rTensorBot_tmul_one (a : A) : A.rTensorBot (a ⊗ₜ[R] 1) = a :=
  (toSubmodule A).rTensorOne_tmul_one a


@[simp]
theorem rTensorBot_symm_apply (a : A) : A.rTensorBot.symm a = a ⊗ₜ[R] 1 := rfl


@[simp]
theorem comm_trans_lTensorBot :
    (Algebra.TensorProduct.comm R _ _).trans A.lTensorBot = A.rTensorBot :=
  AlgEquiv.toLinearEquiv_injective (toSubmodule A).comm_trans_lTensorOne


@[simp]
theorem comm_trans_rTensorBot :
    (Algebra.TensorProduct.comm R _ _).trans A.rTensorBot = A.lTensorBot :=
  AlgEquiv.toLinearEquiv_injective (toSubmodule A).comm_trans_rTensorOne


/-- Given `R`-algebras `S,T`, there is a natural `R`-linear isomorphism from `S ⊗[R] T` to
`S' ⊗[R] T'` where `S',T'` are the images of `S,T` in `S ⊗[R] T` respectively.
This is promoted to an `R`-algebra isomorphism `Algebra.TensorProduct.algEquivIncludeRange`. -/
def linearEquivIncludeRange :
    S ⊗[R] T ≃ₗ[R] (includeLeft : S →ₐ[R] S ⊗[R] T).range ⊗[R]
      (includeRight : T →ₐ[R] S ⊗[R] T).range := .ofLinear
  (_root_.TensorProduct.map
    includeLeft.toLinearMap.rangeRestrict includeRight.toLinearMap.rangeRestrict)
  ((LinearMap.range includeLeft).mulMap (LinearMap.range includeRight))
  (_root_.TensorProduct.ext' <| by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      ⊢ ∀ (x : Subtype fun x => Membership.mem Algebra.TensorProduct.includeLeft.ran …
    -/
    rintro ⟨x', x, rfl : x ⊗ₜ 1 = x'⟩ ⟨y', y, rfl : 1 ⊗ₜ y = y'⟩
    /-
      case mk.intro.mk.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      x : S
      y : T
      ⊢ Eq (((_root_.TensorProduct.map Algebra.TensorProduct.includeLeft.toLinearMap …
    -/
    rw [LinearMap.comp_apply, LinearMap.id_apply]
    /-
      case mk.intro.mk.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      x : S
      y : T
      ⊢ Eq ((_root_.TensorProduct.map Algebra.TensorProduct.includeLeft.toLinearMap. …
    -/
    erw [Submodule.mulMap_tmul]
    /-
      case mk.intro.mk.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      x : S
      y : T
      ⊢ Eq ((_root_.TensorProduct.map Algebra.TensorProduct.includeLeft.toLinearMap. …
    -/
    rw [tmul_mul_tmul, mul_one, one_mul, _root_.TensorProduct.map_tmul]
    /-
      case mk.intro.mk.intro
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      x : S
      y : T
      ⊢ Eq (TensorProduct.tmul R (Algebra.TensorProduct.includeLeft.toLinearMap.rang …
    -/
    rfl)
    /-
      🎉 no goals
    -/
  (_root_.TensorProduct.ext' fun x y ↦ by
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      x : S
      y : T
      ⊢ Eq ((((LinearMap.range Algebra.TensorProduct.includeLeft).mulMap (LinearMap. …
    -/
    rw [LinearMap.comp_apply, LinearMap.id_apply, _root_.TensorProduct.map_tmul]
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      x : S
      y : T
      ⊢ Eq (((LinearMap.range Algebra.TensorProduct.includeLeft).mulMap (LinearMap.r …
    -/
    erw [Submodule.mulMap_tmul]
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      x : S
      y : T
      ⊢ Eq (HMul.hMul ↑(Algebra.TensorProduct.includeLeft.toLinearMap.rangeRestrict  …
    -/
    change (x ⊗ₜ 1) * (1 ⊗ₜ y) = _
    /-
      R : Type u_1
      S : Type u_2
      T : Type u_3
      inst✝⁴ : CommSemiring R
      inst✝³ : Semiring S
      inst✝² : Algebra R S
      inst✝¹ : Semiring T
      inst✝ : Algebra R T
      x : S
      y : T
      ⊢ Eq (HMul.hMul (TensorProduct.tmul R x 1) (TensorProduct.tmul R 1 y)) (Tensor …
    -/
    rw [tmul_mul_tmul, mul_one, one_mul])
    /-
      🎉 no goals
    -/


theorem linearEquivIncludeRange_toLinearMap :
    (linearEquivIncludeRange R S T).toLinearMap =
      _root_.TensorProduct.map includeLeft.toLinearMap.rangeRestrict
        includeRight.toLinearMap.rangeRestrict := rfl


theorem linearEquivIncludeRange_symm_toLinearMap :
    (linearEquivIncludeRange R S T).symm.toLinearMap =
      (LinearMap.range includeLeft).mulMap (LinearMap.range includeRight) := rfl


@[simp]
theorem linearEquivIncludeRange_tmul (x y) :
    linearEquivIncludeRange R S T (x ⊗ₜ[R] y) =
      ((includeLeft : S →ₐ[R] S ⊗[R] T).rangeRestrict x) ⊗ₜ[R]
        ((includeRight : T →ₐ[R] S ⊗[R] T).rangeRestrict y) := rfl


@[simp]
theorem linearEquivIncludeRange_symm_tmul (x y) :
    (linearEquivIncludeRange R S T).symm (x ⊗ₜ[R] y) = x.1 * y.1 := rfl


/-- Given `R`-algebras `S,T`, there is a natural `R`-algebra isomorphism from `S ⊗[R] T` to
`S' ⊗[R] T'` where `S',T'` are the images of `S,T` in `S ⊗[R] T` respectively. -/
def algEquivIncludeRange :
    S ⊗[R] T ≃ₐ[R] (includeLeft : S →ₐ[R] S ⊗[R] T).range ⊗[R]
      (includeRight : T →ₐ[R] S ⊗[R] T).range :=
                                                                         /-
                                                                           R : Type u_1
                                                                           S : Type u_2
                                                                           T : Type u_3
                                                                           inst✝⁴ : CommSemiring R
                                                                           inst✝³ : Semiring S
                                                                           inst✝² : Algebra R S
                                                                           inst✝¹ : Semiring T
                                                                           inst✝ : Algebra R T
                                                                           ⊢ ∀ (a₁ a₂ : S) (b₁ b₂ : T), Eq ((Algebra.TensorProduct.linearEquivIncludeRang …
                                                                         -/
  algEquivOfLinearEquivTensorProduct (linearEquivIncludeRange R S T) (by simp) rfl
                                                                         /-
                                                                           🎉 no goals
                                                                         -/


theorem algEquivIncludeRange_toAlgHom :
    (algEquivIncludeRange R S T).toAlgHom =
      map includeLeft.rangeRestrict includeRight.rangeRestrict := rfl


@[simp]
theorem algEquivIncludeRange_tmul (x y) :
    algEquivIncludeRange R S T (x ⊗ₜ[R] y) =
      ((includeLeft : S →ₐ[R] S ⊗[R] T).rangeRestrict x) ⊗ₜ[R]
        ((includeRight : T →ₐ[R] S ⊗[R] T).rangeRestrict y) := rfl


@[simp]
theorem algEquivIncludeRange_symm_tmul (x y) :
    (algEquivIncludeRange R S T).symm (x ⊗ₜ[R] y) = x.1 * y.1 := rfl


/-- If `A` and `B` are subalgebras in a commutative algebra `S` over `R`,
there is the natural `R`-algebra homomorphism
`A ⊗[R] B →ₐ[R] S` induced by multiplication in `S`. -/
def Subalgebra.mulMap : A ⊗[R] B →ₐ[R] S := Algebra.TensorProduct.productMap A.val B.val


variable (R S T) in
theorem Algebra.TensorProduct.algEquivIncludeRange_symm_toAlgHom :
    (algEquivIncludeRange R S T).symm.toAlgHom =
      (includeLeft : S →ₐ[R] S ⊗[R] T).range.mulMap includeRight.range := rfl


@[simp]
theorem mulMap_tmul (a : A) (b : B) : mulMap A B (a ⊗ₜ[R] b) = a.1 * b.1 := rfl


theorem mulMap_map_comp_eq (f : S →ₐ[R] T) :
    (mulMap (A.map f) (B.map f)).comp
      (Algebra.TensorProduct.map (f.subalgebraMap A) (f.subalgebraMap B))
        = f.comp (mulMap A B) := by
  /-
    R : Type u_1
    S : Type u_2
    T : Type u_3
    inst✝⁴ : CommSemiring R
    inst✝³ : CommSemiring S
    inst✝² : Algebra R S
    inst✝¹ : CommSemiring T
    inst✝ : Algebra R T
    A B : Subalgebra R S
    f : AlgHom R S T
    ⊢ Eq (((Subalgebra.map f A).mulMap (Subalgebra.map f B)).comp (Algebra.TensorP …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


theorem mulMap_toLinearMap : (A.mulMap B).toLinearMap = (toSubmodule A).mulMap (toSubmodule B) :=
  rfl


theorem mulMap_comm : mulMap B A = (mulMap A B).comp (Algebra.TensorProduct.comm R B A) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ⊢ Eq (B.mulMap A) ((A.mulMap B).comp ↑(Algebra.TensorProduct.comm R (Subtype f …
  -/
          /-
            🎉 no goals
          -/
  ext <;> simp
          /-
            🎉 no goals
          -/


theorem mulMap_range : (A.mulMap B).range = A ⊔ B := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : CommSemiring S
    inst✝ : Algebra R S
    A B : Subalgebra R S
    ⊢ Eq (A.mulMap B).range (Max.max A B)
  -/
  simp_rw [mulMap, Algebra.TensorProduct.productMap_range, Subalgebra.range_val]
  /-
    🎉 no goals
  -/


theorem mulMap_bot_left_eq : mulMap ⊥ A = A.val.comp A.lTensorBot.toAlgHom :=
  AlgHom.toLinearMap_injective (toSubmodule A).mulMap_one_left_eq


theorem mulMap_bot_right_eq : mulMap A ⊥ = A.val.comp A.rTensorBot.toAlgHom :=
  AlgHom.toLinearMap_injective (toSubmodule A).mulMap_one_right_eq


/-- If `A` and `B` are subalgebras in a commutative algebra `S` over `R`,
there is the natural `R`-algebra homomorphism
`A ⊗[R] B →ₐ[R] A ⊔ B` induced by multiplication in `S`,
which is surjective (`Subalgebra.mulMap'_surjective`). -/
def mulMap' : A ⊗[R] B →ₐ[R] ↥(A ⊔ B) :=
  (equivOfEq _ _ (mulMap_range A B)).toAlgHom.comp (mulMap A B).rangeRestrict


variable {A B} in
@[simp]
theorem val_mulMap'_tmul (a : A) (b : B) : (mulMap' A B (a ⊗ₜ[R] b) : S) = a.1 * b.1 := rfl


theorem mulMap'_surjective : Function.Surjective (mulMap' A B) := by
  simp_rw [mulMap', AlgEquiv.toAlgHom_eq_coe, AlgHom.coe_comp, AlgHom.coe_coe,
    EquivLike.comp_surjective, AlgHom.rangeRestrict_surjective]


