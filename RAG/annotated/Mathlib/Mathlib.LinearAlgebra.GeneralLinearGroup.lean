/-- The group of invertible linear maps from `M` to itself -/
abbrev GeneralLinearGroup :=
  (M →ₗ[R] M)ˣ


/-- An invertible linear map `f` determines an equivalence from `M` to itself. -/
def toLinearEquiv (f : GeneralLinearGroup R M) : M ≃ₗ[R] M :=
  { f.val with
    invFun := f.inv.toFun
                                                      /-
                                                        R : Type u_1
                                                        M : Type u_2
                                                        inst✝² : Semiring R
                                                        inst✝¹ : AddCommMonoid M
                                                        inst✝ : Module R M
                                                        f : LinearMap.GeneralLinearGroup R M
                                                        m : M
                                                        ⊢ Eq ((HMul.hMul f.inv ↑f) m) m
                                                      -/
    left_inv := fun m ↦ show (f.inv * f.val) m = m by rw [f.inv_val]; simp
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
                                                       /-
                                                         R : Type u_1
                                                         M : Type u_2
                                                         inst✝² : Semiring R
                                                         inst✝¹ : AddCommMonoid M
                                                         inst✝ : Module R M
                                                         f : LinearMap.GeneralLinearGroup R M
                                                         m : M
                                                         ⊢ Eq ((HMul.hMul (↑f) f.inv) m) m
                                                       -/
    right_inv := fun m ↦ show (f.val * f.inv) m = m by rw [f.val_inv]; simp }
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


/-- An equivalence from `M` to itself determines an invertible linear map. -/
def ofLinearEquiv (f : M ≃ₗ[R] M) : GeneralLinearGroup R M where
  val := f
  inv := (f.symm : M →ₗ[R] M)
  val_inv := LinearMap.ext fun _ ↦ f.apply_symm_apply _
  inv_val := LinearMap.ext fun _ ↦ f.symm_apply_apply _


/-- The general linear group on `R` and `M` is multiplicatively equivalent to the type of linear
equivalences between `M` and itself. -/
def generalLinearEquiv : GeneralLinearGroup R M ≃* M ≃ₗ[R] M where
  toFun := toLinearEquiv
  invFun := ofLinearEquiv
                   /-
                     R : Type u_1
                     M : Type u_2
                     inst✝² : Semiring R
                     inst✝¹ : AddCommMonoid M
                     inst✝ : Module R M
                     f : LinearMap.GeneralLinearGroup R M
                     ⊢ Eq (LinearMap.GeneralLinearGroup.ofLinearEquiv f.toLinearEquiv) f
                   -/
  left_inv f := by ext; rfl
                        /-
                          🎉 no goals
                        -/
                    /-
                      R : Type u_1
                      M : Type u_2
                      inst✝² : Semiring R
                      inst✝¹ : AddCommMonoid M
                      inst✝ : Module R M
                      f : LinearEquiv (RingHom.id R) M M
                      ⊢ Eq (LinearMap.GeneralLinearGroup.ofLinearEquiv f).toLinearEquiv f
                    -/
  right_inv f := by ext; rfl
                         /-
                           🎉 no goals
                         -/
                     /-
                       R : Type u_1
                       M : Type u_2
                       inst✝² : Semiring R
                       inst✝¹ : AddCommMonoid M
                       inst✝ : Module R M
                       x y : LinearMap.GeneralLinearGroup R M
                       ⊢ Eq ({ toFun := LinearMap.GeneralLinearGroup.toLinearEquiv, invFun := LinearM …
                     -/
  map_mul' x y := by ext; rfl
                          /-
                            🎉 no goals
                          -/


@[simp]
theorem generalLinearEquiv_to_linearMap (f : GeneralLinearGroup R M) :
                                                     /-
                                                       R : Type u_1
                                                       M : Type u_2
                                                       inst✝² : Semiring R
                                                       inst✝¹ : AddCommMonoid M
                                                       inst✝ : Module R M
                                                       f : LinearMap.GeneralLinearGroup R M
                                                       ⊢ Eq ↑((LinearMap.GeneralLinearGroup.generalLinearEquiv R M) f) ↑f
                                                     -/
    (generalLinearEquiv R M f : M →ₗ[R] M) = f := by ext; rfl
                                                          /-
                                                            🎉 no goals
                                                          -/


@[simp]
theorem coeFn_generalLinearEquiv (f : GeneralLinearGroup R M) :
    (generalLinearEquiv R M f) = (f : M → M) := rfl


