/-- If `M` and `N` are submodules in an algebra `S` over `R`, there is the natural `R`-linear map
`M ⊗[R] N →ₗ[R] S` induced by multiplication in `S`. -/
def mulMap : M ⊗[R] N →ₗ[R] S := TensorProduct.lift ((LinearMap.mul R S).domRestrict₁₂ M N)


@[simp]
theorem mulMap_tmul (m : M) (n : N) : mulMap M N (m ⊗ₜ[R] n) = m.1 * n.1 := rfl


theorem mulMap_map_comp_eq {T : Type w} [Semiring T] [Algebra R T]
    {F : Type*} [FunLike F S T] [AlgHomClass F R S T] (f : F) :
    mulMap (M.map f) (N.map f) ∘ₗ
      TensorProduct.map ((f : S →ₗ[R] T).submoduleMap M) ((f : S →ₗ[R] T).submoduleMap N)
        = f ∘ₗ mulMap M N := by
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
    ⊢ Eq (((Submodule.map f M).mulMap (Submodule.map f N)).comp (TensorProduct.map …
  -/
  ext
  simp only [TensorProduct.AlgebraTensorModule.curry_apply, LinearMap.restrictScalars_comp,
    TensorProduct.curry_apply, LinearMap.coe_comp, LinearMap.coe_restrictScalars,
    Function.comp_apply, TensorProduct.map_tmul, mulMap_tmul, LinearMap.coe_coe, map_mul]
  /-
    case a.h.h
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
    x✝¹ : Subtype fun x => Membership.mem M x
    x✝ : Subtype fun x => Membership.mem N x
    ⊢ Eq (HMul.hMul ↑(((↑f).submoduleMap M) x✝¹) ↑(((↑f).submoduleMap N) x✝)) (HMu …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem mulMap_op :
    mulMap (equivOpposite.symm (MulOpposite.op M)) (equivOpposite.symm (MulOpposite.op N)) =
    (MulOpposite.opLinearEquiv R).toLinearMap ∘ₗ mulMap N M ∘ₗ
    (TensorProduct.congr
      (LinearEquiv.ofSubmodule' (MulOpposite.opLinearEquiv R).symm M)
      (LinearEquiv.ofSubmodule' (MulOpposite.opLinearEquiv R).symm N) ≪≫ₗ
    TensorProduct.comm R M N).toLinearMap :=
  TensorProduct.ext' fun _ _ ↦ rfl


theorem mulMap_comm_of_commute (hc : ∀ (m : M) (n : N), Commute m.1 n.1) :
    mulMap N M = mulMap M N ∘ₗ TensorProduct.comm R N M := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    hc : ∀ (m : Subtype fun x => Membership.mem M x) (n : Subtype fun x => Members …
    ⊢ Eq (N.mulMap M) ((M.mulMap N).comp ↑(TensorProduct.comm R (Subtype fun x =>  …
  -/
  refine TensorProduct.ext' fun n m ↦ ?_
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    hc : ∀ (m : Subtype fun x => Membership.mem M x) (n : Subtype fun x => Members …
    n : Subtype fun x => Membership.mem N x
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq ((N.mulMap M) (TensorProduct.tmul R n m)) (((M.mulMap N).comp ↑(TensorPro …
  -/
  simp_rw [LinearMap.comp_apply, LinearEquiv.coe_coe, TensorProduct.comm_tmul, mulMap_tmul]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    hc : ∀ (m : Subtype fun x => Membership.mem M x) (n : Subtype fun x => Members …
    n : Subtype fun x => Membership.mem N x
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq (HMul.hMul ↑n ↑m) (HMul.hMul ↑m ↑n)
  -/
  exact (hc m n).symm
  /-
    🎉 no goals
  -/


variable {M} in
theorem mulMap_comp_rTensor {M' : Submodule R S} (hM : M' ≤ M) :
    mulMap M N ∘ₗ (inclusion hM).rTensor N = mulMap M' N :=
  TensorProduct.ext' fun _ _ ↦ rfl


variable {N} in
theorem mulMap_comp_lTensor {N' : Submodule R S} (hN : N' ≤ N) :
    mulMap M N ∘ₗ (inclusion hN).lTensor M = mulMap M N' :=
  TensorProduct.ext' fun _ _ ↦ rfl


variable {M N} in
theorem mulMap_comp_map_inclusion {M' N' : Submodule R S} (hM : M' ≤ M) (hN : N' ≤ N) :
    mulMap M N ∘ₗ TensorProduct.map (inclusion hM) (inclusion hN) = mulMap M' N' :=
  TensorProduct.ext' fun _ _ ↦ rfl


theorem mulMap_eq_mul'_comp_mapIncl : mulMap M N = .mul' R S ∘ₗ TensorProduct.mapIncl M N :=
  TensorProduct.ext' fun _ _ ↦ rfl


theorem mulMap_range : LinearMap.range (mulMap M N) = M * N := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    ⊢ Eq (LinearMap.range (M.mulMap N)) (HMul.hMul M N)
  -/
  refine le_antisymm ?_ (mul_le.2 fun m hm n hn ↦ ⟨⟨m, hm⟩ ⊗ₜ[R] ⟨n, hn⟩, rfl⟩)
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    ⊢ LE.le (LinearMap.range (M.mulMap N)) (HMul.hMul M N)
  -/
  rintro _ ⟨x, rfl⟩
  induction x with
  | zero => rw [_root_.map_zero]; exact zero_mem _
  | tmul a b => exact mul_mem_mul a.2 b.2
  | add a b ha hb => rw [_root_.map_add]; exact add_mem ha hb


/-- If `M` and `N` are submodules in an algebra `S` over `R`, there is the natural `R`-linear map
`M ⊗[R] N →ₗ[R] M * N` induced by multiplication in `S`,
which is surjective (`Submodule.mulMap'_surjective`). -/
def mulMap' : M ⊗[R] N →ₗ[R] ↥(M * N) :=
  (LinearEquiv.ofEq _ _ (mulMap_range M N)).toLinearMap ∘ₗ (mulMap M N).rangeRestrict


variable {M N} in
@[simp]
theorem val_mulMap'_tmul (m : M) (n : N) : (mulMap' M N (m ⊗ₜ[R] n) : S) = m.1 * n.1 := rfl


theorem mulMap'_surjective : Function.Surjective (mulMap' M N) := by
  simp_rw [mulMap', LinearMap.coe_comp, LinearEquiv.coe_coe, EquivLike.comp_surjective,
    LinearMap.surjective_rangeRestrict]


/-- If `N` is a submodule in an algebra `S` over `R`, there is the natural `R`-linear map
`i(R) ⊗[R] N →ₗ[R] N` induced by multiplication in `S`, here `i : R → S` is the structure map.
This is promoted to an isomorphism of `R`-modules as `Submodule.lTensorOne`. Use that instead. -/
def lTensorOne' : (⊥ : Subalgebra R S) ⊗[R] N →ₗ[R] N :=
  show Subalgebra.toSubmodule ⊥ ⊗[R] N →ₗ[R] N from
                              /-
                                R : Type u
                                S : Type v
                                inst✝² : CommSemiring R
                                inst✝¹ : Semiring S
                                inst✝ : Algebra R S
                                M N : Submodule R S
                                ⊢ Eq (LinearMap.range ((Subalgebra.toSubmodule Bot.bot).mulMap N)) N
                              -/
    (LinearEquiv.ofEq _ _ (by rw [Algebra.toSubmodule_bot, mulMap_range, one_mul])).toLinearMap ∘ₗ
                              /-
                                🎉 no goals
                              -/
      (mulMap _ N).rangeRestrict


variable {N} in
@[simp]
theorem lTensorOne'_tmul (y : R) (n : N) :
    N.lTensorOne' (algebraMap R _ y ⊗ₜ[R] n) = y • n := Subtype.val_injective <| by
  simp_rw [lTensorOne', LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply,
    LinearEquiv.coe_ofEq_apply, LinearMap.codRestrict_apply, SetLike.val_smul, Algebra.smul_def]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    N : Submodule R S
    y : R
    n : Subtype fun x => Membership.mem N x
    ⊢ Eq (((Subalgebra.toSubmodule Bot.bot).mulMap N) (TensorProduct.tmul R ((alge …
  -/
  exact mulMap_tmul _ N _ _
  /-
    🎉 no goals
  -/


variable {N} in
@[simp]
theorem lTensorOne'_one_tmul (n : N) : N.lTensorOne' (1 ⊗ₜ[R] n) = n := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    N : Submodule R S
    n : Subtype fun x => Membership.mem N x
    ⊢ Eq (N.lTensorOne' (TensorProduct.tmul R 1 n)) n
  -/
  simpa using lTensorOne'_tmul 1 n
  /-
    🎉 no goals
  -/


/-- If `N` is a submodule in an algebra `S` over `R`,
there is the natural isomorphism of `R`-modules between
`i(R) ⊗[R] N` and `N` induced by multiplication in `S`, here `i : R → S` is the structure map.
This generalizes `TensorProduct.lid` as `i(R)` is not necessarily isomorphic to `R`. -/
def lTensorOne : (⊥ : Subalgebra R S) ⊗[R] N ≃ₗ[R] N :=
  LinearEquiv.ofLinear N.lTensorOne' (TensorProduct.mk R (⊥ : Subalgebra R S) N 1)
        /-
          R : Type u
          S : Type v
          inst✝² : CommSemiring R
          inst✝¹ : Semiring S
          inst✝ : Algebra R S
          M N : Submodule R S
          ⊢ Eq (N.lTensorOne'.comp ((TensorProduct.mk R (Subtype fun x => Membership.mem …
        -/
    (by ext; simp) <| TensorProduct.ext' fun r n ↦ by
             /-
               🎉 no goals
             -/
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    r : Subtype fun x => Membership.mem Bot.bot x
    n : Subtype fun x => Membership.mem N x
    ⊢ Eq ((((TensorProduct.mk R (Subtype fun x => Membership.mem Bot.bot x) (Subty …
  -/
  change 1 ⊗ₜ[R] lTensorOne' N _ = r ⊗ₜ[R] n
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    r : Subtype fun x => Membership.mem Bot.bot x
    n : Subtype fun x => Membership.mem N x
    ⊢ Eq (TensorProduct.tmul R 1 (N.lTensorOne' (TensorProduct.tmul R r n))) (Tens …
  -/
  obtain ⟨x, h⟩ := Algebra.mem_bot.1 r.2
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    r : Subtype fun x => Membership.mem Bot.bot x
    n : Subtype fun x => Membership.mem N x
    x : R
    h : Eq ((algebraMap R S) x) ↑r
    ⊢ Eq (TensorProduct.tmul R 1 (N.lTensorOne' (TensorProduct.tmul R r n))) (Tens …
  -/
  replace h : algebraMap R _ x = r := Subtype.val_injective h
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    r : Subtype fun x => Membership.mem Bot.bot x
    n : Subtype fun x => Membership.mem N x
    x : R
    h : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x) r
    ⊢ Eq (TensorProduct.tmul R 1 (N.lTensorOne' (TensorProduct.tmul R r n))) (Tens …
  -/
  rw [← h, lTensorOne'_tmul, ← TensorProduct.smul_tmul, Algebra.smul_def, mul_one]
  /-
    🎉 no goals
  -/


variable {N} in
@[simp]
theorem lTensorOne_tmul (y : R) (n : N) : N.lTensorOne (algebraMap R _ y ⊗ₜ[R] n) = y • n :=
  N.lTensorOne'_tmul y n


variable {N} in
@[simp]
theorem lTensorOne_one_tmul (n : N) : N.lTensorOne (1 ⊗ₜ[R] n) = n :=
  N.lTensorOne'_one_tmul n


variable {N} in
@[simp]
theorem lTensorOne_symm_apply (n : N) : N.lTensorOne.symm n = 1 ⊗ₜ[R] n := rfl


theorem mulMap_one_left_eq :
    mulMap (Subalgebra.toSubmodule ⊥) N = N.subtype ∘ₗ N.lTensorOne.toLinearMap :=
  TensorProduct.ext' fun _ _ ↦ rfl


/-- If `M` is a submodule in an algebra `S` over `R`, there is the natural `R`-linear map
`M ⊗[R] i(R) →ₗ[R] M` induced by multiplication in `S`, here `i : R → S` is the structure map.
This is promoted to an isomorphism of `R`-modules as `Submodule.rTensorOne`. Use that instead. -/
def rTensorOne' : M ⊗[R] (⊥ : Subalgebra R S) →ₗ[R] M :=
  show M ⊗[R] Subalgebra.toSubmodule ⊥ →ₗ[R] M from
                              /-
                                R : Type u
                                S : Type v
                                inst✝² : CommSemiring R
                                inst✝¹ : Semiring S
                                inst✝ : Algebra R S
                                M N : Submodule R S
                                ⊢ Eq (LinearMap.range (M.mulMap (Subalgebra.toSubmodule Bot.bot))) M
                              -/
    (LinearEquiv.ofEq _ _ (by rw [Algebra.toSubmodule_bot, mulMap_range, mul_one])).toLinearMap ∘ₗ
                              /-
                                🎉 no goals
                              -/
      (mulMap M _).rangeRestrict


variable {M} in
@[simp]
theorem rTensorOne'_tmul (y : R) (m : M) :
    M.rTensorOne' (m ⊗ₜ[R] algebraMap R _ y) = y • m := Subtype.val_injective <| by
  simp_rw [rTensorOne', LinearMap.coe_comp, LinearEquiv.coe_coe, Function.comp_apply,
    LinearEquiv.coe_ofEq_apply, LinearMap.codRestrict_apply, SetLike.val_smul]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    y : R
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq ((M.mulMap (Subalgebra.toSubmodule Bot.bot)) (TensorProduct.tmul R m ((al …
  -/
  rw [Algebra.smul_def, Algebra.commutes]
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    y : R
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq ((M.mulMap (Subalgebra.toSubmodule Bot.bot)) (TensorProduct.tmul R m ((al …
  -/
  exact mulMap_tmul M _ _ _
  /-
    🎉 no goals
  -/


variable {M} in
@[simp]
theorem rTensorOne'_tmul_one (m : M) : M.rTensorOne' (m ⊗ₜ[R] 1) = m := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq (M.rTensorOne' (TensorProduct.tmul R m 1)) m
  -/
  simpa using rTensorOne'_tmul 1 m
  /-
    🎉 no goals
  -/


/-- If `M` is a submodule in an algebra `S` over `R`,
there is the natural isomorphism of `R`-modules between
`M ⊗[R] i(R)` and `M` induced by multiplication in `S`, here `i : R → S` is the structure map.
This generalizes `TensorProduct.rid` as `i(R)` is not necessarily isomorphic to `R`. -/
def rTensorOne : M ⊗[R] (⊥ : Subalgebra R S) ≃ₗ[R] M :=
  LinearEquiv.ofLinear M.rTensorOne' ((TensorProduct.comm R _ _).toLinearMap ∘ₗ
                                                     /-
                                                       R : Type u
                                                       S : Type v
                                                       inst✝² : CommSemiring R
                                                       inst✝¹ : Semiring S
                                                       inst✝ : Algebra R S
                                                       M N : Submodule R S
                                                       ⊢ Eq (M.rTensorOne'.comp ((↑(TensorProduct.comm R (Subtype fun x => Membership …
                                                     -/
    TensorProduct.mk R (⊥ : Subalgebra R S) M 1) (by ext; simp) <| TensorProduct.ext' fun n r ↦ by
                                                          /-
                                                            🎉 no goals
                                                          -/
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    n : Subtype fun x => Membership.mem M x
    r : Subtype fun x => Membership.mem Bot.bot x
    ⊢ Eq ((((↑(TensorProduct.comm R (Subtype fun x => Membership.mem Bot.bot x) (S …
  -/
  change rTensorOne' M _ ⊗ₜ[R] 1 = n ⊗ₜ[R] r
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    n : Subtype fun x => Membership.mem M x
    r : Subtype fun x => Membership.mem Bot.bot x
    ⊢ Eq (TensorProduct.tmul R (M.rTensorOne' (TensorProduct.tmul R n r)) 1) (Tens …
  -/
  obtain ⟨x, h⟩ := Algebra.mem_bot.1 r.2
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    n : Subtype fun x => Membership.mem M x
    r : Subtype fun x => Membership.mem Bot.bot x
    x : R
    h : Eq ((algebraMap R S) x) ↑r
    ⊢ Eq (TensorProduct.tmul R (M.rTensorOne' (TensorProduct.tmul R n r)) 1) (Tens …
  -/
  replace h : algebraMap R _ x = r := Subtype.val_injective h
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M N : Submodule R S
    n : Subtype fun x => Membership.mem M x
    r : Subtype fun x => Membership.mem Bot.bot x
    x : R
    h : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x) r
    ⊢ Eq (TensorProduct.tmul R (M.rTensorOne' (TensorProduct.tmul R n r)) 1) (Tens …
  -/
  rw [← h, rTensorOne'_tmul, TensorProduct.smul_tmul, Algebra.smul_def, mul_one]
  /-
    🎉 no goals
  -/


variable {M} in
@[simp]
theorem rTensorOne_tmul (y : R) (m : M) : M.rTensorOne (m ⊗ₜ[R] algebraMap R _ y) = y • m :=
  M.rTensorOne'_tmul y m


variable {M} in
@[simp]
theorem rTensorOne_tmul_one (m : M) : M.rTensorOne (m ⊗ₜ[R] 1) = m :=
  M.rTensorOne'_tmul_one m


variable {M} in
@[simp]
theorem rTensorOne_symm_apply (m : M) : M.rTensorOne.symm m = m ⊗ₜ[R] 1 := rfl


theorem mulMap_one_right_eq :
    mulMap M (Subalgebra.toSubmodule ⊥) = M.subtype ∘ₗ M.rTensorOne.toLinearMap :=
  TensorProduct.ext' fun _ _ ↦ rfl


@[simp]
theorem comm_trans_lTensorOne :
    (TensorProduct.comm R _ _).trans M.lTensorOne = M.rTensorOne := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    ⊢ Eq ((TensorProduct.comm R (Subtype fun x => Membership.mem M x) (Subtype fun …
  -/
  refine LinearEquiv.toLinearMap_injective <| TensorProduct.ext' fun m r ↦ ?_
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    m : Subtype fun x => Membership.mem M x
    r : Subtype fun x => Membership.mem Bot.bot x
    ⊢ Eq (↑((TensorProduct.comm R (Subtype fun x => Membership.mem M x) (Subtype f …
  -/
  obtain ⟨x, h⟩ := Algebra.mem_bot.1 r.2
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    m : Subtype fun x => Membership.mem M x
    r : Subtype fun x => Membership.mem Bot.bot x
    x : R
    h : Eq ((algebraMap R S) x) ↑r
    ⊢ Eq (↑((TensorProduct.comm R (Subtype fun x => Membership.mem M x) (Subtype f …
  -/
  replace h : algebraMap R _ x = r := Subtype.val_injective h
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    m : Subtype fun x => Membership.mem M x
    r : Subtype fun x => Membership.mem Bot.bot x
    x : R
    h : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x) r
    ⊢ Eq (↑((TensorProduct.comm R (Subtype fun x => Membership.mem M x) (Subtype f …
  -/
  rw [← h]; simp
            /-
              🎉 no goals
            -/


@[simp]
theorem comm_trans_rTensorOne :
    (TensorProduct.comm R _ _).trans M.rTensorOne = M.lTensorOne := by
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    ⊢ Eq ((TensorProduct.comm R (Subtype fun x => Membership.mem Bot.bot x) (Subty …
  -/
  refine LinearEquiv.toLinearMap_injective <| TensorProduct.ext' fun r m ↦ ?_
  /-
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    r : Subtype fun x => Membership.mem Bot.bot x
    m : Subtype fun x => Membership.mem M x
    ⊢ Eq (↑((TensorProduct.comm R (Subtype fun x => Membership.mem Bot.bot x) (Sub …
  -/
  obtain ⟨x, h⟩ := Algebra.mem_bot.1 r.2
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    r : Subtype fun x => Membership.mem Bot.bot x
    m : Subtype fun x => Membership.mem M x
    x : R
    h : Eq ((algebraMap R S) x) ↑r
    ⊢ Eq (↑((TensorProduct.comm R (Subtype fun x => Membership.mem Bot.bot x) (Sub …
  -/
  replace h : algebraMap R _ x = r := Subtype.val_injective h
  /-
    case intro
    R : Type u
    S : Type v
    inst✝² : CommSemiring R
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    M : Submodule R S
    r : Subtype fun x => Membership.mem Bot.bot x
    m : Subtype fun x => Membership.mem M x
    x : R
    h : Eq ((algebraMap R (Subtype fun x => Membership.mem Bot.bot x)) x) r
    ⊢ Eq (↑((TensorProduct.comm R (Subtype fun x => Membership.mem Bot.bot x) (Sub …
  -/
  rw [← h]; simp
            /-
              🎉 no goals
            -/


variable {M} in
theorem mulLeftMap_eq_mulMap_comp {ι : Type*} [DecidableEq ι] (m : ι → M) :
    mulLeftMap N m = mulMap M N ∘ₗ LinearMap.rTensor N (Finsupp.linearCombination R m) ∘ₗ
      (TensorProduct.finsuppScalarLeft R N ι).symm.toLinearMap := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : Semiring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    ι : Type u_1
    inst✝ : DecidableEq ι
    m : ι → Subtype fun x => Membership.mem M x
    ⊢ Eq (Submodule.mulLeftMap N m) ((M.mulMap N).comp ((LinearMap.rTensor (Subtyp …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


variable {N} in
theorem mulRightMap_eq_mulMap_comp {ι : Type*} [DecidableEq ι] (n : ι → N) :
    mulRightMap M n = mulMap M N ∘ₗ LinearMap.lTensor M (Finsupp.linearCombination R n) ∘ₗ
      (TensorProduct.finsuppScalarRight R M ι).symm.toLinearMap := by
  /-
    R : Type u
    S : Type v
    inst✝³ : CommSemiring R
    inst✝² : Semiring S
    inst✝¹ : Algebra R S
    M N : Submodule R S
    ι : Type u_1
    inst✝ : DecidableEq ι
    n : ι → Subtype fun x => Membership.mem N x
    ⊢ Eq (M.mulRightMap n) ((M.mulMap N).comp ((LinearMap.lTensor (Subtype fun x = …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


theorem mulMap_comm : mulMap N M = (mulMap M N).comp (TensorProduct.comm R N M).toLinearMap :=
  mulMap_comm_of_commute M N fun _ _ ↦ mul_comm _ _


