/-- It is useful to define the bracket via this auxiliary function so that we have a type-theoretic
expression of the fact that `L` acts by linear endomorphisms. It simplifies the proofs in
`lieRingModule` below. -/
def hasBracketAux (x : L) : Module.End R (M ⊗[R] N) :=
  (toEnd R L M x).rTensor N + (toEnd R L N x).lTensor M


/-- The tensor product of two Lie modules is a Lie ring module. -/
instance lieRingModule : LieRingModule L (M ⊗[R] N) where
  bracket x := hasBracketAux x
  add_lie x y t := by
    simp only [hasBracketAux, LinearMap.lTensor_add, LinearMap.rTensor_add, LieHom.map_add,
      LinearMap.add_apply]
    /-
      R : Type u
      inst✝¹⁸ : CommRing R
      L : Type v
      M : Type w
      N : Type w₁
      P : Type w₂
      Q : Type w₃
      inst✝¹⁷ : LieRing L
      inst✝¹⁶ : LieAlgebra R L
      inst✝¹⁵ : AddCommGroup M
      inst✝¹⁴ : Module R M
      inst✝¹³ : LieRingModule L M
      inst✝¹² : LieModule R L M
      inst✝¹¹ : AddCommGroup N
      inst✝¹⁰ : Module R N
      inst✝⁹ : LieRingModule L N
      inst✝⁸ : LieModule R L N
      inst✝⁷ : AddCommGroup P
      inst✝⁶ : Module R P
      inst✝⁵ : LieRingModule L P
      inst✝⁴ : LieModule R L P
      inst✝³ : AddCommGroup Q
      inst✝² : Module R Q
      inst✝¹ : LieRingModule L Q
      inst✝ : LieModule R L Q
      x y : L
      t : TensorProduct R M N
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd ((LinearMap.rTensor N ((LieModule.toEnd R L M) x))  …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/
  lie_add _ := LinearMap.map_add _
  leibniz_lie x y t := by
    suffices (hasBracketAux x).comp (hasBracketAux y) =
        hasBracketAux ⁅x, y⁆ + (hasBracketAux y).comp (hasBracketAux x) by
      simp only [← LinearMap.add_apply]; rw [← LinearMap.comp_apply, this]; rfl
    /-
      R : Type u
      inst✝¹⁸ : CommRing R
      L : Type v
      M : Type w
      N : Type w₁
      P : Type w₂
      Q : Type w₃
      inst✝¹⁷ : LieRing L
      inst✝¹⁶ : LieAlgebra R L
      inst✝¹⁵ : AddCommGroup M
      inst✝¹⁴ : Module R M
      inst✝¹³ : LieRingModule L M
      inst✝¹² : LieModule R L M
      inst✝¹¹ : AddCommGroup N
      inst✝¹⁰ : Module R N
      inst✝⁹ : LieRingModule L N
      inst✝⁸ : LieModule R L N
      inst✝⁷ : AddCommGroup P
      inst✝⁶ : Module R P
      inst✝⁵ : LieRingModule L P
      inst✝⁴ : LieModule R L P
      inst✝³ : AddCommGroup Q
      inst✝² : Module R Q
      inst✝¹ : LieRingModule L Q
      inst✝ : LieModule R L Q
      x y : L
      t : TensorProduct R M N
      ⊢ Eq (LinearMap.comp (TensorProduct.LieModule.hasBracketAux x) (TensorProduct. …
    -/
    ext m n
    simp only [hasBracketAux, AlgebraTensorModule.curry_apply, curry_apply, sub_tmul, tmul_sub,
      LinearMap.coe_restrictScalars, Function.comp_apply, LinearMap.coe_comp,
      LinearMap.rTensor_tmul, LieHom.map_lie, toEnd_apply_apply, LinearMap.add_apply,
      LinearMap.map_add, LieHom.lie_apply, Module.End.lie_apply, LinearMap.lTensor_tmul]
    /-
      case a.h.h
      R : Type u
      inst✝¹⁸ : CommRing R
      L : Type v
      M : Type w
      N : Type w₁
      P : Type w₂
      Q : Type w₃
      inst✝¹⁷ : LieRing L
      inst✝¹⁶ : LieAlgebra R L
      inst✝¹⁵ : AddCommGroup M
      inst✝¹⁴ : Module R M
      inst✝¹³ : LieRingModule L M
      inst✝¹² : LieModule R L M
      inst✝¹¹ : AddCommGroup N
      inst✝¹⁰ : Module R N
      inst✝⁹ : LieRingModule L N
      inst✝⁸ : LieModule R L N
      inst✝⁷ : AddCommGroup P
      inst✝⁶ : Module R P
      inst✝⁵ : LieRingModule L P
      inst✝⁴ : LieModule R L P
      inst✝³ : AddCommGroup Q
      inst✝² : Module R Q
      inst✝¹ : LieRingModule L Q
      inst✝ : LieModule R L Q
      x y : L
      t : TensorProduct R M N
      m : M
      n : N
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd (TensorProduct.tmul R (Bracket.bracket x (Bracket.b …
    -/
    /-
      🎉 no goals
    -/
    abel
    /-
      🎉 no goals
    -/


/-- The tensor product of two Lie modules is a Lie module. -/
instance lieModule : LieModule R L (M ⊗[R] N) where
  smul_lie c x t := by
    /-
      R : Type u
      inst✝¹⁸ : CommRing R
      L : Type v
      M : Type w
      N : Type w₁
      P : Type w₂
      Q : Type w₃
      inst✝¹⁷ : LieRing L
      inst✝¹⁶ : LieAlgebra R L
      inst✝¹⁵ : AddCommGroup M
      inst✝¹⁴ : Module R M
      inst✝¹³ : LieRingModule L M
      inst✝¹² : LieModule R L M
      inst✝¹¹ : AddCommGroup N
      inst✝¹⁰ : Module R N
      inst✝⁹ : LieRingModule L N
      inst✝⁸ : LieModule R L N
      inst✝⁷ : AddCommGroup P
      inst✝⁶ : Module R P
      inst✝⁵ : LieRingModule L P
      inst✝⁴ : LieModule R L P
      inst✝³ : AddCommGroup Q
      inst✝² : Module R Q
      inst✝¹ : LieRingModule L Q
      inst✝ : LieModule R L Q
      c : R
      x : L
      t : TensorProduct R M N
      ⊢ Eq (Bracket.bracket (HSMul.hSMul c x) t) (HSMul.hSMul c (Bracket.bracket x t))
    -/
    change hasBracketAux (c • x) _ = c • hasBracketAux _ _
    simp only [hasBracketAux, smul_add, LinearMap.rTensor_smul, LinearMap.smul_apply,
      LinearMap.lTensor_smul, LieHom.map_smul, LinearMap.add_apply]
  lie_smul c _ := LinearMap.map_smul _ c


@[simp]
theorem lie_tmul_right (x : L) (m : M) (n : N) : ⁅x, m ⊗ₜ[R] n⁆ = ⁅x, m⁆ ⊗ₜ n + m ⊗ₜ ⁅x, n⁆ :=
  show hasBracketAux x (m ⊗ₜ[R] n) = _ by
    simp only [hasBracketAux, LinearMap.rTensor_tmul, toEnd_apply_apply,
      LinearMap.add_apply, LinearMap.lTensor_tmul]


/-- The universal property for tensor product of modules of a Lie algebra: the `R`-linear
tensor-hom adjunction is equivariant with respect to the `L` action. -/
def lift : (M →ₗ[R] N →ₗ[R] P) ≃ₗ⁅R,L⁆ M ⊗[R] N →ₗ[R] P :=
  { TensorProduct.lift.equiv R M N P with
    map_lie' := fun {x f} => by
      /-
        R : Type u
        inst✝¹⁸ : CommRing R
        L : Type v
        M : Type w
        N : Type w₁
        P : Type w₂
        Q : Type w₃
        inst✝¹⁷ : LieRing L
        inst✝¹⁶ : LieAlgebra R L
        inst✝¹⁵ : AddCommGroup M
        inst✝¹⁴ : Module R M
        inst✝¹³ : LieRingModule L M
        inst✝¹² : LieModule R L M
        inst✝¹¹ : AddCommGroup N
        inst✝¹⁰ : Module R N
        inst✝⁹ : LieRingModule L N
        inst✝⁸ : LieModule R L N
        inst✝⁷ : AddCommGroup P
        inst✝⁶ : Module R P
        inst✝⁵ : LieRingModule L P
        inst✝⁴ : LieModule R L P
        inst✝³ : AddCommGroup Q
        inst✝² : Module R Q
        inst✝¹ : LieRingModule L Q
        inst✝ : LieModule R L Q
        x : L
        f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
        ⊢ Eq ((↑__src✝).toFun (Bracket.bracket x f)) (Bracket.bracket x ((↑__src✝).toF …
      -/
      ext m n
      simp only [AddHom.toFun_eq_coe, LinearMap.coe_toAddHom, LinearEquiv.coe_coe,
        AlgebraTensorModule.curry_apply, curry_apply, LinearMap.coe_restrictScalars,
        lift.equiv_apply, LieHom.lie_apply, LinearMap.sub_apply, lie_tmul_right, map_add]
      /-
        case a.h.h
        R : Type u
        inst✝¹⁸ : CommRing R
        L : Type v
        M : Type w
        N : Type w₁
        P : Type w₂
        Q : Type w₃
        inst✝¹⁷ : LieRing L
        inst✝¹⁶ : LieAlgebra R L
        inst✝¹⁵ : AddCommGroup M
        inst✝¹⁴ : Module R M
        inst✝¹³ : LieRingModule L M
        inst✝¹² : LieModule R L M
        inst✝¹¹ : AddCommGroup N
        inst✝¹⁰ : Module R N
        inst✝⁹ : LieRingModule L N
        inst✝⁸ : LieModule R L N
        inst✝⁷ : AddCommGroup P
        inst✝⁶ : Module R P
        inst✝⁵ : LieRingModule L P
        inst✝⁴ : LieModule R L P
        inst✝³ : AddCommGroup Q
        inst✝² : Module R Q
        inst✝¹ : LieRingModule L Q
        inst✝ : LieModule R L Q
        x : L
        f : LinearMap (RingHom.id R) M (LinearMap (RingHom.id R) N P)
        m : M
        n : N
        ⊢ Eq (HSub.hSub (HSub.hSub (Bracket.bracket x ((f m) n)) ((f m) (Bracket.brack …
      -/
      /-
        🎉 no goals
      -/
      abel }
      /-
        🎉 no goals
      -/


@[simp]
theorem lift_apply (f : M →ₗ[R] N →ₗ[R] P) (m : M) (n : N) : lift R L M N P f (m ⊗ₜ n) = f m n :=
  rfl


/-- A weaker form of the universal property for tensor product of modules of a Lie algebra.

Note that maps `f` of type `M →ₗ⁅R,L⁆ N →ₗ[R] P` are exactly those `R`-bilinear maps satisfying
`⁅x, f m n⁆ = f ⁅x, m⁆ n + f m ⁅x, n⁆` for all `x, m, n` (see e.g, `LieModuleHom.map_lie₂`). -/
def liftLie : (M →ₗ⁅R,L⁆ N →ₗ[R] P) ≃ₗ[R] M ⊗[R] N →ₗ⁅R,L⁆ P :=
  maxTrivLinearMapEquivLieModuleHom.symm ≪≫ₗ ↑(maxTrivEquiv (lift R L M N P)) ≪≫ₗ
    maxTrivLinearMapEquivLieModuleHom


@[simp]
theorem coe_liftLie_eq_lift_coe (f : M →ₗ⁅R,L⁆ N →ₗ[R] P) :
    ⇑(liftLie R L M N P f) = lift R L M N P f := by
  suffices (liftLie R L M N P f : M ⊗[R] N →ₗ[R] P) = lift R L M N P f by
    rw [← this, LieModuleHom.coe_toLinearMap]
  /-
    R : Type u
    inst✝¹⁴ : CommRing R
    L : Type v
    M : Type w
    N : Type w₁
    P : Type w₂
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    inst✝⁹ : LieRingModule L M
    inst✝⁸ : LieModule R L M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R N
    inst✝⁵ : LieRingModule L N
    inst✝⁴ : LieModule R L N
    inst✝³ : AddCommGroup P
    inst✝² : Module R P
    inst✝¹ : LieRingModule L P
    inst✝ : LieModule R L P
    f : LieModuleHom R L M (LinearMap (RingHom.id R) N P)
    ⊢ Eq (↑((TensorProduct.LieModule.liftLie R L M N P) f)) ((TensorProduct.LieMod …
  -/
  ext m n
  simp only [liftLie, LinearEquiv.trans_apply, LieModuleEquiv.coe_toLinearEquiv,
    toLinearMap_maxTrivLinearMapEquivLieModuleHom, coe_maxTrivEquiv_apply,
    toLinearMap_maxTrivLinearMapEquivLieModuleHom_symm]


theorem liftLie_apply (f : M →ₗ⁅R,L⁆ N →ₗ[R] P) (m : M) (n : N) :
    liftLie R L M N P f (m ⊗ₜ n) = f m n := by
  /-
    R : Type u
    inst✝¹⁴ : CommRing R
    L : Type v
    M : Type w
    N : Type w₁
    P : Type w₂
    inst✝¹³ : LieRing L
    inst✝¹² : LieAlgebra R L
    inst✝¹¹ : AddCommGroup M
    inst✝¹⁰ : Module R M
    inst✝⁹ : LieRingModule L M
    inst✝⁸ : LieModule R L M
    inst✝⁷ : AddCommGroup N
    inst✝⁶ : Module R N
    inst✝⁵ : LieRingModule L N
    inst✝⁴ : LieModule R L N
    inst✝³ : AddCommGroup P
    inst✝² : Module R P
    inst✝¹ : LieRingModule L P
    inst✝ : LieModule R L P
    f : LieModuleHom R L M (LinearMap (RingHom.id R) N P)
    m : M
    n : N
    ⊢ Eq (((TensorProduct.LieModule.liftLie R L M N P) f) (TensorProduct.tmul R m  …
  -/
  simp only [coe_liftLie_eq_lift_coe, LieModuleHom.coe_toLinearMap, lift_apply]
  /-
    🎉 no goals
  -/


/-- A pair of Lie module morphisms `f : M → P` and `g : N → Q`, induce a Lie module morphism:
`M ⊗ N → P ⊗ Q`. -/
nonrec def map (f : M →ₗ⁅R,L⁆ P) (g : N →ₗ⁅R,L⁆ Q) : M ⊗[R] N →ₗ⁅R,L⁆ P ⊗[R] Q :=
  { map (f : M →ₗ[R] P) (g : N →ₗ[R] Q) with
    map_lie' := fun {x t} => by
      /-
        R : Type u
        inst✝¹⁸ : CommRing R
        L : Type v
        M : Type w
        N : Type w₁
        P : Type w₂
        Q : Type w₃
        inst✝¹⁷ : LieRing L
        inst✝¹⁶ : LieAlgebra R L
        inst✝¹⁵ : AddCommGroup M
        inst✝¹⁴ : Module R M
        inst✝¹³ : LieRingModule L M
        inst✝¹² : LieModule R L M
        inst✝¹¹ : AddCommGroup N
        inst✝¹⁰ : Module R N
        inst✝⁹ : LieRingModule L N
        inst✝⁸ : LieModule R L N
        inst✝⁷ : AddCommGroup P
        inst✝⁶ : Module R P
        inst✝⁵ : LieRingModule L P
        inst✝⁴ : LieModule R L P
        inst✝³ : AddCommGroup Q
        inst✝² : Module R Q
        inst✝¹ : LieRingModule L Q
        inst✝ : LieModule R L Q
        f : LieModuleHom R L M P
        g : LieModuleHom R L N Q
        x : L
        t : TensorProduct R M N
        ⊢ Eq (__src✝.toFun (Bracket.bracket x t)) (Bracket.bracket x (__src✝.toFun t))
      -/
      simp only [LinearMap.toFun_eq_coe]
      /-
        R : Type u
        inst✝¹⁸ : CommRing R
        L : Type v
        M : Type w
        N : Type w₁
        P : Type w₂
        Q : Type w₃
        inst✝¹⁷ : LieRing L
        inst✝¹⁶ : LieAlgebra R L
        inst✝¹⁵ : AddCommGroup M
        inst✝¹⁴ : Module R M
        inst✝¹³ : LieRingModule L M
        inst✝¹² : LieModule R L M
        inst✝¹¹ : AddCommGroup N
        inst✝¹⁰ : Module R N
        inst✝⁹ : LieRingModule L N
        inst✝⁸ : LieModule R L N
        inst✝⁷ : AddCommGroup P
        inst✝⁶ : Module R P
        inst✝⁵ : LieRingModule L P
        inst✝⁴ : LieModule R L P
        inst✝³ : AddCommGroup Q
        inst✝² : Module R Q
        inst✝¹ : LieRingModule L Q
        inst✝ : LieModule R L Q
        f : LieModuleHom R L M P
        g : LieModuleHom R L N Q
        x : L
        t : TensorProduct R M N
        ⊢ Eq ((TensorProduct.map ↑f ↑g) (Bracket.bracket x t)) (Bracket.bracket x ((Te …
      -/
      refine t.induction_on ?_ ?_ ?_
        /-
          case refine_1
          R : Type u
          inst✝¹⁸ : CommRing R
          L : Type v
          M : Type w
          N : Type w₁
          P : Type w₂
          Q : Type w₃
          inst✝¹⁷ : LieRing L
          inst✝¹⁶ : LieAlgebra R L
          inst✝¹⁵ : AddCommGroup M
          inst✝¹⁴ : Module R M
          inst✝¹³ : LieRingModule L M
          inst✝¹² : LieModule R L M
          inst✝¹¹ : AddCommGroup N
          inst✝¹⁰ : Module R N
          inst✝⁹ : LieRingModule L N
          inst✝⁸ : LieModule R L N
          inst✝⁷ : AddCommGroup P
          inst✝⁶ : Module R P
          inst✝⁵ : LieRingModule L P
          inst✝⁴ : LieModule R L P
          inst✝³ : AddCommGroup Q
          inst✝² : Module R Q
          inst✝¹ : LieRingModule L Q
          inst✝ : LieModule R L Q
          f : LieModuleHom R L M P
          g : LieModuleHom R L N Q
          x : L
          t : TensorProduct R M N
          ⊢ Eq ((TensorProduct.map ↑f ↑g) (Bracket.bracket x 0)) (Bracket.bracket x ((Te …
        -/
      · simp only [LinearMap.map_zero, lie_zero]
        /-
          🎉 no goals
        -/
        /-
          case refine_2
          R : Type u
          inst✝¹⁸ : CommRing R
          L : Type v
          M : Type w
          N : Type w₁
          P : Type w₂
          Q : Type w₃
          inst✝¹⁷ : LieRing L
          inst✝¹⁶ : LieAlgebra R L
          inst✝¹⁵ : AddCommGroup M
          inst✝¹⁴ : Module R M
          inst✝¹³ : LieRingModule L M
          inst✝¹² : LieModule R L M
          inst✝¹¹ : AddCommGroup N
          inst✝¹⁰ : Module R N
          inst✝⁹ : LieRingModule L N
          inst✝⁸ : LieModule R L N
          inst✝⁷ : AddCommGroup P
          inst✝⁶ : Module R P
          inst✝⁵ : LieRingModule L P
          inst✝⁴ : LieModule R L P
          inst✝³ : AddCommGroup Q
          inst✝² : Module R Q
          inst✝¹ : LieRingModule L Q
          inst✝ : LieModule R L Q
          f : LieModuleHom R L M P
          g : LieModuleHom R L N Q
          x : L
          t : TensorProduct R M N
          ⊢ ∀ (x_1 : M) (y : N), Eq ((TensorProduct.map ↑f ↑g) (Bracket.bracket x (Tenso …
        -/
      · intro m n
        simp only [LieModuleHom.coe_toLinearMap, lie_tmul_right, LieModuleHom.map_lie, map_tmul,
          LinearMap.map_add]
        /-
          case refine_3
          R : Type u
          inst✝¹⁸ : CommRing R
          L : Type v
          M : Type w
          N : Type w₁
          P : Type w₂
          Q : Type w₃
          inst✝¹⁷ : LieRing L
          inst✝¹⁶ : LieAlgebra R L
          inst✝¹⁵ : AddCommGroup M
          inst✝¹⁴ : Module R M
          inst✝¹³ : LieRingModule L M
          inst✝¹² : LieModule R L M
          inst✝¹¹ : AddCommGroup N
          inst✝¹⁰ : Module R N
          inst✝⁹ : LieRingModule L N
          inst✝⁸ : LieModule R L N
          inst✝⁷ : AddCommGroup P
          inst✝⁶ : Module R P
          inst✝⁵ : LieRingModule L P
          inst✝⁴ : LieModule R L P
          inst✝³ : AddCommGroup Q
          inst✝² : Module R Q
          inst✝¹ : LieRingModule L Q
          inst✝ : LieModule R L Q
          f : LieModuleHom R L M P
          g : LieModuleHom R L N Q
          x : L
          t : TensorProduct R M N
          ⊢ ∀ (x_1 y : TensorProduct R M N), Eq ((TensorProduct.map ↑f ↑g) (Bracket.brac …
        -/
      · intro t₁ t₂ ht₁ ht₂; simp only [ht₁, ht₂, lie_add, LinearMap.map_add] }
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem toLinearMap_map (f : M →ₗ⁅R,L⁆ P) (g : N →ₗ⁅R,L⁆ Q) :
    (map f g : M ⊗[R] N →ₗ[R] P ⊗[R] Q) = TensorProduct.map (f : M →ₗ[R] P) (g : N →ₗ[R] Q) :=
  rfl


@[deprecated (since := "2024-12-30")] alias coe_linearMap_map := toLinearMap_map


@[simp]
nonrec theorem map_tmul (f : M →ₗ⁅R,L⁆ P) (g : N →ₗ⁅R,L⁆ Q) (m : M) (n : N) :
    map f g (m ⊗ₜ n) = f m ⊗ₜ g n :=
  map_tmul _ _ _ _


/-- Given Lie submodules `M' ⊆ M` and `N' ⊆ N`, this is the natural map: `M' ⊗ N' → M ⊗ N`. -/
def mapIncl (M' : LieSubmodule R L M) (N' : LieSubmodule R L N) : M' ⊗[R] N' →ₗ⁅R,L⁆ M ⊗[R] N :=
  map M'.incl N'.incl


@[simp]
theorem mapIncl_def (M' : LieSubmodule R L M) (N' : LieSubmodule R L N) :
    mapIncl M' N' = map M'.incl N'.incl :=
  rfl


/-- The action of the Lie algebra on one of its modules, regarded as a morphism of Lie modules. -/
def toModuleHom : L ⊗[R] M →ₗ⁅R,L⁆ M :=
  TensorProduct.LieModule.liftLie R L L M M
    { (toEnd R L M : L →ₗ[R] M →ₗ[R] M) with
                                  /-
                                    R : Type u
                                    inst✝⁶ : CommRing R
                                    L : Type v
                                    M : Type w
                                    inst✝⁵ : LieRing L
                                    inst✝⁴ : LieAlgebra R L
                                    inst✝³ : AddCommGroup M
                                    inst✝² : Module R M
                                    inst✝¹ : LieRingModule L M
                                    inst✝ : LieModule R L M
                                    x m : L
                                    ⊢ Eq (__src✝.toFun (Bracket.bracket x m)) (Bracket.bracket x (__src✝.toFun m))
                                  -/
      map_lie' := fun {x m} => by ext n; simp [LieRing.of_associative_ring_bracket] }
                                         /-
                                           🎉 no goals
                                         -/


@[simp]
theorem toModuleHom_apply (x : L) (m : M) : toModuleHom R L M (x ⊗ₜ m) = ⁅x, m⁆ := by
  simp only [toModuleHom, TensorProduct.LieModule.liftLie_apply, LieModuleHom.coe_mk,
    LinearMap.coe_mk, LinearMap.coe_toAddHom, LieHom.coe_toLinearMap, toEnd_apply_apply]


/-- A useful alternative characterisation of Lie ideal operations on Lie submodules.

Given a Lie ideal `I ⊆ L` and a Lie submodule `N ⊆ M`, by tensoring the inclusion maps and then
applying the action of `L` on `M`, we obtain morphism of Lie modules `f : I ⊗ N → L ⊗ M → M`.

This lemma states that `⁅I, N⁆ = range f`. -/
theorem lieIdeal_oper_eq_tensor_map_range :
    ⁅I, N⁆ = ((toModuleHom R L M).comp (mapIncl I N : I ⊗[R] N →ₗ⁅R,L⁆ L ⊗[R] M)).range := by
  rw [← toSubmodule_inj, lieIdeal_oper_eq_linear_span, LieModuleHom.toSubmodule_range,
    LieModuleHom.toLinearMap_comp, LinearMap.range_comp, mapIncl_def, toLinearMap_map,
    TensorProduct.map_range_eq_span_tmul, Submodule.map_span]
  /-
    R : Type u
    inst✝⁶ : CommRing R
    L : Type v
    M : Type w
    inst✝⁵ : LieRing L
    inst✝⁴ : LieAlgebra R L
    inst✝³ : AddCommGroup M
    inst✝² : Module R M
    inst✝¹ : LieRingModule L M
    inst✝ : LieModule R L M
    I : LieIdeal R L
    N : LieSubmodule R L M
    ⊢ Eq (Submodule.span R (setOf fun m => Exists fun x => Exists fun n => Eq (Bra …
  -/
  congr; ext m; constructor
    /-
      case e_s.h.mp
      R : Type u
      inst✝⁶ : CommRing R
      L : Type v
      M : Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      I : LieIdeal R L
      N : LieSubmodule R L M
      m : M
      ⊢ Membership.mem (setOf fun m => Exists fun x => Exists fun n => Eq (Bracket.b …
    -/
  · rintro ⟨⟨x, hx⟩, ⟨n, hn⟩, rfl⟩; use x ⊗ₜ n; constructor
      /-
        case h.left
        R : Type u
        inst✝⁶ : CommRing R
        L : Type v
        M : Type w
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        I : LieIdeal R L
        N : LieSubmodule R L M
        x : L
        hx : Membership.mem I x
        n : M
        hn : Membership.mem N n
        ⊢ Membership.mem (setOf fun t => Exists fun m => Exists fun n => Eq (TensorPro …
      -/
    · use ⟨x, hx⟩, ⟨n, hn⟩; rfl
                            /-
                              🎉 no goals
                            -/
      /-
        case h.right
        R : Type u
        inst✝⁶ : CommRing R
        L : Type v
        M : Type w
        inst✝⁵ : LieRing L
        inst✝⁴ : LieAlgebra R L
        inst✝³ : AddCommGroup M
        inst✝² : Module R M
        inst✝¹ : LieRingModule L M
        inst✝ : LieModule R L M
        I : LieIdeal R L
        N : LieSubmodule R L M
        x : L
        hx : Membership.mem I x
        n : M
        hn : Membership.mem N n
        ⊢ Eq (↑(LieModule.toModuleHom R L M) (TensorProduct.tmul R x n)) (Bracket.brac …
      -/
    · simp
      /-
        🎉 no goals
      -/
    /-
      case e_s.h.mpr
      R : Type u
      inst✝⁶ : CommRing R
      L : Type v
      M : Type w
      inst✝⁵ : LieRing L
      inst✝⁴ : LieAlgebra R L
      inst✝³ : AddCommGroup M
      inst✝² : Module R M
      inst✝¹ : LieRingModule L M
      inst✝ : LieModule R L M
      I : LieIdeal R L
      N : LieSubmodule R L M
      m : M
      ⊢ Membership.mem (Set.image (⇑↑(LieModule.toModuleHom R L M)) (setOf fun t =>  …
    -/
  · rintro ⟨t, ⟨⟨x, hx⟩, ⟨n, hn⟩, rfl⟩, h⟩; rw [← h]; use ⟨x, hx⟩, ⟨n, hn⟩; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


