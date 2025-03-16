/-- `ZeroHom M N` is the type of functions `M → N` that preserve zero.

When possible, instead of parametrizing results over `(f : ZeroHom M N)`,
you should parametrize over `(F : Type*) [ZeroHomClass F M N] (f : F)`.

When you extend this structure, make sure to also extend `ZeroHomClass`.
-/
structure ZeroHom (M : Type*) (N : Type*) [Zero M] [Zero N] where
  /-- The underlying function -/
  protected toFun : M → N
  /-- The proposition that the function preserves 0 -/
  protected map_zero' : toFun 0 = 0


/-- `ZeroHomClass F M N` states that `F` is a type of zero-preserving homomorphisms.

You should extend this typeclass when you extend `ZeroHom`.
-/
class ZeroHomClass (F : Type*) (M N : outParam Type*) [Zero M] [Zero N] [FunLike F M N] :
    Prop where
  /-- The proposition that the function preserves 0 -/
  map_zero : ∀ f : F, f 0 = 0

-- Instances and lemmas are defined below through `@[to_additive]`.

/-- `M →ₙ+ N` is the type of functions `M → N` that preserve addition. The `ₙ` in the notation
stands for "non-unital" because it is intended to match the notation for `NonUnitalAlgHom` and
`NonUnitalRingHom`, so a `AddHom` is a non-unital additive monoid hom.

When possible, instead of parametrizing results over `(f : AddHom M N)`,
you should parametrize over `(F : Type*) [AddHomClass F M N] (f : F)`.

When you extend this structure, make sure to extend `AddHomClass`.
-/
structure AddHom (M : Type*) (N : Type*) [Add M] [Add N] where
  /-- The underlying function -/
  protected toFun : M → N
  /-- The proposition that the function preserves addition -/
  protected map_add' : ∀ x y, toFun (x + y) = toFun x + toFun y


/-- `M →ₙ+ N` denotes the type of addition-preserving maps from `M` to `N`. -/
infixr:25 " →ₙ+ " => AddHom


/-- `AddHomClass F M N` states that `F` is a type of addition-preserving homomorphisms.
You should declare an instance of this typeclass when you extend `AddHom`.
-/
class AddHomClass (F : Type*) (M N : outParam Type*) [Add M] [Add N] [FunLike F M N] : Prop where
  /-- The proposition that the function preserves addition -/
  map_add : ∀ (f : F) (x y : M), f (x + y) = f x + f y

-- Instances and lemmas are defined below through `@[to_additive]`.

/-- `M →+ N` is the type of functions `M → N` that preserve the `AddZeroClass` structure.

`AddMonoidHom` is also used for group homomorphisms.

When possible, instead of parametrizing results over `(f : M →+ N)`,
you should parametrize over `(F : Type*) [AddMonoidHomClass F M N] (f : F)`.

When you extend this structure, make sure to extend `AddMonoidHomClass`.
-/
structure AddMonoidHom (M : Type*) (N : Type*) [AddZeroClass M] [AddZeroClass N] extends
  ZeroHom M N, AddHom M N


/-- `M →+ N` denotes the type of additive monoid homomorphisms from `M` to `N`. -/
infixr:25 " →+ " => AddMonoidHom


/-- `AddMonoidHomClass F M N` states that `F` is a type of `AddZeroClass`-preserving
homomorphisms.

You should also extend this typeclass when you extend `AddMonoidHom`.
-/
class AddMonoidHomClass (F : Type*) (M N : outParam Type*)
    [AddZeroClass M] [AddZeroClass N] [FunLike F M N]
    extends AddHomClass F M N, ZeroHomClass F M N : Prop

-- Instances and lemmas are defined below through `@[to_additive]`.

/-- `OneHom M N` is the type of functions `M → N` that preserve one.

When possible, instead of parametrizing results over `(f : OneHom M N)`,
you should parametrize over `(F : Type*) [OneHomClass F M N] (f : F)`.

When you extend this structure, make sure to also extend `OneHomClass`.
-/
@[to_additive]
structure OneHom (M : Type*) (N : Type*) [One M] [One N] where
  /-- The underlying function -/
  protected toFun : M → N
  /-- The proposition that the function preserves 1 -/
  protected map_one' : toFun 1 = 1


/-- `OneHomClass F M N` states that `F` is a type of one-preserving homomorphisms.
You should extend this typeclass when you extend `OneHom`.
-/
@[to_additive]
class OneHomClass (F : Type*) (M N : outParam Type*) [One M] [One N] [FunLike F M N] : Prop where
  /-- The proposition that the function preserves 1 -/
  map_one : ∀ f : F, f 1 = 1


@[to_additive]
instance OneHom.funLike : FunLike (OneHom M N) M N where
  coe := OneHom.toFun
                             /-
                               ι : Type u_1
                               α : Type u_2
                               β : Type u_3
                               M : Type u_4
                               N : Type u_5
                               P : Type u_6
                               G : Type u_7
                               H : Type u_8
                               F : Type u_9
                               inst✝¹ : One M
                               inst✝ : One N
                               f g : OneHom M N
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive]
instance OneHom.oneHomClass : OneHomClass (OneHom M N) M N where
  map_one := OneHom.map_one'


/-- See note [low priority simp lemmas] -/
@[to_additive (attr := simp low)]
theorem map_one [OneHomClass F M N] (f : F) : f 1 = 1 :=
  OneHomClass.map_one f


                                                                                          /-
                                                                                            ι : Type u_1
                                                                                            M : Type u_4
                                                                                            N : Type u_5
                                                                                            F : Type u_9
                                                                                            inst✝³ : One M
                                                                                            inst✝² : One N
                                                                                            inst✝¹ : FunLike F M N
                                                                                            inst✝ : OneHomClass F M N
                                                                                            f : F
                                                                                            ⊢ Eq (Function.comp (⇑f) 1) 1
                                                                                          -/
@[to_additive] lemma map_comp_one [OneHomClass F M N] (f : F) : f ∘ (1 : ι → M) = 1 := by simp
                                                                                          /-
                                                                                            🎉 no goals
                                                                                          -/


/-- In principle this could be an instance, but in practice it causes performance issues. -/
@[to_additive]
theorem Subsingleton.of_oneHomClass [Subsingleton M] [OneHomClass F M N] :
    Subsingleton F where
                                           /-
                                             M : Type u_4
                                             N : Type u_5
                                             F : Type u_9
                                             inst✝⁴ : One M
                                             inst✝³ : One N
                                             inst✝² : FunLike F M N
                                             inst✝¹ : Subsingleton M
                                             inst✝ : OneHomClass F M N
                                             f g : F
                                             x : M
                                             ⊢ Eq (f x) (g x)
                                           -/
  allEq f g := DFunLike.ext _ _ fun x ↦ by simp [Subsingleton.elim x 1]
                                           /-
                                             🎉 no goals
                                           -/


@[to_additive] instance [Subsingleton M] : Subsingleton (OneHom M N) := .of_oneHomClass


@[to_additive]
theorem map_eq_one_iff [OneHomClass F M N] (f : F) (hf : Function.Injective f)
    {x : M} :
    f x = 1 ↔ x = 1 := hf.eq_iff' (map_one f)


@[to_additive]
theorem map_ne_one_iff {R S F : Type*} [One R] [One S] [FunLike F R S] [OneHomClass F R S] (f : F)
    (hf : Function.Injective f) {x : R} : f x ≠ 1 ↔ x ≠ 1 := (map_eq_one_iff f hf).not


@[to_additive]
theorem ne_one_of_map {R S F : Type*} [One R] [One S] [FunLike F R S] [OneHomClass F R S]
                                                                      /-
                                                                        R : Type u_10
                                                                        S : Type u_11
                                                                        F : Type u_12
                                                                        inst✝³ : One R
                                                                        inst✝² : One S
                                                                        inst✝¹ : FunLike F R S
                                                                        inst✝ : OneHomClass F R S
                                                                        f : F
                                                                        x : R
                                                                        hx : Ne (f x) 1
                                                                        ⊢ Ne (f x) (f 1)
                                                                      -/
    {f : F} {x : R} (hx : f x ≠ 1) : x ≠ 1 := ne_of_apply_ne f <| (by rwa [(map_one f)])
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


/-- Turn an element of a type `F` satisfying `OneHomClass F M N` into an actual
`OneHom`. This is declared as the default coercion from `F` to `OneHom M N`. -/
@[to_additive (attr := coe)
"Turn an element of a type `F` satisfying `ZeroHomClass F M N` into an actual
`ZeroHom`. This is declared as the default coercion from `F` to `ZeroHom M N`."]
def OneHomClass.toOneHom [OneHomClass F M N] (f : F) : OneHom M N where
  toFun := f
  map_one' := map_one f


/-- Any type satisfying `OneHomClass` can be cast into `OneHom` via `OneHomClass.toOneHom`. -/
@[to_additive "Any type satisfying `ZeroHomClass` can be cast into `ZeroHom` via
`ZeroHomClass.toZeroHom`. "]
instance [OneHomClass F M N] : CoeTC F (OneHom M N) :=
  ⟨OneHomClass.toOneHom⟩


@[to_additive (attr := simp)]
theorem OneHom.coe_coe [OneHomClass F M N] (f : F) :
    ((f : OneHom M N) : M → N) = f := rfl


/-- `M →ₙ* N` is the type of functions `M → N` that preserve multiplication. The `ₙ` in the notation
stands for "non-unital" because it is intended to match the notation for `NonUnitalAlgHom` and
`NonUnitalRingHom`, so a `MulHom` is a non-unital monoid hom.

When possible, instead of parametrizing results over `(f : M →ₙ* N)`,
you should parametrize over `(F : Type*) [MulHomClass F M N] (f : F)`.
When you extend this structure, make sure to extend `MulHomClass`.
-/
@[to_additive]
structure MulHom (M : Type*) (N : Type*) [Mul M] [Mul N] where
  /-- The underlying function -/
  protected toFun : M → N
  /-- The proposition that the function preserves multiplication -/
  protected map_mul' : ∀ x y, toFun (x * y) = toFun x * toFun y


/-- `M →ₙ* N` denotes the type of multiplication-preserving maps from `M` to `N`. -/
infixr:25 " →ₙ* " => MulHom


/-- `MulHomClass F M N` states that `F` is a type of multiplication-preserving homomorphisms.

You should declare an instance of this typeclass when you extend `MulHom`.
-/
@[to_additive]
class MulHomClass (F : Type*) (M N : outParam Type*) [Mul M] [Mul N] [FunLike F M N] : Prop where
  /-- The proposition that the function preserves multiplication -/
  map_mul : ∀ (f : F) (x y : M), f (x * y) = f x * f y


@[to_additive]
instance MulHom.funLike : FunLike (M →ₙ* N) M N where
  coe := MulHom.toFun
                             /-
                               ι : Type u_1
                               α : Type u_2
                               β : Type u_3
                               M : Type u_4
                               N : Type u_5
                               P : Type u_6
                               G : Type u_7
                               H : Type u_8
                               F : Type u_9
                               inst✝¹ : Mul M
                               inst✝ : Mul N
                               f g : MulHom M N
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


/-- `MulHom` is a type of multiplication-preserving homomorphisms -/
@[to_additive "`AddHom` is a type of addition-preserving homomorphisms"]
instance MulHom.mulHomClass : MulHomClass (M →ₙ* N) M N where
  map_mul := MulHom.map_mul'


/-- See note [low priority simp lemmas] -/
@[to_additive (attr := simp low)]
theorem map_mul [MulHomClass F M N] (f : F) (x y : M) : f (x * y) = f x * f y :=
  MulHomClass.map_mul f x y


@[to_additive (attr := simp)]
lemma map_comp_mul [MulHomClass F M N] (f : F) (g h : ι → M) : f ∘ (g * h) = f ∘ g * f ∘ h := by
  /-
    ι : Type u_1
    M : Type u_4
    N : Type u_5
    F : Type u_9
    inst✝³ : Mul M
    inst✝² : Mul N
    inst✝¹ : FunLike F M N
    inst✝ : MulHomClass F M N
    f : F
    g h : ι → M
    ⊢ Eq (Function.comp (⇑f) (HMul.hMul g h)) (HMul.hMul (Function.comp (⇑f) g) (F …
  -/
  ext; simp
       /-
         🎉 no goals
       -/


/-- Turn an element of a type `F` satisfying `MulHomClass F M N` into an actual
`MulHom`. This is declared as the default coercion from `F` to `M →ₙ* N`. -/
@[to_additive (attr := coe)
"Turn an element of a type `F` satisfying `AddHomClass F M N` into an actual
`AddHom`. This is declared as the default coercion from `F` to `M →ₙ+ N`."]
def MulHomClass.toMulHom [MulHomClass F M N] (f : F) : M →ₙ* N where
  toFun := f
  map_mul' := map_mul f


/-- Any type satisfying `MulHomClass` can be cast into `MulHom` via `MulHomClass.toMulHom`. -/
@[to_additive "Any type satisfying `AddHomClass` can be cast into `AddHom` via
`AddHomClass.toAddHom`."]
instance [MulHomClass F M N] : CoeTC F (M →ₙ* N) :=
  ⟨MulHomClass.toMulHom⟩


@[to_additive (attr := simp)]
theorem MulHom.coe_coe [MulHomClass F M N] (f : F) : ((f : MulHom M N) : M → N) = f := rfl


/-- `M →* N` is the type of functions `M → N` that preserve the `Monoid` structure.
`MonoidHom` is also used for group homomorphisms.

When possible, instead of parametrizing results over `(f : M →* N)`,
you should parametrize over `(F : Type*) [MonoidHomClass F M N] (f : F)`.

When you extend this structure, make sure to extend `MonoidHomClass`.
-/
@[to_additive]
structure MonoidHom (M : Type*) (N : Type*) [MulOneClass M] [MulOneClass N] extends
  OneHom M N, M →ₙ* N


/-- `M →* N` denotes the type of monoid homomorphisms from `M` to `N`. -/
infixr:25 " →* " => MonoidHom


/-- `MonoidHomClass F M N` states that `F` is a type of `Monoid`-preserving homomorphisms.
You should also extend this typeclass when you extend `MonoidHom`. -/
@[to_additive]
class MonoidHomClass (F : Type*) (M N : outParam Type*) [MulOneClass M] [MulOneClass N]
  [FunLike F M N]
  extends MulHomClass F M N, OneHomClass F M N : Prop


@[to_additive]
instance MonoidHom.instFunLike : FunLike (M →* N) M N where
  coe f := f.toFun
  coe_injective' f g h := by
    /-
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      M : Type u_4
      N : Type u_5
      P : Type u_6
      G : Type u_7
      H : Type u_8
      F : Type u_9
      inst✝¹ : MulOneClass M
      inst✝ : MulOneClass N
      f g : MonoidHom M N
      h : Eq ((fun f => f.toFun) f) ((fun f => f.toFun) g)
      ⊢ Eq f g
    -/
    cases f
    /-
      case mk
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      M : Type u_4
      N : Type u_5
      P : Type u_6
      G : Type u_7
      H : Type u_8
      F : Type u_9
      inst✝¹ : MulOneClass M
      inst✝ : MulOneClass N
      g : MonoidHom M N
      toOneHom✝ : OneHom M N
      map_mul'✝ : ∀ (x y : M), Eq (toOneHom✝.toFun (HMul.hMul x y)) (HMul.hMul (toOn …
      h : Eq ((fun f => f.toFun) { toOneHom := toOneHom✝, map_mul' := map_mul'✝ }) ( …
      ⊢ Eq { toOneHom := toOneHom✝, map_mul' := map_mul'✝ } g
    -/
    cases g
    /-
      case mk.mk
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      M : Type u_4
      N : Type u_5
      P : Type u_6
      G : Type u_7
      H : Type u_8
      F : Type u_9
      inst✝¹ : MulOneClass M
      inst✝ : MulOneClass N
      toOneHom✝¹ : OneHom M N
      map_mul'✝¹ : ∀ (x y : M), Eq (toOneHom✝¹.toFun (HMul.hMul x y)) (HMul.hMul (to …
      toOneHom✝ : OneHom M N
      map_mul'✝ : ∀ (x y : M), Eq (toOneHom✝.toFun (HMul.hMul x y)) (HMul.hMul (toOn …
      h : Eq ((fun f => f.toFun) { toOneHom := toOneHom✝¹, map_mul' := map_mul'✝¹ }) …
      ⊢ Eq { toOneHom := toOneHom✝¹, map_mul' := map_mul'✝¹ } { toOneHom := toOneHom …
    -/
    congr
    /-
      case mk.mk.e_toOneHom
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      M : Type u_4
      N : Type u_5
      P : Type u_6
      G : Type u_7
      H : Type u_8
      F : Type u_9
      inst✝¹ : MulOneClass M
      inst✝ : MulOneClass N
      toOneHom✝¹ : OneHom M N
      map_mul'✝¹ : ∀ (x y : M), Eq (toOneHom✝¹.toFun (HMul.hMul x y)) (HMul.hMul (to …
      toOneHom✝ : OneHom M N
      map_mul'✝ : ∀ (x y : M), Eq (toOneHom✝.toFun (HMul.hMul x y)) (HMul.hMul (toOn …
      h : Eq ((fun f => f.toFun) { toOneHom := toOneHom✝¹, map_mul' := map_mul'✝¹ }) …
      ⊢ Eq toOneHom✝¹ toOneHom✝
    -/
    apply DFunLike.coe_injective'
    /-
      case mk.mk.e_toOneHom.a
      ι : Type u_1
      α : Type u_2
      β : Type u_3
      M : Type u_4
      N : Type u_5
      P : Type u_6
      G : Type u_7
      H : Type u_8
      F : Type u_9
      inst✝¹ : MulOneClass M
      inst✝ : MulOneClass N
      toOneHom✝¹ : OneHom M N
      map_mul'✝¹ : ∀ (x y : M), Eq (toOneHom✝¹.toFun (HMul.hMul x y)) (HMul.hMul (to …
      toOneHom✝ : OneHom M N
      map_mul'✝ : ∀ (x y : M), Eq (toOneHom✝.toFun (HMul.hMul x y)) (HMul.hMul (toOn …
      h : Eq ((fun f => f.toFun) { toOneHom := toOneHom✝¹, map_mul' := map_mul'✝¹ }) …
      ⊢ Eq ⇑toOneHom✝¹ ⇑toOneHom✝
    -/
    exact h
    /-
      🎉 no goals
    -/


@[to_additive]
instance MonoidHom.instMonoidHomClass : MonoidHomClass (M →* N) M N where
  map_mul := MonoidHom.map_mul'
  map_one f := f.toOneHom.map_one'


@[to_additive] instance [Subsingleton M] : Subsingleton (M →* N) := .of_oneHomClass


/-- Turn an element of a type `F` satisfying `MonoidHomClass F M N` into an actual
`MonoidHom`. This is declared as the default coercion from `F` to `M →* N`. -/
@[to_additive (attr := coe)
"Turn an element of a type `F` satisfying `AddMonoidHomClass F M N` into an
actual `MonoidHom`. This is declared as the default coercion from `F` to `M →+ N`."]
def MonoidHomClass.toMonoidHom [MonoidHomClass F M N] (f : F) : M →* N :=
  { (f : M →ₙ* N), (f : OneHom M N) with }


/-- Any type satisfying `MonoidHomClass` can be cast into `MonoidHom` via
`MonoidHomClass.toMonoidHom`. -/
@[to_additive "Any type satisfying `AddMonoidHomClass` can be cast into `AddMonoidHom` via
`AddMonoidHomClass.toAddMonoidHom`."]
instance [MonoidHomClass F M N] : CoeTC F (M →* N) :=
  ⟨MonoidHomClass.toMonoidHom⟩


@[to_additive (attr := simp)]
theorem MonoidHom.coe_coe [MonoidHomClass F M N] (f : F) : ((f : M →* N) : M → N) = f := rfl


@[to_additive]
theorem map_mul_eq_one [MonoidHomClass F M N] (f : F) {a b : M} (h : a * b = 1) :
    f a * f b = 1 := by
  /-
    M : Type u_4
    N : Type u_5
    F : Type u_9
    inst✝³ : MulOneClass M
    inst✝² : MulOneClass N
    inst✝¹ : FunLike F M N
    inst✝ : MonoidHomClass F M N
    f : F
    a b : M
    h : Eq (HMul.hMul a b) 1
    ⊢ Eq (HMul.hMul (f a) (f b)) 1
  -/
  rw [← map_mul, h, map_one]
  /-
    🎉 no goals
  -/


@[to_additive]
theorem map_div' [DivInvMonoid G] [DivInvMonoid H] [MonoidHomClass F G H]
    (f : F) (hf : ∀ a, f a⁻¹ = (f a)⁻¹) (a b : G) : f (a / b) = f a / f b := by
  /-
    G : Type u_7
    H : Type u_8
    F : Type u_9
    inst✝³ : FunLike F G H
    inst✝² : DivInvMonoid G
    inst✝¹ : DivInvMonoid H
    inst✝ : MonoidHomClass F G H
    f : F
    hf : ∀ (a : G), Eq (f (Inv.inv a)) (Inv.inv (f a))
    a b : G
    ⊢ Eq (f (HDiv.hDiv a b)) (HDiv.hDiv (f a) (f b))
  -/
  rw [div_eq_mul_inv, div_eq_mul_inv, map_mul, hf]
  /-
    🎉 no goals
  -/


@[to_additive]
lemma map_comp_div' [DivInvMonoid G] [DivInvMonoid H] [MonoidHomClass F G H] (f : F)
    (hf : ∀ a, f a⁻¹ = (f a)⁻¹) (g h : ι → G) : f ∘ (g / h) = f ∘ g / f ∘ h := by
  /-
    ι : Type u_1
    G : Type u_7
    H : Type u_8
    F : Type u_9
    inst✝³ : FunLike F G H
    inst✝² : DivInvMonoid G
    inst✝¹ : DivInvMonoid H
    inst✝ : MonoidHomClass F G H
    f : F
    hf : ∀ (a : G), Eq (f (Inv.inv a)) (Inv.inv (f a))
    g h : ι → G
    ⊢ Eq (Function.comp (⇑f) (HDiv.hDiv g h)) (HDiv.hDiv (Function.comp (⇑f) g) (F …
  -/
  ext; simp [map_div' f hf]
       /-
         🎉 no goals
       -/


/-- Group homomorphisms preserve inverse.

See note [low priority simp lemmas] -/
@[to_additive (attr := simp low) "Additive group homomorphisms preserve negation."]
theorem map_inv [Group G] [DivisionMonoid H] [MonoidHomClass F G H]
    (f : F) (a : G) : f a⁻¹ = (f a)⁻¹ :=
  eq_inv_of_mul_eq_one_left <| map_mul_eq_one f <| inv_mul_cancel _


@[to_additive (attr := simp)]
lemma map_comp_inv [Group G] [DivisionMonoid H] [MonoidHomClass F G H] (f : F) (g : ι → G) :
                              /-
                                ι : Type u_1
                                G : Type u_7
                                H : Type u_8
                                F : Type u_9
                                inst✝³ : FunLike F G H
                                inst✝² : Group G
                                inst✝¹ : DivisionMonoid H
                                inst✝ : MonoidHomClass F G H
                                f : F
                                g : ι → G
                                ⊢ Eq (Function.comp (⇑f) (Inv.inv g)) (Inv.inv (Function.comp (⇑f) g))
                              -/
    f ∘ g⁻¹ = (f ∘ g)⁻¹ := by ext; simp
                                   /-
                                     🎉 no goals
                                   -/


/-- Group homomorphisms preserve division. -/
@[to_additive "Additive group homomorphisms preserve subtraction."]
theorem map_mul_inv [Group G] [DivisionMonoid H] [MonoidHomClass F G H] (f : F) (a b : G) :
                                      /-
                                        G : Type u_7
                                        H : Type u_8
                                        F : Type u_9
                                        inst✝³ : FunLike F G H
                                        inst✝² : Group G
                                        inst✝¹ : DivisionMonoid H
                                        inst✝ : MonoidHomClass F G H
                                        f : F
                                        a b : G
                                        ⊢ Eq (f (HMul.hMul a (Inv.inv b))) (HMul.hMul (f a) (Inv.inv (f b)))
                                      -/
    f (a * b⁻¹) = f a * (f b)⁻¹ := by rw [map_mul, map_inv]
                                      /-
                                        🎉 no goals
                                      -/


@[to_additive]
lemma map_comp_mul_inv [Group G] [DivisionMonoid H] [MonoidHomClass F G H] (f : F) (g h : ι → G) :
                                            /-
                                              ι : Type u_1
                                              G : Type u_7
                                              H : Type u_8
                                              F : Type u_9
                                              inst✝³ : FunLike F G H
                                              inst✝² : Group G
                                              inst✝¹ : DivisionMonoid H
                                              inst✝ : MonoidHomClass F G H
                                              f : F
                                              g h : ι → G
                                              ⊢ Eq (Function.comp (⇑f) (HMul.hMul g (Inv.inv h))) (HMul.hMul (Function.comp  …
                                            -/
    f ∘ (g * h⁻¹) = f ∘ g * (f ∘ h)⁻¹ := by simp
                                            /-
                                              🎉 no goals
                                            -/


/-- Group homomorphisms preserve division.

See note [low priority simp lemmas] -/
@[to_additive (attr := simp low) "Additive group homomorphisms preserve subtraction."]
theorem map_div [Group G] [DivisionMonoid H] [MonoidHomClass F G H] (f : F) :
    ∀ a b, f (a / b) = f a / f b := map_div' _ <| map_inv f


@[to_additive (attr := simp)]
lemma map_comp_div [Group G] [DivisionMonoid H] [MonoidHomClass F G H] (f : F) (g h : ι → G) :
                                      /-
                                        ι : Type u_1
                                        G : Type u_7
                                        H : Type u_8
                                        F : Type u_9
                                        inst✝³ : FunLike F G H
                                        inst✝² : Group G
                                        inst✝¹ : DivisionMonoid H
                                        inst✝ : MonoidHomClass F G H
                                        f : F
                                        g h : ι → G
                                        ⊢ Eq (Function.comp (⇑f) (HDiv.hDiv g h)) (HDiv.hDiv (Function.comp (⇑f) g) (F …
                                      -/
    f ∘ (g / h) = f ∘ g / f ∘ h := by ext; simp
                                           /-
                                             🎉 no goals
                                           -/


/-- See note [low priority simp lemmas] -/
@[to_additive (attr := simp low) (reorder := 9 10)]
theorem map_pow [Monoid G] [Monoid H] [MonoidHomClass F G H] (f : F) (a : G) :
    ∀ n : ℕ, f (a ^ n) = f a ^ n
            /-
              G : Type u_7
              H : Type u_8
              F : Type u_9
              inst✝³ : FunLike F G H
              inst✝² : Monoid G
              inst✝¹ : Monoid H
              inst✝ : MonoidHomClass F G H
              f : F
              a : G
              ⊢ Eq (f (HPow.hPow a 0)) (HPow.hPow (f a) 0)
            -/
  | 0 => by rw [pow_zero, pow_zero, map_one]
            /-
              🎉 no goals
            -/
                /-
                  G : Type u_7
                  H : Type u_8
                  F : Type u_9
                  inst✝³ : FunLike F G H
                  inst✝² : Monoid G
                  inst✝¹ : Monoid H
                  inst✝ : MonoidHomClass F G H
                  f : F
                  a : G
                  n : Nat
                  ⊢ Eq (f (HPow.hPow a (HAdd.hAdd n 1))) (HPow.hPow (f a) (HAdd.hAdd n 1))
                -/
  | n + 1 => by rw [pow_succ, pow_succ, map_mul, map_pow f a n]
                /-
                  🎉 no goals
                -/


@[to_additive (attr := simp)]
lemma map_comp_pow [Monoid G] [Monoid H] [MonoidHomClass F G H] (f : F) (g : ι → G) (n : ℕ) :
                                  /-
                                    ι : Type u_1
                                    G : Type u_7
                                    H : Type u_8
                                    F : Type u_9
                                    inst✝³ : FunLike F G H
                                    inst✝² : Monoid G
                                    inst✝¹ : Monoid H
                                    inst✝ : MonoidHomClass F G H
                                    f : F
                                    g : ι → G
                                    n : Nat
                                    ⊢ Eq (Function.comp (⇑f) (HPow.hPow g n)) (HPow.hPow (Function.comp (⇑f) g) n)
                                  -/
    f ∘ (g ^ n) = f ∘ g ^ n := by ext; simp
                                       /-
                                         🎉 no goals
                                       -/


@[to_additive]
theorem map_zpow' [DivInvMonoid G] [DivInvMonoid H] [MonoidHomClass F G H]
    (f : F) (hf : ∀ x : G, f x⁻¹ = (f x)⁻¹) (a : G) : ∀ n : ℤ, f (a ^ n) = f a ^ n
                  /-
                    G : Type u_7
                    H : Type u_8
                    F : Type u_9
                    inst✝³ : FunLike F G H
                    inst✝² : DivInvMonoid G
                    inst✝¹ : DivInvMonoid H
                    inst✝ : MonoidHomClass F G H
                    f : F
                    hf : ∀ (x : G), Eq (f (Inv.inv x)) (Inv.inv (f x))
                    a : G
                    n : Nat
                    ⊢ Eq (f (HPow.hPow a ↑n)) (HPow.hPow (f a) ↑n)
                  -/
  | (n : ℕ) => by rw [zpow_natCast, map_pow, zpow_natCast]
                  /-
                    🎉 no goals
                  -/
                        /-
                          G : Type u_7
                          H : Type u_8
                          F : Type u_9
                          inst✝³ : FunLike F G H
                          inst✝² : DivInvMonoid G
                          inst✝¹ : DivInvMonoid H
                          inst✝ : MonoidHomClass F G H
                          f : F
                          hf : ∀ (x : G), Eq (f (Inv.inv x)) (Inv.inv (f x))
                          a : G
                          n : Nat
                          ⊢ Eq (f (HPow.hPow a (Int.negSucc n))) (HPow.hPow (f a) (Int.negSucc n))
                        -/
  | Int.negSucc n => by rw [zpow_negSucc, hf, map_pow, ← zpow_negSucc]
                        /-
                          🎉 no goals
                        -/


@[to_additive (attr := simp)]
lemma map_comp_zpow' [DivInvMonoid G] [DivInvMonoid H] [MonoidHomClass F G H] (f : F)
    (hf : ∀ x : G, f x⁻¹ = (f x)⁻¹) (g : ι → G) (n : ℤ) : f ∘ (g ^ n) = f ∘ g ^ n := by
  /-
    ι : Type u_1
    G : Type u_7
    H : Type u_8
    F : Type u_9
    inst✝³ : FunLike F G H
    inst✝² : DivInvMonoid G
    inst✝¹ : DivInvMonoid H
    inst✝ : MonoidHomClass F G H
    f : F
    hf : ∀ (x : G), Eq (f (Inv.inv x)) (Inv.inv (f x))
    g : ι → G
    n : Int
    ⊢ Eq (Function.comp (⇑f) (HPow.hPow g n)) (HPow.hPow (Function.comp (⇑f) g) n)
  -/
  ext; simp [map_zpow' f hf]
       /-
         🎉 no goals
       -/


/-- Group homomorphisms preserve integer power.

See note [low priority simp lemmas] -/
@[to_additive (attr := simp low) (reorder := 9 10)
"Additive group homomorphisms preserve integer scaling."]
theorem map_zpow [Group G] [DivisionMonoid H] [MonoidHomClass F G H]
    (f : F) (g : G) (n : ℤ) : f (g ^ n) = f g ^ n := map_zpow' f (map_inv f) g n


@[to_additive]
lemma map_comp_zpow [Group G] [DivisionMonoid H] [MonoidHomClass F G H] (f : F) (g : ι → G)
                                            /-
                                              ι : Type u_1
                                              G : Type u_7
                                              H : Type u_8
                                              F : Type u_9
                                              inst✝³ : FunLike F G H
                                              inst✝² : Group G
                                              inst✝¹ : DivisionMonoid H
                                              inst✝ : MonoidHomClass F G H
                                              f : F
                                              g : ι → G
                                              n : Int
                                              ⊢ Eq (Function.comp (⇑f) (HPow.hPow g n)) (HPow.hPow (Function.comp (⇑f) g) n)
                                            -/
    (n : ℤ) : f ∘ (g ^ n) = f ∘ g ^ n := by simp
                                            /-
                                              🎉 no goals
                                            -/


/-- `MonoidHom` down-cast to a `OneHom`, forgetting the multiplicative property. -/
@[to_additive "`AddMonoidHom` down-cast to a `ZeroHom`, forgetting the additive property"]
instance MonoidHom.coeToOneHom [MulOneClass M] [MulOneClass N] :
  Coe (M →* N) (OneHom M N) := ⟨MonoidHom.toOneHom⟩


/-- `MonoidHom` down-cast to a `MulHom`, forgetting the 1-preserving property. -/
@[to_additive "`AddMonoidHom` down-cast to an `AddHom`, forgetting the 0-preserving property."]
instance MonoidHom.coeToMulHom [MulOneClass M] [MulOneClass N] :
  Coe (M →* N) (M →ₙ* N) := ⟨MonoidHom.toMulHom⟩

-- these must come after the coe_toFun definitions

@[to_additive (attr := simp)]
theorem OneHom.coe_mk [One M] [One N] (f : M → N) (h1) : (OneHom.mk f h1 : M → N) = f := rfl


@[to_additive (attr := simp)]
theorem OneHom.toFun_eq_coe [One M] [One N] (f : OneHom M N) : f.toFun = f := rfl


@[to_additive (attr := simp)]
theorem MulHom.coe_mk [Mul M] [Mul N] (f : M → N) (hmul) : (MulHom.mk f hmul : M → N) = f := rfl


@[to_additive (attr := simp)]
theorem MulHom.toFun_eq_coe [Mul M] [Mul N] (f : M →ₙ* N) : f.toFun = f := rfl


@[to_additive (attr := simp)]
theorem MonoidHom.coe_mk [MulOneClass M] [MulOneClass N] (f hmul) :
    (MonoidHom.mk f hmul : M → N) = f := rfl


@[to_additive (attr := simp)]
theorem MonoidHom.toOneHom_coe [MulOneClass M] [MulOneClass N] (f : M →* N) :
    (f.toOneHom : M → N) = f := rfl


@[to_additive (attr := simp)]
theorem MonoidHom.toMulHom_coe [MulOneClass M] [MulOneClass N] (f : M →* N) :
    f.toMulHom.toFun = f := rfl


@[to_additive]
theorem MonoidHom.toFun_eq_coe [MulOneClass M] [MulOneClass N] (f : M →* N) : f.toFun = f := rfl


@[to_additive (attr := ext)]
theorem OneHom.ext [One M] [One N] ⦃f g : OneHom M N⦄ (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


@[to_additive (attr := ext)]
theorem MulHom.ext [Mul M] [Mul N] ⦃f g : M →ₙ* N⦄ (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


@[to_additive (attr := ext)]
theorem MonoidHom.ext [MulOneClass M] [MulOneClass N] ⦃f g : M →* N⦄ (h : ∀ x, f x = g x) : f = g :=
  DFunLike.ext _ _ h


/-- Makes a group homomorphism from a proof that the map preserves multiplication. -/
@[to_additive (attr := simps (config := .asFn))
  "Makes an additive group homomorphism from a proof that the map preserves addition."]
def mk' (f : M → G) (map_mul : ∀ a b : M, f (a * b) = f a * f b) : M →* G where
  toFun := f
  map_mul' := map_mul
                 /-
                   ι : Type u_1
                   α : Type u_2
                   β : Type u_3
                   M : Type u_4
                   N : Type u_5
                   P : Type u_6
                   G : Type u_7
                   H : Type u_8
                   F : Type u_9
                   inst✝¹ : Group G
                   inst✝ : MulOneClass M
                   f : M → G
                   map_mul : ∀ (a b : M), Eq (f (HMul.hMul a b)) (HMul.hMul (f a) (f b))
                   ⊢ Eq (f 1) 1
                 -/
  map_one' := by rw [← mul_right_cancel_iff, ← map_mul _ 1, one_mul, one_mul]
                 /-
                   🎉 no goals
                 -/


@[to_additive (attr := simp)]
theorem OneHom.mk_coe [One M] [One N] (f : OneHom M N) (h1) : OneHom.mk f h1 = f :=
  OneHom.ext fun _ => rfl


@[to_additive (attr := simp)]
theorem MulHom.mk_coe [Mul M] [Mul N] (f : M →ₙ* N) (hmul) : MulHom.mk f hmul = f :=
  MulHom.ext fun _ => rfl


@[to_additive (attr := simp)]
theorem MonoidHom.mk_coe [MulOneClass M] [MulOneClass N] (f : M →* N) (hmul) :
    MonoidHom.mk f hmul = f := MonoidHom.ext fun _ => rfl


/-- Copy of a `OneHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
@[to_additive
  "Copy of a `ZeroHom` with a new `toFun` equal to the old one. Useful to fix
  definitional equalities."]
protected def OneHom.copy [One M] [One N] (f : OneHom M N) (f' : M → N) (h : f' = f) :
    OneHom M N where
  toFun := f'
  map_one' := h.symm ▸ f.map_one'


@[to_additive (attr := simp)]
theorem OneHom.coe_copy {_ : One M} {_ : One N} (f : OneHom M N) (f' : M → N) (h : f' = f) :
    (f.copy f' h) = f' :=
  rfl


@[to_additive]
theorem OneHom.coe_copy_eq {_ : One M} {_ : One N} (f : OneHom M N) (f' : M → N) (h : f' = f) :
    f.copy f' h = f :=
  DFunLike.ext' h


/-- Copy of a `MulHom` with a new `toFun` equal to the old one. Useful to fix definitional
equalities. -/
@[to_additive
  "Copy of an `AddHom` with a new `toFun` equal to the old one. Useful to fix
  definitional equalities."]
protected def MulHom.copy [Mul M] [Mul N] (f : M →ₙ* N) (f' : M → N) (h : f' = f) :
    M →ₙ* N where
  toFun := f'
  map_mul' := h.symm ▸ f.map_mul'


@[to_additive (attr := simp)]
theorem MulHom.coe_copy {_ : Mul M} {_ : Mul N} (f : M →ₙ* N) (f' : M → N) (h : f' = f) :
    (f.copy f' h) = f' :=
  rfl


@[to_additive]
theorem MulHom.coe_copy_eq {_ : Mul M} {_ : Mul N} (f : M →ₙ* N) (f' : M → N) (h : f' = f) :
    f.copy f' h = f :=
  DFunLike.ext' h


/-- Copy of a `MonoidHom` with a new `toFun` equal to the old one. Useful to fix
definitional equalities. -/
@[to_additive
  "Copy of an `AddMonoidHom` with a new `toFun` equal to the old one. Useful to fix
  definitional equalities."]
protected def MonoidHom.copy [MulOneClass M] [MulOneClass N] (f : M →* N) (f' : M → N)
    (h : f' = f) : M →* N :=
  { f.toOneHom.copy f' h, f.toMulHom.copy f' h with }


@[to_additive (attr := simp)]
theorem MonoidHom.coe_copy {_ : MulOneClass M} {_ : MulOneClass N} (f : M →* N) (f' : M → N)
    (h : f' = f) : (f.copy f' h) = f' :=
  rfl


@[to_additive]
theorem MonoidHom.copy_eq {_ : MulOneClass M} {_ : MulOneClass N} (f : M →* N) (f' : M → N)
    (h : f' = f) : f.copy f' h = f :=
  DFunLike.ext' h


@[to_additive]
protected theorem OneHom.map_one [One M] [One N] (f : OneHom M N) : f 1 = 1 :=
  f.map_one'


/-- If `f` is a monoid homomorphism then `f 1 = 1`. -/
@[to_additive "If `f` is an additive monoid homomorphism then `f 0 = 0`."]
protected theorem MonoidHom.map_one [MulOneClass M] [MulOneClass N] (f : M →* N) : f 1 = 1 :=
  f.map_one'


@[to_additive]
protected theorem MulHom.map_mul [Mul M] [Mul N] (f : M →ₙ* N) (a b : M) : f (a * b) = f a * f b :=
  f.map_mul' a b


/-- If `f` is a monoid homomorphism then `f (a * b) = f a * f b`. -/
@[to_additive "If `f` is an additive monoid homomorphism then `f (a + b) = f a + f b`."]
protected theorem MonoidHom.map_mul [MulOneClass M] [MulOneClass N] (f : M →* N) (a b : M) :
    f (a * b) = f a * f b := f.map_mul' a b


/-- Given a monoid homomorphism `f : M →* N` and an element `x : M`, if `x` has a right inverse,
then `f x` has a right inverse too. For elements invertible on both sides see `IsUnit.map`. -/
@[to_additive
  "Given an AddMonoid homomorphism `f : M →+ N` and an element `x : M`, if `x` has
  a right inverse, then `f x` has a right inverse too."]
theorem map_exists_right_inv (f : F) {x : M} (hx : ∃ y, x * y = 1) : ∃ y, f x * y = 1 :=
  let ⟨y, hy⟩ := hx
  ⟨f y, map_mul_eq_one f hy⟩


/-- Given a monoid homomorphism `f : M →* N` and an element `x : M`, if `x` has a left inverse,
then `f x` has a left inverse too. For elements invertible on both sides see `IsUnit.map`. -/
@[to_additive
  "Given an AddMonoid homomorphism `f : M →+ N` and an element `x : M`, if `x` has
  a left inverse, then `f x` has a left inverse too. For elements invertible on both sides see
  `IsAddUnit.map`."]
theorem map_exists_left_inv (f : F) {x : M} (hx : ∃ y, y * x = 1) : ∃ y, y * f x = 1 :=
  let ⟨y, hy⟩ := hx
  ⟨f y, map_mul_eq_one f hy⟩


/-- The identity map from a type with 1 to itself. -/
@[to_additive (attr := simps) "The identity map from a type with zero to itself."]
def OneHom.id (M : Type*) [One M] : OneHom M M where
  toFun x := x
  map_one' := rfl


/-- The identity map from a type with multiplication to itself. -/
@[to_additive (attr := simps) "The identity map from a type with addition to itself."]
def MulHom.id (M : Type*) [Mul M] : M →ₙ* M where
  toFun x := x
  map_mul' _ _ := rfl


/-- The identity map from a monoid to itself. -/
@[to_additive (attr := simps) "The identity map from an additive monoid to itself."]
def MonoidHom.id (M : Type*) [MulOneClass M] : M →* M where
  toFun x := x
  map_one' := rfl
  map_mul' _ _ := rfl


/-- Composition of `OneHom`s as a `OneHom`. -/
@[to_additive "Composition of `ZeroHom`s as a `ZeroHom`."]
def OneHom.comp [One M] [One N] [One P] (hnp : OneHom N P) (hmn : OneHom M N) : OneHom M P where
  toFun := hnp ∘ hmn
                 /-
                   ι : Type u_1
                   α : Type u_2
                   β : Type u_3
                   M : Type u_4
                   N : Type u_5
                   P : Type u_6
                   G : Type u_7
                   H : Type u_8
                   F : Type u_9
                   inst✝² : One M
                   inst✝¹ : One N
                   inst✝ : One P
                   hnp : OneHom N P
                   hmn : OneHom M N
                   ⊢ Eq (Function.comp (⇑hnp) (⇑hmn) 1) 1
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/


/-- Composition of `MulHom`s as a `MulHom`. -/
@[to_additive "Composition of `AddHom`s as an `AddHom`."]
def MulHom.comp [Mul M] [Mul N] [Mul P] (hnp : N →ₙ* P) (hmn : M →ₙ* N) : M →ₙ* P where
  toFun := hnp ∘ hmn
                     /-
                       ι : Type u_1
                       α : Type u_2
                       β : Type u_3
                       M : Type u_4
                       N : Type u_5
                       P : Type u_6
                       G : Type u_7
                       H : Type u_8
                       F : Type u_9
                       inst✝² : Mul M
                       inst✝¹ : Mul N
                       inst✝ : Mul P
                       hnp : MulHom N P
                       hmn : MulHom M N
                       x y : M
                       ⊢ Eq (Function.comp (⇑hnp) (⇑hmn) (HMul.hMul x y)) (HMul.hMul (Function.comp ( …
                     -/
  map_mul' x y := by simp
                     /-
                       🎉 no goals
                     -/


/-- Composition of monoid morphisms as a monoid morphism. -/
@[to_additive "Composition of additive monoid morphisms as an additive monoid morphism."]
def MonoidHom.comp [MulOneClass M] [MulOneClass N] [MulOneClass P] (hnp : N →* P) (hmn : M →* N) :
    M →* P where
  toFun := hnp ∘ hmn
                 /-
                   ι : Type u_1
                   α : Type u_2
                   β : Type u_3
                   M : Type u_4
                   N : Type u_5
                   P : Type u_6
                   G : Type u_7
                   H : Type u_8
                   F : Type u_9
                   inst✝² : MulOneClass M
                   inst✝¹ : MulOneClass N
                   inst✝ : MulOneClass P
                   hnp : MonoidHom N P
                   hmn : MonoidHom M N
                   ⊢ Eq (Function.comp (⇑hnp) (⇑hmn) 1) 1
                 -/
  map_one' := by simp
                 /-
                   🎉 no goals
                 -/
                 /-
                   ι : Type u_1
                   α : Type u_2
                   β : Type u_3
                   M : Type u_4
                   N : Type u_5
                   P : Type u_6
                   G : Type u_7
                   H : Type u_8
                   F : Type u_9
                   inst✝² : MulOneClass M
                   inst✝¹ : MulOneClass N
                   inst✝ : MulOneClass P
                   hnp : MonoidHom N P
                   hmn : MonoidHom M N
                   ⊢ ∀ (x y : M), Eq ({ toFun := Function.comp ⇑hnp ⇑hmn, map_one' := ⋯ }.toFun ( …
                 -/
  map_mul' := by simp
                 /-
                   🎉 no goals
                 -/


@[to_additive (attr := simp)]
theorem OneHom.coe_comp [One M] [One N] [One P] (g : OneHom N P) (f : OneHom M N) :
    ↑(g.comp f) = g ∘ f := rfl


@[to_additive (attr := simp)]
theorem MulHom.coe_comp [Mul M] [Mul N] [Mul P] (g : N →ₙ* P) (f : M →ₙ* N) :
    ↑(g.comp f) = g ∘ f := rfl


@[to_additive (attr := simp)]
theorem MonoidHom.coe_comp [MulOneClass M] [MulOneClass N] [MulOneClass P]
    (g : N →* P) (f : M →* N) : ↑(g.comp f) = g ∘ f := rfl


@[to_additive]
theorem OneHom.comp_apply [One M] [One N] [One P] (g : OneHom N P) (f : OneHom M N) (x : M) :
    g.comp f x = g (f x) := rfl


@[to_additive]
theorem MulHom.comp_apply [Mul M] [Mul N] [Mul P] (g : N →ₙ* P) (f : M →ₙ* N) (x : M) :
    g.comp f x = g (f x) := rfl


@[to_additive]
theorem MonoidHom.comp_apply [MulOneClass M] [MulOneClass N] [MulOneClass P]
    (g : N →* P) (f : M →* N) (x : M) : g.comp f x = g (f x) := rfl


/-- Composition of monoid homomorphisms is associative. -/
@[to_additive "Composition of additive monoid homomorphisms is associative."]
theorem OneHom.comp_assoc {Q : Type*} [One M] [One N] [One P] [One Q]
    (f : OneHom M N) (g : OneHom N P) (h : OneHom P Q) :
    (h.comp g).comp f = h.comp (g.comp f) := rfl


@[to_additive]
theorem MulHom.comp_assoc {Q : Type*} [Mul M] [Mul N] [Mul P] [Mul Q]
    (f : M →ₙ* N) (g : N →ₙ* P) (h : P →ₙ* Q) : (h.comp g).comp f = h.comp (g.comp f) := rfl


@[to_additive]
theorem MonoidHom.comp_assoc {Q : Type*} [MulOneClass M] [MulOneClass N] [MulOneClass P]
    [MulOneClass Q] (f : M →* N) (g : N →* P) (h : P →* Q) :
    (h.comp g).comp f = h.comp (g.comp f) := rfl


@[to_additive]
theorem OneHom.cancel_right [One M] [One N] [One P] {g₁ g₂ : OneHom N P} {f : OneHom M N}
    (hf : Function.Surjective f) : g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => OneHom.ext <| hf.forall.2 (DFunLike.ext_iff.1 h), fun h => h ▸ rfl⟩


@[to_additive]
theorem MulHom.cancel_right [Mul M] [Mul N] [Mul P] {g₁ g₂ : N →ₙ* P} {f : M →ₙ* N}
    (hf : Function.Surjective f) : g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => MulHom.ext <| hf.forall.2 (DFunLike.ext_iff.1 h), fun h => h ▸ rfl⟩


@[to_additive]
theorem MonoidHom.cancel_right [MulOneClass M] [MulOneClass N] [MulOneClass P]
    {g₁ g₂ : N →* P} {f : M →* N} (hf : Function.Surjective f) :
    g₁.comp f = g₂.comp f ↔ g₁ = g₂ :=
  ⟨fun h => MonoidHom.ext <| hf.forall.2 (DFunLike.ext_iff.1 h), fun h => h ▸ rfl⟩


@[to_additive]
theorem OneHom.cancel_left [One M] [One N] [One P] {g : OneHom N P} {f₁ f₂ : OneHom M N}
    (hg : Function.Injective g) : g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                         /-
                                           M : Type u_4
                                           N : Type u_5
                                           P : Type u_6
                                           inst✝² : One M
                                           inst✝¹ : One N
                                           inst✝ : One P
                                           g : OneHom N P
                                           f₁ f₂ : OneHom M N
                                           hg : Function.Injective ⇑g
                                           h : Eq (g.comp f₁) (g.comp f₂)
                                           x : M
                                           ⊢ Eq (g (f₁ x)) (g (f₂ x))
                                         -/
  ⟨fun h => OneHom.ext fun x => hg <| by rw [← OneHom.comp_apply, h, OneHom.comp_apply],
                                         /-
                                           🎉 no goals
                                         -/
    fun h => h ▸ rfl⟩


@[to_additive]
theorem MulHom.cancel_left [Mul M] [Mul N] [Mul P] {g : N →ₙ* P} {f₁ f₂ : M →ₙ* N}
    (hg : Function.Injective g) : g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                         /-
                                           M : Type u_4
                                           N : Type u_5
                                           P : Type u_6
                                           inst✝² : Mul M
                                           inst✝¹ : Mul N
                                           inst✝ : Mul P
                                           g : MulHom N P
                                           f₁ f₂ : MulHom M N
                                           hg : Function.Injective ⇑g
                                           h : Eq (g.comp f₁) (g.comp f₂)
                                           x : M
                                           ⊢ Eq (g (f₁ x)) (g (f₂ x))
                                         -/
  ⟨fun h => MulHom.ext fun x => hg <| by rw [← MulHom.comp_apply, h, MulHom.comp_apply],
                                         /-
                                           🎉 no goals
                                         -/
    fun h => h ▸ rfl⟩


@[to_additive]
theorem MonoidHom.cancel_left [MulOneClass M] [MulOneClass N] [MulOneClass P]
    {g : N →* P} {f₁ f₂ : M →* N} (hg : Function.Injective g) : g.comp f₁ = g.comp f₂ ↔ f₁ = f₂ :=
                                            /-
                                              M : Type u_4
                                              N : Type u_5
                                              P : Type u_6
                                              inst✝² : MulOneClass M
                                              inst✝¹ : MulOneClass N
                                              inst✝ : MulOneClass P
                                              g : MonoidHom N P
                                              f₁ f₂ : MonoidHom M N
                                              hg : Function.Injective ⇑g
                                              h : Eq (g.comp f₁) (g.comp f₂)
                                              x : M
                                              ⊢ Eq (g (f₁ x)) (g (f₂ x))
                                            -/
  ⟨fun h => MonoidHom.ext fun x => hg <| by rw [← MonoidHom.comp_apply, h, MonoidHom.comp_apply],
                                            /-
                                              🎉 no goals
                                            -/
    fun h => h ▸ rfl⟩


@[to_additive]
theorem MonoidHom.toOneHom_injective [MulOneClass M] [MulOneClass N] :
    Function.Injective (MonoidHom.toOneHom : (M →* N) → OneHom M N) :=
  Function.Injective.of_comp (f := DFunLike.coe) DFunLike.coe_injective


@[to_additive]
theorem MonoidHom.toMulHom_injective [MulOneClass M] [MulOneClass N] :
    Function.Injective (MonoidHom.toMulHom : (M →* N) → M →ₙ* N) :=
  Function.Injective.of_comp (f := DFunLike.coe) DFunLike.coe_injective


@[to_additive (attr := simp)]
theorem OneHom.comp_id [One M] [One N] (f : OneHom M N) : f.comp (OneHom.id M) = f :=
  OneHom.ext fun _ => rfl


@[to_additive (attr := simp)]
theorem MulHom.comp_id [Mul M] [Mul N] (f : M →ₙ* N) : f.comp (MulHom.id M) = f :=
  MulHom.ext fun _ => rfl


@[to_additive (attr := simp)]
theorem MonoidHom.comp_id [MulOneClass M] [MulOneClass N] (f : M →* N) :
    f.comp (MonoidHom.id M) = f := MonoidHom.ext fun _ => rfl


@[to_additive (attr := simp)]
theorem OneHom.id_comp [One M] [One N] (f : OneHom M N) : (OneHom.id N).comp f = f :=
  OneHom.ext fun _ => rfl


@[to_additive (attr := simp)]
theorem MulHom.id_comp [Mul M] [Mul N] (f : M →ₙ* N) : (MulHom.id N).comp f = f :=
  MulHom.ext fun _ => rfl


@[to_additive (attr := simp)]
theorem MonoidHom.id_comp [MulOneClass M] [MulOneClass N] (f : M →* N) :
    (MonoidHom.id N).comp f = f := MonoidHom.ext fun _ => rfl


@[to_additive]
protected theorem MonoidHom.map_pow [Monoid M] [Monoid N] (f : M →* N) (a : M) (n : ℕ) :
    f (a ^ n) = f a ^ n := map_pow f a n


@[to_additive]
protected theorem MonoidHom.map_zpow' [DivInvMonoid M] [DivInvMonoid N] (f : M →* N)
    (hf : ∀ x, f x⁻¹ = (f x)⁻¹) (a : M) (n : ℤ) :
    f (a ^ n) = f a ^ n := map_zpow' f hf a n


/-- Makes a `OneHom` inverse from the bijective inverse of a `OneHom` -/
@[to_additive (attr := simps)
  "Make a `ZeroHom` inverse from the bijective inverse of a `ZeroHom`"]
def OneHom.inverse [One M] [One N]
    (f : OneHom M N) (g : N → M)
    (h₁ : Function.LeftInverse g f) :
  OneHom N M :=
  { toFun := g,
                   /-
                     ι : Type u_1
                     α : Type u_2
                     β : Type u_3
                     M : Type u_4
                     N : Type u_5
                     P : Type u_6
                     G : Type u_7
                     H : Type u_8
                     F : Type u_9
                     inst✝¹ : One M
                     inst✝ : One N
                     f : OneHom M N
                     g : N → M
                     h₁ : Function.LeftInverse g ⇑f
                     ⊢ Eq (g 1) 1
                   -/
    map_one' := by rw [← f.map_one, h₁] }
                   /-
                     🎉 no goals
                   -/


/-- Makes a multiplicative inverse from a bijection which preserves multiplication. -/
@[to_additive (attr := simps)
  "Makes an additive inverse from a bijection which preserves addition."]
def MulHom.inverse [Mul M] [Mul N] (f : M →ₙ* N) (g : N → M)
    (h₁ : Function.LeftInverse g f)
    (h₂ : Function.RightInverse g f) : N →ₙ* M where
  toFun := g
  map_mul' x y :=
    calc
                                              /-
                                                ι : Type u_1
                                                α : Type u_2
                                                β : Type u_3
                                                M : Type u_4
                                                N : Type u_5
                                                P : Type u_6
                                                G : Type u_7
                                                H : Type u_8
                                                F : Type u_9
                                                inst✝¹ : Mul M
                                                inst✝ : Mul N
                                                f : MulHom M N
                                                g : N → M
                                                h₁ : Function.LeftInverse g ⇑f
                                                h₂ : Function.RightInverse g ⇑f
                                                x y : N
                                                ⊢ Eq (g (HMul.hMul x y)) (g (HMul.hMul (f (g x)) (f (g y))))
                                              -/
      g (x * y) = g (f (g x) * f (g y)) := by rw [h₂ x, h₂ y]
                                              /-
                                                🎉 no goals
                                              -/
                                  /-
                                    ι : Type u_1
                                    α : Type u_2
                                    β : Type u_3
                                    M : Type u_4
                                    N : Type u_5
                                    P : Type u_6
                                    G : Type u_7
                                    H : Type u_8
                                    F : Type u_9
                                    inst✝¹ : Mul M
                                    inst✝ : Mul N
                                    f : MulHom M N
                                    g : N → M
                                    h₁ : Function.LeftInverse g ⇑f
                                    h₂ : Function.RightInverse g ⇑f
                                    x y : N
                                    ⊢ Eq (g (HMul.hMul (f (g x)) (f (g y)))) (g (f (HMul.hMul (g x) (g y))))
                                  -/
      _ = g (f (g x * g y)) := by rw [f.map_mul]
                                  /-
                                    🎉 no goals
                                  -/
      _ = g x * g y := h₁ _


/-- The inverse of a bijective `MonoidHom` is a `MonoidHom`. -/
@[to_additive (attr := simps)
  "The inverse of a bijective `AddMonoidHom` is an `AddMonoidHom`."]
def MonoidHom.inverse {A B : Type*} [Monoid A] [Monoid B] (f : A →* B) (g : B → A)
    (h₁ : Function.LeftInverse g f) (h₂ : Function.RightInverse g f) : B →* A :=
  { (f : OneHom A B).inverse g h₁,
    (f : A →ₙ* B).inverse g h₁ h₂ with toFun := g }


/-- The monoid of endomorphisms. -/
protected def End := M →* M


instance instFunLike : FunLike (Monoid.End M) M M := MonoidHom.instFunLike

instance instMonoidHomClass : MonoidHomClass (Monoid.End M) M M := MonoidHom.instMonoidHomClass


instance instOne : One (Monoid.End M) where one := .id _

instance instMul : Mul (Monoid.End M) where mul := .comp


instance : Monoid (Monoid.End M) where
  mul := MonoidHom.comp
  one := MonoidHom.id M
  mul_assoc _ _ _ := MonoidHom.comp_assoc _ _ _
  mul_one := MonoidHom.comp_id
  one_mul := MonoidHom.id_comp
                                             /-
                                               ι : Type u_1
                                               α : Type u_2
                                               β : Type u_3
                                               M : Type u_4
                                               N : Type u_5
                                               P : Type u_6
                                               G : Type u_7
                                               H : Type u_8
                                               F : Type u_9
                                               inst✝ : MulOneClass M
                                               n : Nat
                                               f : Monoid.End M
                                               ⊢ Eq (Nat.iterate (⇑f) n) ⇑(npowRec n f)
                                             -/
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  npow n f := (npowRec n f).copy f^[n] <| by induction n <;> simp [npowRec, *] <;> rfl
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/
  npow_succ _ _ := DFunLike.coe_injective <| Function.iterate_succ _ _


instance : Inhabited (Monoid.End M) := ⟨1⟩


@[simp, norm_cast] lemma coe_pow (f : Monoid.End M) (n : ℕ) : (↑(f ^ n) : M → M) = f^[n] := rfl


@[simp]
theorem coe_one : ((1 : Monoid.End M) : M → M) = id := rfl


@[simp]
theorem coe_mul (f g) : ((f * g : Monoid.End M) : M → M) = f ∘ g := rfl


/-- The monoid of endomorphisms. -/
protected def End := A →+ A


instance instFunLike : FunLike (AddMonoid.End A) A A := AddMonoidHom.instFunLike

instance instAddMonoidHomClass : AddMonoidHomClass (AddMonoid.End A) A A :=
  AddMonoidHom.instAddMonoidHomClass


instance instOne : One (AddMonoid.End A) where one := .id _

instance instMul : Mul (AddMonoid.End A) where mul := .comp


@[simp, norm_cast] lemma coe_one : ((1 : AddMonoid.End A) : A → A) = id := rfl


@[simp, norm_cast] lemma coe_mul (f g : AddMonoid.End A) : (f * g : A → A) = f ∘ g := rfl


instance monoid : Monoid (AddMonoid.End A) where
  mul_assoc _ _ _ := AddMonoidHom.comp_assoc _ _ _
  mul_one := AddMonoidHom.comp_id
  one_mul := AddMonoidHom.id_comp
                                                         /-
                                                           ι : Type u_1
                                                           α : Type u_2
                                                           β : Type u_3
                                                           M : Type u_4
                                                           N : Type u_5
                                                           P : Type u_6
                                                           G : Type u_7
                                                           H : Type u_8
                                                           F : Type u_9
                                                           A : Type u_10
                                                           inst✝ : AddZeroClass A
                                                           n : Nat
                                                           f : AddMonoid.End A
                                                           ⊢ Eq (Nat.iterate (⇑f) n) ⇑(npowRec n f)
                                                         -/
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/
  npow n f := (npowRec n f).copy (Nat.iterate f n) <| by induction n <;> simp [npowRec, *] <;> rfl
                                                                                               /-
                                                                                                 🎉 no goals
                                                                                               -/
  npow_succ _ _ := DFunLike.coe_injective <| Function.iterate_succ _ _


@[simp, norm_cast] lemma coe_pow (f : AddMonoid.End A) (n : ℕ) : (↑(f ^ n) : A → A) = f^[n] := rfl


instance : Inhabited (AddMonoid.End A) := ⟨1⟩


/-- `1` is the homomorphism sending all elements to `1`. -/
@[to_additive "`0` is the homomorphism sending all elements to `0`."]
instance [One M] [One N] : One (OneHom M N) := ⟨⟨fun _ => 1, rfl⟩⟩


/-- `1` is the multiplicative homomorphism sending all elements to `1`. -/
@[to_additive "`0` is the additive homomorphism sending all elements to `0`"]
instance [Mul M] [MulOneClass N] : One (M →ₙ* N) :=
  ⟨⟨fun _ => 1, fun _ _ => (one_mul 1).symm⟩⟩


/-- `1` is the monoid homomorphism sending all elements to `1`. -/
@[to_additive "`0` is the additive monoid homomorphism sending all elements to `0`."]
instance [MulOneClass M] [MulOneClass N] : One (M →* N) :=
  ⟨⟨⟨fun _ => 1, rfl⟩, fun _ _ => (one_mul 1).symm⟩⟩


@[to_additive (attr := simp)]
theorem OneHom.one_apply [One M] [One N] (x : M) : (1 : OneHom M N) x = 1 := rfl


@[to_additive (attr := simp)]
theorem MonoidHom.one_apply [MulOneClass M] [MulOneClass N] (x : M) : (1 : M →* N) x = 1 := rfl


@[to_additive (attr := simp)]
theorem OneHom.one_comp [One M] [One N] [One P] (f : OneHom M N) :
    (1 : OneHom N P).comp f = 1 := rfl


@[to_additive (attr := simp)]
theorem OneHom.comp_one [One M] [One N] [One P] (f : OneHom N P) : f.comp (1 : OneHom M N) = 1 := by
  /-
    M : Type u_4
    N : Type u_5
    P : Type u_6
    inst✝² : One M
    inst✝¹ : One N
    inst✝ : One P
    f : OneHom N P
    ⊢ Eq (f.comp 1) 1
  -/
  ext
  /-
    case h
    M : Type u_4
    N : Type u_5
    P : Type u_6
    inst✝² : One M
    inst✝¹ : One N
    inst✝ : One P
    f : OneHom N P
    x✝ : M
    ⊢ Eq ((f.comp 1) x✝) (1 x✝)
  -/
  simp only [OneHom.map_one, OneHom.coe_comp, Function.comp_apply, OneHom.one_apply]
  /-
    🎉 no goals
  -/


@[to_additive]
instance [One M] [One N] : Inhabited (OneHom M N) := ⟨1⟩


@[to_additive]
instance [Mul M] [MulOneClass N] : Inhabited (M →ₙ* N) := ⟨1⟩


@[to_additive]
instance [MulOneClass M] [MulOneClass N] : Inhabited (M →* N) := ⟨1⟩


@[to_additive (attr := simp)]
theorem one_comp [MulOneClass M] [MulOneClass N] [MulOneClass P] (f : M →* N) :
    (1 : N →* P).comp f = 1 := rfl


@[to_additive (attr := simp)]
theorem comp_one [MulOneClass M] [MulOneClass N] [MulOneClass P] (f : N →* P) :
    f.comp (1 : M →* N) = 1 := by
  /-
    M : Type u_4
    N : Type u_5
    P : Type u_6
    inst✝² : MulOneClass M
    inst✝¹ : MulOneClass N
    inst✝ : MulOneClass P
    f : MonoidHom N P
    ⊢ Eq (f.comp 1) 1
  -/
  ext
  /-
    case h
    M : Type u_4
    N : Type u_5
    P : Type u_6
    inst✝² : MulOneClass M
    inst✝¹ : MulOneClass N
    inst✝ : MulOneClass P
    f : MonoidHom N P
    x✝ : M
    ⊢ Eq ((f.comp 1) x✝) (1 x✝)
  -/
  simp only [map_one, coe_comp, Function.comp_apply, one_apply]
  /-
    🎉 no goals
  -/


/-- Group homomorphisms preserve inverse. -/
@[to_additive "Additive group homomorphisms preserve negation."]
protected theorem map_inv [Group α] [DivisionMonoid β] (f : α →* β) (a : α) : f a⁻¹ = (f a)⁻¹ :=
  map_inv f _


/-- Group homomorphisms preserve integer power. -/
@[to_additive "Additive group homomorphisms preserve integer scaling."]
protected theorem map_zpow [Group α] [DivisionMonoid β] (f : α →* β) (g : α) (n : ℤ) :
    f (g ^ n) = f g ^ n := map_zpow f g n


/-- Group homomorphisms preserve division. -/
@[to_additive "Additive group homomorphisms preserve subtraction."]
protected theorem map_div [Group α] [DivisionMonoid β] (f : α →* β) (g h : α) :
    f (g / h) = f g / f h := map_div f g h


/-- Group homomorphisms preserve division. -/
@[to_additive "Additive group homomorphisms preserve subtraction."]
protected theorem map_mul_inv [Group α] [DivisionMonoid β] (f : α →* β) (g h : α) :
                                      /-
                                        α : Type u_2
                                        β : Type u_3
                                        inst✝¹ : Group α
                                        inst✝ : DivisionMonoid β
                                        f : MonoidHom α β
                                        g h : α
                                        ⊢ Eq (f (HMul.hMul g (Inv.inv h))) (HMul.hMul (f g) (Inv.inv (f h)))
                                      -/
    f (g * h⁻¹) = f g * (f h)⁻¹ := by simp
                                      /-
                                        🎉 no goals
                                      -/


