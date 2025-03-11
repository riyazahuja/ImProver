/-- Equivariant functions :
When `φ : M → N` is a function, and types `X` and `Y` are endowed with additive actions
of `M` and `N`, a function `f : X → Y` is `φ`-equivariant if `f (m +ᵥ x) = (φ m) +ᵥ (f x)`. -/
structure AddActionHom {M N : Type*} (φ: M → N) (X : Type*) [VAdd M X] (Y : Type*) [VAdd N Y] where
  /-- The underlying function. -/
  protected toFun : X → Y
  /-- The proposition that the function commutes with the additive actions. -/
  protected map_vadd' : ∀ (m : M) (x : X), toFun (m +ᵥ x) = (φ m) +ᵥ toFun x


/-- Equivariant functions :
When `φ : M → N` is a function, and types `X` and `Y` are endowed with actions of `M` and `N`,
a function `f : X → Y` is `φ`-equivariant if `f (m • x) = (φ m) • (f x)`. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
@[to_additive]
structure MulActionHom where
  /-- The underlying function. -/
  protected toFun : X → Y
  /-- The proposition that the function commutes with the actions. -/
  protected map_smul' : ∀ (m : M) (x : X), toFun (m • x) = (φ m) • toFun x

/- Porting note: local notation given a name, conflict with Algebra.Hom.GroupAction
 see https://github.com/leanprover/lean4/issues/2000 -/

/-- `φ`-equivariant functions `X → Y`,
where `φ : M → N`, where `M` and `N` act on `X` and `Y` respectively.-/
notation:25 (name := «MulActionHomLocal≺») X " →ₑ[" φ:25 "] " Y:0 => MulActionHom φ X Y


/-- `M`-equivariant functions `X → Y` with respect to the action of `M`.
This is the same as `X →ₑ[@id M] Y`. -/
notation:25 (name := «MulActionHomIdLocal≺») X " →[" M:25 "] " Y:0 => MulActionHom (@id M) X Y


/-- `φ`-equivariant functions `X → Y`,
where `φ : M → N`, where `M` and `N` act additively on `X` and `Y` respectively

We use the same notation as for multiplicative actions, as conflicts are unlikely. -/
notation:25 (name := «AddActionHomLocal≺») X " →ₑ[" φ:25 "] " Y:0 => AddActionHom φ X Y


/-- `M`-equivariant functions `X → Y` with respect to the additive action of `M`.
This is the same as `X →ₑ[@id M] Y`.

We use the same notation as for multiplicative actions, as conflicts are unlikely. -/
notation:25 (name := «AddActionHomIdLocal≺») X " →[" M:25 "] " Y:0 => AddActionHom (@id M) X Y


/-- `AddActionSemiHomClass F φ X Y` states that
  `F` is a type of morphisms which are `φ`-equivariant.

You should extend this class when you extend `AddActionHom`. -/
class AddActionSemiHomClass (F : Type*)
    {M N : outParam Type*} (φ : outParam (M → N))
    (X Y : outParam Type*) [VAdd M X] [VAdd N Y] [FunLike F X Y] : Prop where
  /-- The proposition that the function preserves the action. -/
  map_vaddₛₗ : ∀ (f : F) (c : M) (x : X), f (c +ᵥ x) = (φ c) +ᵥ (f x)


/-- `MulActionSemiHomClass F φ X Y` states that
  `F` is a type of morphisms which are `φ`-equivariant.

You should extend this class when you extend `MulActionHom`. -/
@[to_additive]
class MulActionSemiHomClass (F : Type*)
    {M N : outParam Type*} (φ : outParam (M → N))
    (X Y : outParam Type*) [SMul M X] [SMul N Y] [FunLike F X Y] : Prop where
  /-- The proposition that the function preserves the action. -/
  map_smulₛₗ : ∀ (f : F) (c : M) (x : X), f (c • x) = (φ c) • (f x)


/-- `MulActionHomClass F M X Y` states that `F` is a type of
morphisms which are equivariant with respect to actions of `M`
This is an abbreviation of `MulActionSemiHomClass`. -/
@[to_additive "`MulActionHomClass F M X Y` states that `F` is a type of
morphisms which are equivariant with respect to actions of `M`
This is an abbreviation of `MulActionSemiHomClass`."]
abbrev MulActionHomClass (F : Type*) (M : outParam Type*)
    (X Y : outParam Type*) [SMul M X] [SMul M Y] [FunLike F X Y] :=
  MulActionSemiHomClass F (@id M) X Y


@[to_additive] instance : FunLike (MulActionHom φ X Y) X Y where
  coe := MulActionHom.toFun
                             /-
                               M' : Type u_1
                               M : Type u_2
                               N : Type u_3
                               P : Type u_4
                               φ : M → N
                               ψ : N → P
                               χ : M → P
                               X : Type u_5
                               inst✝⁴ : SMul M X
                               inst✝³ : SMul M' X
                               Y : Type u_6
                               inst✝² : SMul N Y
                               inst✝¹ : SMul M' Y
                               Z : Type u_7
                               inst✝ : SMul P Z
                               f g : MulActionHom φ X Y
                               h : Eq f.toFun g.toFun
                               ⊢ Eq f g
                             -/
  coe_injective' f g h := by cases f; cases g; congr
                                               /-
                                                 🎉 no goals
                                               -/


@[to_additive (attr := simp)]
theorem map_smul {F M X Y : Type*} [SMul M X] [SMul M Y]
    [FunLike F X Y] [MulActionHomClass F M X Y]
    (f : F) (c : M) (x : X) : f (c • x) = c • f x :=
  map_smulₛₗ f c x


@[to_additive]
instance : MulActionSemiHomClass (X →ₑ[φ] Y) φ X Y where
  map_smulₛₗ := MulActionHom.map_smul'


/-- Turn an element of a type `F` satisfying `MulActionSemiHomClass F φ X Y`
  into an actual `MulActionHom`.
  This is declared as the default coercion from `F` to `MulActionSemiHom φ X Y`. -/
@[to_additive (attr := coe)
  "Turn an element of a type `F` satisfying `AddActionSemiHomClass F φ X Y`
  into an actual `AddActionHom`.
  This is declared as the default coercion from `F` to `AddActionSemiHom φ X Y`."]
def _root_.MulActionSemiHomClass.toMulActionHom [MulActionSemiHomClass F φ X Y] (f : F) :
    X →ₑ[φ] Y where
  toFun := DFunLike.coe f
  map_smul' := map_smulₛₗ f


/-- Any type satisfying `MulActionSemiHomClass` can be cast into `MulActionHom` via
  `MulActionHomSemiClass.toMulActionHom`. -/
@[to_additive]
instance [MulActionSemiHomClass F φ X Y] : CoeTC F (X →ₑ[φ] Y) :=
  ⟨MulActionSemiHomClass.toMulActionHom⟩


variable (M' X Y F) in
/-- If Y/X/M forms a scalar tower, any map X → Y preserving X-action also preserves M-action. -/
@[to_additive]
theorem _root_.IsScalarTower.smulHomClass [MulOneClass X] [SMul X Y] [IsScalarTower M' X Y]
    [MulActionHomClass F X X Y] : MulActionHomClass F M' X Y where
  map_smulₛₗ f m x := by
    rw [← mul_one (m • x), ← smul_eq_mul, map_smul, smul_assoc, ← map_smul,
      smul_eq_mul, mul_one, id_eq]


@[to_additive]
protected theorem map_smul (f : X →[M'] Y) (m : M') (x : X) : f (m • x) = m • f x :=
  map_smul f m x


@[to_additive (attr := ext)]
theorem ext {f g : X →ₑ[φ] Y} :
    (∀ x, f x = g x) → f = g :=
  DFunLike.ext f g


@[to_additive]
protected theorem congr_fun {f g : X →ₑ[φ] Y} (h : f = g) (x : X) :
    f x = g x :=
  DFunLike.congr_fun h _


/-- Two equal maps on scalars give rise to an equivariant map for identity -/
@[to_additive "Two equal maps on scalars give rise to an equivariant map for identity"]
def ofEq {φ' : M → N} (h : φ = φ') (f : X →ₑ[φ] Y) : X →ₑ[φ'] Y where
  toFun := f.toFun
  map_smul' m a := h ▸ f.map_smul' m a


@[to_additive (attr := simp)]
theorem ofEq_coe {φ' : M → N} (h : φ = φ') (f : X →ₑ[φ] Y) :
    (f.ofEq h).toFun = f.toFun := rfl


@[to_additive (attr := simp)]
theorem ofEq_apply {φ' : M → N} (h : φ = φ') (f : X →ₑ[φ] Y) (a : X) :
    (f.ofEq h) a = f a :=
  rfl



/-- The identity map as an equivariant map. -/
@[to_additive "The identity map as an equivariant map."]
protected def id : X →[M] X :=
  ⟨id, fun _ _ => rfl⟩


@[to_additive (attr := simp)]
theorem id_apply (x : X) :
    MulActionHom.id M x = x :=
  rfl


/-- Composition of two equivariant maps. -/
@[to_additive "Composition of two equivariant maps."]
def comp (g : Y →ₑ[ψ] Z) (f : X →ₑ[φ] Y) [κ : CompTriple φ ψ χ] :
    X →ₑ[χ] Z :=
  ⟨g ∘ f, fun m x =>
    calc
                                          /-
                                            M' : Type u_1
                                            M : Type u_2
                                            N : Type u_3
                                            P : Type u_4
                                            φ : M → N
                                            ψ : N → P
                                            χ : M → P
                                            X : Type u_5
                                            inst✝⁴ : SMul M X
                                            inst✝³ : SMul M' X
                                            Y : Type u_6
                                            inst✝² : SMul N Y
                                            inst✝¹ : SMul M' Y
                                            Z : Type u_7
                                            inst✝ : SMul P Z
                                            g : MulActionHom ψ Y Z
                                            f : MulActionHom φ X Y
                                            κ : CompTriple φ ψ χ
                                            m : M
                                            x : X
                                            ⊢ Eq (g (f (HSMul.hSMul m x))) (g (HSMul.hSMul (φ m) (f x)))
                                          -/
      g (f (m • x)) = g (φ m • f x) := by rw [map_smulₛₗ]
                                          /-
                                            🎉 no goals
                                          -/
                                  /-
                                    M' : Type u_1
                                    M : Type u_2
                                    N : Type u_3
                                    P : Type u_4
                                    φ : M → N
                                    ψ : N → P
                                    χ : M → P
                                    X : Type u_5
                                    inst✝⁴ : SMul M X
                                    inst✝³ : SMul M' X
                                    Y : Type u_6
                                    inst✝² : SMul N Y
                                    inst✝¹ : SMul M' Y
                                    Z : Type u_7
                                    inst✝ : SMul P Z
                                    g : MulActionHom ψ Y Z
                                    f : MulActionHom φ X Y
                                    κ : CompTriple φ ψ χ
                                    m : M
                                    x : X
                                    ⊢ Eq (g (HSMul.hSMul (φ m) (f x))) (HSMul.hSMul (ψ (φ m)) (g (f x)))
                                  -/
      _ = ψ (φ m) • g (f x) := by rw [map_smulₛₗ]
                                  /-
                                    🎉 no goals
                                  -/
      _ = (ψ ∘ φ) m • g (f x) := rfl
                              /-
                                M' : Type u_1
                                M : Type u_2
                                N : Type u_3
                                P : Type u_4
                                φ : M → N
                                ψ : N → P
                                χ : M → P
                                X : Type u_5
                                inst✝⁴ : SMul M X
                                inst✝³ : SMul M' X
                                Y : Type u_6
                                inst✝² : SMul N Y
                                inst✝¹ : SMul M' Y
                                Z : Type u_7
                                inst✝ : SMul P Z
                                g : MulActionHom ψ Y Z
                                f : MulActionHom φ X Y
                                κ : CompTriple φ ψ χ
                                m : M
                                x : X
                                ⊢ Eq (HSMul.hSMul (Function.comp ψ φ m) (g (f x))) (HSMul.hSMul (χ m) (g (f x)))
                              -/
      _ = χ m • g (f x) := by rw [κ.comp_eq] ⟩
                              /-
                                🎉 no goals
                              -/


@[to_additive (attr := simp)]
theorem comp_apply
    (g : Y →ₑ[ψ] Z) (f : X →ₑ[φ] Y) [CompTriple φ ψ χ] (x : X) :
    g.comp f x = g (f x) := rfl


@[to_additive (attr := simp)]
theorem id_comp (f : X →ₑ[φ] Y) :
    (MulActionHom.id N).comp f = f :=
                  /-
                    M : Type u_2
                    N : Type u_3
                    φ : M → N
                    X : Type u_5
                    inst✝¹ : SMul M X
                    Y : Type u_6
                    inst✝ : SMul N Y
                    f : MulActionHom φ X Y
                    x : X
                    ⊢ Eq (((MulActionHom.id N).comp f) x) (f x)
                  -/
  ext fun x => by rw [comp_apply, id_apply]
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem comp_id (f : X →ₑ[φ] Y) :
    f.comp (MulActionHom.id M) = f :=
                  /-
                    M : Type u_2
                    N : Type u_3
                    φ : M → N
                    X : Type u_5
                    inst✝¹ : SMul M X
                    Y : Type u_6
                    inst✝ : SMul N Y
                    f : MulActionHom φ X Y
                    x : X
                    ⊢ Eq ((f.comp (MulActionHom.id M)) x) (f x)
                  -/
  ext fun x => by rw [comp_apply, id_apply]
                  /-
                    🎉 no goals
                  -/


@[to_additive (attr := simp)]
theorem comp_assoc {Q T : Type*} [SMul Q T]
    {η : P → Q} {θ : M → Q} {ζ : N → Q}
    (h : Z →ₑ[η] T) (g : Y →ₑ[ψ] Z) (f : X →ₑ[φ] Y)
    [CompTriple φ ψ χ] [CompTriple χ η θ]
    [CompTriple ψ η ζ] [CompTriple φ ζ θ] :
    h.comp (g.comp f) = (h.comp g).comp f :=
  ext fun _ => rfl


/-- The inverse of a bijective equivariant map is equivariant. -/
@[to_additive (attr := simps) "The inverse of a bijective equivariant map is equivariant."]
def inverse (f : X →[M] Y₁) (g : Y₁ → X)
    (h₁ : Function.LeftInverse g f) (h₂ : Function.RightInverse g f) : Y₁ →[M] X where
  toFun := g
  map_smul' m x :=
    calc
                                        /-
                                          M' : Type u_1
                                          M : Type u_2
                                          N : Type u_3
                                          P : Type u_4
                                          φ : M → N
                                          ψ : N → P
                                          χ : M → P
                                          X : Type u_5
                                          inst✝⁵ : SMul M X
                                          inst✝⁴ : SMul M' X
                                          Y : Type u_6
                                          inst✝³ : SMul N Y
                                          inst✝² : SMul M' Y
                                          Z : Type u_7
                                          inst✝¹ : SMul P Z
                                          φ' : N → M
                                          Y₁ : Type u_8
                                          inst✝ : SMul M Y₁
                                          f : MulActionHom id X Y₁
                                          g : Y₁ → X
                                          h₁ : Function.LeftInverse g ⇑f
                                          h₂ : Function.RightInverse g ⇑f
                                          m : M
                                          x : Y₁
                                          ⊢ Eq (g (HSMul.hSMul m x)) (g (HSMul.hSMul m (f (g x))))
                                        -/
      g (m • x) = g (m • f (g x)) := by rw [h₂]
                                        /-
                                          🎉 no goals
                                        -/
                                /-
                                  M' : Type u_1
                                  M : Type u_2
                                  N : Type u_3
                                  P : Type u_4
                                  φ : M → N
                                  ψ : N → P
                                  χ : M → P
                                  X : Type u_5
                                  inst✝⁵ : SMul M X
                                  inst✝⁴ : SMul M' X
                                  Y : Type u_6
                                  inst✝³ : SMul N Y
                                  inst✝² : SMul M' Y
                                  Z : Type u_7
                                  inst✝¹ : SMul P Z
                                  φ' : N → M
                                  Y₁ : Type u_8
                                  inst✝ : SMul M Y₁
                                  f : MulActionHom id X Y₁
                                  g : Y₁ → X
                                  h₁ : Function.LeftInverse g ⇑f
                                  h₂ : Function.RightInverse g ⇑f
                                  m : M
                                  x : Y₁
                                  ⊢ Eq (g (HSMul.hSMul m (f (g x)))) (g (f (HSMul.hSMul m (g x))))
                                -/
      _ = g (f (m • g x)) := by simp only [map_smul, id_eq]
                                /-
                                  🎉 no goals
                                -/
                        /-
                          M' : Type u_1
                          M : Type u_2
                          N : Type u_3
                          P : Type u_4
                          φ : M → N
                          ψ : N → P
                          χ : M → P
                          X : Type u_5
                          inst✝⁵ : SMul M X
                          inst✝⁴ : SMul M' X
                          Y : Type u_6
                          inst✝³ : SMul N Y
                          inst✝² : SMul M' Y
                          Z : Type u_7
                          inst✝¹ : SMul P Z
                          φ' : N → M
                          Y₁ : Type u_8
                          inst✝ : SMul M Y₁
                          f : MulActionHom id X Y₁
                          g : Y₁ → X
                          h₁ : Function.LeftInverse g ⇑f
                          h₂ : Function.RightInverse g ⇑f
                          m : M
                          x : Y₁
                          ⊢ Eq (g (f (HSMul.hSMul m (g x)))) (HSMul.hSMul m (g x))
                        -/
      _ = m • g x := by rw [h₁]
                        /-
                          🎉 no goals
                        -/



/-- The inverse of a bijective equivariant map is equivariant. -/
@[to_additive (attr := simps) "The inverse of a bijective equivariant map is equivariant."]
def inverse' (f : X →ₑ[φ] Y) (g : Y → X) (k : Function.RightInverse φ' φ)
    (h₁ : Function.LeftInverse g f) (h₂ : Function.RightInverse g f) :
    Y →ₑ[φ'] X where
  toFun := g
  map_smul' m x :=
    calc
                                        /-
                                          M' : Type u_1
                                          M : Type u_2
                                          N : Type u_3
                                          P : Type u_4
                                          φ : M → N
                                          ψ : N → P
                                          χ : M → P
                                          X : Type u_5
                                          inst✝⁵ : SMul M X
                                          inst✝⁴ : SMul M' X
                                          Y : Type u_6
                                          inst✝³ : SMul N Y
                                          inst✝² : SMul M' Y
                                          Z : Type u_7
                                          inst✝¹ : SMul P Z
                                          φ' : N → M
                                          Y₁ : Type u_8
                                          inst✝ : SMul M Y₁
                                          f : MulActionHom φ X Y
                                          g : Y → X
                                          k : Function.RightInverse φ' φ
                                          h₁ : Function.LeftInverse g ⇑f
                                          h₂ : Function.RightInverse g ⇑f
                                          m : N
                                          x : Y
                                          ⊢ Eq (g (HSMul.hSMul m x)) (g (HSMul.hSMul m (f (g x))))
                                        -/
      g (m • x) = g (m • f (g x)) := by rw [h₂]
                                        /-
                                          🎉 no goals
                                        -/
                                         /-
                                           M' : Type u_1
                                           M : Type u_2
                                           N : Type u_3
                                           P : Type u_4
                                           φ : M → N
                                           ψ : N → P
                                           χ : M → P
                                           X : Type u_5
                                           inst✝⁵ : SMul M X
                                           inst✝⁴ : SMul M' X
                                           Y : Type u_6
                                           inst✝³ : SMul N Y
                                           inst✝² : SMul M' Y
                                           Z : Type u_7
                                           inst✝¹ : SMul P Z
                                           φ' : N → M
                                           Y₁ : Type u_8
                                           inst✝ : SMul M Y₁
                                           f : MulActionHom φ X Y
                                           g : Y → X
                                           k : Function.RightInverse φ' φ
                                           h₁ : Function.LeftInverse g ⇑f
                                           h₂ : Function.RightInverse g ⇑f
                                           m : N
                                           x : Y
                                           ⊢ Eq (g (HSMul.hSMul m (f (g x)))) (g (HSMul.hSMul (φ (φ' m)) (f (g x))))
                                         -/
      _ = g ((φ (φ' m)) • f (g x)) := by rw [k]
                                         /-
                                           🎉 no goals
                                         -/
                                   /-
                                     M' : Type u_1
                                     M : Type u_2
                                     N : Type u_3
                                     P : Type u_4
                                     φ : M → N
                                     ψ : N → P
                                     χ : M → P
                                     X : Type u_5
                                     inst✝⁵ : SMul M X
                                     inst✝⁴ : SMul M' X
                                     Y : Type u_6
                                     inst✝³ : SMul N Y
                                     inst✝² : SMul M' Y
                                     Z : Type u_7
                                     inst✝¹ : SMul P Z
                                     φ' : N → M
                                     Y₁ : Type u_8
                                     inst✝ : SMul M Y₁
                                     f : MulActionHom φ X Y
                                     g : Y → X
                                     k : Function.RightInverse φ' φ
                                     h₁ : Function.LeftInverse g ⇑f
                                     h₂ : Function.RightInverse g ⇑f
                                     m : N
                                     x : Y
                                     ⊢ Eq (g (HSMul.hSMul (φ (φ' m)) (f (g x)))) (g (f (HSMul.hSMul (φ' m) (g x))))
                                   -/
      _ = g (f (φ' m • g x)) := by rw [map_smulₛₗ]
                                   /-
                                     🎉 no goals
                                   -/
                           /-
                             M' : Type u_1
                             M : Type u_2
                             N : Type u_3
                             P : Type u_4
                             φ : M → N
                             ψ : N → P
                             χ : M → P
                             X : Type u_5
                             inst✝⁵ : SMul M X
                             inst✝⁴ : SMul M' X
                             Y : Type u_6
                             inst✝³ : SMul N Y
                             inst✝² : SMul M' Y
                             Z : Type u_7
                             inst✝¹ : SMul P Z
                             φ' : N → M
                             Y₁ : Type u_8
                             inst✝ : SMul M Y₁
                             f : MulActionHom φ X Y
                             g : Y → X
                             k : Function.RightInverse φ' φ
                             h₁ : Function.LeftInverse g ⇑f
                             h₂ : Function.RightInverse g ⇑f
                             m : N
                             x : Y
                             ⊢ Eq (g (f (HSMul.hSMul (φ' m) (g x)))) (HSMul.hSMul (φ' m) (g x))
                           -/
      _ = φ' m • g x := by rw [h₁]
                           /-
                             🎉 no goals
                           -/


@[to_additive]
lemma inverse_eq_inverse' (f : X →[M] Y₁) (g : Y₁ → X)
    (h₁ : Function.LeftInverse g f) (h₂ : Function.RightInverse g f) :
  inverse f g h₁ h₂ =  inverse' f g (congrFun rfl) h₁ h₂ := by
  /-
    M : Type u_2
    X : Type u_5
    inst✝¹ : SMul M X
    Y₁ : Type u_8
    inst✝ : SMul M Y₁
    f : MulActionHom id X Y₁
    g : Y₁ → X
    h₁ : Function.LeftInverse g ⇑f
    h₂ : Function.RightInverse g ⇑f
    ⊢ Eq (f.inverse g h₁ h₂) (f.inverse' g ⋯ h₁ h₂)
  -/
  rfl
  /-
    🎉 no goals
  -/


@[to_additive]
theorem inverse'_inverse'
    {f : X →ₑ[φ] Y} {g : Y → X}
    {k₁ : Function.LeftInverse φ' φ} {k₂ : Function.RightInverse φ' φ}
    {h₁ : Function.LeftInverse g f} {h₂ : Function.RightInverse g f} :
    inverse' (inverse' f g k₂ h₁ h₂) f k₁ h₂ h₁ = f :=
  ext fun _ => rfl


@[to_additive]
theorem comp_inverse' {f : X →ₑ[φ] Y} {g : Y → X}
    {k₁ : Function.LeftInverse φ' φ} {k₂ : Function.RightInverse φ' φ}
    {h₁ : Function.LeftInverse g f} {h₂ : Function.RightInverse g f} :
    (inverse' f g k₂ h₁ h₂).comp f (κ := CompTriple.comp_inv k₁)
      = MulActionHom.id M := by
  /-
    M : Type u_2
    N : Type u_3
    φ : M → N
    X : Type u_5
    inst✝¹ : SMul M X
    Y : Type u_6
    inst✝ : SMul N Y
    φ' : N → M
    f : MulActionHom φ X Y
    g : Y → X
    k₁ : Function.LeftInverse φ' φ
    k₂ : Function.RightInverse φ' φ
    h₁ : Function.LeftInverse g ⇑f
    h₂ : Function.RightInverse g ⇑f
    ⊢ Eq ((f.inverse' g k₂ h₁ h₂).comp f) (MulActionHom.id M)
  -/
  rw [MulActionHom.ext_iff]
  /-
    M : Type u_2
    N : Type u_3
    φ : M → N
    X : Type u_5
    inst✝¹ : SMul M X
    Y : Type u_6
    inst✝ : SMul N Y
    φ' : N → M
    f : MulActionHom φ X Y
    g : Y → X
    k₁ : Function.LeftInverse φ' φ
    k₂ : Function.RightInverse φ' φ
    h₁ : Function.LeftInverse g ⇑f
    h₂ : Function.RightInverse g ⇑f
    ⊢ ∀ (x : X), Eq (((f.inverse' g k₂ h₁ h₂).comp f) x) ((MulActionHom.id M) x)
  -/
  intro x
  /-
    M : Type u_2
    N : Type u_3
    φ : M → N
    X : Type u_5
    inst✝¹ : SMul M X
    Y : Type u_6
    inst✝ : SMul N Y
    φ' : N → M
    f : MulActionHom φ X Y
    g : Y → X
    k₁ : Function.LeftInverse φ' φ
    k₂ : Function.RightInverse φ' φ
    h₁ : Function.LeftInverse g ⇑f
    h₂ : Function.RightInverse g ⇑f
    x : X
    ⊢ Eq (((f.inverse' g k₂ h₁ h₂).comp f) x) ((MulActionHom.id M) x)
  -/
  simp only [comp_apply, inverse_apply, id_apply]
  /-
    M : Type u_2
    N : Type u_3
    φ : M → N
    X : Type u_5
    inst✝¹ : SMul M X
    Y : Type u_6
    inst✝ : SMul N Y
    φ' : N → M
    f : MulActionHom φ X Y
    g : Y → X
    k₁ : Function.LeftInverse φ' φ
    k₂ : Function.RightInverse φ' φ
    h₁ : Function.LeftInverse g ⇑f
    h₂ : Function.RightInverse g ⇑f
    x : X
    ⊢ Eq ((f.inverse' g k₂ h₁ h₂) (f x)) x
  -/
  exact h₁ x
  /-
    🎉 no goals
  -/


@[to_additive]
theorem inverse'_comp {f : X →ₑ[φ] Y} {g : Y → X}
    {k₂ : Function.RightInverse φ' φ}
    {h₁ : Function.LeftInverse g f} {h₂ : Function.RightInverse g f} :
    f.comp (inverse' f g k₂ h₁ h₂) (κ := CompTriple.comp_inv k₂) = MulActionHom.id N := by
  /-
    M : Type u_2
    N : Type u_3
    φ : M → N
    X : Type u_5
    inst✝¹ : SMul M X
    Y : Type u_6
    inst✝ : SMul N Y
    φ' : N → M
    f : MulActionHom φ X Y
    g : Y → X
    k₂ : Function.RightInverse φ' φ
    h₁ : Function.LeftInverse g ⇑f
    h₂ : Function.RightInverse g ⇑f
    ⊢ Eq (f.comp (f.inverse' g k₂ h₁ h₂)) (MulActionHom.id N)
  -/
  rw [MulActionHom.ext_iff]
  /-
    M : Type u_2
    N : Type u_3
    φ : M → N
    X : Type u_5
    inst✝¹ : SMul M X
    Y : Type u_6
    inst✝ : SMul N Y
    φ' : N → M
    f : MulActionHom φ X Y
    g : Y → X
    k₂ : Function.RightInverse φ' φ
    h₁ : Function.LeftInverse g ⇑f
    h₂ : Function.RightInverse g ⇑f
    ⊢ ∀ (x : Y), Eq ((f.comp (f.inverse' g k₂ h₁ h₂)) x) ((MulActionHom.id N) x)
  -/
  intro x
  /-
    M : Type u_2
    N : Type u_3
    φ : M → N
    X : Type u_5
    inst✝¹ : SMul M X
    Y : Type u_6
    inst✝ : SMul N Y
    φ' : N → M
    f : MulActionHom φ X Y
    g : Y → X
    k₂ : Function.RightInverse φ' φ
    h₁ : Function.LeftInverse g ⇑f
    h₂ : Function.RightInverse g ⇑f
    x : Y
    ⊢ Eq ((f.comp (f.inverse' g k₂ h₁ h₂)) x) ((MulActionHom.id N) x)
  -/
  simp only [comp_apply, inverse_apply, id_apply]
  /-
    M : Type u_2
    N : Type u_3
    φ : M → N
    X : Type u_5
    inst✝¹ : SMul M X
    Y : Type u_6
    inst✝ : SMul N Y
    φ' : N → M
    f : MulActionHom φ X Y
    g : Y → X
    k₂ : Function.RightInverse φ' φ
    h₁ : Function.LeftInverse g ⇑f
    h₂ : Function.RightInverse g ⇑f
    x : Y
    ⊢ Eq (f ((f.inverse' g k₂ h₁ h₂) x)) x
  -/
  exact h₂ x
  /-
    🎉 no goals
  -/


/-- If actions of `M` and `N` on `α` commute,
  then for `c : M`, `(c • · : α → α)` is an `N`-action homomorphism. -/
@[to_additive (attr := simps) "If additive actions of `M` and `N` on `α` commute,
  then for `c : M`, `(c • · : α → α)` is an `N`-additive action homomorphism."]
def _root_.SMulCommClass.toMulActionHom {M} (N α : Type*)
    [SMul M α] [SMul N α] [SMulCommClass M N α] (c : M) :
    α →[N] α where
  toFun := (c • ·)
  map_smul' := smul_comm _


/-- Equivariant additive monoid homomorphisms. -/
structure DistribMulActionHom extends A →ₑ[φ] B, A →+ B


@[inherit_doc]
notation:25 (name := «DistribMulActionHomLocal≺»)
  A " →ₑ+[" φ:25 "] " B:0 => DistribMulActionHom φ A B


@[inherit_doc]
notation:25 (name := «DistribMulActionHomIdLocal≺»)
  A " →+[" M:25 "] " B:0 => DistribMulActionHom (MonoidHom.id M) A B

-- QUESTION/TODO : Impose that `φ` is a morphism of monoids?


/-- `DistribMulActionSemiHomClass F φ A B` states that `F` is a type of morphisms
  preserving the additive monoid structure and equivariant with respect to `φ`.
    You should extend this class when you extend `DistribMulActionSemiHom`. -/
class DistribMulActionSemiHomClass (F : Type*)
    {M N : outParam Type*} (φ : outParam (M → N))
    (A B : outParam Type*)
    [Monoid M] [Monoid N]
    [AddMonoid A] [AddMonoid B] [DistribMulAction M A] [DistribMulAction N B]
    [FunLike F A B]
    extends MulActionSemiHomClass F φ A B, AddMonoidHomClass F A B : Prop


/-- `DistribMulActionHomClass F M A B` states that `F` is a type of morphisms preserving
  the additive monoid structure and equivariant with respect to the action of `M`.
    It is an abbreviation to `DistribMulActionHomClass F (MonoidHom.id M) A B`
You should extend this class when you extend `DistribMulActionHom`. -/
abbrev DistribMulActionHomClass (F : Type*) (M : outParam Type*)
    (A B : outParam Type*) [Monoid M] [AddMonoid A] [AddMonoid B]
    [DistribMulAction M A] [DistribMulAction M B] [FunLike F A B] :=
    DistribMulActionSemiHomClass F (MonoidHom.id M) A B


instance : FunLike (A →ₑ+[φ] B) A B where
  coe m := m.toFun
  coe_injective' f g h := by
    /-
      M : Type u_1
      inst✝¹⁴ : Monoid M
      N : Type u_2
      inst✝¹³ : Monoid N
      P : Type u_3
      inst✝¹² : Monoid P
      φ : MonoidHom M N
      φ' : MonoidHom N M
      ψ : MonoidHom N P
      χ : MonoidHom M P
      A : Type u_4
      inst✝¹¹ : AddMonoid A
      inst✝¹⁰ : DistribMulAction M A
      B : Type u_5
      inst✝⁹ : AddMonoid B
      inst✝⁸ : DistribMulAction N B
      B₁ : Type u_6
      inst✝⁷ : AddMonoid B₁
      inst✝⁶ : DistribMulAction M B₁
      C : Type u_7
      inst✝⁵ : AddMonoid C
      inst✝⁴ : DistribMulAction P C
      A' : Type u_8
      inst✝³ : AddGroup A'
      inst✝² : DistribMulAction M A'
      B' : Type u_9
      inst✝¹ : AddGroup B'
      inst✝ : DistribMulAction N B'
      f g : DistribMulActionHom φ A B
      h : Eq ((fun m => m.toFun) f) ((fun m => m.toFun) g)
      ⊢ Eq f g
    -/
    rcases f with ⟨tF, _, _⟩; rcases g with ⟨tG, _, _⟩
    /-
      case mk.mk
      M : Type u_1
      inst✝¹⁴ : Monoid M
      N : Type u_2
      inst✝¹³ : Monoid N
      P : Type u_3
      inst✝¹² : Monoid P
      φ : MonoidHom M N
      φ' : MonoidHom N M
      ψ : MonoidHom N P
      χ : MonoidHom M P
      A : Type u_4
      inst✝¹¹ : AddMonoid A
      inst✝¹⁰ : DistribMulAction M A
      B : Type u_5
      inst✝⁹ : AddMonoid B
      inst✝⁸ : DistribMulAction N B
      B₁ : Type u_6
      inst✝⁷ : AddMonoid B₁
      inst✝⁶ : DistribMulAction M B₁
      C : Type u_7
      inst✝⁵ : AddMonoid C
      inst✝⁴ : DistribMulAction P C
      A' : Type u_8
      inst✝³ : AddGroup A'
      inst✝² : DistribMulAction M A'
      B' : Type u_9
      inst✝¹ : AddGroup B'
      inst✝ : DistribMulAction N B'
      tF : MulActionHom (⇑φ) A B
      map_zero'✝¹ : Eq (tF.toFun 0) 0
      map_add'✝¹ : ∀ (x y : A), Eq (tF.toFun (HAdd.hAdd x y)) (HAdd.hAdd (tF.toFun x …
      tG : MulActionHom (⇑φ) A B
      map_zero'✝ : Eq (tG.toFun 0) 0
      map_add'✝ : ∀ (x y : A), Eq (tG.toFun (HAdd.hAdd x y)) (HAdd.hAdd (tG.toFun x) …
      h : Eq ((fun m => m.toFun) { toMulActionHom := tF, map_zero' := map_zero'✝¹, m …
      ⊢ Eq { toMulActionHom := tF, map_zero' := map_zero'✝¹, map_add' := map_add'✝¹  …
    -/
    cases tF; cases tG; congr
                        /-
                          🎉 no goals
                        -/


instance : DistribMulActionSemiHomClass (A →ₑ+[φ] B) φ A B where
  map_smulₛₗ m := m.map_smul'
  map_zero := DistribMulActionHom.map_zero'
  map_add := DistribMulActionHom.map_add'


/-- Turn an element of a type `F` satisfying `MulActionHomClass F M X Y` into an actual
`MulActionHom`. This is declared as the default coercion from `F` to `MulActionHom M X Y`. -/
@[coe]
def _root_.DistribMulActionSemiHomClass.toDistribMulActionHom
    [DistribMulActionSemiHomClass F φ A B]
    (f : F) : A →ₑ+[φ] B :=
  { (f : A →+ B),  (f : A →ₑ[φ] B) with }


/-- Any type satisfying `MulActionHomClass` can be cast into `MulActionHom`
via `MulActionHomClass.toMulActionHom`. -/
instance [DistribMulActionSemiHomClass F φ A B] :
  CoeTC F (A →ₑ+[φ] B) :=
  ⟨DistribMulActionSemiHomClass.toDistribMulActionHom⟩


/-- If `DistribMulAction` of `M` and `N` on `A` commute,
  then for each `c : M`, `(c • ·)` is an `N`-action additive homomorphism. -/
@[simps]
def _root_.SMulCommClass.toDistribMulActionHom {M} (N A : Type*) [Monoid N] [AddMonoid A]
    [DistribSMul M A] [DistribMulAction N A] [SMulCommClass M N A] (c : M) : A →+[N] A :=
  { SMulCommClass.toMulActionHom N A c,
    DistribSMul.toAddMonoidHom _ c with
    toFun := (c • ·) }


@[simp]
theorem toFun_eq_coe (f : A →ₑ+[φ] B) : f.toFun = f := rfl


@[norm_cast]
theorem coe_fn_coe (f : A →ₑ+[φ] B) : ⇑(f : A →+ B) = f :=
  rfl


@[norm_cast]
theorem coe_fn_coe' (f : A →ₑ+[φ] B) : ⇑(f : A →ₑ[φ] B) = f :=
  rfl


@[ext]
theorem ext {f g : A →ₑ+[φ] B} : (∀ x, f x = g x) → f = g :=
  DFunLike.ext f g


protected theorem congr_fun {f g : A →ₑ+[φ] B} (h : f = g) (x : A) : f x = g x :=
  DFunLike.congr_fun h _


theorem toMulActionHom_injective {f g : A →ₑ+[φ] B} (h : (f : A →ₑ[φ] B) = (g : A →ₑ[φ] B)) :
    f = g := by
  /-
    M : Type u_1
    inst✝⁵ : Monoid M
    N : Type u_2
    inst✝⁴ : Monoid N
    φ : MonoidHom M N
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : DistribMulAction M A
    B : Type u_5
    inst✝¹ : AddMonoid B
    inst✝ : DistribMulAction N B
    f g : DistribMulActionHom φ A B
    h : Eq ↑f ↑g
    ⊢ Eq f g
  -/
  ext a
  /-
    case a
    M : Type u_1
    inst✝⁵ : Monoid M
    N : Type u_2
    inst✝⁴ : Monoid N
    φ : MonoidHom M N
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : DistribMulAction M A
    B : Type u_5
    inst✝¹ : AddMonoid B
    inst✝ : DistribMulAction N B
    f g : DistribMulActionHom φ A B
    h : Eq ↑f ↑g
    a : A
    ⊢ Eq (f a) (g a)
  -/
  exact MulActionHom.congr_fun h a
  /-
    🎉 no goals
  -/


theorem toAddMonoidHom_injective {f g : A →ₑ+[φ] B} (h : (f : A →+ B) = (g : A →+ B)) : f = g := by
  /-
    M : Type u_1
    inst✝⁵ : Monoid M
    N : Type u_2
    inst✝⁴ : Monoid N
    φ : MonoidHom M N
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : DistribMulAction M A
    B : Type u_5
    inst✝¹ : AddMonoid B
    inst✝ : DistribMulAction N B
    f g : DistribMulActionHom φ A B
    h : Eq ↑f ↑g
    ⊢ Eq f g
  -/
  ext a
  /-
    case a
    M : Type u_1
    inst✝⁵ : Monoid M
    N : Type u_2
    inst✝⁴ : Monoid N
    φ : MonoidHom M N
    A : Type u_4
    inst✝³ : AddMonoid A
    inst✝² : DistribMulAction M A
    B : Type u_5
    inst✝¹ : AddMonoid B
    inst✝ : DistribMulAction N B
    f g : DistribMulActionHom φ A B
    h : Eq ↑f ↑g
    a : A
    ⊢ Eq (f a) (g a)
  -/
  exact DFunLike.congr_fun h a
  /-
    🎉 no goals
  -/


protected theorem map_zero (f : A →ₑ+[φ] B) : f 0 = 0 :=
  map_zero f


protected theorem map_add (f : A →ₑ+[φ] B) (x y : A) : f (x + y) = f x + f y :=
  map_add f x y


protected theorem map_neg (f : A' →ₑ+[φ] B') (x : A') : f (-x) = -f x :=
  map_neg f x


protected theorem map_sub (f : A' →ₑ+[φ] B') (x y : A') : f (x - y) = f x - f y :=
  map_sub f x y


protected theorem map_smulₑ (f : A →ₑ+[φ] B) (m : M) (x : A) : f (m • x) = (φ m) • f x :=
  map_smulₛₗ f m x


/-- The identity map as an equivariant additive monoid homomorphism. -/
protected def id : A →+[M] A :=
  ⟨MulActionHom.id _, rfl, fun _ _ => rfl⟩


@[simp]
theorem id_apply (x : A) : DistribMulActionHom.id M x = x := by
  /-
    M : Type u_1
    inst✝² : Monoid M
    A : Type u_4
    inst✝¹ : AddMonoid A
    inst✝ : DistribMulAction M A
    x : A
    ⊢ Eq ((DistribMulActionHom.id M) x) x
  -/
  rfl
  /-
    🎉 no goals
  -/


instance : Zero (A →ₑ+[φ] B) :=
                                                  /-
                                                    M : Type u_1
                                                    inst✝¹⁵ : Monoid M
                                                    N : Type u_2
                                                    inst✝¹⁴ : Monoid N
                                                    P : Type u_3
                                                    inst✝¹³ : Monoid P
                                                    φ : MonoidHom M N
                                                    φ' : MonoidHom N M
                                                    ψ : MonoidHom N P
                                                    χ : MonoidHom M P
                                                    A : Type u_4
                                                    inst✝¹² : AddMonoid A
                                                    inst✝¹¹ : DistribMulAction M A
                                                    B : Type u_5
                                                    inst✝¹⁰ : AddMonoid B
                                                    inst✝⁹ : DistribMulAction N B
                                                    B₁ : Type u_6
                                                    inst✝⁸ : AddMonoid B₁
                                                    inst✝⁷ : DistribMulAction M B₁
                                                    C : Type u_7
                                                    inst✝⁶ : AddMonoid C
                                                    inst✝⁵ : DistribMulAction P C
                                                    A' : Type u_8
                                                    inst✝⁴ : AddGroup A'
                                                    inst✝³ : DistribMulAction M A'
                                                    B' : Type u_9
                                                    inst✝² : AddGroup B'
                                                    inst✝¹ : DistribMulAction N B'
                                                    F : Type u_10
                                                    inst✝ : FunLike F A B
                                                    m : M
                                                    x✝ : A
                                                    ⊢ Eq ((↑__src✝).toFun (HSMul.hSMul m x✝)) (HSMul.hSMul (φ m) ((↑__src✝).toFun  …
                                                  -/
  ⟨{ (0 : A →+ B) with map_smul' := fun m _ => by simp }⟩
                                                  /-
                                                    🎉 no goals
                                                  -/


instance : One (A →+[M] A) :=
  ⟨DistribMulActionHom.id M⟩


@[simp]
theorem coe_zero : ⇑(0 : A →ₑ+[φ] B) = 0 :=
  rfl


@[simp]
theorem coe_one : ⇑(1 : A →+[M] A) = id :=
  rfl


theorem zero_apply (a : A) : (0 : A →ₑ+[φ] B) a = 0 :=
  rfl


theorem one_apply (a : A) : (1 : A →+[M] A) a = a :=
  rfl


instance : Inhabited (A →ₑ+[φ] B) :=
  ⟨0⟩


set_option linter.unusedVariables false in
/-- Composition of two equivariant additive monoid homomorphisms. -/
def comp (g : B →ₑ+[ψ] C) (f : A →ₑ+[φ] B) [κ : MonoidHom.CompTriple φ ψ χ] :
    A →ₑ+[χ] C :=
  { MulActionHom.comp (g : B →ₑ[ψ] C) (f : A →ₑ[φ] B),
    AddMonoidHom.comp (g : B →+ C) (f : A →+ B) with }


@[simp]
theorem comp_apply
    (g : B →ₑ+[ψ] C) (f : A →ₑ+[φ] B) [MonoidHom.CompTriple φ ψ χ] (x : A) : g.comp f x = g (f x) :=
  rfl


@[simp]
theorem id_comp (f : A →ₑ+[φ] B) : comp (DistribMulActionHom.id N) f = f :=
                  /-
                    M : Type u_1
                    inst✝⁵ : Monoid M
                    N : Type u_2
                    inst✝⁴ : Monoid N
                    φ : MonoidHom M N
                    A : Type u_4
                    inst✝³ : AddMonoid A
                    inst✝² : DistribMulAction M A
                    B : Type u_5
                    inst✝¹ : AddMonoid B
                    inst✝ : DistribMulAction N B
                    f : DistribMulActionHom φ A B
                    x : A
                    ⊢ Eq (((DistribMulActionHom.id N).comp f) x) (f x)
                  -/
  ext fun x => by rw [comp_apply, id_apply]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem comp_id (f : A →ₑ+[φ] B) : f.comp (DistribMulActionHom.id M) = f :=
                  /-
                    M : Type u_1
                    inst✝⁵ : Monoid M
                    N : Type u_2
                    inst✝⁴ : Monoid N
                    φ : MonoidHom M N
                    A : Type u_4
                    inst✝³ : AddMonoid A
                    inst✝² : DistribMulAction M A
                    B : Type u_5
                    inst✝¹ : AddMonoid B
                    inst✝ : DistribMulAction N B
                    f : DistribMulActionHom φ A B
                    x : A
                    ⊢ Eq ((f.comp (DistribMulActionHom.id M)) x) (f x)
                  -/
  ext fun x => by rw [comp_apply, id_apply]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem comp_assoc {Q D : Type*} [Monoid Q] [AddMonoid D] [DistribMulAction Q D]
    {η : P →* Q} {θ : M →* Q} {ζ : N →* Q}
    (h : C →ₑ+[η] D) (g : B →ₑ+[ψ] C) (f : A →ₑ+[φ] B)
    [MonoidHom.CompTriple φ ψ χ] [MonoidHom.CompTriple χ η θ]
    [MonoidHom.CompTriple ψ η ζ] [MonoidHom.CompTriple φ ζ θ] :
    h.comp (g.comp f) = (h.comp g).comp f :=
  ext fun _ => rfl


/-- The inverse of a bijective `DistribMulActionHom` is a `DistribMulActionHom`. -/
@[simps]
def inverse (f : A →+[M] B₁) (g : B₁ → A) (h₁ : Function.LeftInverse g f)
    (h₂ : Function.RightInverse g f) : B₁ →+[M] A :=
  { (f : A →+ B₁).inverse g h₁ h₂, f.toMulActionHom.inverse g h₁ h₂ with toFun := g }


@[ext]
theorem ext_ring {f g : R →ₑ+[σ] N'} (h : f 1 = g 1) : f = g := by
  /-
    R : Type u_11
    inst✝³ : Semiring R
    S : Type u_12
    inst✝² : Semiring S
    N' : Type u_14
    inst✝¹ : AddMonoid N'
    inst✝ : DistribMulAction S N'
    σ : MonoidHom R S
    f g : DistribMulActionHom σ R N'
    h : Eq (f 1) (g 1)
    ⊢ Eq f g
  -/
  ext x
  /-
    case a
    R : Type u_11
    inst✝³ : Semiring R
    S : Type u_12
    inst✝² : Semiring S
    N' : Type u_14
    inst✝¹ : AddMonoid N'
    inst✝ : DistribMulAction S N'
    σ : MonoidHom R S
    f g : DistribMulActionHom σ R N'
    h : Eq (f 1) (g 1)
    x : R
    ⊢ Eq (f x) (g x)
  -/
  rw [← mul_one x, ← smul_eq_mul R, f.map_smulₑ, g.map_smulₑ, h]
  /-
    🎉 no goals
  -/



/-- Equivariant ring homomorphisms. -/
-- Porting note (https://github.com/leanprover-community/mathlib4/issues/5171): this linter isn't ported yet.
-- @[nolint has_nonempty_instance]
structure MulSemiringActionHom extends R →ₑ+[φ] S, R →+* S

/-
/-- Equivariant ring homomorphism -/
abbrev MulSemiringActionHom
  (M : Type*) [Monoid M]
  (R : Type*) [Semiring R] [MulSemiringAction M R]
  (S : Type*) [Semiring S] [MulSemiringAction M S]:= MulSemiringActionHom (MonoidHom.id M) R S
-/


@[inherit_doc]
notation:25 (name := «MulSemiringActionHomLocal≺»)
  R " →ₑ+*[" φ:25 "] " S:0 => MulSemiringActionHom φ R S


@[inherit_doc]
notation:25 (name := «MulSemiringActionHomIdLocal≺»)
  R " →+*[" M:25 "] " S:0 => MulSemiringActionHom (MonoidHom.id M) R S


/-- `MulSemiringActionHomClass F φ R S` states that `F` is a type of morphisms preserving
the ring structure and equivariant with respect to `φ`.

You should extend this class when you extend `MulSemiringActionHom`. -/
class MulSemiringActionSemiHomClass (F : Type*)
    {M N : outParam Type*} [Monoid M] [Monoid N]
    (φ : outParam (M → N))
    (R S : outParam Type*) [Semiring R] [Semiring S]
    [DistribMulAction M R] [DistribMulAction N S] [FunLike F R S]
    extends DistribMulActionSemiHomClass F φ R S, RingHomClass F R S : Prop


/-- `MulSemiringActionHomClass F M R S` states that `F` is a type of morphisms preserving
the ring structure and equivariant with respect to a `DistribMulAction`of `M` on `R` and `S` .
 -/
abbrev MulSemiringActionHomClass
    (F : Type*)
    {M : outParam Type*} [Monoid M]
    (R S : outParam Type*) [Semiring R] [Semiring S]
    [DistribMulAction M R] [DistribMulAction M S] [FunLike F R S] :=
  MulSemiringActionSemiHomClass F (MonoidHom.id M) R S


instance : FunLike (R →ₑ+*[φ] S) R S where
  coe m := m.toFun
  coe_injective' f g h := by
    /-
      M : Type u_1
      inst✝²⁴ : Monoid M
      N : Type u_2
      inst✝²³ : Monoid N
      P : Type u_3
      inst✝²² : Monoid P
      φ : MonoidHom M N
      φ' : MonoidHom N M
      ψ : MonoidHom N P
      χ : MonoidHom M P
      A : Type u_4
      inst✝²¹ : AddMonoid A
      inst✝²⁰ : DistribMulAction M A
      B : Type u_5
      inst✝¹⁹ : AddMonoid B
      inst✝¹⁸ : DistribMulAction N B
      B₁ : Type u_6
      inst✝¹⁷ : AddMonoid B₁
      inst✝¹⁶ : DistribMulAction M B₁
      C : Type u_7
      inst✝¹⁵ : AddMonoid C
      inst✝¹⁴ : DistribMulAction P C
      A' : Type u_8
      inst✝¹³ : AddGroup A'
      inst✝¹² : DistribMulAction M A'
      B' : Type u_9
      inst✝¹¹ : AddGroup B'
      inst✝¹⁰ : DistribMulAction N B'
      R : Type u_10
      inst✝⁹ : Semiring R
      inst✝⁸ : MulSemiringAction M R
      R' : Type u_11
      inst✝⁷ : Ring R'
      inst✝⁶ : MulSemiringAction M R'
      S : Type u_12
      inst✝⁵ : Semiring S
      inst✝⁴ : MulSemiringAction N S
      S' : Type u_13
      inst✝³ : Ring S'
      inst✝² : MulSemiringAction N S'
      T : Type u_14
      inst✝¹ : Semiring T
      inst✝ : MulSemiringAction P T
      f g : MulSemiringActionHom φ R S
      h : Eq ((fun m => m.toFun) f) ((fun m => m.toFun) g)
      ⊢ Eq f g
    -/
    rcases f with ⟨⟨tF, _, _⟩, _, _⟩; rcases g with ⟨⟨tG, _, _⟩, _, _⟩
    /-
      case mk.mk.mk.mk
      M : Type u_1
      inst✝²⁴ : Monoid M
      N : Type u_2
      inst✝²³ : Monoid N
      P : Type u_3
      inst✝²² : Monoid P
      φ : MonoidHom M N
      φ' : MonoidHom N M
      ψ : MonoidHom N P
      χ : MonoidHom M P
      A : Type u_4
      inst✝²¹ : AddMonoid A
      inst✝²⁰ : DistribMulAction M A
      B : Type u_5
      inst✝¹⁹ : AddMonoid B
      inst✝¹⁸ : DistribMulAction N B
      B₁ : Type u_6
      inst✝¹⁷ : AddMonoid B₁
      inst✝¹⁶ : DistribMulAction M B₁
      C : Type u_7
      inst✝¹⁵ : AddMonoid C
      inst✝¹⁴ : DistribMulAction P C
      A' : Type u_8
      inst✝¹³ : AddGroup A'
      inst✝¹² : DistribMulAction M A'
      B' : Type u_9
      inst✝¹¹ : AddGroup B'
      inst✝¹⁰ : DistribMulAction N B'
      R : Type u_10
      inst✝⁹ : Semiring R
      inst✝⁸ : MulSemiringAction M R
      R' : Type u_11
      inst✝⁷ : Ring R'
      inst✝⁶ : MulSemiringAction M R'
      S : Type u_12
      inst✝⁵ : Semiring S
      inst✝⁴ : MulSemiringAction N S
      S' : Type u_13
      inst✝³ : Ring S'
      inst✝² : MulSemiringAction N S'
      T : Type u_14
      inst✝¹ : Semiring T
      inst✝ : MulSemiringAction P T
      tF : MulActionHom (⇑φ) R S
      map_zero'✝¹ : Eq (tF.toFun 0) 0
      map_add'✝¹ : ∀ (x y : R), Eq (tF.toFun (HAdd.hAdd x y)) (HAdd.hAdd (tF.toFun x …
      map_one'✝¹ : Eq ({ toMulActionHom := tF, map_zero' := map_zero'✝¹, map_add' := …
      map_mul'✝¹ : ∀ (x y : R), Eq ({ toMulActionHom := tF, map_zero' := map_zero'✝¹ …
      tG : MulActionHom (⇑φ) R S
      map_zero'✝ : Eq (tG.toFun 0) 0
      map_add'✝ : ∀ (x y : R), Eq (tG.toFun (HAdd.hAdd x y)) (HAdd.hAdd (tG.toFun x) …
      map_one'✝ : Eq ({ toMulActionHom := tG, map_zero' := map_zero'✝, map_add' := m …
      map_mul'✝ : ∀ (x y : R), Eq ({ toMulActionHom := tG, map_zero' := map_zero'✝,  …
      h : Eq ((fun m => m.toFun) { toMulActionHom := tF, map_zero' := map_zero'✝¹, m …
      ⊢ Eq { toMulActionHom := tF, map_zero' := map_zero'✝¹, map_add' := map_add'✝¹, …
    -/
    cases tF; cases tG; congr
                        /-
                          🎉 no goals
                        -/


instance : MulSemiringActionSemiHomClass (R →ₑ+*[φ] S) φ R S where
  map_zero m := m.map_zero'
  map_add m := m.map_add'
  map_one := MulSemiringActionHom.map_one'
  map_mul := MulSemiringActionHom.map_mul'
  map_smulₛₗ m := m.map_smul'


/-- Turn an element of a type `F` satisfying `MulSemiringActionHomClass F M R S` into an actual
`MulSemiringActionHom`. This is declared as the default coercion from `F` to
`MulSemiringActionHom M X Y`. -/
@[coe]
def _root_.MulSemiringActionHomClass.toMulSemiringActionHom
    [MulSemiringActionSemiHomClass F φ R S]
    (f : F) : R →ₑ+*[φ] S :=
 { (f : R →+* S),  (f : R →ₑ+[φ] S) with }


/-- Any type satisfying `MulSemiringActionHomClass` can be cast into `MulSemiringActionHom` via
  `MulSemiringActionHomClass.toMulSemiringActionHom`. -/
instance [MulSemiringActionSemiHomClass F φ R S] :
    CoeTC F (R →ₑ+*[φ] S) :=
  ⟨MulSemiringActionHomClass.toMulSemiringActionHom⟩


@[norm_cast]
theorem coe_fn_coe (f : R →ₑ+*[φ] S) : ⇑(f : R →+* S) = f :=
  rfl


@[norm_cast]
theorem coe_fn_coe' (f : R →ₑ+*[φ] S) : ⇑(f : R →ₑ+[φ] S) = f :=
  rfl


@[ext]
theorem ext {f g : R →ₑ+*[φ] S} : (∀ x, f x = g x) → f = g :=
  DFunLike.ext f g


protected theorem map_zero (f : R →ₑ+*[φ] S) : f 0 = 0 :=
  map_zero f


protected theorem map_add (f : R →ₑ+*[φ] S) (x y : R) : f (x + y) = f x + f y :=
  map_add f x y


protected theorem map_neg (f : R' →ₑ+*[φ] S') (x : R') : f (-x) = -f x :=
  map_neg f x


protected theorem map_sub (f : R' →ₑ+*[φ] S') (x y : R') : f (x - y) = f x - f y :=
  map_sub f x y


protected theorem map_one (f : R →ₑ+*[φ] S) : f 1 = 1 :=
  map_one f


protected theorem map_mul (f : R →ₑ+*[φ] S) (x y : R) : f (x * y) = f x * f y :=
  map_mul f x y


protected theorem map_smulₛₗ (f : R →ₑ+*[φ] S) (m : M) (x : R) : f (m • x) = φ m • f x :=
  map_smulₛₗ f m x


protected theorem map_smul [MulSemiringAction M S] (f : R →+*[M] S) (m : M) (x : R) :
    f (m • x) = m • f x :=
  map_smulₛₗ f m x


/-- The identity map as an equivariant ring homomorphism. -/
protected def id : R →+*[M] R :=
  ⟨DistribMulActionHom.id _, rfl, (fun _ _ => rfl)⟩


@[simp]
theorem id_apply (x : R) : MulSemiringActionHom.id M x = x :=
  rfl



set_option linter.unusedVariables false in
/-- Composition of two equivariant additive ring homomorphisms. -/
def comp (g : S →ₑ+*[ψ] T) (f : R →ₑ+*[φ] S) [κ : MonoidHom.CompTriple φ ψ χ] : R →ₑ+*[χ] T :=
  { DistribMulActionHom.comp (g : S →ₑ+[ψ] T) (f : R →ₑ+[φ] S),
    RingHom.comp (g : S →+* T) (f : R →+* S) with }


@[simp]
theorem comp_apply (g : S →ₑ+*[ψ] T) (f : R →ₑ+*[φ] S) [MonoidHom.CompTriple φ ψ χ] (x : R) :
    g.comp f x = g (f x) := rfl


@[simp]
theorem id_comp (f : R →ₑ+*[φ] S) : (MulSemiringActionHom.id N).comp f = f :=
                  /-
                    M : Type u_1
                    inst✝⁵ : Monoid M
                    N : Type u_2
                    inst✝⁴ : Monoid N
                    φ : MonoidHom M N
                    R : Type u_10
                    inst✝³ : Semiring R
                    inst✝² : MulSemiringAction M R
                    S : Type u_12
                    inst✝¹ : Semiring S
                    inst✝ : MulSemiringAction N S
                    f : MulSemiringActionHom φ R S
                    x : R
                    ⊢ Eq (((MulSemiringActionHom.id N).comp f) x) (f x)
                  -/
  ext fun x => by rw [comp_apply, id_apply]
                  /-
                    🎉 no goals
                  -/


@[simp]
theorem comp_id (f : R →ₑ+*[φ] S) : f.comp (MulSemiringActionHom.id M) = f :=
                  /-
                    M : Type u_1
                    inst✝⁵ : Monoid M
                    N : Type u_2
                    inst✝⁴ : Monoid N
                    φ : MonoidHom M N
                    R : Type u_10
                    inst✝³ : Semiring R
                    inst✝² : MulSemiringAction M R
                    S : Type u_12
                    inst✝¹ : Semiring S
                    inst✝ : MulSemiringAction N S
                    f : MulSemiringActionHom φ R S
                    x : R
                    ⊢ Eq ((f.comp (MulSemiringActionHom.id M)) x) (f x)
                  -/
  ext fun x => by rw [comp_apply, id_apply]
                  /-
                    🎉 no goals
                  -/


/-- The inverse of a bijective `MulSemiringActionHom` is a `MulSemiringActionHom`. -/
@[simps]
def inverse' (f : R →ₑ+*[φ] S) (g : S → R) (k : Function.RightInverse φ' φ)
    (h₁ : Function.LeftInverse g f) (h₂ : Function.RightInverse g f) :
    S →ₑ+*[φ'] R :=
  { (f : R →+ S).inverse g h₁ h₂,
    (f : R →* S).inverse g h₁ h₂,
    (f : R →ₑ[φ] S).inverse' g k h₁ h₂ with
    toFun := g }


/-- The inverse of a bijective `MulSemiringActionHom` is a `MulSemiringActionHom`. -/
@[simps]
def inverse {S₁ : Type*} [Semiring S₁] [MulSemiringAction M S₁]
    (f : R →+*[M] S₁) (g : S₁ → R)
    (h₁ : Function.LeftInverse g f) (h₂ : Function.RightInverse g f) :
    S₁ →+*[M] R :=
  { (f : R →+ S₁).inverse g h₁ h₂,
    (f : R →* S₁).inverse g h₁ h₂,
    f.toMulActionHom.inverse g h₁ h₂ with
    toFun := g }


