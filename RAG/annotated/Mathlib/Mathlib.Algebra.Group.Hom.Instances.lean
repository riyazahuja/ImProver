/-- `(M →* N)` is a `CommMonoid` if `N` is commutative. -/
@[to_additive "`(M →+ N)` is an `AddCommMonoid` if `N` is commutative."]
instance MonoidHom.commMonoid [MulOneClass M] [CommMonoid N] :
    CommMonoid (M →* N) where
  mul := (· * ·)
                  /-
                    M : Type uM
                    N : Type uN
                    P : Type uP
                    Q : Type uQ
                    inst✝¹ : MulOneClass M
                    inst✝ : CommMonoid N
                    ⊢ ∀ (a b c : MonoidHom M N), Eq (HMul.hMul (HMul.hMul a b) c) (HMul.hMul a (HM …
                  -/
  mul_assoc := by intros; ext; apply mul_assoc
                               /-
                                 🎉 no goals
                               -/
  one := 1
                /-
                  M : Type uM
                  N : Type uN
                  P : Type uP
                  Q : Type uQ
                  inst✝¹ : MulOneClass M
                  inst✝ : CommMonoid N
                  ⊢ ∀ (a : MonoidHom M N), Eq (HMul.hMul 1 a) a
                -/
  one_mul := by intros; ext; apply one_mul
                             /-
                               🎉 no goals
                             -/
                /-
                  M : Type uM
                  N : Type uN
                  P : Type uP
                  Q : Type uQ
                  inst✝¹ : MulOneClass M
                  inst✝ : CommMonoid N
                  ⊢ ∀ (a : MonoidHom M N), Eq (HMul.hMul a 1) a
                -/
  mul_one := by intros; ext; apply mul_one
                             /-
                               🎉 no goals
                             -/
                 /-
                   M : Type uM
                   N : Type uN
                   P : Type uP
                   Q : Type uQ
                   inst✝¹ : MulOneClass M
                   inst✝ : CommMonoid N
                   ⊢ ∀ (a b : MonoidHom M N), Eq (HMul.hMul a b) (HMul.hMul b a)
                 -/
  mul_comm := by intros; ext; apply mul_comm
                                                /-
                                                  M : Type uM
                                                  N : Type uN
                                                  P : Type uP
                                                  Q : Type uQ
                                                  inst✝¹ : MulOneClass M
                                                  inst✝ : CommMonoid N
                                                  n : Nat
                                                  f : MonoidHom M N
                                                  ⊢ Eq ((fun x => HPow.hPow (f x) n) 1) 1
                                                -/
                                                /-
                                                  🎉 no goals
                                                -/
                              /-
                                🎉 no goals
                              -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  npow n f :=
    /-
      M : Type uM
      N : Type uN
      P : Type uP
      Q : Type uQ
      inst✝¹ : MulOneClass M
      inst✝ : CommMonoid N
      f : MonoidHom M N
      ⊢ Eq ((fun n f => { toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯, map_mu …
    -/
    { toFun := fun x => f x ^ n, map_one' := by simp, map_mul' := fun x y => by simp [mul_pow] }
    /-
      case h
      M : Type uM
      N : Type uN
      P : Type uP
      Q : Type uQ
      inst✝¹ : MulOneClass M
      inst✝ : CommMonoid N
      f : MonoidHom M N
      x : M
      ⊢ Eq (((fun n f => { toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯, map_m …
    -/
  npow_zero f := by
    /-
      🎉 no goals
    -/
    ext x
    /-
      M : Type uM
      N : Type uN
      P : Type uP
      Q : Type uQ
      inst✝¹ : MulOneClass M
      inst✝ : CommMonoid N
      n : Nat
      f : MonoidHom M N
      ⊢ Eq ((fun n f => { toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯, map_mu …
    -/
    simp
    /-
      case h
      M : Type uM
      N : Type uN
      P : Type uP
      Q : Type uQ
      inst✝¹ : MulOneClass M
      inst✝ : CommMonoid N
      n : Nat
      f : MonoidHom M N
      x : M
      ⊢ Eq (((fun n f => { toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯, map_m …
    -/
  npow_succ n f := by
    /-
      🎉 no goals
    -/
    ext x
    simp [pow_succ]


/-- If `G` is a commutative group, then `M →* G` is a commutative group too. -/
@[to_additive "If `G` is an additive commutative group, then `M →+ G` is an additive commutative
      group too."]
instance MonoidHom.commGroup {M G} [MulOneClass M] [CommGroup G] : CommGroup (M →* G) :=
  { MonoidHom.commMonoid with
    inv := Inv.inv,
    div := Div.div,
    div_eq_mul_inv := by
      /-
        M✝ : Type uM
        N : Type uN
        P : Type uP
        Q : Type uQ
        M : Type ?u.2847
        G : Type ?u.2850
        inst✝¹ : MulOneClass M
        inst✝ : CommGroup G
        ⊢ ∀ (a b : MonoidHom M G), Eq (HDiv.hDiv a b) (HMul.hMul a (Inv.inv b))
      -/
      intros
      /-
        M✝ : Type uM
        N : Type uN
        P : Type uP
        Q : Type uQ
        M : Type ?u.2847
        G : Type ?u.2850
        inst✝¹ : MulOneClass M
        inst✝ : CommGroup G
        a✝ b✝ : MonoidHom M G
        ⊢ Eq (HDiv.hDiv a✝ b✝) (HMul.hMul a✝ (Inv.inv b✝))
      -/
      ext
      /-
        case h
        M✝ : Type uM
        N : Type uN
        P : Type uP
        Q : Type uQ
        M : Type ?u.2847
        G : Type ?u.2850
        inst✝¹ : MulOneClass M
        inst✝ : CommGroup G
        a✝ b✝ : MonoidHom M G
        x✝ : M
        ⊢ Eq ((HDiv.hDiv a✝ b✝) x✝) ((HMul.hMul a✝ (Inv.inv b✝)) x✝)
      -/
      apply div_eq_mul_inv,
      /-
        🎉 no goals
      -/
                         /-
                           M✝ : Type uM
                           N : Type uN
                           P : Type uP
                           Q : Type uQ
                           M : Type ?u.2847
                           G : Type ?u.2850
                           inst✝¹ : MulOneClass M
                           inst✝ : CommGroup G
                           ⊢ ∀ (a : MonoidHom M G), Eq (HMul.hMul (Inv.inv a) a) 1
                         -/
    inv_mul_cancel := by intros; ext; apply inv_mul_cancel,
                                      /-
                                        🎉 no goals
                                      -/
                       /-
                         M✝ : Type uM
                         N : Type uN
                         P : Type uP
                         Q : Type uQ
                         M : Type ?u.2847
                         G : Type ?u.2850
                         inst✝¹ : MulOneClass M
                         inst✝ : CommGroup G
                         n : Int
                         f : MonoidHom M G
                         ⊢ Eq ((fun x => HPow.hPow (f x) n) 1) 1
                       -/
    zpow := fun n f =>
                       /-
                         🎉 no goals
                       -/
                                  /-
                                    M✝ : Type uM
                                    N : Type uN
                                    P : Type uP
                                    Q : Type uQ
                                    M : Type ?u.2847
                                    G : Type ?u.2850
                                    inst✝¹ : MulOneClass M
                                    inst✝ : CommGroup G
                                    n : Int
                                    f : MonoidHom M G
                                    x y : M
                                    ⊢ Eq ({ toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯ }.toFun (HMul.hMul  …
                                  -/
      { toFun := fun x => f x ^ n,
                                  /-
                                    🎉 no goals
                                  -/
        map_one' := by simp,
      /-
        M✝ : Type uM
        N : Type uN
        P : Type uP
        Q : Type uQ
        M : Type ?u.2847
        G : Type ?u.2850
        inst✝¹ : MulOneClass M
        inst✝ : CommGroup G
        f : MonoidHom M G
        ⊢ Eq ((fun n f => { toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯, map_mu …
      -/
        map_mul' := fun x y => by simp [mul_zpow] },
      /-
        case h
        M✝ : Type uM
        N : Type uN
        P : Type uP
        Q : Type uQ
        M : Type ?u.2847
        G : Type ?u.2850
        inst✝¹ : MulOneClass M
        inst✝ : CommGroup G
        f : MonoidHom M G
        x : M
        ⊢ Eq (((fun n f => { toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯, map_m …
      -/
    zpow_zero' := fun f => by
      /-
        🎉 no goals
      -/
      ext x
      /-
        M✝ : Type uM
        N : Type uN
        P : Type uP
        Q : Type uQ
        M : Type ?u.2847
        G : Type ?u.2850
        inst✝¹ : MulOneClass M
        inst✝ : CommGroup G
        n : Nat
        f : MonoidHom M G
        ⊢ Eq ((fun n f => { toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯, map_mu …
      -/
      simp,
      /-
        case h
        M✝ : Type uM
        N : Type uN
        P : Type uP
        Q : Type uQ
        M : Type ?u.2847
        G : Type ?u.2850
        inst✝¹ : MulOneClass M
        inst✝ : CommGroup G
        n : Nat
        f : MonoidHom M G
        x : M
        ⊢ Eq (((fun n f => { toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯, map_m …
      -/
    zpow_succ' := fun n f => by
      /-
        🎉 no goals
      -/
      ext x
      /-
        M✝ : Type uM
        N : Type uN
        P : Type uP
        Q : Type uQ
        M : Type ?u.2847
        G : Type ?u.2850
        inst✝¹ : MulOneClass M
        inst✝ : CommGroup G
        n : Nat
        f : MonoidHom M G
        ⊢ Eq ((fun n f => { toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯, map_mu …
      -/
      simp [zpow_add_one],
      /-
        case h
        M✝ : Type uM
        N : Type uN
        P : Type uP
        Q : Type uQ
        M : Type ?u.2847
        G : Type ?u.2850
        inst✝¹ : MulOneClass M
        inst✝ : CommGroup G
        n : Nat
        f : MonoidHom M G
        x : M
        ⊢ Eq (((fun n f => { toFun := fun x => HPow.hPow (f x) n, map_one' := ⋯, map_m …
      -/
    zpow_neg' := fun n f => by
      /-
        🎉 no goals
      -/
      ext x
      simp [zpow_natCast, -Int.natCast_add] }


instance AddMonoid.End.instAddCommMonoid [AddCommMonoid M] : AddCommMonoid (AddMonoid.End M) :=
  AddMonoidHom.addCommMonoid


@[simp]
theorem AddMonoid.End.zero_apply [AddCommMonoid M] (m : M) : (0 : AddMonoid.End M) m = 0 :=
  rfl

-- Note: `@[simp]` omitted because `(1 : AddMonoid.End M) = id` by `AddMonoid.coe_one`

theorem AddMonoid.End.one_apply [AddCommMonoid M] (m : M) : (1 : AddMonoid.End M) m = m :=
  rfl


instance AddMonoid.End.instAddCommGroup [AddCommGroup M] : AddCommGroup (AddMonoid.End M) :=
  AddMonoidHom.addCommGroup


instance AddMonoid.End.instIntCast [AddCommGroup M] : IntCast (AddMonoid.End M) :=
  { intCast := fun z => z • (1 : AddMonoid.End M) }


/-- See also `AddMonoid.End.intCast_def`. -/
@[simp]
theorem AddMonoid.End.intCast_apply [AddCommGroup M] (z : ℤ) (m : M) :
    (↑z : AddMonoid.End M) m = z • m :=
  rfl


@[deprecated (since := "2024-04-17")]
alias AddMonoid.End.int_cast_apply := AddMonoid.End.intCast_apply


@[to_additive (attr := simp)] lemma MonoidHom.pow_apply {M N : Type*} [MulOneClass M]
    [CommMonoid N] (f : M →* N) (n : ℕ) (x : M) :
    (f ^ n) x = (f x) ^ n :=
  rfl


@[to_additive]
theorem ext_iff₂ {_ : MulOneClass M} {_ : MulOneClass N} {_ : CommMonoid P} {f g : M →* N →* P} :
    f = g ↔ ∀ x y, f x y = g x y :=
  DFunLike.ext_iff.trans <| forall_congr' fun _ => DFunLike.ext_iff


/-- `flip` arguments of `f : M →* N →* P` -/
@[to_additive "`flip` arguments of `f : M →+ N →+ P`"]
def flip {mM : MulOneClass M} {mN : MulOneClass N} {mP : CommMonoid P} (f : M →* N →* P) :
    N →* M →* P where
  toFun y :=
    { toFun := fun x => f x y,
                     /-
                       M : Type uM
                       N : Type uN
                       P : Type uP
                       Q : Type uQ
                       mM : MulOneClass M
                       mN : MulOneClass N
                       mP : CommMonoid P
                       f : MonoidHom M (MonoidHom N P)
                       y : N
                       ⊢ Eq ((fun x => (f x) y) 1) 1
                     -/
      map_one' := by simp [f.map_one, one_apply],
                     /-
                       🎉 no goals
                     -/
                                  /-
                                    M : Type uM
                                    N : Type uN
                                    P : Type uP
                                    Q : Type uQ
                                    mM : MulOneClass M
                                    mN : MulOneClass N
                                    mP : CommMonoid P
                                    f : MonoidHom M (MonoidHom N P)
                                    y : N
                                    x₁ x₂ : M
                                    ⊢ Eq ({ toFun := fun x => (f x) y, map_one' := ⋯ }.toFun (HMul.hMul x₁ x₂)) (H …
                                  -/
      map_mul' := fun x₁ x₂ => by simp [f.map_mul, mul_apply] }
                                  /-
                                    🎉 no goals
                                  -/
  map_one' := ext fun x => (f x).map_one
  map_mul' y₁ y₂ := ext fun x => (f x).map_mul y₁ y₂


@[to_additive (attr := simp)]
theorem flip_apply {_ : MulOneClass M} {_ : MulOneClass N} {_ : CommMonoid P} (f : M →* N →* P)
    (x : M) (y : N) : f.flip y x = f x y :=
  rfl


@[to_additive]
theorem map_one₂ {_ : MulOneClass M} {_ : MulOneClass N} {_ : CommMonoid P} (f : M →* N →* P)
    (n : N) : f 1 n = 1 :=
  (flip f n).map_one


@[to_additive]
theorem map_mul₂ {_ : MulOneClass M} {_ : MulOneClass N} {_ : CommMonoid P} (f : M →* N →* P)
    (m₁ m₂ : M) (n : N) : f (m₁ * m₂) n = f m₁ n * f m₂ n :=
  (flip f n).map_mul _ _


@[to_additive]
theorem map_inv₂ {_ : Group M} {_ : MulOneClass N} {_ : CommGroup P} (f : M →* N →* P) (m : M)
    (n : N) : f m⁻¹ n = (f m n)⁻¹ :=
  (flip f n).map_inv _


@[to_additive]
theorem map_div₂ {_ : Group M} {_ : MulOneClass N} {_ : CommGroup P} (f : M →* N →* P)
    (m₁ m₂ : M) (n : N) : f (m₁ / m₂) n = f m₁ n / f m₂ n :=
  (flip f n).map_div _ _


/-- Evaluation of a `MonoidHom` at a point as a monoid homomorphism. See also `MonoidHom.apply`
for the evaluation of any function at a point. -/
@[to_additive (attr := simps!)
      "Evaluation of an `AddMonoidHom` at a point as an additive monoid homomorphism.
      See also `AddMonoidHom.apply` for the evaluation of any function at a point."]
def eval [MulOneClass M] [CommMonoid N] : M →* (M →* N) →* N :=
  (MonoidHom.id (M →* N)).flip


/-- The expression `fun g m ↦ g (f m)` as a `MonoidHom`.
Equivalently, `(fun g ↦ MonoidHom.comp g f)` as a `MonoidHom`. -/
@[to_additive (attr := simps!)
      "The expression `fun g m ↦ g (f m)` as an `AddMonoidHom`.
      Equivalently, `(fun g ↦ AddMonoidHom.comp g f)` as an `AddMonoidHom`.

      This also exists in a `LinearMap` version, `LinearMap.lcomp`."]
def compHom' [MulOneClass M] [MulOneClass N] [CommMonoid P] (f : M →* N) : (N →* P) →* M →* P :=
  flip <| eval.comp f


/-- Composition of monoid morphisms (`MonoidHom.comp`) as a monoid morphism.

Note that unlike `MonoidHom.comp_hom'` this requires commutativity of `N`. -/
@[to_additive (attr := simps)
      "Composition of additive monoid morphisms (`AddMonoidHom.comp`) as an additive
      monoid morphism.

      Note that unlike `AddMonoidHom.comp_hom'` this requires commutativity of `N`.

      This also exists in a `LinearMap` version, `LinearMap.llcomp`."]
def compHom [MulOneClass M] [CommMonoid N] [CommMonoid P] :
    (N →* P) →* (M →* N) →* M →* P where
  toFun g := { toFun := g.comp, map_one' := comp_one g, map_mul' := comp_mul g }
  map_one' := by
    /-
      M : Type uM
      N : Type uN
      P : Type uP
      Q : Type uQ
      inst✝² : MulOneClass M
      inst✝¹ : CommMonoid N
      inst✝ : CommMonoid P
      ⊢ Eq ((fun g => { toFun := g.comp, map_one' := ⋯, map_mul' := ⋯ }) 1) 1
    -/
    ext1 f
    /-
      case h
      M : Type uM
      N : Type uN
      P : Type uP
      Q : Type uQ
      inst✝² : MulOneClass M
      inst✝¹ : CommMonoid N
      inst✝ : CommMonoid P
      f : MonoidHom M N
      ⊢ Eq (((fun g => { toFun := g.comp, map_one' := ⋯, map_mul' := ⋯ }) 1) f) (1 f)
    -/
    exact one_comp f
    /-
      🎉 no goals
    -/
  map_mul' g₁ g₂ := by
    /-
      M : Type uM
      N : Type uN
      P : Type uP
      Q : Type uQ
      inst✝² : MulOneClass M
      inst✝¹ : CommMonoid N
      inst✝ : CommMonoid P
      g₁ g₂ : MonoidHom N P
      ⊢ Eq ({ toFun := fun g => { toFun := g.comp, map_one' := ⋯, map_mul' := ⋯ }, m …
    -/
    ext1 f
    /-
      case h
      M : Type uM
      N : Type uN
      P : Type uP
      Q : Type uQ
      inst✝² : MulOneClass M
      inst✝¹ : CommMonoid N
      inst✝ : CommMonoid P
      g₁ g₂ : MonoidHom N P
      f : MonoidHom M N
      ⊢ Eq (({ toFun := fun g => { toFun := g.comp, map_one' := ⋯, map_mul' := ⋯ },  …
    -/
    exact mul_comp g₁ g₂ f
    /-
      🎉 no goals
    -/


/-- Flipping arguments of monoid morphisms (`MonoidHom.flip`) as a monoid morphism. -/
@[to_additive (attr := simps)
      "Flipping arguments of additive monoid morphisms (`AddMonoidHom.flip`)
      as an additive monoid morphism."]
def flipHom {_ : MulOneClass M} {_ : MulOneClass N} {_ : CommMonoid P} :
    (M →* N →* P) →* N →* M →* P where
  toFun := MonoidHom.flip
  map_one' := rfl
  map_mul' _ _ := rfl


/-- The expression `fun m q ↦ f m (g q)` as a `MonoidHom`.

Note that the expression `fun q n ↦ f (g q) n` is simply `MonoidHom.comp`. -/
@[to_additive
      "The expression `fun m q ↦ f m (g q)` as an `AddMonoidHom`.

      Note that the expression `fun q n ↦ f (g q) n` is simply `AddMonoidHom.comp`.

      This also exists as a `LinearMap` version, `LinearMap.compl₂`"]
def compl₂ [MulOneClass M] [MulOneClass N] [CommMonoid P] [MulOneClass Q] (f : M →* N →* P)
    (g : Q →* N) : M →* Q →* P :=
  (compHom' g).comp f


@[to_additive (attr := simp)]
theorem compl₂_apply [MulOneClass M] [MulOneClass N] [CommMonoid P] [MulOneClass Q]
    (f : M →* N →* P) (g : Q →* N) (m : M) (q : Q) : (compl₂ f g) m q = f m (g q) :=
  rfl


/-- The expression `fun m n ↦ g (f m n)` as a `MonoidHom`. -/
@[to_additive
      "The expression `fun m n ↦ g (f m n)` as an `AddMonoidHom`.

      This also exists as a `LinearMap` version, `LinearMap.compr₂`"]
def compr₂ [MulOneClass M] [MulOneClass N] [CommMonoid P] [CommMonoid Q] (f : M →* N →* P)
    (g : P →* Q) : M →* N →* Q :=
  (compHom g).comp f


@[to_additive (attr := simp)]
theorem compr₂_apply [MulOneClass M] [MulOneClass N] [CommMonoid P] [CommMonoid Q] (f : M →* N →* P)
    (g : P →* Q) (m : M) (n : N) : (compr₂ f g) m n = g (f m n) :=
  rfl


